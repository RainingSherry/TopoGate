from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from .config import V23Config
from .data import PreparedSemanticInput
from .model import CycleAutoEncoder, LatentLinearDecoder


@dataclass(frozen=True)
class FitResult:
    model: CycleAutoEncoder
    linear_decoder: LatentLinearDecoder
    clean_embedding: np.ndarray
    history: list[dict[str, float]]
    linear_history: list[dict[str, float]]


def seed_runtime(seed: int, device: torch.device) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed))


def _semantic_corruption(
    clean_semantic: torch.Tensor,
    *,
    mask_ratio: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clean_semantic.shape[0] <= 1:
        return clean_semantic.clone(), torch.zeros_like(clean_semantic)
    offset = int(
        torch.randint(
            1,
            clean_semantic.shape[0],
            (1,),
            device=clean_semantic.device,
            generator=generator,
        ).item()
    )
    donor = torch.roll(clean_semantic, shifts=offset, dims=0)
    requested = torch.rand(
        clean_semantic.shape,
        device=clean_semantic.device,
        generator=generator,
    ) < float(mask_ratio)
    corrupted = torch.where(requested, donor, clean_semantic)
    effective = requested & corrupted.ne(clean_semantic)
    return corrupted, effective.to(dtype=clean_semantic.dtype)


def _to_model(semantic: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (semantic - mean[None, :]) / scale[None, :]


@torch.no_grad()
def extract_embedding(
    model: CycleAutoEncoder,
    model_view: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, model_view.shape[0], batch_size):
        batch = torch.as_tensor(model_view[start : start + batch_size], dtype=torch.float32, device=device)
        chunks.append(model.encode(batch).cpu().numpy())
    result = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def fit_backbone(
    prepared: PreparedSemanticInput,
    *,
    config: V23Config,
    seed: int,
    device: torch.device,
) -> FitResult:
    """Fit canonical scMAE and a frozen-encoder linear decoder without labels or K."""

    config.validate()
    seed_runtime(seed, device)
    semantic_tensor = torch.as_tensor(prepared.semantic, dtype=torch.float32)
    loader_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 101)
    loader = DataLoader(
        TensorDataset(semantic_tensor),
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        drop_last=False,
    )
    model = CycleAutoEncoder(
        num_genes=prepared.model.shape[1],
        hidden_size=config.hidden_size,
        masked_data_weight=config.masked_data_weight,
        mask_loss_weight=config.mask_loss_weight,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    mean = torch.as_tensor(prepared.preprocessor.mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(prepared.preprocessor.scale, dtype=torch.float32, device=device)
    corruption_generator = torch.Generator(device=device).manual_seed(int(seed) + 211)
    history: list[dict[str, float]] = []
    for epoch in range(int(config.epochs)):
        model.train()
        total = 0.0
        rec_total = 0.0
        mask_total = 0.0
        effective_total = 0.0
        batches = 0
        for (semantic_cpu,) in loader:
            clean_semantic = semantic_cpu.to(device=device)
            corrupted_semantic, effective = _semantic_corruption(
                clean_semantic,
                mask_ratio=config.training_mask_ratio,
                generator=corruption_generator,
            )
            clean_model = _to_model(clean_semantic, mean, scale)
            corrupted_model = _to_model(corrupted_semantic, mean, scale)
            _, loss, parts = model.loss_mask(corrupted_model, clean_model, effective, return_parts=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            rec_total += float(parts["reconstruction_loss"])
            mask_total += float(parts["mask_loss"])
            effective_total += float(effective.mean())
            batches += 1
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total / max(1, batches),
                "reconstruction_loss": rec_total / max(1, batches),
                "mask_loss": mask_total / max(1, batches),
                "effective_training_mask_rate": effective_total / max(1, batches),
            }
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    linear_decoder = LatentLinearDecoder(config.hidden_size, prepared.model.shape[1]).to(device)
    linear_optimizer = torch.optim.Adam(
        linear_decoder.parameters(),
        lr=config.latent_linear_learning_rate,
    )
    linear_generator = torch.Generator(device=device).manual_seed(int(seed) + 307)
    linear_history: list[dict[str, float]] = []
    for epoch in range(int(config.latent_linear_epochs)):
        linear_decoder.train()
        total = 0.0
        batches = 0
        for (semantic_cpu,) in loader:
            clean_semantic = semantic_cpu.to(device=device)
            corrupted_semantic, effective = _semantic_corruption(
                clean_semantic,
                mask_ratio=config.training_mask_ratio,
                generator=linear_generator,
            )
            clean_model = _to_model(clean_semantic, mean, scale)
            corrupted_model = _to_model(corrupted_semantic, mean, scale)
            with torch.no_grad():
                latent = model.encode(corrupted_model)
            reconstruction = linear_decoder(latent)
            raw = functional.mse_loss(reconstruction, clean_model, reduction="none")
            weights = effective * config.masked_data_weight + (1.0 - effective) * (1.0 - config.masked_data_weight)
            loss = (weights * raw).mean()
            linear_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            linear_optimizer.step()
            total += float(loss.detach())
            batches += 1
        linear_history.append({"epoch": float(epoch + 1), "loss": total / max(1, batches)})

    clean_embedding = extract_embedding(
        model,
        prepared.model,
        batch_size=config.profile_batch_size,
        device=device,
    )
    return FitResult(
        model=model,
        linear_decoder=linear_decoder,
        clean_embedding=clean_embedding,
        history=history,
        linear_history=linear_history,
    )


def checkpoint_payload(result: FitResult, config: V23Config, seed: int) -> dict[str, Any]:
    return {
        "model": result.model.state_dict(),
        "linear_decoder": result.linear_decoder.state_dict(),
        "config": config.to_dict(),
        "seed": int(seed),
        "labels_accessible_during_fit": False,
        "K_accessible_during_fit": False,
    }
