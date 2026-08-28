from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import V20Config
from .graph import build_svd_knn_graph, compute_topology_statistics
from .model import (
    FeatureGate,
    V20AutoEncoder,
    cyclic_donor,
    random_bernoulli_mask,
    random_topk_mask,
    straight_through_topk,
)


ALLOWED_PHYSICAL_GPUS = frozenset({1, 2, 3, 4, 5, 6})


def resolve_device(device: str | torch.device, gpu: int | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cpu":
        return torch.device("cpu")
    if gpu is not None and int(gpu) not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(f"physical GPU {gpu} is forbidden; allowed={sorted(ALLOWED_PHYSICAL_GPUS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda:0")


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _grad_norm(module: torch.nn.Module) -> float:
    values = [p.grad.detach().norm().item() for p in module.parameters() if p.grad is not None]
    return float(np.sqrt(np.sum(np.square(values)))) if values else 0.0


def _batch_indices(n_samples: int, batch_size: int, generator: torch.Generator, device: torch.device) -> list[torch.Tensor]:
    order = torch.randperm(n_samples, generator=generator, device=device)
    return [order[start : min(n_samples, start + batch_size)] for start in range(0, n_samples, batch_size)]


def _sample_random_mask(
    shape: tuple[int, int],
    *,
    k_mask: int,
    mask_ratio: float,
    mode: str,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if mode == "topk":
        return random_topk_mask(shape, k_mask, device=device, generator=generator)
    if mode == "bernoulli":
        return random_bernoulli_mask(shape, mask_ratio, device=device, generator=generator)
    raise ValueError(f"unsupported random mask mode: {mode}")


def _clean_embedding(model: V20AutoEncoder, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model.feature(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def fit_full(
    X_model: np.ndarray,
    X_graph: Any,
    *,
    config: V20Config,
    seed: int,
    device: str | torch.device,
    stats_cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit Full without labels, K, or benchmark metadata."""
    config.validate()
    seed_all(seed)
    runtime_device = torch.device(device)
    X_np = np.ascontiguousarray(np.asarray(X_model, dtype=np.float32))
    if X_np.ndim != 2 or X_np.shape[0] == 0 or X_np.shape[1] == 0:
        raise ValueError("X_model must be a non-empty 2D matrix")
    graph = build_svd_knn_graph(
        X_graph,
        neighbor_k=config.neighbor_k,
        svd_target=config.graph_svd_target,
        svd_min_dim=config.graph_svd_min_dim,
        svd_max_dim=config.graph_svd_max_dim,
        seed=seed,
    )
    stats, stats_profile = compute_topology_statistics(
        X_np,
        graph,
        block_size=config.stats_block_size,
        cache_dir=stats_cache_dir,
        cache_dtype=config.stats_cache_dtype,
        clip=config.stats_clip,
    )
    n_samples, n_features = X_np.shape
    k_mask = max(1, min(n_features, int(round(config.mask_ratio * n_features))))
    model = V20AutoEncoder(
        num_genes=n_features,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        masked_data_weight=config.masked_data_weight,
        mask_loss_weight=config.mask_loss_weight,
    ).to(runtime_device)
    gate = FeatureGate(config.gate_hidden).to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    gate_optimizer = torch.optim.Adam(gate.parameters(), lr=float(config.gate_lr))
    batch_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 101)
    mask_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 202)
    gate_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 303)
    history: list[dict[str, float]] = []
    global_step = 0
    gate_updates = 0
    gate_nonzero_updates = 0
    for epoch in range(config.epochs):
        model.train()
        gate.train()
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "mask_loss": 0.0, "requested_mask_rate": 0.0, "effective_mask_rate": 0.0, "gate_loss": 0.0}
        batches = 0
        for batch_ids in _batch_indices(n_samples, config.batch_size, batch_rng, runtime_device):
            row_ids = batch_ids.detach().cpu().numpy()
            batch = torch.as_tensor(X_np[row_ids], dtype=torch.float32, device=runtime_device)
            stat_batch = torch.as_tensor(np.asarray(stats[row_ids], dtype=np.float32), dtype=torch.float32, device=runtime_device)
            if epoch < config.warmup_epochs:
                requested = _sample_random_mask(
                    (batch.shape[0], n_features),
                    k_mask=k_mask,
                    mask_ratio=config.mask_ratio,
                    mode=config.random_mask_mode,
                    device=runtime_device,
                    generator=mask_rng,
                )
            else:
                _set_requires_grad(gate, False)
                with torch.no_grad():
                    gate_logits = gate(stat_batch)
                    _, requested = straight_through_topk(gate_logits, k_mask, generator=mask_rng, gumbel_scale=config.gumbel_scale, tau_ste=config.tau_ste)
                _set_requires_grad(gate, True)
            donor = cyclic_donor(batch, generator=mask_rng)
            changed = (donor != batch).to(dtype=batch.dtype)
            effective = requested * changed
            corrupted = batch + requested * (donor - batch)
            _set_requires_grad(model, True)
            optimizer.zero_grad(set_to_none=True)
            training_mask = effective if config.mask_target_mode == "effective" else requested
            _, parts = model.loss_encoder(corrupted, batch, training_mask)
            parts["loss"].backward()
            optimizer.step()
            totals["loss"] += float(parts["total_loss"].cpu())
            totals["reconstruction_loss"] += float(parts["reconstruction_loss"].cpu())
            totals["mask_loss"] += float(parts["mask_loss"].cpu())
            totals["requested_mask_rate"] += float(requested.mean().cpu())
            totals["effective_mask_rate"] += float(effective.mean().cpu())
            global_step += 1
            if epoch >= config.warmup_epochs and global_step % config.gate_update_every == 0:
                gate_updates += 1
                _set_requires_grad(model, False)
                _set_requires_grad(gate, True)
                gate_optimizer.zero_grad(set_to_none=True)
                gate_logits = gate(stat_batch)
                mask_st, _hard = straight_through_topk(gate_logits, k_mask, generator=gate_rng, gumbel_scale=config.gumbel_scale, tau_ste=config.tau_ste)
                donor_gate = cyclic_donor(batch, generator=gate_rng)
                changed_gate = (donor_gate != batch).to(dtype=batch.dtype)
                effective_st = mask_st * changed_gate
                corrupted_gate = batch + mask_st * (donor_gate - batch)
                _, _, reconstruction = model.forward_mask(corrupted_gate)
                rec_gate = model.reconstruction_loss(reconstruction, batch, effective_st)
                gate_loss = -rec_gate
                gate_loss.backward()
                gate_grad = _grad_norm(gate)
                if gate_grad > 0.0 and np.isfinite(gate_grad):
                    gate_nonzero_updates += 1
                gate_optimizer.step()
                _set_requires_grad(model, True)
                totals["gate_loss"] += float(gate_loss.detach().cpu())
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()} | {"epoch": float(epoch + 1)})
    embedding = _clean_embedding(model, X_np, config.batch_size, runtime_device)
    diagnostics = {
        "graph_profile": graph.profile,
        "stats_profile": stats_profile,
        "history": history,
        "requested_mask_rate": float(k_mask / n_features),
        "mask_target_mode": config.mask_target_mode,
        "random_mask_mode": config.random_mask_mode,
        "gate_updates": int(gate_updates),
        "gate_nonzero_update_rate": float(gate_nonzero_updates / max(1, gate_updates)),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "gate_parameter_count": int(sum(p.numel() for p in gate.parameters())),
        "labels_used_during_fit": False,
        "K_used_during_fit": False,
        "model": model,
        "gate": gate,
    }
    return embedding, diagnostics


def fit_scmae_only(
    X_model: np.ndarray,
    *,
    config: V20Config,
    seed: int,
    device: str | torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train the matched vanilla scMAE branch without graph or Gate state."""
    config.validate()
    if config.variant != "scmae_only":
        raise ValueError("fit_scmae_only requires variant='scmae_only'")
    seed_all(seed)
    runtime_device = torch.device(device)
    X_np = np.ascontiguousarray(np.asarray(X_model, dtype=np.float32))
    if X_np.ndim != 2 or X_np.shape[0] == 0 or X_np.shape[1] == 0:
        raise ValueError("X_model must be a non-empty 2D matrix")
    n_samples, n_features = X_np.shape
    k_mask = max(1, min(n_features, int(round(config.mask_ratio * n_features))))
    model = V20AutoEncoder(
        num_genes=n_features,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        masked_data_weight=config.masked_data_weight,
        mask_loss_weight=config.mask_loss_weight,
    ).to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    batch_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 101)
    mask_rng = torch.Generator(device=runtime_device).manual_seed(int(seed) + 202)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "mask_loss": 0.0,
            "requested_mask_rate": 0.0,
            "effective_mask_rate": 0.0,
        }
        batches = 0
        for batch_ids in _batch_indices(n_samples, config.batch_size, batch_rng, runtime_device):
            row_ids = batch_ids.detach().cpu().numpy()
            batch = torch.as_tensor(X_np[row_ids], dtype=torch.float32, device=runtime_device)
            requested = _sample_random_mask(
                (batch.shape[0], n_features),
                k_mask=k_mask,
                mask_ratio=config.mask_ratio,
                mode=config.random_mask_mode,
                device=runtime_device,
                generator=mask_rng,
            )
            donor = cyclic_donor(batch, generator=mask_rng)
            changed = (donor != batch).to(dtype=batch.dtype)
            effective = requested * changed
            corrupted = batch + requested * (donor - batch)
            optimizer.zero_grad(set_to_none=True)
            training_mask = effective if config.mask_target_mode == "effective" else requested
            _, parts = model.loss_encoder(corrupted, batch, training_mask)
            parts["loss"].backward()
            optimizer.step()
            totals["loss"] += float(parts["total_loss"].cpu())
            totals["reconstruction_loss"] += float(parts["reconstruction_loss"].cpu())
            totals["mask_loss"] += float(parts["mask_loss"].cpu())
            totals["requested_mask_rate"] += float(requested.mean().cpu())
            totals["effective_mask_rate"] += float(effective.mean().cpu())
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()} | {"epoch": float(epoch + 1)})
    embedding = _clean_embedding(model, X_np, config.batch_size, runtime_device)
    diagnostics = {
        "graph_profile": {"enabled": False, "reason": "scmae_only"},
        "stats_profile": {"enabled": False, "reason": "scmae_only"},
        "history": history,
        "requested_mask_rate": float(k_mask / n_features),
        "mask_target_mode": config.mask_target_mode,
        "random_mask_mode": config.random_mask_mode,
        "gate_updates": 0,
        "gate_nonzero_update_rate": 0.0,
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "gate_parameter_count": 0,
        "labels_used_during_fit": False,
        "K_used_during_fit": False,
        "model": model,
        "gate": None,
    }
    return embedding, diagnostics
