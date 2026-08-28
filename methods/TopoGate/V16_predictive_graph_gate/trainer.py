from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from .config import V16Config
from .model import SparseCountMAE, SphericalPrototypeHead, masked_poisson_loss
from .sparse import PreparedCounts


@dataclass
class StageAResult:
    model: SparseCountMAE
    head: SphericalPrototypeHead
    embedding: np.ndarray
    probabilities: np.ndarray
    history: list[dict[str, float]]
    seconds: float


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(config: V16Config) -> torch.device:
    if config.no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    if config.gpu in {0, 7}:
        raise ValueError("V16 cannot use physical GPU 0 or 7")
    return torch.device(f"cuda:{int(config.gpu)}")


def _mask_batch(
    x: torch.Tensor,
    ratio: float,
    zero_ratio: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observed = x > 0.0
    random_values = torch.rand(x.shape, device=x.device, generator=generator)
    mask = observed & (random_values < float(ratio))
    for row in torch.where(observed.sum(dim=1).gt(0) & mask.sum(dim=1).eq(0))[0].tolist():
        positions = torch.where(observed[row])[0]
        chosen = torch.randint(positions.numel(), (1,), device=x.device, generator=generator)
        mask[row, positions[chosen]] = True
    corrupted = x.masked_fill(mask, 0.0)
    zero_mask = (~observed) & (torch.rand(x.shape, device=x.device, generator=generator) < float(zero_ratio))
    return corrupted, mask, zero_mask


def _encode_all(
    prepared: PreparedCounts,
    model: SparseCountMAE,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, prepared.n_samples, int(batch_size)):
            indices = np.arange(start, min(start + int(batch_size), prepared.n_samples), dtype=np.int64)
            x = torch.as_tensor(prepared.rows(indices), dtype=torch.float32, device=device)
            chunks.append(model.encode(x).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, model.encoder[-1].out_features), dtype=np.float32)


def train_stage_a(
    prepared: PreparedCounts,
    n_clusters: int,
    config: V16Config,
    device: torch.device,
) -> StageAResult:
    """Train topology-disabled sparse count MAE and freeze its assignment head."""
    set_seed(config.seed)
    model = SparseCountMAE(prepared.n_features, config.hidden_dim, config.latent_dim, config.dropout).to(device)
    head = SphericalPrototypeHead(n_clusters, config.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.seed) + 17)
    order_rng = np.random.default_rng(int(config.seed) + 19)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(int(config.epochs)):
        model.train()
        order = order_rng.permutation(prepared.n_samples)
        losses: list[float] = []
        for start in range(0, prepared.n_samples, int(config.batch_size)):
            batch_indices = order[start : start + int(config.batch_size)]
            x = torch.as_tensor(prepared.rows(batch_indices), dtype=torch.float32, device=device)
            target = torch.as_tensor(prepared.raw_rows(batch_indices), dtype=torch.float32, device=device)
            exposure = target.sum(dim=1, keepdim=True).clamp_min(1.0).log()
            corrupted, mask, zero_mask = _mask_batch(x, config.mask_ratio, config.zero_sample_ratio, generator)
            z = model.encode(corrupted)
            log_rate = model.decode(z) + exposure
            loss = masked_poisson_loss(log_rate, target, mask, zero_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses)) if losses else 0.0})
    embedding = _encode_all(prepared, model, device, config.batch_size)
    head.initialise(embedding, seed=config.seed, n_init=config.n_init)
    model.eval()
    head.eval()
    with torch.no_grad():
        probabilities = head(torch.as_tensor(embedding, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float32)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return StageAResult(
        model=model,
        head=head,
        embedding=embedding,
        probabilities=probabilities,
        history=history,
        seconds=float(time.perf_counter() - started),
    )
