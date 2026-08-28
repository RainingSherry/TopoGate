"""Small reconstruction probe used identically by every V26 corruption arm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import corruption, protocol
from .data import SparseDataset


@dataclass(frozen=True)
class FitResult:
    embedding: np.ndarray
    history: list[float]
    batch_size: int
    peak_allocated_mib: float
    peak_reserved_mib: float
    mask_audit: dict[str, Any]


def _imports() -> tuple[Any, Any]:
    import torch
    from torch import nn

    return torch, nn


def _make_model(input_dim: int, device: Any) -> tuple[Any, Any]:
    torch, nn = _imports()
    model = nn.Sequential(
        nn.Linear(input_dim, protocol.HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(protocol.HIDDEN_DIM, protocol.LATENT_DIM),
        nn.ReLU(),
        nn.Linear(protocol.LATENT_DIM, protocol.HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(protocol.HIDDEN_DIM, input_dim),
    ).to(device)
    encoder = nn.Sequential(*list(model.children())[:4]).to(device)
    return model, encoder


def select_batch_size(dataset: SparseDataset, device: Any, seed: int) -> tuple[int, dict[str, Any]]:
    """Use an outcome-independent forward/backward preflight to select batch size."""
    torch, _ = _imports()
    for batch_size in protocol.BATCH_CANDIDATES:
        if batch_size > dataset.n_samples:
            continue
        model = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.manual_seed(seed)
            model, _ = _make_model(dataset.n_features, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=protocol.LEARNING_RATE)
            indices = np.arange(batch_size, dtype=np.int64)
            clean = dataset.x[indices].toarray().astype(np.float32, copy=False)
            xb = torch.from_numpy(clean).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - xb) ** 2)
            loss.backward()
            optimizer.step()
            peak_reserved = float(torch.cuda.max_memory_reserved(device) / (1024**2))
            peak_allocated = float(torch.cuda.max_memory_allocated(device) / (1024**2))
            del xb, optimizer, model
            torch.cuda.empty_cache()
            return batch_size, {"peak_reserved_mib": peak_reserved, "peak_allocated_mib": peak_allocated, "status": "completed_valid"}
        except RuntimeError as exc:
            if model is not None:
                del model
            torch.cuda.empty_cache()
            if "out of memory" not in str(exc).lower():
                raise
    raise RuntimeError("V26 preflight OOM at every frozen batch candidate")


def fit(
    dataset: SparseDataset,
    *,
    arm: str,
    seed: int,
    device: Any,
    epochs: int,
    batch_size: int,
    oracle: corruption.OracleScores | None,
) -> FitResult:
    torch, _ = _imports()
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)
    model, encoder = _make_model(dataset.n_features, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol.LEARNING_RATE)
    history: list[float] = []
    cumulative = {"pair_count_total": 0, "changed_coordinate_total": 0, "support_crossing_total": 0}
    for epoch in range(int(epochs)):
        rng = np.random.default_rng(int(seed) + 1009 * epoch)
        order = rng.permutation(dataset.n_samples)
        epoch_loss = 0.0
        epoch_count = 0
        model.train()
        for start in range(0, dataset.n_samples, batch_size):
            indices = order[start : start + batch_size]
            clean = dataset.x[indices].toarray().astype(np.float32, copy=False)
            corrupted, audit = corruption.corrupt_batch(clean, indices, arm=arm, seed=seed, epoch=epoch, oracle=oracle)
            for key in cumulative:
                cumulative[key] += int(audit.get(key, 0))
            xb = torch.from_numpy(corrupted).to(device)
            target = torch.from_numpy(clean).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - target) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * int(indices.size)
            epoch_count += int(indices.size)
        history.append(epoch_loss / max(epoch_count, 1))
    model.eval()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, dataset.n_samples, max(batch_size, 256)):
            clean = dataset.x[start : start + max(batch_size, 256)].toarray().astype(np.float32, copy=False)
            embeddings.append(encoder(torch.from_numpy(clean).to(device)).detach().cpu().numpy())
    result = FitResult(
        embedding=np.concatenate(embeddings, axis=0),
        history=history,
        batch_size=int(batch_size),
        peak_allocated_mib=float(torch.cuda.max_memory_allocated(device) / (1024**2)),
        peak_reserved_mib=float(torch.cuda.max_memory_reserved(device) / (1024**2)),
        mask_audit={**cumulative, "value_multiset_preserved": True},
    )
    del model, encoder, optimizer
    torch.cuda.empty_cache()
    return result
