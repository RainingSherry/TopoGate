"""Unsupervised diagnostics for TopoGate V0.

These helpers never consume benchmark labels or a requested number of clusters.
They are engineering diagnostics and are kept separate from the clustering
readout so they cannot accidentally influence training or variant selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, TensorDataset

from .graph import NeighborGraph


def embedding_geometry(embedding: np.ndarray) -> dict[str, float]:
    """Summarize finite latent geometry without labels."""

    values = np.asarray(embedding, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embedding must be two-dimensional")
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "n_samples": int(finite.shape[0]),
        "n_features": int(finite.shape[1]),
        "mean_abs": float(np.mean(np.abs(finite))) if finite.size else 0.0,
        "feature_std_mean": float(np.mean(np.std(finite, axis=0))) if finite.size else 0.0,
        "sample_norm_mean": float(np.mean(np.linalg.norm(finite, axis=1))) if finite.size else 0.0,
        "sample_norm_p95": float(np.percentile(np.linalg.norm(finite, axis=1), 95))
        if finite.size
        else 0.0,
        "finite": bool(np.all(np.isfinite(values))),
    }


def neighbor_overlap(reference_indices: np.ndarray, embedding: np.ndarray) -> float:
    """Measure overlap between a reference graph and latent kNN neighbours."""

    reference = np.asarray(reference_indices, dtype=np.int64)
    values = np.asarray(embedding, dtype=np.float32)
    if reference.ndim != 2 or reference.shape[1] == 0 or values.shape[0] < 2:
        return 0.0
    k = min(int(reference.shape[1]), values.shape[0] - 1)
    normalized = normalize(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    nearest = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(normalized)
    _, indices = nearest.kneighbors(normalized)
    overlap = []
    for row in range(values.shape[0]):
        candidate = [int(value) for value in indices[row] if int(value) != row][:k]
        overlap.append(len(set(candidate).intersection(reference[row, :k].tolist())) / float(k))
    return float(np.mean(overlap)) if overlap else 0.0


@torch.no_grad()
def evaluate_unsupervised_views(
    *,
    model: torch.nn.Module,
    data_np: np.ndarray,
    clean_embedding: np.ndarray,
    graph: NeighborGraph,
    batch_size: int,
    mask_ratio: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate deterministic masked views and latent stability without labels."""

    from .corruption import apply_scmae_noise

    data = np.ascontiguousarray(np.asarray(data_np, dtype=np.float32))
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("data_np must be a non-empty matrix")
    tensor = torch.as_tensor(data, dtype=torch.float32)
    view_cosines: list[float] = []
    total_loss = 0.0
    total_count = 0
    for view_id in range(2):
        generator = torch.Generator(device=device.type)
        generator.manual_seed(int(seed) + 100_003 + view_id * 7_919)
        latent_rows: list[np.ndarray] = []
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            drop_last=False,
        )
        offset = 0
        for (clean_cpu,) in loader:
            clean = clean_cpu.to(device)
            # A view is generated from its own row permutation, keeping the
            # diagnostic independent of the training streams.
            replacement = clean[torch.randperm(clean.shape[0], device=device, generator=generator)]
            selected = torch.rand(
                clean.shape, device=device, generator=generator
            ) < float(mask_ratio)
            corrupted = torch.where(selected, replacement, clean)
            mask = (corrupted != clean).to(dtype=clean.dtype)
            latent, loss, _parts = model.loss_mask_weighted(corrupted, clean, mask)
            latent_rows.append(latent.detach().cpu().numpy().astype(np.float32, copy=False))
            total_loss += float(loss.detach().cpu()) * int(clean.shape[0])
            total_count += int(clean.shape[0])
            offset += int(clean.shape[0])
        view_embedding = np.concatenate(latent_rows, axis=0)
        clean_norm = np.linalg.norm(clean_embedding, axis=1)
        view_norm = np.linalg.norm(view_embedding, axis=1)
        cosine = np.sum(clean_embedding * view_embedding, axis=1) / np.clip(
            clean_norm * view_norm, 1e-8, None
        )
        view_cosines.append(float(np.mean(np.nan_to_num(cosine, nan=0.0))))

    return {
        # ``total_count`` already includes both deterministic views.  Dividing
        # by two again would report exactly half of the mean per-view loss.
        "eval_mask_loss": float(total_loss / max(1, total_count)),
        "latent_view_cosine_mean": float(np.mean(view_cosines)) if view_cosines else 0.0,
        "latent_view_cosine_std": float(np.std(view_cosines)) if view_cosines else 0.0,
        "input_neighbor_overlap": neighbor_overlap(graph.indices, clean_embedding),
        "latent_mean_feature_std": float(np.mean(np.std(clean_embedding, axis=0)))
        if clean_embedding.size
        else 0.0,
    }


__all__ = ["embedding_geometry", "evaluate_unsupervised_views", "neighbor_overlap"]
