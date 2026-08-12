from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class SVDKNNGraph:
    indices: np.ndarray
    adjacency: sp.csr_matrix
    embedding: np.ndarray
    profile: dict[str, Any]


def build_svd_knn_graph(
    X_graph: sp.spmatrix,
    *,
    neighbor_k: int,
    svd_target: float,
    svd_min_dim: int,
    svd_max_dim: int,
    seed: int,
) -> SVDKNNGraph:
    matrix = sp.csr_matrix(X_graph, dtype=np.float32)
    n_samples, n_features = matrix.shape
    if n_samples <= 1:
        return SVDKNNGraph(
            np.zeros((n_samples, 0), dtype=np.int64),
            sp.csr_matrix((n_samples, n_samples)),
            np.zeros((n_samples, 0), dtype=np.float32),
            {"neighbor_k": 0, "svd_dim": 0, "variance_target_reached": False, "self_edges": 0},
        )
    max_dim = max(1, min(int(svd_max_dim), n_features, n_samples - 1))
    min_dim = min(max(1, int(svd_min_dim)), max_dim)
    if max_dim == 1:
        embedding = matrix.toarray().astype(np.float32, copy=False)
        svd_dim = 1
        target_reached = False
        explained = []
    else:
        svd = TruncatedSVD(n_components=max_dim, random_state=int(seed))
        full = svd.fit_transform(matrix).astype(np.float32, copy=False)
        ratios = np.nan_to_num(svd.explained_variance_ratio_, nan=0.0)
        cumulative = np.cumsum(ratios)
        eligible = np.flatnonzero(cumulative >= float(svd_target))
        svd_dim = max(min_dim, int(eligible[0] + 1)) if eligible.size else max_dim
        svd_dim = min(svd_dim, max_dim)
        target_reached = bool(cumulative[svd_dim - 1] >= float(svd_target))
        embedding = full[:, :svd_dim]
        explained = ratios[:svd_dim].tolist()
    embedding = normalize(np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0), axis=1).astype(np.float32)
    k_eff = min(int(neighbor_k), n_samples - 1)
    nearest = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
    nearest.fit(embedding)
    distances, raw_indices = nearest.kneighbors(embedding)
    indices = np.empty((n_samples, k_eff), dtype=np.int64)
    neighbor_distances = np.empty((n_samples, k_eff), dtype=np.float32)
    for row in range(n_samples):
        keep = raw_indices[row] != row
        row_indices = raw_indices[row][keep][:k_eff]
        row_distances = distances[row][keep][:k_eff]
        if row_indices.size != k_eff:
            raise RuntimeError("kNN query did not return enough non-self neighbours")
        indices[row] = row_indices
        neighbor_distances[row] = row_distances
    rows = np.repeat(np.arange(n_samples, dtype=np.int64), k_eff)
    cols = indices.reshape(-1)
    values = np.full(rows.shape, 1.0 / float(k_eff), dtype=np.float32)
    adjacency = sp.csr_matrix((values, (rows, cols)), shape=(n_samples, n_samples))
    return SVDKNNGraph(
        indices=indices,
        adjacency=adjacency,
        embedding=embedding,
        profile={
            "neighbor_k_requested": int(neighbor_k),
            "neighbor_k_effective": int(k_eff),
            "svd_target": float(svd_target),
            "svd_min_dim": int(min_dim),
            "svd_max_dim": int(max_dim),
            "svd_dim": int(svd_dim),
            "variance_target_reached": bool(target_reached),
            "explained_variance_ratio_sum": float(sum(explained)),
            "mean_neighbor_cosine_distance": float(np.mean(neighbor_distances)),
            "self_edges": int(np.count_nonzero(adjacency.diagonal())),
            "label_leakage_diagnostic": False,
        },
    )


def compute_topology_statistics(
    X_model: np.ndarray,
    graph: SVDKNNGraph,
    *,
    support_matrix: sp.spmatrix | np.ndarray | None = None,
    block_size: int = 512,
    cache_dir: str | Path | None = None,
    cache_dtype: str = "float32",
    clip: float = 5.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return [sample, feature, (deviation, dispersion, support, stability)]."""
    X = np.asarray(X_model, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != graph.indices.shape[0]:
        raise ValueError("X_model and graph sample counts must match")
    n_samples, n_features = X.shape
    if cache_dir is None:
        stats: np.ndarray = np.empty((n_samples, n_features, 4), dtype=np.float32)
        storage_profile: dict[str, Any] = {"storage": "memory"}
    else:
        root = Path(cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "topology_statistics.dat"
        stats = np.memmap(path, mode="w+", dtype=np.dtype(cache_dtype), shape=(n_samples, n_features, 4))
        storage_profile = {"storage": "memmap", "path": str(path.resolve()), "dtype": cache_dtype}
    support = None if support_matrix is None else sp.csr_matrix(support_matrix)
    for start in range(0, n_features, int(block_size)):
        stop = min(n_features, start + int(block_size))
        block = X[:, start:stop]
        mean = graph.adjacency.dot(block)
        second = graph.adjacency.dot(block * block)
        dispersion = np.sqrt(np.maximum(second - mean * mean, 0.0)).astype(np.float32)
        deviation = np.abs(block - mean).astype(np.float32)
        if support is None:
            support_fraction = graph.adjacency.dot((np.abs(block) > 1e-8).astype(np.float32))
        else:
            support_block = support[:, start:stop]
            support_fraction = graph.adjacency.dot((support_block != 0).astype(np.float32))
            if sp.issparse(support_fraction):
                support_fraction = support_fraction.toarray()
        stability = 1.0 / (1.0 + np.maximum(dispersion, 0.0))
        values = np.stack([np.log1p(deviation), np.log1p(dispersion), support_fraction, stability], axis=2)
        # Keep support/stability interpretable in [0, 1], while standardising
        # the two unbounded topology residuals across samples per feature.
        residual = values[:, :, :2]
        mean_residual = residual.mean(axis=0, keepdims=True)
        std_residual = residual.std(axis=0, keepdims=True)
        values[:, :, :2] = np.clip((residual - mean_residual) / np.maximum(std_residual, 1e-6), -float(clip), float(clip))
        stats[:, start:stop, :] = values.astype(np.dtype(cache_dtype), copy=False)
    if isinstance(stats, np.memmap):
        stats.flush()
    profile = storage_profile | {
        "shape": [int(n_samples), int(n_features), 4],
        "block_size": int(block_size),
        "clip": float(clip),
        "feature_names": ["deviation_z", "dispersion_z", "neighbor_support", "stability"],
        "support_is_label_free": True,
    }
    return stats, profile
