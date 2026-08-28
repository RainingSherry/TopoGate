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
        empty = np.zeros((n_samples, 0), dtype=np.int64)
        return SVDKNNGraph(
            empty,
            sp.csr_matrix((n_samples, n_samples)),
            np.zeros((n_samples, 0), dtype=np.float32),
            {"neighbor_k": 0, "svd_dim": 0, "variance_target_reached": False},
        )
    max_dim = max(1, min(int(svd_max_dim), n_features, n_samples - 1))
    min_dim = min(max(1, int(svd_min_dim)), max_dim)
    if max_dim == 1:
        embedding = matrix.toarray().astype(np.float32, copy=False)
        target_reached = False
        svd_dim = 1
        explained = []
    else:
        svd = TruncatedSVD(n_components=max_dim, random_state=int(seed))
        full_embedding = svd.fit_transform(matrix).astype(np.float32, copy=False)
        explained_values = np.nan_to_num(svd.explained_variance_ratio_, nan=0.0)
        cumulative = np.cumsum(explained_values)
        eligible = np.flatnonzero(cumulative >= float(svd_target))
        if eligible.size:
            svd_dim = max(min_dim, int(eligible[0] + 1))
            target_reached = bool(cumulative[svd_dim - 1] >= float(svd_target))
        else:
            svd_dim = max_dim
            target_reached = False
        embedding = full_embedding[:, :svd_dim]
        explained = explained_values[:svd_dim].tolist()
    embedding = normalize(np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0), axis=1).astype(np.float32)
    k_eff = min(int(neighbor_k), n_samples - 1)
    nearest = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
    nearest.fit(embedding)
    distances, raw_indices = nearest.kneighbors(embedding)
    # Ties between duplicate rows can place the query index after the first
    # returned neighbor.  Filter by identity explicitly so topology statistics
    # never include a sample's own value as a neighbor.
    indices = np.empty((n_samples, k_eff), dtype=np.int64)
    neighbor_distances = np.empty((n_samples, k_eff), dtype=np.float32)
    for row in range(n_samples):
        keep = raw_indices[row] != row
        row_indices = raw_indices[row][keep][:k_eff]
        row_distances = distances[row][keep][:k_eff]
        if row_indices.size != k_eff:
            raise RuntimeError("kNN query did not return enough non-self neighbors")
        indices[row] = row_indices.astype(np.int64, copy=False)
        neighbor_distances[row] = row_distances.astype(np.float32, copy=False)
    rows = np.repeat(np.arange(n_samples, dtype=np.int64), k_eff)
    cols = indices.reshape(-1)
    values = np.full(rows.shape, 1.0 / float(k_eff), dtype=np.float32)
    adjacency = sp.csr_matrix((values, (rows, cols)), shape=(n_samples, n_samples))
    profile = {
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
    }
    return SVDKNNGraph(indices, adjacency, embedding, profile)


def compute_topology_statistics(
    X_model: np.ndarray,
    graph: SVDKNNGraph,
    *,
    block_size: int,
    cache_dir: str | Path | None = None,
    cache_dtype: str = "float32",
    clip: float = 5.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute normalized [sample, feature, (deviation, dispersion)] statistics."""
    X = np.asarray(X_model, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != graph.indices.shape[0]:
        raise ValueError("X_model and graph sample counts must match")
    n_samples, n_features = X.shape
    shape = (n_samples, n_features, 2)
    if cache_dir is None:
        stats: np.ndarray = np.empty(shape, dtype=np.float32)
        cache_profile = {"storage": "memory"}
    else:
        root = Path(cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "topology_statistics.dat"
        stats = np.memmap(path, mode="w+", dtype=np.dtype(cache_dtype), shape=shape)
        cache_profile = {"storage": "memmap", "path": str(path.resolve()), "dtype": cache_dtype}
    for start in range(0, n_features, int(block_size)):
        stop = min(n_features, start + int(block_size))
        block = X[:, start:stop]
        mean = graph.adjacency.dot(block)
        second = graph.adjacency.dot(block * block)
        dispersion = np.sqrt(np.maximum(second - mean * mean, 0.0)).astype(np.float32)
        deviation = np.abs(block - mean).astype(np.float32)
        stats[:, start:stop, 0] = deviation
        stats[:, start:stop, 1] = dispersion
    for start in range(0, n_features, int(block_size)):
        stop = min(n_features, start + int(block_size))
        values = np.log1p(np.maximum(np.asarray(stats[:, start:stop, :], dtype=np.float32), 0.0))
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        normalized_values = (values - mean) / np.maximum(std, 1e-6)
        stats[:, start:stop, :] = np.clip(normalized_values, -float(clip), float(clip)).astype(np.dtype(cache_dtype))
    if isinstance(stats, np.memmap):
        stats.flush()
    cache_profile.update({"shape": list(shape), "block_size": int(block_size), "clip": float(clip)})
    return stats, cache_profile
