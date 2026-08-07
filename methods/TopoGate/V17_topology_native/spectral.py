from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class SpectralResult:
    labels: np.ndarray
    embedding: np.ndarray
    abstained: np.ndarray
    profile: dict[str, Any]


def normalized_spectral_readout(
    affinity: sp.spmatrix,
    n_clusters: int,
    *,
    seed: int,
    n_init: int,
    degree_epsilon: float,
) -> SpectralResult:
    """Partition the supported graph and mark degree-zero samples as abstained."""
    matrix = sp.csr_matrix(affinity, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("affinity must be square")
    if int(n_clusters) <= 0:
        raise ValueError("n_clusters must be positive")
    if matrix.data.size and np.min(matrix.data) < 0.0:
        raise ValueError("affinity must be non-negative")
    matrix = (0.5 * (matrix + matrix.transpose())).tocsr()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    n = int(matrix.shape[0])
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    active = degree > float(degree_epsilon)
    abstained = ~active
    labels = np.full(n, -1, dtype=np.int64)
    embedding = np.zeros((n, int(n_clusters)), dtype=np.float32)
    active_count = int(active.sum())
    if active_count == 0:
        return SpectralResult(
            labels,
            embedding,
            abstained,
            {
                "status": "all_abstained",
                "active_nodes": 0,
                "abstained_nodes": int(n),
                "abstention_rate": 1.0 if n else 0.0,
                "connected_components": 0,
                "K_used_only_in_readout": True,
            },
        )
    if active_count < int(n_clusters):
        raise ValueError("fewer topology-supported samples than requested clusters")
    active_affinity = matrix[active][:, active]
    component_count, _ = connected_components(active_affinity, directed=False, return_labels=True)
    if int(n_clusters) == 1:
        active_embedding = np.ones((active_count, 1), dtype=np.float32)
        active_labels = np.zeros(active_count, dtype=np.int64)
    elif active_count == int(n_clusters):
        active_embedding = np.eye(active_count, dtype=np.float32)
        active_labels = np.arange(active_count, dtype=np.int64)
    else:
        active_embedding = spectral_embedding(
            active_affinity,
            n_components=int(n_clusters),
            eigen_solver="arpack",
            random_state=int(seed),
            eigen_tol="auto",
            norm_laplacian=True,
            drop_first=False,
        )
        active_embedding = normalize(active_embedding, norm="l2", axis=1, copy=False).astype(np.float32)
        active_labels = KMeans(
            n_clusters=int(n_clusters),
            n_init=int(n_init),
            random_state=int(seed),
        ).fit_predict(active_embedding)
    labels[active] = active_labels
    embedding[active] = active_embedding
    return SpectralResult(
        labels,
        embedding,
        abstained,
        {
            "status": "partial_abstention" if abstained.any() else "ok",
            "active_nodes": active_count,
            "abstained_nodes": int(abstained.sum()),
            "abstention_rate": float(np.mean(abstained)) if n else 0.0,
            "connected_components": int(component_count),
            "n_clusters": int(n_clusters),
            "K_used_only_in_readout": True,
            "readout": "normalized_spectral_embedding_then_kmeans",
        },
    )
