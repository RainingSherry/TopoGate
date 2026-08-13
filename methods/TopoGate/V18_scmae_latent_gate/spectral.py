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
class ReadoutResult:
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
) -> ReadoutResult:
    """Use the exact C-derived affinity; K is consulted only here."""
    matrix = sp.csr_matrix(affinity, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("affinity must be square")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    matrix = (0.5 * (matrix + matrix.T)).tocsr()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    n = int(matrix.shape[0])
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    active = degree > float(degree_epsilon)
    labels = np.full(n, -1, dtype=np.int64)
    embedding = np.zeros((n, int(n_clusters)), dtype=np.float32)
    if not np.any(active):
        return ReadoutResult(labels, embedding, ~active, {
            "status": "all_abstained", "active_nodes": 0, "abstention_rate": 1.0 if n else 0.0,
            "connected_components": 0, "K_used_only_in_readout": True,
        })
    active_affinity = matrix[active][:, active]
    active_count = int(active.sum())
    if active_count < int(n_clusters):
        return ReadoutResult(labels, embedding, ~active, {
            "status": "insufficient_supported_nodes", "active_nodes": active_count,
            "requested_clusters": int(n_clusters), "abstention_rate": float(np.mean(~active)),
            "connected_components": int(connected_components(active_affinity, directed=False, return_labels=False)),
            "K_used_only_in_readout": True,
        })
    component_count = int(connected_components(active_affinity, directed=False, return_labels=False))
    if n_clusters == 1:
        active_embedding = np.ones((active_count, 1), dtype=np.float32)
        active_labels = np.zeros(active_count, dtype=np.int64)
    elif active_count == n_clusters:
        active_embedding = np.eye(active_count, dtype=np.float32)
        active_labels = np.arange(active_count, dtype=np.int64)
    else:
        active_embedding = spectral_embedding(
            active_affinity,
            n_components=int(n_clusters),
            eigen_solver="arpack",
            random_state=int(seed),
            norm_laplacian=True,
            drop_first=False,
        )
        active_embedding = normalize(np.asarray(active_embedding), norm="l2", axis=1).astype(np.float32)
        active_labels = KMeans(n_clusters=int(n_clusters), n_init=int(n_init), random_state=int(seed)).fit_predict(active_embedding)
    labels[active] = active_labels
    return ReadoutResult(labels, _place_embedding(embedding, active, active_embedding), ~active, {
        "status": "partial_abstention" if np.any(~active) else "ok",
        "active_nodes": active_count,
        "abstained_nodes": int((~active).sum()),
        "abstention_rate": float(np.mean(~active)) if n else 0.0,
        "connected_components": component_count,
        "n_clusters": int(n_clusters),
        "K_used_only_in_readout": True,
        "readout": "normalized_spectral_embedding_then_kmeans",
    })


def _place_embedding(output: np.ndarray, active: np.ndarray, values: np.ndarray) -> np.ndarray:
    output[active] = values
    return output


def leiden_readout(affinity: sp.spmatrix, *, resolution: float) -> ReadoutResult:
    """Optional graph-only readout from the same affinity matrix."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Leiden readout requires python-igraph and leidenalg") from exc
    matrix = sp.csr_matrix(affinity, dtype=np.float64)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    active = degree > 0.0
    labels = np.full(matrix.shape[0], -1, dtype=np.int64)
    if np.any(active):
        sub = matrix[active][:, active].tocoo()
        graph = ig.Graph(n=int(active.sum()), edges=list(zip(sub.row.tolist(), sub.col.tolist(), strict=True)), directed=False)
        graph.es["weight"] = sub.data.tolist()
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                              weights="weight", resolution_parameter=float(resolution))
        labels[active] = np.asarray(partition.membership, dtype=np.int64)
    return ReadoutResult(labels, np.zeros((matrix.shape[0], 0), dtype=np.float32), ~active, {
        "status": "ok" if np.any(active) else "all_abstained", "readout": "leiden_same_C_affinity",
        "active_nodes": int(active.sum()), "n_clusters_observed": int(len(set(labels[labels >= 0]))),
        "K_used_only_in_readout": False,
    })
