"""Label-free PCA/cosine neighbourhood graph for TopoGate V0.

This module is shared by both F and T parameterizations.  F consumes ``probs``
directly; T multiplies them by the analytic edge-reliability factors below.
No function in this module accepts labels, which keeps the graph boundary
auditable by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class NeighborGraph:
    indices: np.ndarray
    probs: np.ndarray
    similarity: np.ndarray
    distance: np.ndarray
    embedding: np.ndarray
    mutual: np.ndarray
    snn: np.ndarray
    profile: dict


def _empty_graph(n_samples: int, tau: float, *, pca_dim: int = 0) -> NeighborGraph:
    empty_i = np.zeros((int(n_samples), 0), dtype=np.int64)
    empty_f = np.zeros((int(n_samples), 0), dtype=np.float32)
    return NeighborGraph(
        indices=empty_i,
        probs=empty_f,
        similarity=empty_f,
        distance=empty_f,
        embedding=np.zeros((int(n_samples), 0), dtype=np.float32),
        mutual=empty_f.astype(bool),
        snn=empty_f,
        profile={
            "neighbor_k": 0,
            "tau": float(tau),
            "knn_pca_dim": int(pca_dim),
            "graph_enabled": False,
            "label_leakage_diagnostic": False,
        },
    )


def _validate_data(data_np: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(np.asarray(data_np, dtype=np.float32))
    if data.ndim != 2:
        raise ValueError(f"data_np must be two-dimensional, got shape {data.shape}")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("data_np must contain at least one sample and one feature")
    if not np.all(np.isfinite(data)):
        raise ValueError("data_np contains non-finite values")
    return data


def build_pca_knn_graph(
    data_np: np.ndarray,
    k: int,
    pca_dim: int,
    tau: float,
    seed: int,
    precomputed_embedding: np.ndarray | None = None,
) -> NeighborGraph:
    """Build the historical PCA + cosine kNN graph without labels."""

    data = _validate_data(data_np)
    if float(tau) <= 0.0:
        raise ValueError("tau must be positive")
    n_samples, n_features = data.shape
    if int(k) <= 0 or n_samples <= 1:
        return _empty_graph(n_samples, tau, pca_dim=0)

    dim = max(1, min(int(pca_dim), n_features, n_samples - 1))
    if precomputed_embedding is None:
        raw_embedding = (
            PCA(n_components=dim, random_state=int(seed)).fit_transform(data)
            if dim < min(data.shape)
            else data
        )
        embedding = normalize(
            np.nan_to_num(raw_embedding, nan=0.0, posinf=0.0, neginf=0.0),
            axis=1,
        ).astype(np.float32)
    else:
        embedding = np.ascontiguousarray(np.asarray(precomputed_embedding, dtype=np.float32))
        if embedding.shape != (n_samples, dim):
            raise ValueError(
                f"precomputed_embedding must have shape {(n_samples, dim)}, "
                f"got {embedding.shape}"
            )
        if not np.all(np.isfinite(embedding)):
            raise ValueError("precomputed_embedding contains non-finite values")

    k_eff = min(int(k), n_samples - 1)
    nearest = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
    nearest.fit(embedding)
    distances, indices = nearest.kneighbors(embedding)
    indices = indices[:, 1 : k_eff + 1].astype(np.int64, copy=False)
    distances = distances[:, 1 : k_eff + 1].astype(np.float32, copy=False)
    similarity = (1.0 - distances).astype(np.float32)
    scaled = similarity / max(float(tau), 1e-8)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled).astype(np.float32)
    probs = exp_scaled / np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-12, None)

    neighbor_sets = [set(row.tolist()) for row in indices]
    mutual = np.zeros_like(indices, dtype=bool)
    snn = np.zeros_like(similarity, dtype=np.float32)
    for sample in range(n_samples):
        sample_set = neighbor_sets[sample]
        for position, neighbor in enumerate(indices[sample]):
            mutual[sample, position] = sample in neighbor_sets[int(neighbor)]
            union = sample_set.union(neighbor_sets[int(neighbor)])
            snn[sample, position] = len(sample_set.intersection(neighbor_sets[int(neighbor)])) / float(
                max(1, len(union))
            )

    profile = {
        "neighbor_k": int(k_eff),
        "tau": float(tau),
        "knn_pca_dim": int(dim),
        "mean_neighbor_similarity": float(np.mean(similarity)),
        "mean_mutual_ratio": float(np.mean(mutual)),
        "mean_snn": float(np.mean(snn)),
        "mean_max_neighbor_prob": float(np.mean(np.max(probs, axis=1))),
        "graph_enabled": True,
        "label_leakage_diagnostic": False,
    }
    return NeighborGraph(
        indices=indices,
        probs=probs.astype(np.float32),
        similarity=similarity,
        distance=distances,
        embedding=embedding,
        mutual=mutual,
        snn=snn,
        profile=profile,
    )


def compute_edge_reliability(
    graph: NeighborGraph,
    mode: str,
    gamma_sim: float,
    gamma_mutual: float,
    gamma_snn: float,
    gamma_distance: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute analytic reliability and row-normalized edge weights."""

    allowed = {"none", "sim", "sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}
    if mode not in allowed:
        raise ValueError(f"unknown edge reliability mode: {mode!r}")
    if graph.indices.shape[1] == 0 or mode == "none":
        weights = graph.probs.copy()
        reliability = np.ones_like(weights, dtype=np.float32)
        return reliability, weights, summarize_edge_weights(weights)

    reliability = np.ones_like(graph.similarity, dtype=np.float32)
    if mode in {"sim", "sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
        reliability *= np.exp(float(gamma_sim) * graph.similarity).astype(np.float32)
    if mode in {"sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
        reliability *= 1.0 + float(gamma_mutual) * graph.mutual.astype(np.float32)
    if mode in {"sim_mutual_snn", "sim_mutual_snn_distance"}:
        reliability *= 1.0 + float(gamma_snn) * graph.snn
    if mode == "sim_mutual_snn_distance":
        reliability *= np.exp(-float(gamma_distance) * graph.distance).astype(np.float32)
    reliability = np.clip(reliability, 1e-6, 1e6).astype(np.float32)
    weights = graph.probs * reliability
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    return reliability, weights.astype(np.float32), summarize_edge_weights(weights)


def summarize_edge_weights(weights: np.ndarray) -> dict:
    """Return compact, label-free diagnostics for a row-normalized edge matrix."""

    values = np.asarray(weights, dtype=np.float32)
    if values.size == 0:
        return {
            "edge_weight_entropy": 0.0,
            "effective_neighbor_count": 0.0,
            "max_edge_weight_mean": 0.0,
            "max_edge_weight_p95": 0.0,
            "fraction_effective_neighbors_lt_2": 1.0,
        }
    entropy = -np.sum(values * np.log(np.clip(values, 1e-12, None)), axis=1)
    effective = np.exp(entropy)
    max_weight = np.max(values, axis=1)
    return {
        "edge_weight_entropy": float(np.mean(entropy)),
        "effective_neighbor_count": float(np.mean(effective)),
        "max_edge_weight_mean": float(np.mean(max_weight)),
        "max_edge_weight_p95": float(np.percentile(max_weight, 95)),
        "fraction_effective_neighbors_lt_2": float(np.mean(effective < 2.0)),
    }


__all__ = [
    "NeighborGraph",
    "build_pca_knn_graph",
    "compute_edge_reliability",
    "summarize_edge_weights",
]
