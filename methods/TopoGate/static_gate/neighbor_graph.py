from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


@dataclass
class NeighborGraph:
    indices: np.ndarray
    probs: np.ndarray
    similarity: np.ndarray
    distance: np.ndarray
    embedding: np.ndarray
    mutual: np.ndarray
    snn: np.ndarray
    profile: dict


def build_pca_knn_graph(
    data_np: np.ndarray,
    k: int,
    pca_dim: int,
    tau: float,
    seed: int,
    labels: np.ndarray | None = None,
    stress_bad_edge_ratio: float = 0.0,
) -> NeighborGraph:
    stress_ratio = float(stress_bad_edge_ratio)
    if not 0.0 <= stress_ratio <= 1.0:
        raise ValueError("stress_bad_edge_ratio must be between 0 and 1")
    data = np.asarray(data_np, dtype=np.float32)
    n_cells, n_genes = data.shape
    if k <= 0 or n_cells <= 1:
        if stress_ratio > 0.0:
            raise ValueError("Cross-label edge stress requires at least one graph edge")
        empty_i = np.zeros((n_cells, 0), dtype=np.int64)
        empty_f = np.zeros((n_cells, 0), dtype=np.float32)
        return NeighborGraph(
            empty_i,
            empty_f,
            empty_f,
            empty_f,
            empty_f,
            empty_f.astype(bool),
            empty_f,
            {
                "neighbor_k": 0,
                "stress_bad_edge_ratio": stress_ratio,
                "stress_bad_edge_ratio_realized": 0.0,
                "label_leakage_diagnostic": False,
            },
        )
    dim = max(1, min(int(pca_dim), n_genes, n_cells - 1))
    emb = PCA(n_components=dim, random_state=seed).fit_transform(data) if dim < min(data.shape) else data
    emb = normalize(np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0), axis=1).astype(np.float32)
    k_eff = min(int(k), n_cells - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
    nn.fit(emb)
    distances, indices = nn.kneighbors(emb)
    indices = indices[:, 1 : k_eff + 1].astype(np.int64, copy=False)
    distances = distances[:, 1 : k_eff + 1].astype(np.float32, copy=False)
    similarity = (1.0 - distances).astype(np.float32)
    scaled = similarity / max(float(tau), 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled).astype(np.float32)
    probs = exp_scaled / np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-12, None)

    neighbor_sets = [set(row.tolist()) for row in indices]
    mutual = np.zeros_like(indices, dtype=bool)
    snn = np.zeros_like(similarity, dtype=np.float32)
    for i in range(n_cells):
        set_i = neighbor_sets[i]
        for pos, j in enumerate(indices[i]):
            mutual[i, pos] = i in neighbor_sets[j]
            union = set_i.union(neighbor_sets[j])
            snn[i, pos] = len(set_i.intersection(neighbor_sets[j])) / float(max(1, len(union)))

    profile = {
        "neighbor_k": int(k_eff),
        "tau": float(tau),
        "knn_pca_dim": int(dim),
        "mean_neighbor_similarity": float(np.mean(similarity)),
        "mean_mutual_ratio": float(np.mean(mutual)),
        "mean_snn": float(np.mean(snn)),
        "mean_max_neighbor_prob": float(np.mean(np.max(probs, axis=1))),
        "stress_bad_edge_ratio": stress_ratio,
        "stress_bad_edge_ratio_realized": 0.0,
        "label_leakage_diagnostic": False,
    }
    graph = NeighborGraph(indices, probs.astype(np.float32), similarity, distances, emb, mutual, snn, profile)
    if stress_ratio == 0.0:
        return graph
    if labels is None:
        raise ValueError("labels are required only when stress_bad_edge_ratio is greater than zero")
    return inject_cross_label_edges(graph, labels, stress_ratio, tau=tau, seed=seed)


def _recompute_graph_edges(indices: np.ndarray, embedding: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_cells, _ = indices.shape
    similarity = np.einsum("ij,ikj->ik", embedding, embedding[indices]).astype(np.float32)
    distance = (1.0 - similarity).astype(np.float32)
    scaled = similarity / max(float(tau), 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled).astype(np.float32)
    probs = exp_scaled / np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-12, None)

    neighbor_sets = [set(row.tolist()) for row in indices]
    mutual = np.zeros_like(indices, dtype=bool)
    snn = np.zeros_like(similarity, dtype=np.float32)
    for i in range(n_cells):
        set_i = neighbor_sets[i]
        for pos, j in enumerate(indices[i]):
            mutual[i, pos] = i in neighbor_sets[j]
            union = set_i.union(neighbor_sets[j])
            snn[i, pos] = len(set_i.intersection(neighbor_sets[j])) / float(max(1, len(union)))
    return probs.astype(np.float32), similarity, distance, mutual, snn


def inject_cross_label_edges(
    graph: NeighborGraph,
    labels: np.ndarray,
    ratio: float,
    tau: float,
    seed: int,
) -> NeighborGraph:
    """Replace a fixed share of each row with cross-label diagnostic edges.

    Labels are consulted only in this explicitly non-default stress path.  A
    separate RNG keeps the clean training RNG stream unchanged.
    """

    stress_ratio = float(ratio)
    if not 0.0 <= stress_ratio <= 1.0:
        raise ValueError("stress_bad_edge_ratio must be between 0 and 1")
    if stress_ratio == 0.0:
        return graph
    n_cells, k = graph.indices.shape
    if k == 0:
        raise ValueError("Cross-label edge stress requires at least one graph edge")
    label_values = np.asarray(labels).reshape(-1)
    if label_values.shape != (n_cells,):
        raise ValueError(f"labels must have shape ({n_cells},), got {label_values.shape}")

    rng = np.random.default_rng(int(seed) + 104729)
    stressed_indices = graph.indices.copy()
    all_cells = np.arange(n_cells, dtype=np.int64)
    total_edges = n_cells * k
    total_replacements = int(np.rint(stress_ratio * total_edges))
    selected_flat = rng.choice(total_edges, size=total_replacements, replace=False)
    replacement_mask = np.zeros(total_edges, dtype=bool)
    replacement_mask[selected_flat] = True
    replacement_mask = replacement_mask.reshape(n_cells, k)
    for cell in range(n_cells):
        positions = np.flatnonzero(replacement_mask[cell])
        replacements_per_row = int(positions.size)
        if replacements_per_row == 0:
            continue
        cross_label = all_cells[label_values != label_values[cell]]
        if cross_label.size == 0:
            raise ValueError(f"Cell {cell} has no cross-label candidate for diagnostic edge stress")
        keep_positions = np.ones(k, dtype=bool)
        keep_positions[positions] = False
        preserved = stressed_indices[cell, keep_positions]
        # Prefer genuinely new cross-label endpoints so the requested stress
        # fraction is also the realized changed-edge fraction.
        candidates = np.setdiff1d(cross_label, graph.indices[cell], assume_unique=False)
        if candidates.size == 0:
            candidates = np.setdiff1d(cross_label, preserved, assume_unique=False)
        if candidates.size == 0:
            candidates = cross_label
        stressed_indices[cell, positions] = rng.choice(
            candidates,
            size=replacements_per_row,
            replace=candidates.size < replacements_per_row,
        )

    probs, similarity, distance, mutual, snn = _recompute_graph_edges(stressed_indices, graph.embedding, tau)
    profile = dict(graph.profile)
    per_row = replacement_mask.sum(axis=1)
    profile.update(
        {
            "stress_bad_edge_ratio": stress_ratio,
            "stress_bad_edge_ratio_realized": float(total_replacements / total_edges),
            "stress_bad_edges_per_row_mean": float(per_row.mean()),
            "stress_bad_edges_per_row_min": int(per_row.min()),
            "stress_bad_edges_per_row_max": int(per_row.max()),
            "stress_cross_label_edge_fraction": float(
                np.mean(label_values[stressed_indices] != label_values[:, None])
            ),
            "label_leakage_diagnostic": True,
            "label_usage": "cross-label edge stress diagnostic only",
            "mean_neighbor_similarity": float(np.mean(similarity)),
            "mean_mutual_ratio": float(np.mean(mutual)),
            "mean_snn": float(np.mean(snn)),
            "mean_max_neighbor_prob": float(np.mean(np.max(probs, axis=1))),
        }
    )
    return NeighborGraph(
        stressed_indices,
        probs,
        similarity,
        distance,
        graph.embedding,
        mutual,
        snn,
        profile,
    )


def compute_edge_reliability(
    graph: NeighborGraph,
    mode: str,
    gamma_sim: float,
    gamma_mutual: float,
    gamma_snn: float,
    gamma_distance: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if graph.indices.shape[1] == 0 or mode == "none":
        weights = graph.probs.copy()
        rel = np.ones_like(weights, dtype=np.float32)
        return rel, weights, summarize_edge_weights(weights)

    rel = np.ones_like(graph.similarity, dtype=np.float32)
    if mode in {"sim", "sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
        rel *= np.exp(float(gamma_sim) * graph.similarity).astype(np.float32)
    if mode in {"sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
        rel *= (1.0 + float(gamma_mutual) * graph.mutual.astype(np.float32))
    if mode in {"sim_mutual_snn", "sim_mutual_snn_distance"}:
        rel *= (1.0 + float(gamma_snn) * graph.snn)
    if mode == "sim_mutual_snn_distance":
        rel *= np.exp(-float(gamma_distance) * graph.distance).astype(np.float32)
    rel = np.clip(rel, 1e-6, 1e6).astype(np.float32)
    weights = graph.probs * rel
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    return rel, weights.astype(np.float32), summarize_edge_weights(weights)


def summarize_edge_weights(weights: np.ndarray) -> dict:
    if weights.size == 0:
        return {
            "edge_weight_entropy": 0.0,
            "effective_neighbor_count": 0.0,
            "max_edge_weight_mean": 0.0,
            "max_edge_weight_p95": 0.0,
            "fraction_effective_neighbors_lt_2": 1.0,
        }
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, None)), axis=1)
    effective = np.exp(entropy)
    max_w = np.max(weights, axis=1)
    return {
        "edge_weight_entropy": float(np.mean(entropy)),
        "effective_neighbor_count": float(np.mean(effective)),
        "max_edge_weight_mean": float(np.mean(max_w)),
        "max_edge_weight_p95": float(np.percentile(max_w, 95)),
        "fraction_effective_neighbors_lt_2": float(np.mean(effective < 2.0)),
    }


def build_random_neighbors(n_cells: int, k: int, rng: np.random.Generator, exclude: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros((n_cells, k), dtype=np.int64)
    all_idx = np.arange(n_cells)
    for i in range(n_cells):
        banned = {i}
        if exclude is not None:
            banned.update(exclude[i].tolist())
        candidates = np.setdiff1d(all_idx, np.fromiter(banned, dtype=np.int64), assume_unique=False)
        if candidates.size == 0:
            candidates = all_idx[all_idx != i]
        out[i] = rng.choice(candidates, size=k, replace=candidates.size < k)
    return out


def build_far_neighbors(embedding: np.ndarray, k: int, rng: np.random.Generator, candidate_pool: int = 96) -> np.ndarray:
    n_cells = int(embedding.shape[0])
    out = np.zeros((n_cells, k), dtype=np.int64)
    all_idx = np.arange(n_cells)
    for i in range(n_cells):
        candidates = rng.choice(all_idx[all_idx != i], size=min(candidate_pool, n_cells - 1), replace=False)
        sim = embedding[candidates] @ embedding[i]
        far = candidates[np.argsort(sim)[:k]]
        if far.size < k:
            far = rng.choice(candidates, size=k, replace=True)
        out[i] = far[:k]
    return out
