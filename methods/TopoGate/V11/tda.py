"""Sparse H0 persistent-homology priors for TopoGate V11.

The implementation is deliberately narrow. It computes exact zero-dimensional
persistence on a fixed, sparse Vietoris--Rips 1-skeleton: vertices are born at
filtration value zero and a component dies when a sorted skeleton edge first
merges it with another component. The resulting merge-edge scores are mapped
to the current candidate graph as detached priors.

This is not a dense Vietoris--Rips computation, and it does not compute H1.
Keeping that distinction explicit prevents ordinary kNN statistics from being
presented as a full persistence diagram.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TdaPriorMode = Literal[
    "none",
    "h0_mst",
    "h0_early_mst",
    "fixed_filtration",
    "random",
]
TdaScaleMode = Literal["median", "quantile", "max", "none"]


@dataclass(frozen=True)
class H0Persistence:
    """Finite H0 intervals of a fixed sparse Vietoris--Rips 1-skeleton."""

    edge_pairs: np.ndarray
    edge_distances: np.ndarray
    merge_mask: np.ndarray
    persistence_norm: np.ndarray
    persistence_score: np.ndarray
    scale: float
    n_nodes: int
    n_components: int
    filtration_metric: str
    scale_mode: str

    @property
    def merge_count(self) -> int:
        return int(np.sum(self.merge_mask))


def _unit_rows(embedding: np.ndarray) -> np.ndarray:
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("embedding must be a two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("embedding must contain only finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _skeleton_edges(
    embedding: np.ndarray,
    raw_knn_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique undirected edges and chord distances.

    The neighbour relation is fixed by ``raw_knn_indices``. Distances are
    Euclidean chord distances between unit-normalised rows, so the filtration
    is a genuine metric filtration on the embedded points even though the
    1-skeleton is sparse.
    """
    unit = _unit_rows(embedding)
    indices = np.asarray(raw_knn_indices)
    if indices.ndim != 2 or indices.shape[0] != unit.shape[0]:
        raise ValueError("raw_knn_indices must have one row per embedding point")
    n = int(unit.shape[0])
    edges: dict[tuple[int, int], float] = {}
    for left in range(n):
        for value in indices[left].ravel().tolist():
            right = int(value)
            if right < 0 or right >= n or right == left:
                continue
            pair = (min(left, right), max(left, right))
            distance = float(np.linalg.norm(unit[left] - unit[right]))
            previous = edges.get(pair)
            if previous is None or distance < previous:
                edges[pair] = distance
    if not edges:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    pairs = np.asarray(sorted(edges), dtype=np.int64)
    distances = np.asarray([edges[tuple(pair)] for pair in pairs], dtype=np.float64)
    order = np.lexsort((pairs[:, 1], pairs[:, 0], distances))
    return pairs[order], distances[order]


def _distance_scale(
    distances: np.ndarray,
    mode: TdaScaleMode,
    quantile: float,
    floor: float,
) -> float:
    positive = np.asarray(distances, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if mode == "none" or positive.size == 0:
        scale = 1.0
    elif mode == "median":
        scale = float(np.median(positive))
    elif mode == "quantile":
        scale = float(np.quantile(positive, float(quantile)))
    elif mode == "max":
        scale = float(np.max(positive))
    else:
        raise ValueError("unknown TDA scale mode")
    return max(scale, float(floor))


def compute_h0_persistence(
    embedding: np.ndarray,
    raw_knn_indices: np.ndarray,
    *,
    scale_mode: TdaScaleMode = "median",
    scale_quantile: float = 0.50,
    scale_floor: float = 1e-6,
) -> H0Persistence:
    """Compute H0 persistence on a fixed sparse raw-kNN 1-skeleton.

    Every vertex is born at zero. Union-find records the edge that kills a
    finite component. Since all births are zero, finite persistence is the
    corresponding edge filtration value. The final component in each
    connected component has an infinite bar and is excluded from the prior.
    """
    if scale_mode not in {"median", "quantile", "max", "none"}:
        raise ValueError("unknown TDA scale mode")
    if not 0.0 <= float(scale_quantile) <= 1.0:
        raise ValueError("scale_quantile must be in [0, 1]")
    if float(scale_floor) <= 0.0:
        raise ValueError("scale_floor must be positive")
    values = np.asarray(embedding)
    if values.ndim != 2:
        raise ValueError("embedding must be a two-dimensional array")
    pairs, distances = _skeleton_edges(values, raw_knn_indices)
    n = int(values.shape[0])
    scale = _distance_scale(distances, scale_mode, scale_quantile, scale_floor)
    if pairs.shape[0] == 0:
        return H0Persistence(
            edge_pairs=pairs,
            edge_distances=distances,
            merge_mask=np.zeros(0, dtype=bool),
            persistence_norm=np.zeros(0, dtype=np.float64),
            persistence_score=np.zeros(0, dtype=np.float64),
            scale=scale,
            n_nodes=n,
            n_components=n,
            filtration_metric="unit_row_euclidean_chord",
            scale_mode=scale_mode,
        )

    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    merge_mask = np.zeros(pairs.shape[0], dtype=bool)
    components = n
    for edge_index, (left, right) in enumerate(pairs):
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left == root_right:
            continue
        # Union by size is deterministic for equal-size roots because the
        # sorted skeleton and the left root are processed first.
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]
        merge_mask[edge_index] = True
        components -= 1

    persistence_norm = distances / scale
    # A bounded score is easier to combine with the existing raw prior while
    # retaining the ordering of H0 lifetimes.
    persistence_score = np.where(
        merge_mask,
        -np.expm1(-persistence_norm),
        0.0,
    ).astype(np.float64)
    return H0Persistence(
        edge_pairs=pairs,
        edge_distances=distances,
        merge_mask=merge_mask,
        persistence_norm=persistence_norm,
        persistence_score=persistence_score,
        scale=scale,
        n_nodes=n,
        n_components=int(components),
        filtration_metric="unit_row_euclidean_chord",
        scale_mode=scale_mode,
    )


def _stable_uniform(seed: int, left: int, right: int) -> float:
    """Return a reproducible undirected pseudo-random edge score."""
    a, b = sorted((int(left), int(right)))
    mask = (1 << 64) - 1
    value = (int(seed) & mask) ^ ((a + 0x9E3779B9) * 0xD1B54A32D192ED03)
    value ^= (b + 0x94D049BB133111EB) * 0x9E3779B97F4A7C15
    value &= mask
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return float((value >> 11) / float(1 << 53))


def candidate_prior_from_h0(
    persistence: H0Persistence | None,
    candidate_indices: np.ndarray,
    valid: np.ndarray,
    *,
    mode: TdaPriorMode,
    seed: int = 42,
) -> np.ndarray:
    """Map a fixed H0/filtration/control prior to directed candidate edges.

    ``h0_mst`` keeps only edges that merge H0 components and weights them by
    bounded persistence, so late component deaths receive larger values.
    ``h0_early_mst`` keeps the same merge-edge mask but reverses that ordering:
    early component deaths receive larger values and late bridge-like merges
    receive smaller values. ``fixed_filtration`` uses the same fixed raw
    skeleton but scores every skeleton edge by proximity, providing a
    non-persistent distance control. ``random`` is a deterministic edge-shared
    control and does not use labels or learned quantities.
    """
    if mode not in {"none", "h0_mst", "h0_early_mst", "fixed_filtration", "random"}:
        raise ValueError("unknown TDA prior mode")
    indices = np.asarray(candidate_indices)
    valid_array = np.asarray(valid, dtype=bool)
    if indices.ndim != 2 or valid_array.shape != indices.shape:
        raise ValueError("candidate_indices and valid must have the same [n, k] shape")
    output = np.zeros(indices.shape, dtype=np.float32)
    if mode == "none":
        return output
    if mode in {"h0_mst", "h0_early_mst", "fixed_filtration"} and persistence is None:
        raise ValueError(f"TDA prior mode {mode} requires H0 persistence")

    lookup: dict[tuple[int, int], int] = {}
    if persistence is not None:
        lookup = {
            (int(pair[0]), int(pair[1])): index
            for index, pair in enumerate(persistence.edge_pairs)
        }

    for left in range(indices.shape[0]):
        for column in np.flatnonzero(valid_array[left]):
            right = int(indices[left, column])
            if right == left or right < 0 or right >= indices.shape[0]:
                continue
            if mode == "random":
                output[left, column] = _stable_uniform(seed, left, right)
                continue
            pair = (min(left, right), max(left, right))
            edge_index = lookup.get(pair)
            if edge_index is None:
                continue
            assert persistence is not None
            if mode == "h0_mst":
                output[left, column] = float(persistence.persistence_score[edge_index])
            elif mode == "h0_early_mst":
                output[left, column] = float(
                    np.exp(-persistence.persistence_norm[edge_index])
                    if persistence.merge_mask[edge_index]
                    else 0.0
                )
            else:
                output[left, column] = float(
                    np.exp(-persistence.persistence_norm[edge_index])
                )
    return output
