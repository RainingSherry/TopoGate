from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


FEATURE_NAMES = ("cosine", "mutual", "snn_jaccard", "view_recurrence", "view_stability")


@dataclass(frozen=True)
class CandidateGraph:
    """Fixed-width sparse candidate support; invalid slots are never edges."""

    indices: np.ndarray
    features: np.ndarray
    valid: np.ndarray
    profile: dict[str, Any]

    @property
    def n_nodes(self) -> int:
        return int(self.indices.shape[0])

    @property
    def width(self) -> int:
        return int(self.indices.shape[1])

    @property
    def n_edges(self) -> int:
        return int(self.valid.sum())


def _knn_view(view: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    values = normalize(np.nan_to_num(np.asarray(view, dtype=np.float32)), norm="l2", axis=1)
    n = values.shape[0]
    if n <= 1:
        return np.empty((n, 0), dtype=np.int64), np.empty((n, 0), dtype=np.float32)
    take = min(int(k) + 1, n)
    model = NearestNeighbors(n_neighbors=take, metric="cosine", algorithm="brute", n_jobs=1)
    model.fit(values)
    distances, raw_indices = model.kneighbors(values, return_distance=True)
    indices = np.full((n, min(int(k), n - 1)), -1, dtype=np.int64)
    cosine = np.zeros(indices.shape, dtype=np.float32)
    for row in range(n):
        keep = raw_indices[row] != row
        chosen = raw_indices[row][keep][: indices.shape[1]]
        chosen_distance = distances[row][keep][: indices.shape[1]]
        count = len(chosen)
        indices[row, :count] = chosen
        cosine[row, :count] = np.clip(1.0 - chosen_distance, -1.0, 1.0)
    return indices, cosine


def build_candidate_graph(
    views: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    k: int = 20,
    width: int = 40,
) -> CandidateGraph:
    """Build a union of latent cosine kNN graphs and five fixed edge features."""
    if not views:
        raise ValueError("at least one latent view is required")
    arrays = [normalize(np.nan_to_num(np.asarray(view, dtype=np.float32)), norm="l2", axis=1)
              for view in views]
    if any(a.ndim != 2 for a in arrays):
        raise ValueError("latent views must be two-dimensional")
    n = arrays[0].shape[0]
    if n == 0 or any(a.shape[0] != n for a in arrays):
        raise ValueError("latent views must have identical non-empty row counts")
    if k <= 0 or width <= 0:
        raise ValueError("k and width must be positive")
    if n == 1:
        return CandidateGraph(np.full((1, width), -1, dtype=np.int64),
                              np.zeros((1, width, len(FEATURE_NAMES)), dtype=np.float32),
                              np.zeros((1, width), dtype=bool),
                              {"kind": "latent_cosine_snn_union", "n_views": len(arrays), "k": k,
                               "width": width, "feature_names": list(FEATURE_NAMES),
                               "full_pairwise_matrix_materialized": False})

    neighbor_indices: list[np.ndarray] = []
    neighbor_cosines: list[np.ndarray] = []
    neighbor_sets: list[list[set[int]]] = []
    for view in arrays:
        idx, cos = _knn_view(view, min(k, n - 1))
        neighbor_indices.append(idx)
        neighbor_cosines.append(cos)
        neighbor_sets.append([{int(j) for j in row if j >= 0} for row in idx])

    actual_width = min(width, n - 1, len(arrays) * min(k, n - 1))
    indices = np.full((n, actual_width), -1, dtype=np.int64)
    features = np.zeros((n, actual_width, len(FEATURE_NAMES)), dtype=np.float32)
    valid = np.zeros((n, actual_width), dtype=bool)
    for i in range(n):
        donors = set().union(*(neighbor_sets[m][i] for m in range(len(arrays))))
        scored: list[tuple[int, float, np.ndarray]] = []
        for j in donors:
            cos_values = np.asarray([
                float(np.dot(arrays[m][i], arrays[m][j])) for m in range(len(arrays))
            ], dtype=np.float32)
            # The latent rows are normalized once above; dot products are cosine.
            cos_values = np.clip(cos_values, -1.0, 1.0)
            recurrence = float(np.mean([j in neighbor_sets[m][i] for m in range(len(arrays))]))
            mutual = float(np.mean([
                j in neighbor_sets[m][i] and i in neighbor_sets[m][j]
                for m in range(len(arrays))
            ]))
            jaccard = float(np.mean([
                len(neighbor_sets[m][i] & neighbor_sets[m][j]) /
                max(1, len(neighbor_sets[m][i] | neighbor_sets[m][j]))
                for m in range(len(arrays))
            ]))
            mean_cos = float(np.mean(cos_values))
            stability = mean_cos - float(np.std(cos_values))
            scored.append((int(j), mean_cos, np.asarray(
                [mean_cos, mutual, jaccard, recurrence, stability], dtype=np.float32)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        chosen = scored[:actual_width]
        for slot, (_, _, edge_features) in enumerate(chosen):
            indices[i, slot] = int(chosen[slot][0])
            features[i, slot] = edge_features
            valid[i, slot] = True
    return CandidateGraph(
        indices=indices,
        features=features,
        valid=valid,
        profile={
            "kind": "latent_cosine_snn_union",
            "n_views": int(len(arrays)),
            "k_per_view": int(k),
            "candidate_width_requested": int(width),
            "candidate_width_actual": int(actual_width),
            "candidate_edges": int(valid.sum()),
            "feature_names": list(FEATURE_NAMES),
            "full_pairwise_matrix_materialized": False,
            "self_loops": 0,
        },
    )


def _normalized_views(views: tuple[np.ndarray, ...] | list[np.ndarray]) -> list[np.ndarray]:
    arrays = [normalize(np.nan_to_num(np.asarray(view, dtype=np.float32)), norm="l2", axis=1)
              for view in views]
    if not arrays or any(array.ndim != 2 for array in arrays):
        raise ValueError("views must contain two-dimensional arrays")
    n = arrays[0].shape[0]
    if n == 0 or any(array.shape[0] != n for array in arrays):
        raise ValueError("views must have identical non-empty row counts")
    return arrays


def _edge_features(
    arrays: list[np.ndarray],
    neighbor_sets: list[list[set[int]]],
    i: int,
    j: int,
) -> np.ndarray:
    cos_values = np.asarray([float(np.dot(arrays[m][i], arrays[m][j]))
                             for m in range(len(arrays))], dtype=np.float32)
    cos_values = np.clip(cos_values, -1.0, 1.0)
    mutual = float(np.mean([
        j in neighbor_sets[m][i] and i in neighbor_sets[m][j]
        for m in range(len(arrays))
    ]))
    jaccard = float(np.mean([
        len(neighbor_sets[m][i] & neighbor_sets[m][j]) /
        max(1, len(neighbor_sets[m][i] | neighbor_sets[m][j]))
        for m in range(len(arrays))
    ]))
    mean_cos = float(np.mean(cos_values))
    recurrence = float(np.mean([j in neighbor_sets[m][i] for m in range(len(arrays))]))
    stability = mean_cos - float(np.std(cos_values))
    return np.asarray([mean_cos, mutual, jaccard, recurrence, stability], dtype=np.float32)


def shuffle_candidate_graph(
    graph: CandidateGraph,
    *,
    seed: int,
    views: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
) -> CandidateGraph:
    """Degree-preserving, label-free shuffled-E0 control.

    When latent views are supplied, features are recomputed for the shuffled
    edges. Permuting feature rows would create an invalid graph/scorer pair.
    """
    rng = np.random.default_rng(int(seed))
    n, width = graph.indices.shape
    indices = np.full_like(graph.indices, -1)
    features = np.zeros_like(graph.features)
    valid = graph.valid.copy()
    universe = np.arange(n, dtype=np.int64)
    arrays = None if views is None else _normalized_views(views)
    neighbor_sets = None
    if arrays is not None:
        k = int(graph.profile.get("k_per_view", graph.profile.get("k", width)))
        neighbor_sets = []
        for view in arrays:
            view_indices, _ = _knn_view(view, min(k, n - 1))
            neighbor_sets.append([{int(j) for j in row if j >= 0} for row in view_indices])
    for i in range(n):
        count = int(graph.valid[i].sum())
        if count == 0 or n <= 1:
            valid[i] = False
            continue
        donors = universe[universe != i]
        selected = rng.choice(donors, size=min(count, n - 1), replace=False)
        indices[i, : len(selected)] = selected
        valid[i, len(selected):] = False
        if arrays is not None and neighbor_sets is not None:
            for slot, j in enumerate(selected):
                features[i, slot] = _edge_features(arrays, neighbor_sets, i, int(j))
        else:
            order = rng.permutation(count)
            features[i, : len(selected)] = graph.features[i, :count][order]
    profile = dict(graph.profile)
    profile.update({"kind": "shuffled_latent_candidate_control", "shuffle_seed": int(seed),
                    "features_recomputed_for_edges": arrays is not None})
    return CandidateGraph(indices, features, valid, profile)
