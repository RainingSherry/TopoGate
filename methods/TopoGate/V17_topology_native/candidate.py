from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateSet:
    indices: np.ndarray
    similarity: np.ndarray
    valid: np.ndarray
    view_count: np.ndarray
    profile: dict[str, Any]

    @property
    def n_nodes(self) -> int:
        return int(self.indices.shape[0])

    @property
    def width(self) -> int:
        return int(self.indices.shape[1])


def _topk_cosine_view(view: np.ndarray, *, k: int, block_size: int) -> CandidateSet:
    values = np.asarray(view, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("each projection view must be a 2D array")
    n = int(values.shape[0])
    width = min(int(k), max(0, n - 1))
    indices = np.full((n, width), -1, dtype=np.int64)
    similarity = np.zeros((n, width), dtype=np.float32)
    valid = np.zeros((n, width), dtype=bool)
    max_block_rows = 0
    for start in range(0, n, int(block_size)):
        end = min(start + int(block_size), n)
        scores = values[start:end] @ values.T
        max_block_rows = max(max_block_rows, end - start)
        for local, anchor in enumerate(range(start, end)):
            row = scores[local]
            row[anchor] = -np.inf
            if width == 0:
                continue
            selected = np.argpartition(-row, width - 1)[:width]
            selected = selected[np.argsort(-row[selected], kind="stable")]
            selected_scores = row[selected]
            keep = np.isfinite(selected_scores) & (selected_scores > 0.0)
            count = int(np.sum(keep))
            if count:
                indices[anchor, :count] = selected[keep]
                similarity[anchor, :count] = selected_scores[keep]
                valid[anchor, :count] = True
    return CandidateSet(
        indices=indices,
        similarity=similarity,
        valid=valid,
        view_count=valid.astype(np.int16),
        profile={
            "kind": "blockwise_projected_cosine",
            "k": int(width),
            "block_size": int(block_size),
            "max_similarity_block_shape": [int(max_block_rows), int(n)],
            "full_pairwise_matrix_materialized": False,
            "mean_candidates": float(valid.sum(axis=1).mean()) if n else 0.0,
        },
    )


def build_candidate_union(
    views: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    k_per_view: int,
    union_k: int,
    block_size: int,
) -> CandidateSet:
    """Union small projected-neighbor sets; the union is only a support mask."""
    if not views:
        raise ValueError("at least one projection view is required")
    graphs = [_topk_cosine_view(view, k=k_per_view, block_size=block_size) for view in views]
    n = graphs[0].n_nodes
    if any(graph.n_nodes != n for graph in graphs):
        raise ValueError("all projection views must have the same number of rows")
    width = min(int(union_k), max(0, n - 1))
    indices = np.full((n, width), -1, dtype=np.int64)
    similarity = np.zeros((n, width), dtype=np.float32)
    valid = np.zeros((n, width), dtype=bool)
    view_count = np.zeros((n, width), dtype=np.int16)
    for anchor in range(n):
        observed: dict[int, tuple[int, float]] = {}
        for graph in graphs:
            for position in np.flatnonzero(graph.valid[anchor]):
                donor = int(graph.indices[anchor, position])
                count, total = observed.get(donor, (0, 0.0))
                observed[donor] = (count + 1, total + float(graph.similarity[anchor, position]))
        ranked = [
            (donor, count, total / count)
            for donor, (count, total) in observed.items()
            if donor != anchor
        ]
        ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
        for position, (donor, count, mean_similarity) in enumerate(ranked[:width]):
            indices[anchor, position] = donor
            similarity[anchor, position] = mean_similarity
            valid[anchor, position] = True
            view_count[anchor, position] = count
    return CandidateSet(
        indices=indices,
        similarity=similarity,
        valid=valid,
        view_count=view_count,
        profile={
            "kind": "multi_projection_candidate_union",
            "n_views": int(len(graphs)),
            "k_per_view": int(k_per_view),
            "union_k": int(width),
            "mean_candidates": float(valid.sum(axis=1).mean()) if n else 0.0,
            "empty_candidate_row_fraction": float(np.mean(~valid.any(axis=1))) if n else 1.0,
            "mean_edge_view_recurrence": float(view_count[valid].mean()) if valid.any() else 0.0,
            "full_pairwise_matrix_materialized": False,
        },
    )


def shuffle_candidate_donors(candidates: CandidateSet, *, seed: int) -> CandidateSet:
    """Label-free negative control that replaces candidate donors uniformly."""
    rng = np.random.default_rng(int(seed))
    n, width = candidates.indices.shape
    indices = np.full((n, width), -1, dtype=np.int64)
    similarity = np.zeros((n, width), dtype=np.float32)
    valid = np.zeros((n, width), dtype=bool)
    view_count = np.zeros((n, width), dtype=np.int16)
    universe = np.arange(n, dtype=np.int64)
    for anchor in range(n):
        count = int(candidates.valid[anchor].sum())
        if count == 0:
            continue
        donors = universe[universe != anchor]
        chosen = rng.choice(donors, size=min(count, donors.size), replace=False)
        count = int(chosen.size)
        indices[anchor, :count] = chosen
        similarity[anchor, :count] = 1.0
        valid[anchor, :count] = True
        view_count[anchor, :count] = 1
    profile = dict(candidates.profile)
    profile.update({"kind": "shuffled_candidate_control", "seed": int(seed)})
    return CandidateSet(indices, similarity, valid, view_count, profile)
