from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


@dataclass
class CandidateGraph:
    indices: np.ndarray
    similarity: np.ndarray
    valid: np.ndarray
    profile: dict

    @property
    def n_nodes(self) -> int:
        return int(self.indices.shape[0])

    @property
    def width(self) -> int:
        return int(self.indices.shape[1])

    def edge_purity(self, labels: np.ndarray) -> float:
        y = np.asarray(labels).reshape(-1)
        rows, cols = np.where(self.valid)
        if rows.size == 0:
            return 0.0
        return float(np.mean(y[rows] == y[self.indices[rows, cols]]))

    def budget_normalized_recall(self, labels: np.ndarray) -> float:
        """Measure same-label coverage normalized by the selected edge budget.

        This is not conventional recall over every same-label point: when a
        cluster is larger than the candidate budget, the denominator is the
        number of selected candidates.  The explicit name prevents it from
        being mistaken for a full-neighborhood recall certificate.
        """
        y = np.asarray(labels).reshape(-1)
        values: list[float] = []
        for i in range(self.n_nodes):
            same = np.flatnonzero(y == y[i])
            same = same[same != i]
            selected = self.indices[i, self.valid[i]]
            if same.size and selected.size:
                denominator = min(int(same.size), int(selected.size))
                values.append(float(np.intersect1d(selected, same).size / denominator))
        return float(np.mean(values)) if values else 0.0

    def recall(self, labels: np.ndarray) -> float:
        """Backward-compatible alias for budget-normalized same-label coverage."""
        return self.budget_normalized_recall(labels)


def build_candidate_graph(view_a: sp.spmatrix, k: int = 20, block_size: int = 256) -> CandidateGraph:
    """Build sparse cosine top-k without constructing an ``n x n`` matrix.

    Sparse multiplication retains only rows with at least one shared feature;
    each block is discarded after its row-wise top-k extraction.  This is the
    count-graph operation specified by V16 and avoids the dense pairwise work
    hidden inside a generic brute-force neighbor implementation.
    """
    matrix = sp.csr_matrix(view_a, dtype=np.float32)
    n = int(matrix.shape[0])
    if n <= 1:
        indices = np.empty((n, 0), dtype=np.int64)
        return CandidateGraph(indices, np.empty((n, 0), dtype=np.float32), np.zeros((n, 0), bool), {"k": 0})
    k_eff = min(int(k), n - 1)
    normalized = normalize(matrix, norm="l2", axis=1, copy=True).tocsr()
    indices = np.full((n, k_eff), -1, dtype=np.int64)
    similarity = np.zeros((n, k_eff), dtype=np.float32)
    valid = np.zeros((n, k_eff), dtype=bool)
    transpose = normalized.transpose().tocsr()
    for block_start in range(0, n, int(block_size)):
        block_end = min(block_start + int(block_size), n)
        similarities = (normalized[block_start:block_end] @ transpose).tocsr()
        for local, i in enumerate(range(block_start, block_end)):
            row = similarities.getrow(local)
            keep = row.indices != i
            row_indices = row.indices[keep]
            row_values = row.data[keep].astype(np.float32)
            if row_indices.size > k_eff:
                order = np.argpartition(-row_values, k_eff - 1)[:k_eff]
                order = order[np.argsort(-row_values[order], kind="stable")]
                row_indices = row_indices[order]
                row_values = row_values[order]
            width = int(row_indices.size)
            indices[i, :width] = row_indices
            similarity[i, :width] = row_values
            valid[i, :width] = row_values > 0.0
    profile = {
        "k": int(k_eff),
        "mean_valid_candidates": float(valid.sum(axis=1).mean()) if n else 0.0,
        "empty_candidate_row_fraction": float(np.mean(~valid.any(axis=1))) if n else 1.0,
        "mean_similarity": float(similarity[valid].mean()) if valid.any() else 0.0,
        "median_similarity": float(np.median(similarity[valid])) if valid.any() else 0.0,
        "storage": "sparse_cosine_knn",
    }
    return CandidateGraph(indices, similarity, valid, profile)


def candidate_recurrence(graphs: list[CandidateGraph]) -> float | None:
    if len(graphs) < 2:
        return None
    first = graphs[0]
    scores: list[float] = []
    for other in graphs[1:]:
        for i in range(first.n_nodes):
            a = set(first.indices[i, first.valid[i]].tolist())
            b = set(other.indices[i, other.valid[i]].tolist())
            if a or b:
                scores.append(len(a & b) / max(1, len(a | b)))
    return float(np.mean(scores)) if scores else 1.0
