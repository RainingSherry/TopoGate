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
        return self.budget_normalized_recall(labels)


def build_candidate_graph(view_a: sp.spmatrix, k: int = 20, block_size: int = 256) -> CandidateGraph:
    """Build sparse cosine top-k without constructing an n-by-n dense matrix."""
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


def consensus_graph(
    graphs: list[CandidateGraph],
    *,
    k: int = 20,
    min_repeats: int = 2,
) -> CandidateGraph:
    """Keep edges recurring across split-specific candidate graphs.

    The row dictionaries are bounded by the split graph budgets.  No pairwise
    dense matrix is formed and the final width remains at most ``k``.
    """
    if not graphs:
        raise ValueError("at least one graph is required")
    n = graphs[0].n_nodes
    if any(graph.n_nodes != n for graph in graphs):
        raise ValueError("all graphs must contain the same number of nodes")
    width = min(int(k), max(0, n - 1))
    indices = np.full((n, width), -1, dtype=np.int64)
    similarity = np.zeros((n, width), dtype=np.float32)
    valid = np.zeros((n, width), dtype=bool)
    stable_counts: list[int] = []
    all_counts: list[int] = []
    stable_occurrences = 0
    for i in range(n):
        occurrences: dict[int, tuple[int, float]] = {}
        for graph in graphs:
            for position in np.flatnonzero(graph.valid[i]):
                donor = int(graph.indices[i, position])
                count, best_similarity = occurrences.get(donor, (0, 0.0))
                occurrences[donor] = (count + 1, max(best_similarity, float(graph.similarity[i, position])))
        all_counts.extend(count for count, _ in occurrences.values())
        selected = [
            (donor, count, best_similarity)
            for donor, (count, best_similarity) in occurrences.items()
            if count >= int(min_repeats)
        ]
        selected.sort(key=lambda item: (-item[1], -item[2], item[0]))
        selected = selected[:width]
        stable_counts.append(len(selected))
        stable_occurrences += sum(count for _, count, _ in selected)
        for position, (donor, count, best_similarity) in enumerate(selected):
            indices[i, position] = donor
            similarity[i, position] = best_similarity
            valid[i, position] = best_similarity > 0.0
    profile = {
        "k": int(width),
        "min_repeats": int(min_repeats),
        "source_graphs": int(len(graphs)),
        "mean_valid_candidates": float(valid.sum(axis=1).mean()) if n else 0.0,
        "empty_candidate_row_fraction": float(np.mean(~valid.any(axis=1))) if n else 1.0,
        "stable_edge_rate": float(stable_occurrences / max(1, sum(all_counts))),
        "mean_stable_candidates": float(np.mean(stable_counts)) if stable_counts else 0.0,
        "storage": "sparse_cosine_knn_consensus",
    }
    return CandidateGraph(indices, similarity, valid, profile)
