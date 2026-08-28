"""Deterministic B0--B4 relation selectors and graph construction."""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp

from .relation_features import EdgeTable, FEATURE_FAMILIES


SELECTORS: tuple[str, ...] = ("B0_cosine", "B1_mutual_first", "B2_snn", "B3_stability", "B4_rank_fusion")


def _group_ranges(rows: np.ndarray, n_samples: int) -> list[tuple[int, int]]:
    starts = np.searchsorted(rows, np.arange(n_samples), side="left")
    ends = np.searchsorted(rows, np.arange(n_samples), side="right")
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _row_percentile(values: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    result = np.zeros(values.size, dtype=np.float32)
    for start, end in ranges:
        if end <= start:
            continue
        order = np.argsort(-values[start:end], kind="mergesort")
        count = end - start
        result[start:end][order] = (count - np.arange(count, dtype=np.float32)) / count
    return result


def _select_by_keys(
    table: EdgeTable,
    primary: np.ndarray,
    secondary: np.ndarray,
    tertiary: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(table.rows.size, dtype=bool)
    ranges = _group_ranges(table.rows, table.n_samples)
    for row, (start, end) in enumerate(ranges):
        budget = int(table.budget[row])
        if budget <= 0 or end <= start:
            continue
        slots = np.arange(start, end, dtype=np.int64)
        # np.lexsort uses its last key as the primary key.  Negative values make
        # all selector scores descending; column id is the deterministic final tie-break.
        order = np.lexsort((table.cols[slots], -tertiary[slots], -secondary[slots], -primary[slots]))
        chosen = slots[order[: min(budget, slots.size)]]
        mask[chosen] = True
    counts = np.bincount(table.rows[mask], minlength=table.n_samples)
    if not np.array_equal(counts.astype(np.int64), table.budget.astype(np.int64)):
        raise ValueError("selector did not preserve the frozen row budget")
    return mask


def selector_mask(table: EdgeTable, selector: str) -> np.ndarray:
    """Return a deterministic positive-candidate edge mask for B0--B4."""
    if selector not in SELECTORS:
        raise ValueError(f"unknown selector {selector}; expected {SELECTORS}")
    cosine = table.feature("cosine")
    ranges = _group_ranges(table.rows, table.n_samples)
    if selector == "B0_cosine":
        return _select_by_keys(table, cosine, cosine, np.zeros_like(cosine))
    if selector == "B1_mutual_first":
        mutual = table.feature("mutual")
        return _select_by_keys(table, mutual, cosine, np.zeros_like(cosine))
    if selector == "B2_snn":
        jaccard = table.feature("jaccard")
        return _select_by_keys(table, jaccard, cosine, np.zeros_like(cosine))
    if selector == "B3_stability":
        stability = table.feature("stability_recurrence")
        return _select_by_keys(table, stability, cosine, np.zeros_like(cosine))
    cosine_rank = _row_percentile(cosine, ranges)
    snn_rank = _row_percentile(table.feature("jaccard"), ranges)
    stability_rank = _row_percentile(table.feature("stability_recurrence"), ranges)
    fused = (cosine_rank + snn_rank + stability_rank) / 3.0
    return _select_by_keys(table, fused, table.feature("mutual"), cosine)


def selected_graph(table: EdgeTable, selector: str) -> tuple[sp.csr_matrix, np.ndarray]:
    mask = selector_mask(table, selector)
    graph = sp.csr_matrix(
        (
            table.cosine[mask].astype(np.float32, copy=False),
            (table.rows[mask], table.cols[mask]),
        ),
        shape=(table.n_samples, table.n_samples),
    )
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph = ((graph + graph.T) * 0.5).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph, mask


def selector_contract(selector: str, table: EdgeTable, mask: np.ndarray) -> dict[str, object]:
    if selector not in SELECTORS:
        raise ValueError(selector)
    rows = table.rows[mask]
    return {
        "selector": selector,
        "labels_used": False,
        "feature_names": list(table.feature_names),
        "candidate_edge_count": int(table.rows.size),
        "directed_selected_edge_count": int(mask.sum()),
        "row_budget_hash": table.metadata.get("budget_hash"),
        "row_counts_match": bool(
            np.array_equal(np.bincount(rows, minlength=table.n_samples), table.budget)
        ),
        "weight_rule": "original_positive_H0_cosine",
        "symmetrization": "(W+W.T)/2_then_remove_self_loops",
    }

