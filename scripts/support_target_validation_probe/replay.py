"""Deterministic replay and M1 magnitude-matched control construction."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.sparse_corruption_principle_probe.corruption_library import row_budgets, support_mask, support_thresholds

from . import protocol


def _as_int_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def replay_p2_epoch(clean: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay one C2 P2 epoch and retain the ordered source/destination pairs.

    The old compact C2 artifacts retained only masks.  This function mirrors
    the frozen P2 branch exactly, including NumPy choice ordering, so M0 can
    prove that action identity is reconstructible before any M1 fit starts.
    """

    matrix = np.asarray(clean, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("clean must be a finite two-dimensional matrix")
    active = support_mask(matrix, reference=matrix, ratio=protocol.H0_SUPPORT_THRESHOLD_RATIO)
    requested, pair_counts = row_budgets(
        matrix,
        rate=protocol.CORRUPTION_RATE,
        reference_support=active,
    )
    corrupted = matrix.copy()
    changed = np.zeros(matrix.shape, dtype=bool)
    value_changed = np.zeros(matrix.shape, dtype=bool)
    source_mask = np.zeros(matrix.shape, dtype=bool)
    destination_mask = np.zeros(matrix.shape, dtype=bool)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for row in range(matrix.shape[0]):
        active_idx = np.flatnonzero(active[row])
        inactive_idx = np.flatnonzero(~active[row])
        pair_count = int(pair_counts[row])
        if pair_count <= 0:
            pairs.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
            continue
        sources = _as_int_array(rng.choice(active_idx, size=pair_count, replace=False))
        destinations = _as_int_array(rng.choice(inactive_idx, size=pair_count, replace=False))
        source_mask[row, sources] = True
        destination_mask[row, destinations] = True
        for source, destination in zip(sources, destinations, strict=True):
            source_value = matrix[row, source]
            destination_value = matrix[row, destination]
            corrupted[row, source] = destination_value
            corrupted[row, destination] = source_value
            changed[row, source] = True
            changed[row, destination] = True
        pairs.append((sources, destinations))

    support_after = support_mask(
        corrupted,
        reference=matrix,
        ratio=protocol.H0_SUPPORT_THRESHOLD_RATIO,
    )
    return corrupted, {
        "principle": protocol.P2_PRINCIPLE,
        "requested_changed_counts": requested,
        "effective_changed_counts": np.sum(changed, axis=1).astype(np.int64),
        "active_support_before": active,
        "active_support_after": support_after,
        "changed_mask": changed,
        "value_changed_mask": value_changed,
        "source_mask": source_mask,
        "destination_mask": destination_mask,
        "support_changed_mask": active != support_after,
        "pairs": pairs,
        "labels_used": False,
    }


def _row_value_multiset_equal(before: np.ndarray, after: np.ndarray) -> bool:
    return bool(np.array_equal(np.sort(np.asarray(before, dtype=np.float32)), np.sort(np.asarray(after, dtype=np.float32))))


def build_magnitude_matched_epoch(
    clean: np.ndarray,
    p2_corrupted: np.ndarray,
    p2_audit: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct the support-preserving active-active control for one epoch."""

    matrix = np.asarray(clean, dtype=np.float32)
    p2_values = np.asarray(p2_corrupted, dtype=np.float32)
    active = np.asarray(p2_audit["active_support_before"], dtype=bool)
    if matrix.shape != p2_values.shape or active.shape != matrix.shape:
        raise ValueError("M1 clean/P2/support shapes differ")
    matched = matrix.copy()
    changed = np.zeros(matrix.shape, dtype=bool)
    value_changed = np.zeros(matrix.shape, dtype=bool)
    partner_mask = np.zeros(matrix.shape, dtype=bool)
    row_records: list[dict[str, Any]] = []
    match_failures = 0

    for row, (sources, destinations) in enumerate(p2_audit["pairs"]):
        sources = _as_int_array(sources)
        destinations = _as_int_array(destinations)
        candidates = np.flatnonzero(active[row]).astype(np.int64)
        candidates = candidates[~np.isin(candidates, sources)]
        if candidates.size < sources.size:
            match_failures += 1
            raise ValueError(f"row {row} lacks one-to-one active partners")
        if sources.size == 0:
            row_records.append({"row": row, "p2_l1": 0.0, "mm_l1": 0.0, "relative_mismatch": 0.0})
            continue

        target = 2.0 * np.abs(matrix[row, sources] - matrix[row, destinations]).astype(np.float64)
        candidate_l1 = 2.0 * np.abs(matrix[row, sources, None] - matrix[row, candidates][None, :]).astype(np.float64)
        cost = np.abs(candidate_l1 - target[:, None])
        source_order, candidate_order = linear_sum_assignment(cost)
        if not np.array_equal(source_order, np.arange(sources.size)):
            # This should not happen for a square matrix, but preserving an
            # explicit reorder makes the pairing contract auditable.
            partner_by_source = np.empty(sources.size, dtype=np.int64)
            partner_by_source[source_order] = candidates[candidate_order]
            partners = partner_by_source
        else:
            partners = candidates[candidate_order]
        if np.unique(partners).size != partners.size or np.intersect1d(sources, partners).size:
            match_failures += 1
            raise ValueError(f"row {row} has repeated or source-overlapping partners")

        for source, partner in zip(sources, partners, strict=True):
            source_value = matrix[row, source]
            partner_value = matrix[row, partner]
            matched[row, source] = partner_value
            matched[row, partner] = source_value
            changed[row, source] = True
            changed[row, partner] = True
            value_changed[row, source] = bool(source_value != partner_value)
            value_changed[row, partner] = bool(source_value != partner_value)
        partner_mask[row, partners] = True
        mm_l1 = float(np.sum(np.abs(matched[row] - matrix[row]), dtype=np.float64))
        p2_l1 = float(np.sum(np.abs(p2_values[row] - matrix[row]), dtype=np.float64))
        relative = abs(mm_l1 - p2_l1) / max(abs(p2_l1), protocol.ROW_REL_EPS)
        row_records.append({"row": row, "p2_l1": p2_l1, "mm_l1": mm_l1, "relative_mismatch": float(relative)})

    support_after = support_mask(
        matched,
        reference=matrix,
        ratio=protocol.H0_SUPPORT_THRESHOLD_RATIO,
    )
    row_counts = np.sum(changed, axis=1).astype(np.int64)
    p2_counts = np.asarray(p2_audit["effective_changed_counts"], dtype=np.int64)
    row_relative = np.asarray([record["relative_mismatch"] for record in row_records], dtype=np.float64)
    p2_total = float(np.sum(np.abs(p2_values - matrix), dtype=np.float64))
    mm_total = float(np.sum(np.abs(matched - matrix), dtype=np.float64))
    audit = {
        "principle": protocol.M1_CONTROL,
        "requested_changed_counts": np.asarray(p2_audit["requested_changed_counts"], dtype=np.int64),
        "effective_changed_counts": row_counts,
        "p2_effective_changed_counts": p2_counts,
        "changed_mask": changed,
        "value_changed_mask": value_changed,
        "active_support_before": active,
        "active_support_after": support_after,
        "support_changed_mask": active != support_after,
        "source_mask": np.asarray(p2_audit["source_mask"], dtype=bool),
        "partner_mask": partner_mask,
        "exact_budget": bool(np.array_equal(row_counts, p2_counts)),
        "support_change_rate": float(np.mean(active != support_after)),
        "total_absolute_change": mm_total,
        "p2_total_absolute_change": p2_total,
        "dataset_total_relative_mismatch": abs(mm_total - p2_total) / max(abs(p2_total), protocol.ROW_REL_EPS),
        "median_row_relative_mismatch": float(np.median(row_relative)) if row_relative.size else 0.0,
        "row_relative_mismatch_max": float(np.max(row_relative)) if row_relative.size else 0.0,
        "row_records": row_records,
        "match_failure_count": int(match_failures),
        "row_value_multiset_mismatch_count": int(sum(not _row_value_multiset_equal(matrix[row], matched[row]) for row in range(matrix.shape[0]))),
        "labels_used": False,
    }
    return matched.astype(np.float32, copy=False), audit


def compact_epoch_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Keep only scalar audit fields for publication-safe summaries."""

    keys = (
        "principle",
        "exact_budget",
        "support_change_rate",
        "total_absolute_change",
        "p2_total_absolute_change",
        "dataset_total_relative_mismatch",
        "median_row_relative_mismatch",
        "row_relative_mismatch_max",
        "match_failure_count",
        "row_value_multiset_mismatch_count",
        "labels_used",
    )
    return {key: audit[key] for key in keys}
