"""Deterministic constructive pair matchings for the D1 feasibility map."""
from __future__ import annotations

from typing import Any

import numpy as np

from scripts.sparse_corruption_principle_probe.corruption_library import support_mask, row_budgets

from . import protocol


def _edge_table(left: np.ndarray, right: np.ndarray, *, same_set: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if same_set:
        ii, jj = np.triu_indices(left.size, 1)
        costs = 2.0 * np.abs(left[ii] - right[jj])
    else:
        ii, jj = np.indices((left.size, right.size))
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)
        costs = 2.0 * np.abs(left[ii] - right[jj])
    valid = np.isfinite(costs) & (costs > protocol.PAIR_COST_EPS)
    return ii[valid].astype(np.int64), jj[valid].astype(np.int64), costs[valid].astype(np.float64)


def _tie_key(ii: np.ndarray, jj: np.ndarray, *, seed: int, row: int) -> np.ndarray:
    """Return an edge-specific deterministic hash.

    The closed D1 implementation added a row-wide seed constant, which could
    not change the lexicographic order among tied edges.  Seed, row, and both
    endpoints now interact through splitmix-style mixing, so equal-cost ties
    genuinely reproduce different deterministic matchings across seeds.
    """
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        x = (
            ii.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
            + jj.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
            + np.uint64(int(seed) & int(mask)) * np.uint64(0x165667B19E3779F9)
            + np.uint64(int(row) & int(mask)) * np.uint64(0x85EBCA77C2B2AE63)
        ) & mask
    x ^= x >> np.uint64(30)
    with np.errstate(over="ignore"):
        x = (x * np.uint64(0xBF58476D1CE4E5B9)) & mask
    x ^= x >> np.uint64(27)
    with np.errstate(over="ignore"):
        x = (x * np.uint64(0x94D049BB133111EB)) & mask
    return x ^ (x >> np.uint64(31))


def greedy_matching(
    left: np.ndarray,
    right: np.ndarray,
    count: int,
    *,
    mode: str,
    seed: int,
    row: int,
    same_set: bool = False,
    target_total: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return a valid disjoint matching and its total L1 dose.

    ``min`` and ``max`` are constructive endpoint witnesses.  ``target``
    chooses edges nearest to the target per-pair dose; it is intentionally
    audited after construction rather than treated as an exact optimizer.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.empty((0, 2), dtype=np.int64), 0.0
    ii, jj, costs = _edge_table(left, right, same_set=same_set)
    if target_total is not None:
        if not np.isfinite(target_total) or target_total < 0.0:
            raise ValueError("target_total must be finite and non-negative")
        score = np.abs(costs - float(target_total) / float(count))
    elif mode == "min":
        score = costs
    elif mode == "max":
        score = -costs
    else:
        raise ValueError(f"unknown matching mode {mode!r}")
    order = np.lexsort((_tie_key(ii, jj, seed=seed, row=row), score))
    used_left = np.zeros(np.asarray(left).size, dtype=bool)
    # For active/active matching the two endpoint views are the same vertex
    # set.  A matching must therefore consume both endpoints globally rather
    # than allowing one endpoint to reappear on the opposite side.
    used_right = used_left if same_set else np.zeros(np.asarray(right).size, dtype=bool)
    selected: list[tuple[int, int]] = []
    dose = 0.0
    for position in order:
        i = int(ii[position])
        j = int(jj[position])
        if used_left[i] or used_right[j]:
            continue
        used_left[i] = True
        used_right[j] = True
        selected.append((i, j))
        dose += float(costs[position])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} disjoint positive-dose pairs available; need {count}")
    return np.asarray(selected, dtype=np.int64), float(dose)


def apply_swap(row_values: np.ndarray, left_columns: np.ndarray, right_columns: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Swap values in one row while retaining the exact row multiset."""

    out = np.asarray(row_values, dtype=np.float32).copy()
    left_columns = np.asarray(left_columns, dtype=np.int64)
    right_columns = np.asarray(right_columns, dtype=np.int64)
    for left_idx, right_idx in np.asarray(pairs, dtype=np.int64):
        left = int(left_columns[int(left_idx)])
        right = int(right_columns[int(right_idx)])
        out[left], out[right] = out[right], out[left]
    return out


def _dose(row_values: np.ndarray, corrupted: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(corrupted, dtype=np.float64) - np.asarray(row_values, dtype=np.float64))))


def audit_swap(
    clean_row: np.ndarray,
    corrupted_row: np.ndarray,
    *,
    reference_support: np.ndarray,
    requested_changed_count: int,
    expect_support_change: bool,
    expected_support_change_count: int | None = None,
) -> dict[str, Any]:
    clean_row = np.asarray(clean_row, dtype=np.float32)
    corrupted_row = np.asarray(corrupted_row, dtype=np.float32)
    # A swap edge is retained when its *pair* dose is > PAIR_COST_EPS, i.e.
    # when the per-coordinate value difference is > PAIR_COST_EPS / 2.
    changed = np.abs(corrupted_row - clean_row) > protocol.PAIR_COST_EPS / 2.0
    support_before = np.asarray(reference_support, dtype=bool)
    support_after = np.abs(corrupted_row) >= np.maximum(
        1e-6,
        protocol.SUPPORT_THRESHOLD_RATIO * np.max(np.abs(clean_row)),
    )
    support_changed = support_before != support_after
    multiset_ok = bool(np.array_equal(np.sort(clean_row), np.sort(corrupted_row)))
    support_change_count = int(np.sum(support_changed))
    return {
        "exact_changed_count": bool(int(np.sum(changed)) == int(requested_changed_count)),
        "changed_count": int(np.sum(changed)),
        "requested_changed_count": int(requested_changed_count),
        "support_change_count": support_change_count,
        "support_change_positive": bool(support_change_count > 0),
        "support_change_zero": bool(support_change_count == 0),
        "support_change_exact": (
            True
            if expected_support_change_count is None
            else bool(support_change_count == int(expected_support_change_count))
        ),
        "support_expectation_ok": bool(
            support_change_count > 0 if expect_support_change else support_change_count == 0
        ),
        "row_value_multiset_ok": multiset_ok,
        "dose": _dose(clean_row, corrupted_row),
        "labels_used": False,
    }


def row_constructive_ranges(
    clean_row: np.ndarray,
    *,
    row: int,
    seed: int,
    pair_count: int,
) -> dict[str, Any]:
    """Build Cross/Preserve endpoint witnesses for one positive-budget row."""

    values = np.asarray(clean_row, dtype=np.float32)
    support = support_mask(values[None, :], reference=values[None, :])[0]
    active = np.flatnonzero(support)
    inactive = np.flatnonzero(~support)
    if pair_count <= 0:
        return {
            "pair_count": 0,
            "active_count": int(active.size),
            "inactive_count": int(inactive.size),
            "nonzero_budget": False,
            "common_exists": True,
            "common_width": 0.0,
            "cross_min": 0.0,
            "cross_max": 0.0,
            "preserve_min": 0.0,
            "preserve_max": 0.0,
        }
    try:
        cross_min_pairs, cross_min = greedy_matching(
            values[active], values[inactive], pair_count, mode="min", seed=seed, row=row
        )
        cross_max_pairs, cross_max = greedy_matching(
            values[active], values[inactive], pair_count, mode="max", seed=seed, row=row
        )
        preserve_min_pairs, preserve_min = greedy_matching(
            values[active], values[active], pair_count, mode="min", seed=seed, row=row, same_set=True
        )
        preserve_max_pairs, preserve_max = greedy_matching(
            values[active], values[active], pair_count, mode="max", seed=seed, row=row, same_set=True
        )
    except ValueError:
        # Tie-heavy or otherwise degenerate active rows are a valid feasibility
        # failure, not a runner crash.  D1 records them and keeps D2 locked.
        return {
            "pair_count": int(pair_count),
            "active_count": int(active.size),
            "inactive_count": int(inactive.size),
            "nonzero_budget": True,
            "common_exists": False,
            "common_width": 0.0,
            "range_failure": "insufficient_positive_dose_matching",
        }
    del cross_min_pairs, cross_max_pairs, preserve_min_pairs, preserve_max_pairs
    common_low = max(float(cross_min), float(preserve_min))
    common_high = min(float(cross_max), float(preserve_max))
    return {
        "pair_count": int(pair_count),
        "active_count": int(active.size),
        "inactive_count": int(inactive.size),
        "nonzero_budget": True,
        "common_exists": bool(common_high > common_low + protocol.DOSE_EPS),
        "common_width": float(max(0.0, common_high - common_low)),
        "range_failure": None,
        "common_low": float(common_low),
        "common_high": float(common_high),
        "cross_min": float(cross_min),
        "cross_max": float(cross_max),
        "preserve_min": float(preserve_min),
        "preserve_max": float(preserve_max),
    }


def build_common_dose_row(
    clean_row: np.ndarray,
    *,
    row: int,
    seed: int,
    pair_count: int,
    ranges: dict[str, Any],
) -> dict[str, Any]:
    """Construct both arms at the frozen common-interval midpoint."""

    values = np.asarray(clean_row, dtype=np.float32)
    support = support_mask(values[None, :], reference=values[None, :])[0]
    active = np.flatnonzero(support)
    inactive = np.flatnonzero(~support)
    if pair_count <= 0:
        return {
            "target_dose": 0.0,
            "cross_dose": 0.0,
            "preserve_dose": 0.0,
            "row_relative_mismatch": 0.0,
            "match_ok": True,
            "cross_audit": audit_swap(values, values, reference_support=support, requested_changed_count=0, expect_support_change=False),
            "preserve_audit": audit_swap(values, values, reference_support=support, requested_changed_count=0, expect_support_change=False),
        }
    if not bool(ranges.get("common_exists", False)):
        return {"target_dose": None, "match_ok": False, "reason": "no_common_interval"}
    target = 0.5 * (float(ranges["common_low"]) + float(ranges["common_high"]))
    cross_pairs, _ = greedy_matching(
        values[active], values[inactive], pair_count, mode="target", seed=seed, row=row, target_total=target
    )
    preserve_pairs, _ = greedy_matching(
        values[active], values[active], pair_count, mode="target", seed=seed, row=row, same_set=True, target_total=target
    )
    cross = apply_swap(values, active, inactive, cross_pairs)
    preserve = apply_swap(values, active, active, preserve_pairs)
    cross_audit = audit_swap(
        values,
        cross,
        reference_support=support,
        requested_changed_count=2 * pair_count,
        expect_support_change=True,
        expected_support_change_count=2 * pair_count,
    )
    preserve_audit = audit_swap(
        values,
        preserve,
        reference_support=support,
        requested_changed_count=2 * pair_count,
        expect_support_change=False,
        expected_support_change_count=0,
    )
    cross_dose = float(cross_audit["dose"])
    preserve_dose = float(preserve_audit["dose"])
    mismatch = abs(cross_dose - preserve_dose) / max(cross_dose, protocol.DOSE_EPS)
    return {
        "target_dose": float(target),
        "cross_dose": cross_dose,
        "preserve_dose": preserve_dose,
        "row_relative_mismatch": float(mismatch),
        "match_ok": bool(
            cross_audit["exact_changed_count"]
            and preserve_audit["exact_changed_count"]
            and cross_audit["support_expectation_ok"]
            and preserve_audit["support_expectation_ok"]
            and cross_audit["row_value_multiset_ok"]
            and preserve_audit["row_value_multiset_ok"]
        ),
        "cross_audit": cross_audit,
        "preserve_audit": preserve_audit,
    }


def dataset_budget(clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = support_mask(clean, reference=clean)
    return row_budgets(clean, rate=protocol.CORRUPTION_RATE, reference_support=support)
