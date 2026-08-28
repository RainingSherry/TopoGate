"""Finite, label-free static corruption library for C2.

The six primary arms share an exact per-row changed-coordinate budget.  The
library operates on a matrix and optional frozen difficulty scores only; it
never accepts labels, clustering metrics, or a learned selector.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import protocol


EPS = 1e-7


def support_thresholds(clean: np.ndarray, ratio: float = protocol.H0_SUPPORT_THRESHOLD_RATIO) -> np.ndarray:
    matrix = np.asarray(clean, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("clean matrix must be two-dimensional")
    row_max = np.max(np.abs(matrix), axis=1, keepdims=True)
    return np.maximum(1e-6, float(ratio) * row_max).astype(np.float32)


def support_mask(
    matrix: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    ratio: float = protocol.H0_SUPPORT_THRESHOLD_RATIO,
) -> np.ndarray:
    """Return support using a fixed clean-row threshold when supplied.

    A fixed reference is important here: recomputing a row threshold after a
    corruption can turn a pure value change into an apparent support change.
    The old B1 project is read-only and keeps its historical dynamic helper;
    this new project freezes the more auditable clean-reference semantics.
    """

    values = np.asarray(matrix, dtype=np.float32)
    ref = values if reference is None else np.asarray(reference, dtype=np.float32)
    if values.shape != ref.shape:
        raise ValueError("matrix/reference shapes differ")
    thresholds = support_thresholds(ref, ratio)
    return np.abs(values) >= thresholds


def row_budgets(
    clean: np.ndarray,
    *,
    rate: float = protocol.CORRUPTION_RATE,
    reference_support: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact changed-coordinate budgets and feasible pair counts."""

    matrix = np.asarray(clean, dtype=np.float32)
    if reference_support is None:
        active = support_mask(matrix, reference=matrix)
    else:
        active = np.asarray(reference_support, dtype=bool)
        if active.shape != matrix.shape:
            raise ValueError("reference_support must match clean shape")
    active_count = np.sum(active, axis=1).astype(np.int64)
    inactive_count = matrix.shape[1] - active_count
    pairs = np.minimum.reduce(
        [
            np.ceil(float(rate) * active_count).astype(np.int64),
            active_count // 2,
            inactive_count,
        ]
    )
    return (2 * pairs).astype(np.int64), pairs.astype(np.int64)


def residual_proxy(clean: np.ndarray) -> np.ndarray:
    """Compute a deterministic, label-free residual/difficulty proxy.

    This is a fallback for toy/structural diagnostics only.  Formal C2
    performance runs must pass an explicitly frozen residual artifact from the
    common warm-up protocol; ``corrupt_matrix(P4_ResidualHard)`` therefore
    raises when no score matrix is supplied.
    """

    matrix = np.asarray(clean, dtype=np.float32)
    med = np.median(matrix, axis=0, keepdims=True)
    mad = np.median(np.abs(matrix - med), axis=0, keepdims=True)
    scale = np.maximum(1e-6, 1.4826 * mad)
    return (np.abs(matrix - med) / scale).astype(np.float32)


def geometry_importance(clean: np.ndarray, *, k: int = protocol.GEOMETRY_K) -> np.ndarray:
    """Approximate per-cell/per-feature local-geometry sensitivity.

    ``g_ij = |x_ij| * |x_ij - mean_{k in NN(i)} x_kj|``.  The nearest-neighbour
    search is label-free and uses cosine distance on the supplied matrix.
    """

    matrix = np.asarray(clean, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("clean matrix must be two-dimensional")
    n = matrix.shape[0]
    if n < 2:
        return np.zeros_like(matrix, dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors

    # ``kneighbors(X=None)`` queries the fitted rows and returns neighbours
    # excluding each row itself.  Therefore the largest legal request is
    # ``n - 1``; asking for ``neighbors + 1`` accidentally requests ``n`` on
    # small fixtures and sklearn rejects it (``n_neighbors < n_samples``).
    neighbors = min(max(1, int(k)), n - 1)
    model = NearestNeighbors(n_neighbors=neighbors, metric="cosine", algorithm="auto")
    model.fit(matrix)
    indices = model.kneighbors(return_distance=False)
    local_mean = np.mean(matrix[indices], axis=1)
    return (np.abs(matrix) * np.abs(matrix - local_mean)).astype(np.float32)


def _choice_without_replacement(candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    if count > candidates.size:
        raise ValueError(f"requested {count} positions from {candidates.size} candidates")
    return np.asarray(rng.choice(candidates, size=count, replace=False), dtype=np.int64)


def _other_row(row: int, n: int, rng: np.random.Generator) -> int:
    if n <= 1:
        return row
    donor = int(rng.integers(0, n - 1))
    return donor if donor < row else donor + 1


def _forced_value(
    candidate: float,
    old: float,
    *,
    threshold: float,
    scale: float,
    rng: np.random.Generator,
    preserve_support: bool,
) -> float:
    value = float(candidate)
    if preserve_support and abs(value) < threshold:
        sign = 1.0 if old >= 0.0 else -1.0
        value = sign * max(abs(old) * 1.05, threshold * 1.05, scale * 0.1, 1e-4)
    elif not preserve_support and abs(value) < threshold:
        value = max(threshold * 1.1, scale * 0.1, 1e-4)
    if abs(value - old) <= EPS:
        sign = -1.0 if old > 0.0 else 1.0
        value = sign * max(abs(old) * 1.05, threshold * 1.1, scale * 0.1, 1e-4)
    return float(value)


def _active_scores(
    row: int,
    active_idx: np.ndarray,
    scores: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    descending: bool,
) -> np.ndarray:
    values = np.asarray(scores[row, active_idx], dtype=np.float64)
    # A tiny deterministic-per-call jitter only breaks exact ties; it does not
    # create a tunable sweep and keeps the selected set seed-auditable.
    jitter = rng.uniform(0.0, 1e-12, size=values.size)
    order = np.argsort(-(values + jitter) if descending else (values + jitter), kind="stable")
    return np.asarray(active_idx[order[:count]], dtype=np.int64)


def corrupt_matrix(
    clean: np.ndarray,
    principle: str,
    rng: np.random.Generator,
    *,
    residual_scores: np.ndarray | None = None,
    geometry_scores: np.ndarray | None = None,
    rate: float = protocol.CORRUPTION_RATE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one primary static principle under an exact changed-count budget."""

    if principle not in protocol.PRINCIPLES:
        raise ValueError(f"unknown primary principle {principle!r}")
    matrix = np.asarray(clean, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("clean must be a finite two-dimensional matrix")
    n, d = matrix.shape
    thresholds = support_thresholds(matrix)
    active = support_mask(matrix, reference=matrix)
    requested, pair_counts = row_budgets(matrix, rate=rate)

    if principle == "P4_ResidualHard":
        if residual_scores is None:
            raise ValueError("P4_ResidualHard requires frozen residual_scores")
        residual_scores = np.asarray(residual_scores, dtype=np.float32)
        if residual_scores.shape != matrix.shape or not np.isfinite(residual_scores).all():
            raise ValueError("residual_scores must be finite and match clean shape")
    if principle == "P5_GeometryHard":
        if geometry_scores is None:
            geometry_scores = geometry_importance(matrix)
        geometry_scores = np.asarray(geometry_scores, dtype=np.float32)
        if geometry_scores.shape != matrix.shape or not np.isfinite(geometry_scores).all():
            raise ValueError("geometry_scores must be finite and match clean shape")

    corrupted = matrix.copy()
    changed = np.zeros((n, d), dtype=bool)
    source = np.zeros((n, d), dtype=bool)
    destination = np.zeros((n, d), dtype=bool)
    value_changed = np.zeros((n, d), dtype=bool)
    scale = float(np.median(np.abs(matrix[active]))) if np.any(active) else 1.0
    scale = max(scale, 1e-4)

    prevalence = np.mean(active, axis=0)
    for row in range(n):
        active_idx = np.flatnonzero(active[row])
        inactive_idx = np.flatnonzero(~active[row])
        count = int(requested[row])
        pair_count = int(pair_counts[row])
        if count <= 0:
            continue

        if principle == "P0_Random":
            positions = _choice_without_replacement(np.arange(d), count, rng)
            for col in positions:
                donor = _other_row(row, n, rng)
                old = float(corrupted[row, col])
                candidate = float(matrix[donor, col])
                new_value = _forced_value(
                    candidate,
                    old,
                    threshold=float(thresholds[row, 0]),
                    scale=scale,
                    rng=rng,
                    preserve_support=bool(active[row, col]),
                )
                corrupted[row, col] = new_value
                changed[row, col] = True
                value_changed[row, col] = bool(active[row, col])
        elif principle == "P1_SupportPreserve":
            positions = _choice_without_replacement(active_idx, count, rng)
            for col in positions:
                donor = _other_row(row, n, rng)
                old = float(corrupted[row, col])
                candidate = float(matrix[donor, col])
                corrupted[row, col] = _forced_value(
                    candidate,
                    old,
                    threshold=float(thresholds[row, 0]),
                    scale=scale,
                    rng=rng,
                    preserve_support=True,
                )
                changed[row, col] = True
                value_changed[row, col] = True
        elif principle == "P2_SupportTarget":
            sources = _choice_without_replacement(active_idx, pair_count, rng)
            destinations = _choice_without_replacement(inactive_idx, pair_count, rng)
            source[row, sources] = True
            destination[row, destinations] = True
            for src, dst in zip(sources, destinations, strict=True):
                # H0 is dense and the frozen support is a threshold proxy, so
                # a threshold-inactive destination can still contain a small
                # raw nonzero value. Swap values to move the support role
                # without losing that value from the row multiset.
                source_value = float(matrix[row, src])
                destination_value = float(matrix[row, dst])
                corrupted[row, src] = destination_value
                corrupted[row, dst] = source_value
                changed[row, src] = True
                changed[row, dst] = True
        else:
            if principle == "P3_FrequencyAware":
                # Rare active features are targeted, with no frequency sweep.
                positions = _active_scores(row, active_idx, np.broadcast_to(-prevalence, matrix.shape), count, rng, descending=True)
            elif principle == "P4_ResidualHard":
                positions = _active_scores(row, active_idx, residual_scores, count, rng, descending=True)  # type: ignore[arg-type]
            elif principle == "P5_GeometryHard":
                positions = _active_scores(row, active_idx, geometry_scores, count, rng, descending=True)  # type: ignore[arg-type]
            else:  # pragma: no cover - guarded by PRINCIPLES above
                raise AssertionError(principle)
            for col in positions:
                donor = _other_row(row, n, rng)
                old = float(corrupted[row, col])
                candidate = float(matrix[donor, col])
                corrupted[row, col] = _forced_value(
                    candidate,
                    old,
                    threshold=float(thresholds[row, 0]),
                    scale=scale,
                    rng=rng,
                    preserve_support=True,
                )
                changed[row, col] = True
                value_changed[row, col] = True

    support_after = support_mask(corrupted, reference=matrix)
    support_changed = active != support_after
    row_counts = np.sum(changed, axis=1).astype(np.int64)
    audit: dict[str, Any] = {
        "principle": principle,
        "requested_changed_counts": requested,
        "effective_changed_counts": row_counts,
        "exact_budget": bool(np.array_equal(row_counts, requested)),
        "active_support_before": active,
        "active_support_after": support_after,
        "changed_mask": changed,
        "value_changed_mask": value_changed,
        "support_changed_mask": support_changed,
        "source_mask": source,
        "destination_mask": destination,
        "effective_changed_coordinate_rate": float(np.mean(changed)),
        "support_change_rate": float(np.mean(support_changed)),
        "value_change_rate": float(np.mean(value_changed)),
        "total_absolute_change": float(np.sum(np.abs(corrupted - matrix), dtype=np.float64)),
        "mean_changed_count": float(np.mean(row_counts)),
        "mean_requested_count": float(np.mean(requested)),
        "labels_used": False,
    }
    return corrupted.astype(np.float32, copy=False), audit


def compact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Drop masks from an audit before serializing a compact result."""

    compact: dict[str, Any] = {}
    for key, value in audit.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (np.bool_,)):
            compact[key] = bool(value)
        elif isinstance(value, (np.integer,)):
            compact[key] = int(value)
        elif isinstance(value, (np.floating,)):
            compact[key] = float(value)
        else:
            compact[key] = value
    return compact


def geometry_safe_fixture(
    clean: np.ndarray,
    rng: np.random.Generator,
    *,
    geometry_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Paired low-geometry fixture; never a formal primary arm."""

    if geometry_scores is None:
        geometry_scores = geometry_importance(clean)
    scores = np.asarray(geometry_scores, dtype=np.float32)
    matrix = np.asarray(clean, dtype=np.float32)
    active = support_mask(matrix, reference=matrix)
    requested, _ = row_budgets(matrix)
    low_scores = -scores
    # Reuse the exact P5 mechanism while selecting the low-score tail.  The
    # private score hook avoids adding a seventh public principle name.
    out, audit = _corrupt_value_by_scores(matrix, low_scores, requested, active, rng, "P5_GeometrySafe")
    return out, audit


def _corrupt_value_by_scores(
    matrix: np.ndarray,
    scores: np.ndarray,
    requested: np.ndarray,
    active: np.ndarray,
    rng: np.random.Generator,
    principle: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Internal low-score fixture helper with the same budget contract."""

    corrupted = matrix.copy()
    n, d = matrix.shape
    thresholds = support_thresholds(matrix)
    changed = np.zeros((n, d), dtype=bool)
    value_changed = np.zeros((n, d), dtype=bool)
    scale = float(np.median(np.abs(matrix[active]))) if np.any(active) else 1.0
    for row in range(n):
        idx = np.flatnonzero(active[row])
        count = int(requested[row])
        if count <= 0:
            continue
        positions = _active_scores(row, idx, scores, count, rng, descending=True)
        for col in positions:
            donor = _other_row(row, n, rng)
            corrupted[row, col] = _forced_value(
                float(matrix[donor, col]),
                float(matrix[row, col]),
                threshold=float(thresholds[row, 0]),
                scale=scale,
                rng=rng,
                preserve_support=True,
            )
            changed[row, col] = True
            value_changed[row, col] = True
    after = support_mask(corrupted, reference=matrix)
    return corrupted.astype(np.float32), {
        "principle": principle,
        "requested_changed_counts": requested,
        "effective_changed_counts": np.sum(changed, axis=1).astype(np.int64),
        "exact_budget": bool(np.array_equal(np.sum(changed, axis=1), requested)),
        "active_support_before": active,
        "active_support_after": after,
        "changed_mask": changed,
        "value_changed_mask": value_changed,
        "support_changed_mask": active != after,
        "effective_changed_coordinate_rate": float(np.mean(changed)),
        "support_change_rate": float(np.mean(active != after)),
        "value_change_rate": float(np.mean(value_changed)),
        "total_absolute_change": float(np.sum(np.abs(corrupted - matrix), dtype=np.float64)),
        "labels_used": False,
    }
