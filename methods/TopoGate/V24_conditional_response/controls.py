from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MARGINAL_FEATURE_NAMES = (
    "mean_abs_delta_effective",
    "rms_delta_effective",
    "mean_abs_source_effective",
    "mean_abs_donor_effective",
    "mean_global_nonzero_frequency_effective",
    "mean_standardized_abs_source_effective",
    "mean_standardized_abs_donor_effective",
    "mean_source_two_sided_quantile_surprisal_effective",
    "mean_donor_two_sided_quantile_surprisal_effective",
)


@dataclass(frozen=True)
class MarginalControlBundle:
    support: np.ndarray
    marginal: np.ndarray
    diagnostics: dict[str, object]


def _feature_reference(
    matrix: np.ndarray,
    *,
    relative_scale_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nonzero_frequency = np.mean(matrix != 0.0, axis=0).astype(np.float32)
    # In a 90%-sparse matrix the all-value median and MAD are usually both
    # zero. Reference nonzero values instead, so this nuisance channel does
    # not turn an ordinary nonzero amplitude into an artificial 1e6 outlier.
    median = np.zeros(matrix.shape[1], dtype=np.float32)
    scale = np.ones(matrix.shape[1], dtype=np.float32)
    for feature in range(matrix.shape[1]):
        nonzero = matrix[:, feature][matrix[:, feature] != 0.0]
        if not nonzero.size:
            continue
        center = float(np.median(nonzero))
        mad = float(np.median(np.abs(nonzero - center)))
        fallback = float(np.std(nonzero))
        median[feature] = center
        # A nearly constant nonzero feature can have MAD and standard deviation
        # close to zero. Anchor its scale to the feature magnitude as well; a
        # fixed absolute epsilon would turn an ordinary donor mismatch into an
        # artificial 1e5--1e6 nuisance value on sparse matrices.
        relative_floor = abs(center) * float(relative_scale_floor)
        scale[feature] = max(1.4826 * mad, fallback, relative_floor, 1e-6)
    sorted_values = np.sort(matrix, axis=0).astype(np.float32)
    return nonzero_frequency, median, scale, sorted_values


def _two_sided_quantile_surprisal(values: np.ndarray, sorted_values: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Compute empirical two-sided surprisal for selected feature-value pairs."""

    output = np.zeros(values.shape, dtype=np.float32)
    n_rows = sorted_values.shape[0]
    for local, feature in enumerate(features):
        ranks = np.searchsorted(sorted_values[:, feature], values[:, local], side="right")
        probability = np.clip(ranks.astype(np.float64) / max(1, n_rows), 1.0 / max(1, n_rows), 1.0)
        tail = np.clip(2.0 * np.minimum(probability, 1.0 - probability + 1.0 / max(1, n_rows)), 1e-8, 1.0)
        output[:, local] = -np.log(tail).astype(np.float32)
    return output


def build_marginal_controls(
    matrix: np.ndarray,
    masks: np.ndarray,
    donor_offsets: np.ndarray,
    *,
    standardized_clip: float = 10.0,
    relative_scale_floor: float = 0.01,
) -> MarginalControlBundle:
    """Build sample-by-intervention nuisance controls without labels or K.

    The support channel contains only effective changed count. The separate
    marginal channel deliberately excludes algebraic duplicates such as changed
    fraction, because the feature dimension is fixed by the protocol.
    """

    matrix = np.asarray(matrix, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.bool_)
    donor_offsets = np.asarray(donor_offsets, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or masks.ndim != 2 or masks.shape[1] != matrix.shape[1]:
        raise ValueError("matrix/masks must be two-dimensional with matching feature count")
    if donor_offsets.shape != (masks.shape[0],):
        raise ValueError("donor_offsets must provide one donor per intervention")
    if matrix.shape[0] <= 1:
        raise ValueError("V24 marginal controls require at least two samples")
    if standardized_clip <= 0.0 or not 0.0 < relative_scale_floor < 1.0:
        raise ValueError("invalid standardized nuisance-control bounds")

    n_samples, _ = matrix.shape
    n_masks = masks.shape[0]
    nonzero_frequency, median, scale, sorted_values = _feature_reference(
        matrix,
        relative_scale_floor=relative_scale_floor,
    )
    support = np.zeros((n_samples, n_masks), dtype=np.float32)
    marginal = np.zeros((n_samples, n_masks, len(MARGINAL_FEATURE_NAMES)), dtype=np.float32)
    nonempty = 0
    standardized_clipped = 0
    standardized_effective = 0

    for mask_index, feature_mask in enumerate(masks):
        features = np.flatnonzero(feature_mask)
        if not features.size:
            continue
        donor = np.roll(matrix, shift=int(donor_offsets[mask_index]), axis=0)
        source_values = matrix[:, features]
        donor_values = donor[:, features]
        effective = source_values != donor_values
        count = effective.sum(axis=1).astype(np.float32)
        support[:, mask_index] = count
        safe_count = np.maximum(count, 1.0)
        delta = donor_values - source_values
        effective_float = effective.astype(np.float32)
        source_abs = np.abs(source_values)
        donor_abs = np.abs(donor_values)
        standardized_source_raw = np.abs((source_values - median[features][None, :]) / scale[features][None, :])
        standardized_donor_raw = np.abs((donor_values - median[features][None, :]) / scale[features][None, :])
        standardized_clipped += int(
            np.count_nonzero((standardized_source_raw > standardized_clip) & effective)
            + np.count_nonzero((standardized_donor_raw > standardized_clip) & effective)
        )
        standardized_effective += int(2 * np.count_nonzero(effective))
        standardized_source = np.minimum(standardized_source_raw, standardized_clip)
        standardized_donor = np.minimum(standardized_donor_raw, standardized_clip)
        source_surprisal = _two_sided_quantile_surprisal(source_values, sorted_values, features)
        donor_surprisal = _two_sided_quantile_surprisal(donor_values, sorted_values, features)

        marginal[:, mask_index, 0] = np.sum(np.abs(delta) * effective_float, axis=1) / safe_count
        marginal[:, mask_index, 1] = np.sqrt(np.sum((delta * delta) * effective_float, axis=1) / safe_count)
        marginal[:, mask_index, 2] = np.sum(source_abs * effective_float, axis=1) / safe_count
        marginal[:, mask_index, 3] = np.sum(donor_abs * effective_float, axis=1) / safe_count
        marginal[:, mask_index, 4] = np.sum(nonzero_frequency[features][None, :] * effective_float, axis=1) / safe_count
        marginal[:, mask_index, 5] = np.sum(standardized_source * effective_float, axis=1) / safe_count
        marginal[:, mask_index, 6] = np.sum(standardized_donor * effective_float, axis=1) / safe_count
        marginal[:, mask_index, 7] = np.sum(source_surprisal * effective_float, axis=1) / safe_count
        marginal[:, mask_index, 8] = np.sum(donor_surprisal * effective_float, axis=1) / safe_count
        nonempty += int(np.count_nonzero(count > 0.0))

    if not np.isfinite(support).all() or not np.isfinite(marginal).all():
        raise ValueError("marginal control construction produced non-finite values")
    flattened = np.concatenate((support, marginal.reshape(n_samples, -1)), axis=1)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    positive = singular[singular > 1e-10]
    diagnostics: dict[str, object] = {
        "support_semantics": "effective_changed_count",
        "marginal_feature_names": list(MARGINAL_FEATURE_NAMES),
        "n_samples": int(n_samples),
        "n_masks": int(n_masks),
        "effective_nonempty_sample_interventions": int(nonempty),
        "effective_nonempty_fraction": float(nonempty / max(1, n_samples * n_masks)),
        "control_effective_rank": int(positive.size),
        "control_condition_number": float(positive.max() / positive.min()) if positive.size else float("inf"),
        "marginal_relative_scale_floor": float(relative_scale_floor),
        "marginal_standardized_clip": float(standardized_clip),
        "marginal_standardized_clip_fraction_effective": float(standardized_clipped / max(1, standardized_effective)),
        "reference_scale_min": float(np.min(scale)),
        "labels_accessible": False,
        "K_accessible": False,
    }
    return MarginalControlBundle(support=support, marginal=marginal, diagnostics=diagnostics)
