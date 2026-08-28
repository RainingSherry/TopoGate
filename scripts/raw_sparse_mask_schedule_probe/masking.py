"""Deterministic raw-sparse masking and audit helpers.

The masking functions are intentionally independent of labels, clustering
metrics, and model state.  A mask is sampled per row from the exact raw
zero/non-zero support after the zero-preserving adapter.  The returned audit
scalars are compact and are suitable for a run summary; mask arrays are never
written by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import protocol


@dataclass(frozen=True)
class MaskBatch:
    corrupted: np.ndarray
    mask: np.ndarray
    sampled_ratio: np.ndarray
    audit: dict[str, Any]


def _validate_inputs(x: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(x, dtype=np.float32)
    support = np.asarray(active, dtype=bool)
    if value.ndim != 2 or support.shape != value.shape:
        raise ValueError("x and active must be two-dimensional arrays with equal shape")
    if not np.isfinite(value).all():
        raise ValueError("mask input contains non-finite values")
    if not np.array_equal(value != 0.0, support):
        raise ValueError("active support must equal x != 0.0")
    return value, support


def _seed(seed: int, epoch: int, stream: int = 0) -> int:
    # Keep this arithmetic stable across Python processes and architectures.
    return int((int(seed) * 1_000_003 + int(epoch) * 97_409 + int(stream) * 7_919) % (2**63 - 1))


def _sample_ratio(n_rows: int, schedule: str, seed: int, epoch: int, fixed_ratio: float | None = None, stream: int = 0) -> np.ndarray:
    if schedule == "FIXED":
        ratio = protocol.FIXED_MASK_RATIO if fixed_ratio is None else float(fixed_ratio)
        if not (0.0 < ratio < 1.0):
            raise ValueError("fixed mask ratio must lie in (0,1)")
        return np.full(n_rows, ratio, dtype=np.float32)
    if schedule != "VARIABLE":
        raise ValueError(f"unknown mask schedule: {schedule}")
    rng = np.random.default_rng(_seed(seed, epoch, 11 + int(stream)))
    return rng.uniform(protocol.VARIABLE_MASK_LOW, protocol.VARIABLE_MASK_HIGH, size=n_rows).astype(np.float32)


def _sample_row_indices(candidates: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0 or candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    if count > candidates.size:
        count = int(candidates.size)
    # choice without replacement is deterministic for a fixed Generator and
    # does not depend on values or labels.
    return np.asarray(rng.choice(candidates, size=count, replace=False), dtype=np.int64)


def make_mask(
    x: np.ndarray,
    active: np.ndarray,
    *,
    target_space: str,
    schedule: str,
    seed: int,
    epoch: int,
    stream: int = 0,
    fixed_ratio: float | None = None,
) -> MaskBatch:
    """Return a deterministic zeroing mask and compact audit statistics.

    ``ALL`` samples from all coordinates while ``ACTIVE`` samples only from
    exact non-zero coordinates.  Both use the same nominal per-row budget
    ``ceil(p_i * a_i)`` where ``a_i`` is the number of active coordinates.
    ``ACTIVE`` additionally caps the count at ``a_i``.  Rows with no active
    values have a zero budget and are returned unchanged.
    """
    value, support = _validate_inputs(x, active)
    target = str(target_space).upper()
    schedule = str(schedule).upper()
    if target not in protocol.MASK_TARGETS:
        raise ValueError(f"unknown mask target space: {target_space}")
    ratios = _sample_ratio(value.shape[0], schedule, seed, epoch, fixed_ratio=fixed_ratio, stream=stream)
    rng = np.random.default_rng(_seed(seed, epoch, 23 + int(stream)))
    mask = np.zeros(value.shape, dtype=bool)
    active_counts = support.sum(axis=1).astype(np.int64)
    requested = np.ceil(ratios.astype(np.float64) * active_counts).astype(np.int64)
    requested = np.minimum(requested, value.shape[1])
    selected = np.zeros(value.shape[0], dtype=np.int64)
    for row in range(value.shape[0]):
        count = int(requested[row])
        if target == "ACTIVE":
            candidates = np.flatnonzero(support[row])
            count = min(count, int(candidates.size))
        else:
            candidates = np.arange(value.shape[1], dtype=np.int64)
        picked = _sample_row_indices(candidates, count, rng)
        if picked.size:
            mask[row, picked] = True
            selected[row] = int(picked.size)

    corrupted = value.copy()
    corrupted[mask] = 0.0
    selected_nonzero = int(np.count_nonzero(mask & support))
    changed = int(np.count_nonzero(mask & (value != 0.0)))
    total_selected = int(mask.sum())
    zero_budget_rows = int(np.count_nonzero(active_counts == 0))
    audit = {
        "target_space": target,
        "schedule": schedule,
        "seed": int(seed),
        "epoch": int(epoch),
        "requested_mask_count_total": int(requested.sum()),
        "selected_mask_count_total": total_selected,
        "selected_nonzero_count_total": selected_nonzero,
        "selected_nonzero_fraction": float(selected_nonzero / max(total_selected, 1)),
        "actual_value_change_count_total": changed,
        "actual_value_change_fraction": float(changed / max(value.size, 1)),
        "masked_target_zero_fraction": float(np.count_nonzero(mask & (value == 0.0)) / max(total_selected, 1)),
        "zero_budget_rows": zero_budget_rows,
        "mean_sampled_mask_ratio": float(np.mean(ratios)) if ratios.size else 0.0,
        "std_sampled_mask_ratio": float(np.std(ratios)) if ratios.size else 0.0,
        "sampled_ratio_min": float(np.min(ratios)) if ratios.size else 0.0,
        "sampled_ratio_max": float(np.max(ratios)) if ratios.size else 0.0,
        "nominal_row_budget_min": int(np.min(requested)) if requested.size else 0,
        "nominal_row_budget_max": int(np.max(requested)) if requested.size else 0,
        "mask_count_exact": bool(np.array_equal(selected, np.minimum(requested, active_counts if target == "ACTIVE" else value.shape[1]))),
    }
    if schedule == "FIXED" and audit["std_sampled_mask_ratio"] != 0.0:
        raise AssertionError("fixed schedule ratio is not constant")
    if not (protocol.VARIABLE_MASK_LOW <= audit["sampled_ratio_min"] <= audit["sampled_ratio_max"] <= protocol.VARIABLE_MASK_HIGH):
        raise AssertionError("variable mask ratio escaped the frozen range")
    if target == "ACTIVE" and np.any(mask & ~support):
        raise AssertionError("ACTIVE mask selected an inactive coordinate")
    if np.any(corrupted[~support] != 0.0):
        raise AssertionError("zero-preserving mask changed an inactive coordinate")
    return MaskBatch(corrupted=corrupted, mask=mask, sampled_ratio=ratios, audit=audit)


def arm_to_spec(arm: str) -> tuple[str | None, str | None]:
    arm = str(arm).upper()
    table = {
        "CLEAN_AE": (None, None),
        "ALL_FIXED": ("ALL", "FIXED"),
        "ACTIVE_FIXED": ("ACTIVE", "FIXED"),
        "ALL_VARIABLE": ("ALL", "VARIABLE"),
        "ACTIVE_VARIABLE": ("ACTIVE", "VARIABLE"),
        "Z_FIXED": ("ALL", "FIXED"),
        "Z_VARIABLE": ("ALL", "VARIABLE"),
    }
    if arm not in table:
        raise ValueError(f"unknown main arm: {arm}")
    return table[arm]


def mask_for_arm(x: np.ndarray, active: np.ndarray, arm: str, *, seed: int, epoch: int, stream: int = 0) -> MaskBatch | None:
    target, schedule = arm_to_spec(arm)
    if target is None:
        return None
    return make_mask(x, active, target_space=target, schedule=schedule, seed=seed, epoch=epoch, stream=stream)


def masked_mse(prediction: Any, target: Any, mask: Any) -> Any:
    """Selected-only MSE; return a graph-connected zero for empty masks."""
    import torch

    selected = torch.as_tensor(mask, dtype=torch.bool, device=prediction.device)
    if bool(selected.any()):
        return torch.mean((prediction[selected] - target[selected]) ** 2)
    return prediction.sum() * 0.0


def audit_masked_loss(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    value = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return 0.0
    return float(np.mean((value[selected] - truth[selected]) ** 2))
