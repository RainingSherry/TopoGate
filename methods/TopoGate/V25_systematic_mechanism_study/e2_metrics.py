"""Dataset/seed-level summaries for V25 E2-A feature audits.

Coordinate rows are intentionally treated as descriptive observations.  The
returned inference unit is one dataset/seed summary, so downstream analysis
cannot silently use millions of sample-feature coordinates as independent
replicates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def aggregate_selected_vs_eligible(
    selected: np.ndarray,
    eligible: np.ndarray,
    metrics: Mapping[str, np.ndarray],
    *,
    dataset_id: str,
    seed: int,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    if selected.shape != eligible.shape or selected.ndim != 2:
        raise ValueError("selected and eligible must be matching 2D arrays")
    if np.any(selected & ~eligible):
        raise ValueError("selected coordinates must be a subset of eligible coordinates")
    not_selected = eligible & ~selected
    result: dict[str, Any] = {
        "dataset_id": str(dataset_id),
        "seed": int(seed),
        "statistical_unit": "dataset_seed_summary",
        "coordinate_count": int(selected.size),
        "selected_coordinate_count": int(selected.sum()),
        "eligible_not_selected_coordinate_count": int(not_selected.sum()),
        "coordinate_distribution_is_descriptive_only": True,
        "metrics": {},
    }
    for name, values in metrics.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != selected.shape:
            raise ValueError(f"metric {name!r} shape does not match selection masks")
        if not np.isfinite(array).all():
            raise ValueError(f"metric {name!r} contains non-finite values")
        selected_values = array[selected]
        eligible_values = array[not_selected]
        result["metrics"][str(name)] = {
            "selected_mean": float(selected_values.mean()) if selected_values.size else None,
            "eligible_not_selected_mean": float(eligible_values.mean()) if eligible_values.size else None,
            "difference": float(selected_values.mean() - eligible_values.mean()) if selected_values.size and eligible_values.size else None,
            "selected_n_coordinates": int(selected_values.size),
            "eligible_not_selected_n_coordinates": int(eligible_values.size),
        }
    return result


@dataclass
class CoordinateMetricAccumulator:
    """Streaming reducer whose inferential unit is one dataset/seed.

    ``selected`` and ``eligible`` are coordinate masks for one schedule batch.
    The reducer stores sums and counts only; coordinate rows are never emitted
    as independent observations.  This keeps the audit usable for large
    sample-feature matrices while preserving the preregistered unit.
    """

    dataset_id: str
    seed: int
    coordinate_count: int = 0
    selected_coordinate_count: int = 0
    eligible_not_selected_coordinate_count: int = 0
    _sums_selected: dict[str, float] = field(default_factory=dict)
    _sums_other: dict[str, float] = field(default_factory=dict)

    def update(self, selected: np.ndarray, eligible: np.ndarray, metrics: Mapping[str, np.ndarray]) -> None:
        selected_mask = np.asarray(selected, dtype=bool)
        eligible_mask = np.asarray(eligible, dtype=bool)
        if selected_mask.shape != eligible_mask.shape or selected_mask.ndim != 2:
            raise ValueError("selected and eligible must be matching 2D arrays")
        if np.any(selected_mask & ~eligible_mask):
            raise ValueError("selected coordinates must be a subset of eligible coordinates")
        other_mask = eligible_mask & ~selected_mask
        self.coordinate_count += int(selected_mask.size)
        self.selected_coordinate_count += int(selected_mask.sum())
        self.eligible_not_selected_coordinate_count += int(other_mask.sum())
        for name, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            if array.shape != selected_mask.shape:
                raise ValueError(f"metric {name!r} shape does not match selection masks")
            if not np.isfinite(array).all():
                raise ValueError(f"metric {name!r} contains non-finite values")
            self._sums_selected[str(name)] = self._sums_selected.get(str(name), 0.0) + float(array[selected_mask].sum())
            self._sums_other[str(name)] = self._sums_other.get(str(name), 0.0) + float(array[other_mask].sum())

    def finalize(self) -> dict[str, Any]:
        names = sorted(set(self._sums_selected) | set(self._sums_other))
        metrics: dict[str, Any] = {}
        for name in names:
            selected_mean = (
                self._sums_selected.get(name, 0.0) / self.selected_coordinate_count
                if self.selected_coordinate_count
                else None
            )
            other_mean = (
                self._sums_other.get(name, 0.0) / self.eligible_not_selected_coordinate_count
                if self.eligible_not_selected_coordinate_count
                else None
            )
            metrics[name] = {
                "selected_mean": selected_mean,
                "eligible_not_selected_mean": other_mean,
                "difference": (selected_mean - other_mean) if selected_mean is not None and other_mean is not None else None,
                "selected_n_coordinates": self.selected_coordinate_count,
                "eligible_not_selected_n_coordinates": self.eligible_not_selected_coordinate_count,
            }
        return {
            "dataset_id": self.dataset_id,
            "seed": int(self.seed),
            "statistical_unit": "dataset_seed_summary",
            "coordinate_count": self.coordinate_count,
            "selected_coordinate_count": self.selected_coordinate_count,
            "eligible_not_selected_coordinate_count": self.eligible_not_selected_coordinate_count,
            "coordinate_distribution_is_descriptive_only": True,
            "metrics": metrics,
        }
