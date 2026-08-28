from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


WORLD_NAMES = (
    "W0_global_null",
    "W1_mean_only",
    "W2_support_only",
    "W3_marginal_only",
    "W4_dependency_only",
    "W5_mixed_realistic",
)

PRIMARY_SEEDS = (42, 123, 7, 2025, 2026)


@dataclass(frozen=True)
class V24Q1Config:
    """Frozen primary configuration for the conditional-utility death test."""

    protocol_id: str = "v24_conditional_incremental_response_q1_v2"
    probe_engine: str = "V23_cycle_response_frozen_probe"
    n_samples: int = 3000
    n_features: int = 1000
    n_clusters: int = 6
    zero_fraction: float = 0.90
    block_size: int = 20
    active_blocks_per_sample: int = 5
    fingerprint_masks: int = 64
    fingerprint_mask_ratio: float = 0.10
    dependency_rho: float = 0.65
    dependency_separation_min: float = 0.20
    support_cooccurrence_max: float = 1e-7
    marginal_equality_tolerance: float = 1e-5
    classifier_chance_ceiling: float = 0.52
    null_panel_mean_auc_margin: float = 0.01
    marginal_relative_scale_floor: float = 0.01
    marginal_standardized_clip: float = 10.0
    support_signal_auc_floor: float = 0.60
    mean_shift_min: float = 0.10
    marginal_dispersion_min: float = 0.10
    outer_folds: int = 5
    inner_folds: int = 4
    ridge_alpha: float = 1.0
    pair_count_per_fold: int = 2000
    bootstrap_replicates: int = 200
    calibration_replicates: int = 200
    equivalence_margin: float = 0.02
    null_point_margin: float = 0.01
    w4_delta_min: float = 0.02
    w5_delta_min: float = 0.01
    primary_seeds: tuple[int, ...] = PRIMARY_SEEDS

    def validate(self) -> None:
        if self.n_samples <= 0 or self.n_features <= 0 or self.n_clusters < 2:
            raise ValueError("n_samples/n_features must be positive and n_clusters must be >= 2")
        if self.n_samples % self.n_clusters:
            raise ValueError("n_samples must be divisible by n_clusters")
        if self.n_features % self.block_size:
            raise ValueError("n_features must be divisible by block_size")
        if not 0.0 < self.zero_fraction < 1.0:
            raise ValueError("zero_fraction must be in (0, 1)")
        if not 0 < self.active_blocks_per_sample < self.n_features // self.block_size:
            raise ValueError("active_blocks_per_sample must be within block range")
        expected_nonzero = self.active_blocks_per_sample * self.block_size
        if expected_nonzero != int(round(self.n_features * (1.0 - self.zero_fraction))):
            raise ValueError("block support must exactly realize zero_fraction")
        if self.fingerprint_masks <= 0 or not 0.0 < self.fingerprint_mask_ratio < 1.0:
            raise ValueError("invalid fingerprint mask configuration")
        if not 0.0 < self.dependency_rho < 1.0:
            raise ValueError("dependency_rho must be in (0, 1)")
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("at least two outer and inner folds are required")
        if self.ridge_alpha <= 0.0 or self.pair_count_per_fold <= 0:
            raise ValueError("ridge_alpha and pair_count_per_fold must be positive")
        if not 0.5 < self.support_signal_auc_floor <= 1.0:
            raise ValueError("support_signal_auc_floor must be in (0.5, 1]")
        if not 0.0 < self.null_panel_mean_auc_margin < 0.5:
            raise ValueError("null_panel_mean_auc_margin must be in (0, 0.5)")
        if not 0.0 < self.marginal_relative_scale_floor < 1.0:
            raise ValueError("marginal_relative_scale_floor must be in (0, 1)")
        if self.marginal_standardized_clip <= 0.0:
            raise ValueError("marginal_standardized_clip must be positive")
        if self.mean_shift_min <= 0.0 or self.marginal_dispersion_min <= 0.0:
            raise ValueError("control-signal thresholds must be positive")
        if self.bootstrap_replicates <= 0 or self.calibration_replicates <= 0:
            raise ValueError("bootstrap/calibration replicate counts must be positive")
        if len(self.primary_seeds) != 5 or len(set(self.primary_seeds)) != 5:
            raise ValueError("V24 primary protocol requires exactly five unique seeds")

    @property
    def n_blocks(self) -> int:
        return self.n_features // self.block_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
