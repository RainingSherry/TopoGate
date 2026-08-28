from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class V17Config:
    """Configuration for the label-free V17 reference solver."""

    seed: int = 42
    input_mode: str = "auto"
    projection_views: int = 3
    projection_dim: int = 128
    projection_density: str | float = "auto"
    candidate_k: int = 20
    candidate_union_k: int = 40
    candidate_block_size: int = 128
    lambda_l1: float = 0.02
    lambda_l2: float = 0.001
    lambda_outlier: float = 0.25
    solver_max_iter: int = 300
    solver_tol: float = 1e-5
    coefficient_epsilon: float = 1e-8
    spectral_n_init: int = 20
    degree_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.input_mode not in {"auto", "count", "nonnegative", "continuous"}:
            raise ValueError("input_mode must be auto, count, nonnegative, or continuous")
        if self.projection_views <= 0 or self.projection_dim <= 0:
            raise ValueError("projection_views and projection_dim must be positive")
        if isinstance(self.projection_density, float) and not 0.0 < self.projection_density <= 1.0:
            raise ValueError("projection_density must be 'auto' or a value in (0, 1]")
        if self.projection_density != "auto" and not isinstance(self.projection_density, float):
            raise ValueError("projection_density must be 'auto' or a float")
        if self.candidate_k <= 0 or self.candidate_union_k <= 0:
            raise ValueError("candidate budgets must be positive")
        if self.candidate_union_k < self.candidate_k:
            raise ValueError("candidate_union_k must be at least candidate_k")
        if self.candidate_block_size <= 0:
            raise ValueError("candidate_block_size must be positive")
        if self.lambda_l1 < 0.0 or self.lambda_l2 < 0.0 or self.lambda_outlier <= 0.0:
            raise ValueError("regularization weights must be non-negative and lambda_outlier positive")
        if self.solver_max_iter <= 0 or self.solver_tol <= 0.0:
            raise ValueError("solver_max_iter and solver_tol must be positive")
        if self.coefficient_epsilon < 0.0 or self.degree_epsilon < 0.0:
            raise ValueError("numerical thresholds must be non-negative")
        if self.spectral_n_init <= 0:
            raise ValueError("spectral_n_init must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> V17Config:
    payload: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"V17 config must be a mapping: {path}")
        payload.update(loaded)
    if overrides:
        payload.update({str(key): value for key, value in overrides.items() if value is not None})
    unknown = sorted(set(payload) - set(V17Config.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown V17 config keys: {unknown}")
    return V17Config(**payload)
