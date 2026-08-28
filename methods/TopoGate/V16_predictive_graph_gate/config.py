from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class V16Config:
    """Small, fixed configuration for the predictive-support experiment."""

    seed: int = 42
    hidden_dim: int = 128
    latent_dim: int = 32
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    mask_ratio: float = 0.30
    zero_sample_ratio: float = 0.01
    thinning_fraction: float = 0.50
    support_repeats: int = 3
    graph_k: int = 20
    smoothing: float = 1e-3
    gate_temperature: float = 0.5
    n_init: int = 10
    min_feature_dim: int = 2000
    min_zero_fraction: float = 0.80
    min_median_nnz: float = 5.0
    max_empty_fraction: float = 0.10
    require_sparse_input: bool = True
    enforce_domain: bool = True
    variant: str = "V16_predictive_gate"
    no_cuda: bool = False
    gpu: int = 1

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("hidden_dim and latent_dim must be positive")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if not 0.0 <= self.zero_sample_ratio <= 1.0:
            raise ValueError("zero_sample_ratio must be in [0, 1]")
        if not 0.0 < self.thinning_fraction < 1.0:
            raise ValueError("thinning_fraction must be in (0, 1)")
        if self.support_repeats <= 0 or self.graph_k <= 0:
            raise ValueError("support_repeats and graph_k must be positive")
        if self.smoothing <= 0.0 or self.gate_temperature <= 0.0:
            raise ValueError("smoothing and gate_temperature must be positive")
        if self.min_feature_dim <= 0 or self.min_median_nnz < 0.0:
            raise ValueError("domain thresholds must be non-negative")
        if not 0.0 <= self.min_zero_fraction <= 1.0:
            raise ValueError("min_zero_fraction must be in [0, 1]")
        if not 0.0 <= self.max_empty_fraction <= 1.0:
            raise ValueError("max_empty_fraction must be in [0, 1]")
        valid = {
            "self_only",
            "fixed_predictive_graph",
            "V16_predictive_gate",
            "shuffled_support",
            "output_disabled",
        }
        if self.variant not in valid:
            raise ValueError(f"variant must be one of {sorted(valid)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> V16Config:
    payload: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"V16 config must be a mapping: {path}")
        payload.update(loaded)
    if overrides:
        payload.update({str(k): v for k, v in overrides.items()})
    unknown = sorted(set(payload) - set(V16Config.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown V16 config keys: {unknown}")
    return V16Config(**payload)
