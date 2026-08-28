from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class V16_1Config:
    """Fixed V16.1 protocol configuration.

    The values inherited from V16 are deliberately not exposed as a sweep.
    A different protocol must be represented by a new version, not by a
    per-dataset override.
    """

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
    consensus_min_repeats: int = 2
    smoothing: float = 1e-3
    gate_temperature: float = 0.5
    n_init: int = 10
    min_feature_dim: int = 2000
    min_zero_fraction: float = 0.80
    min_median_nnz: float = 5.0
    max_empty_fraction: float = 0.10
    require_sparse_input: bool = True
    enforce_domain: bool = True
    input_policy: str = "strict_legacy"
    variant: str = "V16_1_predictive_gate"
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
        if self.support_repeats != 3:
            raise ValueError("V16.1 fixes support_repeats to 3")
        if self.graph_k != 20:
            raise ValueError("V16.1 fixes graph_k to 20")
        if self.consensus_min_repeats != 2:
            raise ValueError("V16.1 fixes consensus_min_repeats to 2")
        if self.smoothing <= 0.0 or self.gate_temperature <= 0.0:
            raise ValueError("smoothing and gate_temperature must be positive")
        if self.min_feature_dim <= 0 or self.min_median_nnz < 0.0:
            raise ValueError("domain thresholds must be non-negative")
        if not 0.0 <= self.min_zero_fraction <= 1.0:
            raise ValueError("min_zero_fraction must be in [0, 1]")
        if not 0.0 <= self.max_empty_fraction <= 1.0:
            raise ValueError("max_empty_fraction must be in [0, 1]")
        if self.input_policy not in {"strict_legacy", "expanded_count"}:
            raise ValueError("input_policy must be 'strict_legacy' or 'expanded_count'")
        valid = {
            "self_only",
            "fixed_predictive_graph",
            "V16_1_predictive_gate",
            "shuffled_support",
            "output_disabled",
        }
        if self.variant not in valid:
            raise ValueError(f"variant must be one of {sorted(valid)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> V16_1Config:
    payload: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"V16.1 config must be a mapping: {path}")
        payload.update(loaded)
    if overrides:
        payload.update({str(k): v for k, v in overrides.items()})
    unknown = sorted(set(payload) - set(V16_1Config.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown V16.1 config keys: {unknown}")
    return V16_1Config(**payload)
