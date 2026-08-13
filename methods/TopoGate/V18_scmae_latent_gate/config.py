from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class V18Config:
    """Frozen, label-free protocol parameters for one V18 run."""

    protocol_id: str = "v18_scmae_mainline_v2_2"
    seed: int = 42
    input_mode: str = "auto"
    hidden_size: int = 128
    mask_ratio: float = 0.30
    n_views: int = 3
    candidate_k: int = 20
    candidate_width: int = 40
    batch_size: int = 256
    epochs_mae: int = 80
    epochs_gate: int = 30
    epochs_joint: int = 30
    lr_mae: float = 1e-3
    lr_gate: float = 1e-3
    lr_relation: float = 1e-3
    lr_encoder_joint: float = 1e-4
    dropout: float = 0.0
    masked_data_weight: float = 0.75
    mask_loss_weight: float = 0.70
    huber_delta: float = 1.0
    lambda_topo: float = 1.0
    lambda_gate: float = 1e-3
    lambda_w: float = 1e-3
    lambda_l2: float = 1e-4
    lambda_anchor: float = 0.10
    lambda_var: float = 0.01
    gate_temperature_start: float = 2.0
    gate_temperature_end: float = 0.7
    gate_init_bias: float = -2.0
    gate_gamma: float = -0.1
    gate_zeta: float = 1.1
    solver_max_iter: int = 40
    solver_tolerance: float = 1e-5
    coefficient_epsilon: float = 1e-8
    spectral_n_init: int = 20
    degree_epsilon: float = 1e-12
    leiden_resolution: float = 1.0
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("protocol_id must be non-empty")
        if self.input_mode not in {"auto", "count", "nonnegative", "continuous"}:
            raise ValueError("input_mode must be auto, count, nonnegative, or continuous")
        positive_ints = {
            "hidden_size": self.hidden_size,
            "n_views": self.n_views,
            "candidate_k": self.candidate_k,
            "candidate_width": self.candidate_width,
            "batch_size": self.batch_size,
            "epochs_mae": self.epochs_mae,
            "epochs_gate": self.epochs_gate,
            "epochs_joint": self.epochs_joint,
            "solver_max_iter": self.solver_max_iter,
            "spectral_n_init": self.spectral_n_init,
        }
        if any(int(value) <= 0 for value in positive_ints.values()):
            raise ValueError("all size, epoch, and iteration parameters must be positive")
        if self.candidate_width < self.candidate_k:
            raise ValueError("candidate_width must be >= candidate_k")
        bounded = {
            "mask_ratio": self.mask_ratio,
            "dropout": self.dropout,
            "masked_data_weight": self.masked_data_weight,
            "mask_loss_weight": self.mask_loss_weight,
        }
        if any(not 0.0 <= float(value) < 1.0 for value in bounded.values()):
            raise ValueError("mask_ratio, dropout, and loss weights must be in [0, 1)")
        nonnegative = {
            "lambda_topo": self.lambda_topo,
            "lambda_gate": self.lambda_gate,
            "lambda_w": self.lambda_w,
            "lambda_l2": self.lambda_l2,
            "lambda_anchor": self.lambda_anchor,
            "lambda_var": self.lambda_var,
            "degree_epsilon": self.degree_epsilon,
            "coefficient_epsilon": self.coefficient_epsilon,
        }
        if any(float(value) < 0.0 for value in nonnegative.values()):
            raise ValueError("regularization weights and thresholds must be non-negative")
        if self.huber_delta <= 0.0 or self.lr_mae <= 0.0 or self.lr_gate <= 0.0:
            raise ValueError("huber_delta, learning rates must be positive")
        if self.lr_relation <= 0.0 or self.lr_encoder_joint <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 < self.gate_temperature_end <= self.gate_temperature_start:
            raise ValueError("gate temperatures must satisfy 0 < end <= start")
        if not self.gate_gamma < 0.0 < self.gate_zeta:
            raise ValueError("HardConcrete stretch must straddle [0, 1]")
        if self.leiden_resolution <= 0.0:
            raise ValueError("leiden_resolution must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> V18Config:
    payload: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"V18 config must be a mapping: {path}")
        payload.update(loaded)
    if overrides:
        payload.update({str(k): v for k, v in overrides.items() if v is not None})
    unknown = sorted(set(payload) - set(V18Config.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown V18 config keys: {unknown}")
    return V18Config(**payload)
