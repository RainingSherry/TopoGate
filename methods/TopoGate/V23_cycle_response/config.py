from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class V23Config:
    protocol_id: str = "v23_cycle_response_protocol_a_v1"
    feature_cap: int = 2000
    hidden_size: int = 128
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    training_mask_ratio: float = 0.4
    masked_data_weight: float = 0.75
    mask_loss_weight: float = 0.7
    fingerprint_masks: int = 64
    fingerprint_mask_ratio: float = 0.10
    primary_distance: str = "cosine"
    mad_epsilon: float = 1e-8
    robust_clip: float = 10.0
    latent_linear_epochs: int = 20
    latent_linear_learning_rate: float = 1e-3
    lowrank_rank: int = 64
    lowrank_ridge: float = 1e-3
    profile_batch_size: int = 512

    def validate(self) -> None:
        for name in (
            "feature_cap",
            "hidden_size",
            "epochs",
            "batch_size",
            "fingerprint_masks",
            "latent_linear_epochs",
            "lowrank_rank",
            "profile_batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "learning_rate",
            "latent_linear_learning_rate",
            "lowrank_ridge",
            "mad_epsilon",
            "robust_clip",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("training_mask_ratio", "fingerprint_mask_ratio"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        for name in ("masked_data_weight", "mask_loss_weight"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.primary_distance != "cosine":
            raise ValueError("V23 primary_distance is preregistered as 'cosine'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> V23Config:
    payload: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("V23 config must be a mapping")
        payload.update(loaded)
    if overrides:
        payload.update(overrides)
    config = V23Config(**payload)
    config.validate()
    return config
