from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


V22_VARIANTS = frozenset(
    {
        "scmae_only",
        "scmae_always_visible",
        "scmae_plus_discriminator_random_mask",
        "scmae_plus_discriminator_reconstruction_hard_gate",
        "scmae_plus_discriminator_learned_non_topology_gate",
        "v22_topology_discriminator_hard_gate",
        "v22_topology_discriminator_cooperative_keep_gate",
    }
)


@dataclass(frozen=True)
class V22Config:
    protocol_id: str = "v22_topology_discriminator_hard_mask_v1"
    variant: str = "v22_topology_discriminator_hard_gate"
    hidden_size: int = 128
    discriminator_hidden: int = 96
    coordinate_embedding_dim: int = 16
    gate_hidden: int = 64
    epochs: int = 80
    batch_size: int = 128
    lr: float = 1e-3
    discriminator_lr: float = 5e-4
    gate_lr: float = 5e-4
    dropout: float = 0.0
    masked_data_weight: float = 0.75
    mask_loss_weight: float = 0.7
    random_mask_ratio: float = 0.4
    adversarial_mask_ratio: float = 0.25
    donor_mode: str = "cyclic"
    assignment_change_epsilon: float = 0.0
    lambda_adversarial: float = 0.1
    lambda_gate_reconstruction: float = 0.5
    lambda_gate_coverage: float = 0.01
    gate_reward_clip: float = 4.0
    gate_update_every: int = 1
    discriminator_coordinates_per_row: int = 32
    discriminator_topology_dim: int = 4
    graph_svd_target: float = 0.95
    graph_svd_min_dim: int = 32
    graph_svd_max_dim: int = 128
    neighbor_k: int = 20
    stats_block_size: int = 512
    stats_clip: float = 5.0
    feature_cap: int = 2000
    kmeans_n_init: int = 20
    num_workers: int = 0

    @property
    def uses_discriminator(self) -> bool:
        return self.variant not in {"scmae_only", "scmae_always_visible"}

    @property
    def uses_gate(self) -> bool:
        return self.variant in {
            "scmae_plus_discriminator_reconstruction_hard_gate",
            "scmae_plus_discriminator_learned_non_topology_gate",
            "v22_topology_discriminator_hard_gate",
            "v22_topology_discriminator_cooperative_keep_gate",
        }

    @property
    def uses_topology_gate(self) -> bool:
        return self.variant in {
            "v22_topology_discriminator_hard_gate",
            "v22_topology_discriminator_cooperative_keep_gate",
        }

    @property
    def gate_reward_mode(self) -> str:
        if self.variant == "scmae_plus_discriminator_reconstruction_hard_gate":
            return "reconstruction_error_control"
        if self.variant == "v22_topology_discriminator_cooperative_keep_gate":
            return "cooperative_keep"
        return "discriminator_difficulty"

    def validate(self) -> None:
        if self.variant not in V22_VARIANTS:
            raise ValueError(f"unsupported V22 variant: {self.variant!r}")
        for name in ("epochs", "batch_size", "hidden_size", "discriminator_hidden", "gate_hidden"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("lr", "discriminator_lr", "gate_lr"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(self.random_mask_ratio) <= 1.0:
            raise ValueError("random_mask_ratio must be in [0, 1]")
        if not 0.0 < float(self.adversarial_mask_ratio) <= 1.0:
            raise ValueError("adversarial_mask_ratio must be in (0, 1]")
        if self.gate_reward_mode == "cooperative_keep" and float(self.adversarial_mask_ratio) >= 1.0:
            raise ValueError("cooperative_keep requires adversarial_mask_ratio < 1")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= float(self.mask_loss_weight) <= 1.0:
            raise ValueError("mask_loss_weight must be in [0, 1]")
        if float(self.assignment_change_epsilon) < 0.0:
            raise ValueError("assignment_change_epsilon must be non-negative")
        for name in (
            "lambda_adversarial",
            "lambda_gate_reconstruction",
            "lambda_gate_coverage",
            "gate_reward_clip",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.donor_mode != "cyclic":
            raise ValueError("V22 currently supports only donor_mode='cyclic'")
        if self.gate_update_every <= 0 or self.discriminator_coordinates_per_row <= 0:
            raise ValueError("gate_update_every and discriminator_coordinates_per_row must be positive")
        if self.coordinate_embedding_dim <= 0 or self.discriminator_topology_dim != 4:
            raise ValueError("coordinate_embedding_dim must be positive and topology_dim must be 4")
        if self.graph_svd_min_dim <= 0 or self.graph_svd_max_dim < self.graph_svd_min_dim:
            raise ValueError("invalid graph SVD dimensions")
        if self.neighbor_k <= 0 or self.stats_block_size <= 0 or self.feature_cap <= 0:
            raise ValueError("neighbor_k, stats_block_size, and feature_cap must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> V22Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    if overrides:
        payload.update(overrides)
    config = V22Config(**payload)
    config.validate()
    return config
