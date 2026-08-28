from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class V20Config:
    protocol_id: str = "v20_topology_conditioned_adv_mask_v1"
    variant: str = "topology_adversarial_full"
    hidden_size: int = 128
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    gate_lr: float = 1e-3
    mask_ratio: float = 0.4
    dropout: float = 0.0
    masked_data_weight: float = 0.75
    mask_loss_weight: float = 0.7
    warmup_epochs: int = 40
    gate_update_every: int = 4
    gate_hidden: int = 64
    gumbel_scale: float = 1.0
    tau_ste: float = 0.5
    graph_svd_target: float = 0.95
    graph_svd_min_dim: int = 50
    graph_svd_max_dim: int = 500
    neighbor_k: int = 20
    stats_block_size: int = 1024
    stats_cache_dtype: str = "float32"
    stats_clip: float = 5.0
    kmeans_n_init: int = 20
    num_workers: int = 0
    # V20.1 makes sampled positions versus actual value changes explicit.
    # The v1 defaults remain unchanged for replay.
    mask_target_mode: str = "requested"
    random_mask_mode: str = "topk"

    def validate(self) -> None:
        if self.variant not in {"topology_adversarial_full", "scmae_only"}:
            raise ValueError(f"unsupported V20 variant: {self.variant!r}")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if not 0.0 <= self.mask_loss_weight <= 1.0:
            raise ValueError("mask_loss_weight must be in [0, 1]")
        if self.warmup_epochs < 0 or self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs]")
        if self.gate_update_every <= 0:
            raise ValueError("gate_update_every must be positive")
        if self.graph_svd_min_dim <= 0 or self.graph_svd_max_dim < self.graph_svd_min_dim:
            raise ValueError("invalid SVD dimension bounds")
        if self.neighbor_k <= 0 or self.stats_block_size <= 0:
            raise ValueError("neighbor_k and stats_block_size must be positive")
        if self.mask_target_mode not in {"requested", "effective"}:
            raise ValueError("mask_target_mode must be requested or effective")
        if self.random_mask_mode not in {"topk", "bernoulli"}:
            raise ValueError("random_mask_mode must be topk or bernoulli")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> V20Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    if overrides:
        payload.update(overrides)
    config = V20Config(**payload)
    config.validate()
    return config
