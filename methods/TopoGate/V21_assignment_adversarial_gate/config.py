from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


V21_VARIANTS = frozenset(
    {
        "scmae_only",
        "random_assignment_control",
        "topology_assignment_adversarial",
    }
)


@dataclass(frozen=True)
class V21Config:
    protocol_id: str = "v21_assignment_adversarial_v2_graphfix_v1"
    variant: str = "topology_assignment_adversarial"
    hidden_size: int = 128
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    cluster_lr: float = 1e-3
    gate_lr: float = 5e-4
    mask_ratio: float = 0.4
    dropout: float = 0.0
    masked_data_weight: float = 0.75
    mask_loss_weight: float = 0.7
    mask_target_mode: str = "effective"
    random_mask_mode: str = "bernoulli"
    warmup_epochs: int = 40
    assignment_mask_ratio: float = 0.4
    assignment_budget_scope: str = "changeable"
    assignment_change_epsilon: float = 0.0
    assignment_weight: float = 0.5
    infomax_weight: float = 0.05
    cluster_alpha: float = 1.0
    cluster_distance_reduction: str = "mean"
    cluster_n_init: int = 20
    gate_update_every: int = 1
    gate_hidden: int = 64
    gate_coverage_weight: float = 0.01
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
    readout_mode: str = "student_t_head"
    num_workers: int = 0

    @property
    def uses_cluster_head(self) -> bool:
        return self.variant != "scmae_only"

    @property
    def uses_topology_gate(self) -> bool:
        return self.variant == "topology_assignment_adversarial"

    def validate(self) -> None:
        if self.variant not in V21_VARIANTS:
            raise ValueError(f"unsupported V21 variant: {self.variant!r}")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.hidden_size <= 0 or self.gate_hidden <= 0:
            raise ValueError("hidden_size and gate_hidden must be positive")
        if self.lr <= 0.0 or self.cluster_lr <= 0.0 or self.gate_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if not 0.0 < self.assignment_mask_ratio <= 1.0:
            raise ValueError("assignment_mask_ratio must be in (0, 1]")
        if not 0.0 <= self.mask_loss_weight <= 1.0:
            raise ValueError("mask_loss_weight must be in [0, 1]")
        if self.warmup_epochs < 0 or self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs]")
        if self.uses_cluster_head and self.warmup_epochs >= self.epochs:
            raise ValueError("cluster-head variants require warmup_epochs < epochs")
        if self.mask_target_mode not in {"requested", "effective"}:
            raise ValueError("mask_target_mode must be requested or effective")
        if self.random_mask_mode not in {"topk", "bernoulli"}:
            raise ValueError("random_mask_mode must be topk or bernoulli")
        if self.assignment_budget_scope != "changeable":
            raise ValueError("V21 graph-fix protocol supports only assignment_budget_scope='changeable'")
        if self.assignment_change_epsilon < 0.0:
            raise ValueError("assignment_change_epsilon must be non-negative")
        if self.assignment_weight < 0.0 or self.infomax_weight < 0.0:
            raise ValueError("assignment and InfoMax weights must be non-negative")
        if self.cluster_alpha <= 0.0 or self.cluster_n_init <= 0:
            raise ValueError("cluster_alpha and cluster_n_init must be positive")
        if self.cluster_distance_reduction not in {"mean", "sum"}:
            raise ValueError("cluster_distance_reduction must be mean or sum")
        if self.gate_update_every <= 0 or self.gate_coverage_weight < 0.0:
            raise ValueError("invalid Gate update configuration")
        if self.tau_ste <= 0.0 or self.gumbel_scale < 0.0:
            raise ValueError("tau_ste must be positive and gumbel_scale non-negative")
        if self.graph_svd_min_dim <= 0 or self.graph_svd_max_dim < self.graph_svd_min_dim:
            raise ValueError("invalid SVD dimension bounds")
        if self.neighbor_k <= 0 or self.stats_block_size <= 0:
            raise ValueError("neighbor_k and stats_block_size must be positive")
        if self.readout_mode not in {"student_t_head", "kmeans_embedding"}:
            raise ValueError("readout_mode must be student_t_head or kmeans_embedding")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> V21Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    if overrides:
        payload.update(overrides)
    config = V21Config(**payload)
    config.validate()
    return config
