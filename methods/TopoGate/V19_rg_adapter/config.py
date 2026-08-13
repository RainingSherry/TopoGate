from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


VARIANTS = frozenset({"scmae_only", "rg_full"})
INPUT_PROTOCOLS = frozenset({"rg_native", "clubench_bridge", "shared_text"})


@dataclass(frozen=True)
class V19Config:
    """Frozen RG/scMAE protocol shared by all V19 datasets and seeds."""

    protocol_id: str = "v19_rg_selected_advantage_v1"
    variant: str = "rg_full"
    hidden_size: int = 128
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    mask_ratio: float = 0.4
    dropout: float = 0.0
    masked_data_weight: float = 0.75
    mask_loss_weight: float = 0.70
    neighbor_k: int = 10
    mix_neighbors: int = 4
    knn_pca_dim: int = 50
    tau: float = 0.2
    gate_min: float = 0.0
    gate_max: float = 0.15
    pseudo_weight: float = 0.3
    gamma_sim: float = 1.0
    gamma_mutual: float = 1.0
    gamma_snn: float = 1.0
    gamma_distance: float = 1.0
    beta_mutual: float = 1.0
    beta_snn: float = 1.0
    beta_perturb: float = 2.0
    beta_uncertainty: float = 1.0
    n_top_features: int = 1000
    target_sum: float = 10_000.0
    kmeans_n_init: int = 20
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("protocol_id must be non-empty")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        positive_ints = {
            "hidden_size": self.hidden_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "neighbor_k": self.neighbor_k,
            "mix_neighbors": self.mix_neighbors,
            "knn_pca_dim": self.knn_pca_dim,
            "n_top_features": self.n_top_features,
            "kmeans_n_init": self.kmeans_n_init,
        }
        if any(int(value) <= 0 for value in positive_ints.values()):
            raise ValueError("model, graph, feature, and training sizes must be positive")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers must be non-negative")
        bounded = {
            "mask_ratio": self.mask_ratio,
            "dropout": self.dropout,
            "masked_data_weight": self.masked_data_weight,
            "mask_loss_weight": self.mask_loss_weight,
            "gate_min": self.gate_min,
            "gate_max": self.gate_max,
        }
        if any(not 0.0 <= float(value) < 1.0 for value in bounded.values()):
            raise ValueError("mask, dropout, loss, and gate values must be in [0, 1)")
        if float(self.gate_min) > float(self.gate_max):
            raise ValueError("gate_min must be <= gate_max")
        nonnegative = {
            "pseudo_weight": self.pseudo_weight,
            "gamma_sim": self.gamma_sim,
            "gamma_mutual": self.gamma_mutual,
            "gamma_snn": self.gamma_snn,
            "gamma_distance": self.gamma_distance,
            "beta_mutual": self.beta_mutual,
            "beta_snn": self.beta_snn,
            "beta_perturb": self.beta_perturb,
            "beta_uncertainty": self.beta_uncertainty,
        }
        if any(float(value) < 0.0 for value in nonnegative.values()):
            raise ValueError("RG reliability and gate coefficients must be non-negative")
        if self.lr <= 0.0 or self.tau <= 0.0 or self.target_sum <= 0.0:
            raise ValueError("lr, tau, and target_sum must be positive")

    def for_variant(self, variant: str) -> "V19Config":
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        return replace(self, variant=variant)

    def resolved_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "graph_enabled": self.variant == "rg_full",
                "mix_mode": "reliability" if self.variant == "rg_full" else "none",
                "gate_mode": "topology" if self.variant == "rg_full" else "none",
                "edge_reliability_mode": (
                    "sim_mutual_snn_distance" if self.variant == "rg_full" else "none"
                ),
                "effective_neighbor_k": self.neighbor_k if self.variant == "rg_full" else 0,
                "effective_mix_neighbors": self.mix_neighbors if self.variant == "rg_full" else 0,
                "effective_pseudo_weight": self.pseudo_weight if self.variant == "rg_full" else 0.0,
                "contrast_weight": 0.0,
            }
        )
        return payload


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> V19Config:
    payload: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"V19 config must be a mapping: {path}")
        payload.update(loaded)
    if overrides:
        payload.update({str(key): value for key, value in overrides.items() if value is not None})
    unknown = sorted(set(payload) - set(V19Config.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown V19 config keys: {unknown}")
    return V19Config(**payload)
