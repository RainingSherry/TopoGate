"""Configuration and validation for the independent TopoGate V11 variant.

V11 deliberately does not reuse the mutable V9 runner.  A flat YAML file is
merged with explicit Python overrides and then validated before training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class V11Config:
    # Reproducibility / data
    seed: int = 42
    input_mode: str = "raw"
    scale_input: bool = True
    n_top_features: int = 0
    pca_dim: int = 128
    pca_variance: float = 0.95

    # Encoder / likelihood
    hidden_size: int = 256
    latent_size: int = 64
    dropout: float = 0.1
    weight_decay: float = 1e-4
    reconstruction_distribution: str = "student_t"
    student_t_nu: float = 4.0
    masked_data_weight: float = 0.75
    mask_prediction_weight: float = 0.10
    mask_ratio: float = 0.30
    mask_ratio_end: float = 0.30

    # Training schedule
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    warmup_epochs: int = 20
    ramp_epochs: int = 10
    graph_refresh_interval: int = 5
    ema_decay: float = 0.99
    confidence_threshold: float = 0.55
    confidence_quantile: float = 0.50
    cluster_temperature: float = 0.20
    cluster_assignment_kernel: str = "diagonal_product"
    cluster_logit_normalization: str = "sqrt_dim"
    cluster_scale_floor_ratio: float = 0.00
    cluster_weight: float = 0.50
    clean_cluster_weight: float = 1.00
    mixed_cluster_weight: float = 0.50
    graph_weight: float = 0.20
    edge_consistency_weight: float = 0.10
    edge_alignment_temperature: float = 0.20
    mix_reconstruction_weight: float = 0.25
    dirichlet_strength: float = 1e-3
    gradient_clip: float = 5.0

    # Candidate topology
    neighbor_k: int = 5
    candidate_k: int = 10
    knn_backend: str = "auto"
    knn_exact_max_nodes: int = 5000
    knn_hnsw_m: int = 32
    knn_hnsw_ef_search: int = 64
    edge_temperature: float = 1.00
    raw_prior_temperature: float = 0.20
    raw_prior_weight: float = 1.0
    latent_prior_weight: float = 0.0
    gate_risk_temperature: float = 0.25
    gate_cluster_risk_temperature: float = 0.25
    # How independent reconstruction and assignment evidence is combined
    # into a counterfactual topology target.  ``geometric_mean`` preserves
    # the historical V11.3 candidate; ``harmonic_mean`` is a stricter
    # AND-like rule that suppresses topology when either channel is weak.
    semantic_help_combiner: str = "geometric_mean"
    graph_probe_strength: float = 0.25
    gate_initial_null_bias: float = 2.5
    # ``paired_ema_eval`` evaluates anchor and graph probe with the same
    # deterministic EMA/eval reference. The legacy path is retained only for
    # a reproducibility ablation.
    risk_target_mode: str = "paired_ema_eval"
    # ``input_mix`` is the historical V11 path. ``assignment_residual`` keeps
    # the self/null topology gate but applies its trusted neighbourhood only
    # to a teacher-assignment residual, never to decoder inputs.
    topology_path: str = "input_mix"
    # ``paired_risk`` preserves the historical reconstruction-risk target.
    # ``counterfactual_semantic`` evaluates a teacher-defined topology probe
    # against both reconstruction and clean-assignment risks. The temporal
    # target is a label-free recurrence across two graph refresh events.
    gate_target_source: str = "paired_risk"
    temporal_gate_max: float = 0.25
    residual_assignment_weight: float = 0.50

    # Optional detached persistent-homology prior. ``none`` is the exact
    # default V11 path; the other modes are explicit pilot/control variants.
    tda_prior_mode: str = "none"
    tda_prior_weight: float = 0.0
    tda_scale_mode: str = "median"
    tda_scale_quantile: float = 0.50
    tda_scale_floor: float = 1e-6

    # Ablation switches (all are reversible configuration changes)
    use_dynamic_graph: bool = True
    use_topology: bool = True
    use_edge_reliability: bool = True
    use_teacher: bool = True
    use_cluster_head: bool = True
    use_mixed_reconstruction: bool = True
    use_graph_prior: bool = True
    use_edge_consistency: bool = False

    def validate(self) -> "V11Config":
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.warmup_epochs < 0 or self.ramp_epochs < 0:
            raise ValueError("warmup_epochs and ramp_epochs must be non-negative")
        if self.neighbor_k < 1 or self.candidate_k < self.neighbor_k:
            raise ValueError("candidate_k must be >= neighbor_k >= 1")
        if self.knn_backend not in {"auto", "exact", "faiss_hnsw"}:
            raise ValueError("knn_backend must be auto, exact, or faiss_hnsw")
        if self.knn_exact_max_nodes < 2 or self.knn_hnsw_m < 4 or self.knn_hnsw_ef_search < 2:
            raise ValueError("invalid kNN backend parameters")
        if self.latent_size < 2 or self.hidden_size < self.latent_size:
            raise ValueError("hidden_size must be >= latent_size >= 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.reconstruction_distribution not in {"gaussian", "student_t", "bernoulli", "poisson"}:
            raise ValueError("unsupported reconstruction_distribution")
        if not 0.0 <= self.masked_data_weight <= 1.0:
            raise ValueError("masked_data_weight must be in [0, 1]")
        if not 0.0 <= self.mask_prediction_weight <= 1.0:
            raise ValueError("mask_prediction_weight must be in [0, 1]")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if not 0.0 < self.graph_probe_strength <= 1.0:
            raise ValueError("graph_probe_strength must be in (0, 1]")
        if self.gate_risk_temperature <= 0.0 or self.gate_cluster_risk_temperature <= 0.0:
            raise ValueError("gate risk temperatures must be positive")
        if self.semantic_help_combiner not in {"geometric_mean", "harmonic_mean", "minimum", "product"}:
            raise ValueError(
                "semantic_help_combiner must be geometric_mean, harmonic_mean, "
                "minimum, or product"
            )
        if self.edge_alignment_temperature <= 0.0:
            raise ValueError("edge_alignment_temperature must be positive")
        if self.risk_target_mode not in {"paired_ema_eval", "legacy_student_train"}:
            raise ValueError("risk_target_mode must be paired_ema_eval or legacy_student_train")
        if self.topology_path not in {"input_mix", "assignment_residual"}:
            raise ValueError("topology_path must be input_mix or assignment_residual")
        if self.gate_target_source not in {
            "paired_risk",
            "counterfactual_semantic",
            "temporal_agreement",
        }:
            raise ValueError(
                "gate_target_source must be paired_risk, counterfactual_semantic, "
                "or temporal_agreement"
            )
        if self.gate_target_source == "counterfactual_semantic":
            if not self.use_cluster_head:
                raise ValueError("counterfactual_semantic requires use_cluster_head=true")
            if self.risk_target_mode != "paired_ema_eval":
                raise ValueError(
                    "counterfactual_semantic requires deterministic paired_ema_eval risk"
                )
        # A temporal target is defined from the edge recurrence measured at a
        # later graph refresh.  With a fixed graph there is no second refresh,
        # so the previous implementation silently produced an all-zero target
        # and turned the supposed static-graph ablation into a forced NoMix
        # model.  Fail early instead of reporting that incomparable variant.
        if self.gate_target_source == "temporal_agreement" and not self.use_dynamic_graph:
            raise ValueError(
                "temporal_agreement requires use_dynamic_graph=true; use "
                "counterfactual_semantic or paired_risk for a fixed-graph ablation"
            )
        if not 0.0 <= self.temporal_gate_max <= 1.0:
            raise ValueError("temporal_gate_max must be in [0, 1]")
        if self.residual_assignment_weight < 0.0:
            raise ValueError("residual_assignment_weight must be non-negative")
        if not 0.0 <= self.confidence_quantile < 1.0:
            raise ValueError("confidence_quantile must be in [0, 1)")
        if self.cluster_logit_normalization not in {"none", "sqrt_dim", "mean_dim"}:
            raise ValueError("cluster_logit_normalization must be none, sqrt_dim, or mean_dim")
        if self.cluster_assignment_kernel not in {"diagonal_product", "radial"}:
            raise ValueError("cluster_assignment_kernel must be diagonal_product or radial")
        if not 0.0 <= self.cluster_scale_floor_ratio <= 1.0:
            raise ValueError("cluster_scale_floor_ratio must be in [0, 1]")
        if not 0.0 <= self.mixed_cluster_weight <= 1.0:
            raise ValueError("mixed_cluster_weight must be in [0, 1]")
        if not 0.0 <= self.clean_cluster_weight <= 1.0:
            raise ValueError("clean_cluster_weight must be in [0, 1]")
        if self.edge_consistency_weight < 0.0:
            raise ValueError("edge_consistency_weight must be non-negative")
        if self.tda_prior_mode not in {
            "none",
            "h0_mst",
            "h0_early_mst",
            "fixed_filtration",
            "random",
        }:
            raise ValueError(
                "tda_prior_mode must be none, h0_mst, h0_early_mst, fixed_filtration, or random"
            )
        if self.tda_prior_weight < 0.0:
            raise ValueError("tda_prior_weight must be non-negative")
        if self.tda_scale_mode not in {"median", "quantile", "max", "none"}:
            raise ValueError("tda_scale_mode must be median, quantile, max, or none")
        if not 0.0 <= self.tda_scale_quantile <= 1.0:
            raise ValueError("tda_scale_quantile must be in [0, 1]")
        if self.tda_scale_floor <= 0.0:
            raise ValueError("tda_scale_floor must be positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> V11Config:
    cfg = V11Config()
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        valid = {f.name for f in fields(cfg)}
        unknown = sorted(set(payload) - valid)
        if unknown:
            raise ValueError(f"unknown V11 config keys: {unknown}")
        for key, value in payload.items():
            setattr(cfg, key, value)
    for key, value in (overrides or {}).items():
        if not hasattr(cfg, key):
            raise ValueError(f"unknown V11 override: {key}")
        setattr(cfg, key, value)
    return cfg.validate()
