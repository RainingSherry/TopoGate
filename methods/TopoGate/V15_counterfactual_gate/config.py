from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class V15Config:
    """Configuration for the isolated V15 exploratory implementation."""

    seed: int = 42
    hidden_dim: int = 128
    latent_dim: int = 32
    dropout: float = 0.1
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    mask_ratio: float = 0.4
    # Zeroing observed features is the denoising objective. Row swapping is
    # retained as a graph/feature contamination stress condition, not as the
    # default reconstruction target.
    mask_strategy: str = "zero"
    sparse_transform: str = "tfidf_l2"
    masked_weight: float = 1.0
    visible_weight: float = 0.1
    zero_weight: float = 0.02
    zero_sample_ratio: float = 0.02
    mask_prediction_weight: float = 0.1
    teacher_pretrain_epochs: int = 10
    teacher_selection_warmup_epochs: int = 10
    teacher_view_consistency_weight: float = 0.1
    teacher_variance_weight: float = 0.1
    teacher_variance_floor: float = 0.1
    # The EMA teacher is frozen as a target; the student must remain trainable
    # so the accepted topology signal can shape the representation.
    freeze_backbone_after_teacher: bool = False
    sparse_zero_threshold: float = 0.5
    # Select one topology-disabled view without averaging arbitrary cluster ids.
    teacher_reference_mode: str = "auto"
    teacher_reference_raw_weight: float = 0.5
    teacher_reference_temperature: float = 0.25
    raw_svd_dim: int = 64
    latent_graph_dim: int = 32
    k_raw: int = 10
    k_latent: int = 10
    # Preserve the complete deduplicated 10+10 union. Candidate generation is
    # a recall stage; counterfactual utility, not a quota heuristic, rejects
    # weak latent-only edges.
    candidate_cap: int = 20
    # Candidate construction remains the raw/latent union by default. Scopes
    # are explicit graph ablations and can be promoted after replay evidence.
    candidate_scope: str = "all"
    graph_refresh_interval: int = 10
    graph_backend: str = "exact"
    graph_replacement_fraction: float = 0.0
    gate_mode: str = "counterfactual"
    forced_topk: int = 2
    probe_alpha: float = 0.5
    output_alpha: float = 0.25
    # The main readout must carry the accepted topology residual into the
    # exported representation. Logit transport remains an explicit ablation;
    # it changes assignments but does not define the embedding readout.
    # Assignment-space transport is the main readout; latent/logit/probability
    # transport remain explicit ablations.
    output_mode: str = "assignment"
    output_consensus_scaling: bool = False
    utility_lambda_rec: float = 0.25
    # ``local_consensus`` scores an edge against a leave-one-candidate-out
    # consensus.  This is the label-free target for the learned gate: the
    # tested donor cannot manufacture its own target.
    utility_target_mode: str = "stability"
    utility_reference_mode: str = "teacher"
    utility_reference_temperature: float = 0.05
    # ExactCF normally scores the clean assignment intervention that will be
    # exported. The masked-probe source is retained as a causal ablation.
    direct_utility_source: str = "clean_output"
    # Number of independent two-view intervention pairs averaged to estimate
    # expected utility under the masking distribution.
    utility_probe_pairs: int = 1
    utility_relative_baseline: bool = True
    # A candidate edge must have assignment support in the detached teacher
    # view before its counterfactual response can be useful. This is a semantic
    # term, not a second reliability head.
    utility_assignment_agreement_weight: float = 1.0
    utility_stability_weight: float = 1.0
    utility_confidence_weight: float = 0.5
    utility_opportunity_temperature: float = 0.25
    utility_clip: float = 4.0
    # Minimum robust effect size required before an edge can beat the null
    # branch. This is part of the counterfactual decision rule, not a second
    # reliability gate.
    utility_min_gain: float = 0.0
    utility_temperature: float = 0.5
    utility_holdout_fraction: float = 0.2
    utility_sign_weight: float = 0.25
    # Distil a detached, accepted counterfactual assignment into the student.
    # Zero keeps the direct gate as a readout-only mechanism ablation.
    counterfactual_distill_weight: float = 0.0
    # Keep the EMA teacher topology-disabled until the self representation has
    # matured. At this epoch the teacher and candidate graph become frozen and
    # the counterfactual assignment starts supervising the student.
    counterfactual_distill_start_epoch: int = 80
    # The paper method exports the clean student after topology transfer. The
    # stochastic teacher-side counterfactual readout is retained only as an
    # explicit mechanism-ceiling ablation.
    final_prediction_source: str = "student_clean"
    # Optional teacher-certificate floor for the gate. Zero preserves the
    # unconstrained candidate pool; positive values make unsupported edges
    # abstain before sparsemax rather than pretending their utility is known.
    gate_teacher_agreement_floor: float = 0.0
    gate_temperature: float = 0.5
    gate_opportunity_mode: str = "assignment_uncertainty"
    lambda_cluster: float = 1.0
    # Cross-view co-training is the main semantic correction; zero is the
    # latent-only teacher ablation.
    raw_view_cluster_weight: float = 0.5
    raw_view_cluster_temperature: float = 0.1
    lambda_gate: float = 0.5
    gate_training_mode: str = "detached"
    ema_decay: float = 0.99
    warmup_epochs: int = 10
    student_t_nu: float = 4.0
    cluster_normalize_latent: bool = True
    cluster_cosine_temperature: float = 0.1
    dirichlet_strength: float = 1e-3
    cluster_frequency_decay: float = 0.95
    cluster_frequency_weight: float = 0.1
    # A moving empirical prior can amplify a collapsed teacher. The spherical
    # prototype head therefore starts against a uniform marginal target.
    cluster_frequency_uniform_mix: float = 1.0
    cluster_head: str = "spherical_prototype"
    prototype_separation_weight: float = 0.1
    prototype_separation_margin: float = 0.0
    n_init: int = 10
    no_cuda: bool = False
    gpu: int = 1

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.hidden_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("hidden_dim and latent_dim must be positive")
        if not 0 <= self.teacher_pretrain_epochs < self.epochs:
            raise ValueError("teacher_pretrain_epochs must be in [0, epochs)")
        if self.teacher_selection_warmup_epochs < 0:
            raise ValueError("teacher_selection_warmup_epochs must be non-negative")
        if self.teacher_view_consistency_weight < 0.0 or self.teacher_variance_weight < 0.0:
            raise ValueError("teacher certificate loss weights must be non-negative")
        if self.teacher_variance_floor <= 0.0:
            raise ValueError("teacher_variance_floor must be positive")
        if self.teacher_reference_mode not in {
            "latent",
            "raw",
            "consensus",
            "auto",
            "quality_auto",
        }:
            raise ValueError(
                "teacher_reference_mode must be latent, raw, consensus, auto, or quality_auto"
            )
        if not 0.0 <= self.teacher_reference_raw_weight <= 1.0:
            raise ValueError("teacher_reference_raw_weight must be in [0, 1]")
        if self.teacher_reference_temperature <= 0.0:
            raise ValueError("teacher_reference_temperature must be positive")
        if self.cluster_cosine_temperature <= 0.0:
            raise ValueError("cluster_cosine_temperature must be positive")
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1]")
        if self.mask_strategy not in {"row_swap", "zero"}:
            raise ValueError("mask_strategy must be 'row_swap' or 'zero'")
        if self.sparse_transform not in {"log1p_row", "tfidf_l2"}:
            raise ValueError("sparse_transform must be 'log1p_row' or 'tfidf_l2'")
        if self.cluster_head not in {"spherical_prototype", "student_t"}:
            raise ValueError("cluster_head must be 'spherical_prototype' or 'student_t'")
        if self.prototype_separation_weight < 0.0:
            raise ValueError("prototype_separation_weight must be non-negative")
        if self.prototype_separation_margin < -1.0 or self.prototype_separation_margin > 1.0:
            raise ValueError("prototype_separation_margin must be in [-1, 1]")
        if not 0.0 <= self.sparse_zero_threshold <= 1.0:
            raise ValueError("sparse_zero_threshold must be in [0, 1]")
        if not 0.0 <= self.zero_sample_ratio <= 1.0:
            raise ValueError("zero_sample_ratio must be in [0, 1]")
        if self.k_raw <= 0 or self.k_latent <= 0 or self.candidate_cap <= 0:
            raise ValueError("graph k values must be positive")
        if self.candidate_scope not in {
            "all",
            "both_views",
            "raw_supported",
            "latent_supported",
        }:
            raise ValueError(
                "candidate_scope must be all, both_views, raw_supported, or latent_supported"
            )
        if self.graph_refresh_interval <= 0:
            raise ValueError("graph_refresh_interval must be positive")
        if not 0.0 <= self.graph_replacement_fraction <= 1.0:
            raise ValueError("graph_replacement_fraction must be in [0, 1]")
        if not 0.0 < self.utility_holdout_fraction < 1.0:
            raise ValueError("utility_holdout_fraction must be in (0, 1)")
        if self.utility_sign_weight < 0.0:
            raise ValueError("utility_sign_weight must be non-negative")
        if self.counterfactual_distill_weight < 0.0:
            raise ValueError("counterfactual_distill_weight must be non-negative")
        if self.counterfactual_distill_start_epoch < 1:
            raise ValueError("counterfactual_distill_start_epoch must be positive")
        if (
            self.counterfactual_distill_weight > 0.0
            and self.counterfactual_distill_start_epoch > self.epochs
        ):
            raise ValueError(
                "counterfactual_distill_start_epoch must not exceed epochs when distillation is enabled"
            )
        if self.final_prediction_source not in {"student_clean", "gate_readout"}:
            raise ValueError("final_prediction_source must be student_clean or gate_readout")
        if not 0.0 <= self.gate_teacher_agreement_floor <= 1.0:
            raise ValueError("gate_teacher_agreement_floor must be in [0, 1]")
        if self.gate_opportunity_mode not in {"none", "assignment_uncertainty"}:
            raise ValueError("gate_opportunity_mode must be none or assignment_uncertainty")
        if self.utility_assignment_agreement_weight < 0.0:
            raise ValueError("utility_assignment_agreement_weight must be non-negative")
        if self.utility_target_mode not in {
            "teacher",
            "stability",
            "operator_aligned",
            "local_consensus",
        }:
            raise ValueError(
                "utility_target_mode must be teacher, stability, operator_aligned, or local_consensus"
            )
        if self.utility_reference_mode not in {"teacher", "cross_view", "hybrid", "adaptive"}:
            raise ValueError("utility_reference_mode must be teacher, cross_view, hybrid, or adaptive")
        if self.utility_reference_temperature <= 0.0:
            raise ValueError("utility_reference_temperature must be positive")
        if self.direct_utility_source not in {"clean_output", "masked_probe"}:
            raise ValueError("direct_utility_source must be clean_output or masked_probe")
        if self.utility_probe_pairs <= 0:
            raise ValueError("utility_probe_pairs must be positive")
        if self.utility_stability_weight < 0.0 or self.utility_confidence_weight < 0.0:
            raise ValueError("utility semantic weights must be non-negative")
        if self.utility_min_gain < 0.0:
            raise ValueError("utility_min_gain must be non-negative")
        if self.utility_opportunity_temperature <= 0.0:
            raise ValueError("utility_opportunity_temperature must be positive")
        valid_modes = {
            "counterfactual",
            "counterfactual_learned",
            "direct_target",
            "direct_counterfactual",
            "self_only",
            "union_uniform",
            "forced_topk",
            "shuffled_utility",
            "output_disabled",
        }
        if self.gate_mode not in valid_modes:
            raise ValueError(f"gate_mode must be one of {sorted(valid_modes)}")
        if self.forced_topk <= 0:
            raise ValueError("forced_topk must be positive")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if not 0.0 < self.cluster_frequency_decay < 1.0:
            raise ValueError("cluster_frequency_decay must be in (0, 1)")
        if self.cluster_frequency_weight < 0.0:
            raise ValueError("cluster_frequency_weight must be non-negative")
        if self.raw_view_cluster_weight < 0.0 or self.raw_view_cluster_temperature <= 0.0:
            raise ValueError("raw-view cluster settings must be non-negative and temperature-positive")
        if self.gate_training_mode not in {"detached", "joint"}:
            raise ValueError("gate_training_mode must be detached or joint")
        if self.lambda_gate < 0.0:
            raise ValueError("lambda_gate must be non-negative")
        if not 0.0 <= self.cluster_frequency_uniform_mix <= 1.0:
            raise ValueError("cluster_frequency_uniform_mix must be in [0, 1]")
        if self.probe_alpha <= 0.0 or not 0.0 <= self.output_alpha <= 1.0:
            raise ValueError("probe_alpha must be positive and output_alpha must be in [0, 1]")
        if self.output_mode not in {"logit", "latent", "probability", "assignment"}:
            raise ValueError("output_mode must be logit, latent, probability, or assignment")
        if self.gate_mode in {"direct_counterfactual", "counterfactual_learned"}:
            if self.utility_target_mode not in {"operator_aligned", "local_consensus"}:
                raise ValueError(
                    "counterfactual gate modes require operator_aligned or local_consensus utility"
                )
        if self.gate_mode == "direct_counterfactual" and self.output_mode != "assignment":
            raise ValueError(
                "direct_counterfactual requires assignment output"
            )
        if self.gate_mode == "counterfactual_learned" and self.output_mode not in {
            "assignment",
            "latent",
        }:
            raise ValueError(
                "counterfactual_learned requires assignment or latent output"
            )
        if self.gate_mode in {"direct_counterfactual", "counterfactual_learned"}:
            if self.gate_opportunity_mode != "none":
                raise ValueError(
                    "counterfactual gate modes cannot use a second opportunity gate"
                )
        if self.gate_mode == "counterfactual_learned" and self.lambda_gate <= 0.0:
            raise ValueError("counterfactual_learned requires a positive lambda_gate")
        distill_modes = {
            "direct_counterfactual",
            "counterfactual_learned",
            "shuffled_utility",
            "output_disabled",
        }
        if self.counterfactual_distill_weight > 0.0 and self.gate_mode not in distill_modes:
            raise ValueError(
                "counterfactual distillation requires direct_counterfactual, "
                "counterfactual_learned, shuffled_utility, or output_disabled "
                "gate mode"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_overrides(values: dict[str, Any] | None) -> dict[str, Any]:
    return {} if values is None else {str(k): v for k, v in values.items()}


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> V15Config:
    payload: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"V15 config must be a mapping: {path}")
        payload.update(loaded)
    payload.update(_coerce_overrides(overrides))
    known = set(V15Config.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown V15 config keys: {unknown}")
    return V15Config(**payload)
