from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from methods.TopoGate.V25_systematic_mechanism_study.e1_protocol import E1Config


ACCG_VARIANTS = frozenset(
    {
        "accg_joint",
        "accg_coordinate",
        "accg_shuffled_graph",
        "accg_marginal_only",
        "accg_joint_abstain",
    }
)


@dataclass(frozen=True)
class FeatureConstraintConfig:
    """Frozen label-free feature model and action-constraint protocol."""

    max_features: int = 2000
    transform: str = "robust_zscore"
    transform_clip: float = 8.0
    robust_scale_floor: float = 1e-3
    graph_estimator: str = "nonnegative_cosine_topk"
    graph_k: int = 10
    graph_crossfit_folds: int = 5
    graph_weight_floor: float = 0.0
    residual_scale_floor: float = 1e-3
    epsilon_scope: str = "global"
    epsilon_quantile: float = 0.90
    epsilon_rounds: int = 16
    epsilon_seed_offset: int = 31_001
    selector_mode: str = "joint"
    selector_greedy_passes: int = 2
    selector_pair_lookahead: int = 64
    infeasible_fallback: str = "least_violation"
    barrier_weight: float = 1.0
    graph_control: str = "real"
    selector_audit_rows: int = 32
    exact_solver_max_features: int = 24

    def validate(self) -> None:
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        if self.transform != "robust_zscore":
            raise ValueError("the frozen ACCG transform is robust_zscore")
        if self.transform_clip <= 0.0:
            raise ValueError("transform_clip must be positive")
        if self.robust_scale_floor <= 0.0 or self.residual_scale_floor <= 0.0:
            raise ValueError("robust scale floors must be positive")
        if self.graph_estimator != "nonnegative_cosine_topk":
            raise ValueError("the frozen ACCG graph estimator is nonnegative_cosine_topk")
        if self.graph_k <= 0 or self.graph_crossfit_folds < 2:
            raise ValueError("graph_k must be positive and crossfit folds must be at least two")
        if self.graph_weight_floor < 0.0:
            raise ValueError("graph_weight_floor must be non-negative")
        if self.epsilon_scope not in {"global", "per_sample"}:
            raise ValueError("epsilon_scope must be global or per_sample")
        if not 0.5 < self.epsilon_quantile < 1.0:
            raise ValueError("epsilon_quantile must be in (0.5, 1)")
        if self.epsilon_rounds < 4:
            raise ValueError("epsilon_rounds must be at least four")
        if self.selector_mode not in {"joint", "coordinate"}:
            raise ValueError("selector_mode must be joint or coordinate")
        if self.selector_greedy_passes <= 0:
            raise ValueError("selector_greedy_passes must be positive")
        if self.selector_pair_lookahead < 0:
            raise ValueError("selector_pair_lookahead must be non-negative")
        if self.infeasible_fallback not in {"least_violation", "abstain"}:
            raise ValueError("unsupported infeasible fallback")
        if self.barrier_weight < 0.0:
            raise ValueError("barrier_weight must be non-negative")
        if self.graph_control not in {"real", "shuffled", "marginal"}:
            raise ValueError("graph_control must be real, shuffled, or marginal")
        if self.selector_audit_rows < 0 or self.exact_solver_max_features <= 0:
            raise ValueError("invalid selector audit configuration")


@dataclass(frozen=True)
class ACCGConfig:
    protocol_id: str = "accg_action_conditional_joint_v1"
    variant: str = "accg_joint"
    v21: E1Config = field(default_factory=E1Config)
    constraint: FeatureConstraintConfig = field(default_factory=FeatureConstraintConfig)

    def validate(self) -> None:
        if self.variant not in ACCG_VARIANTS:
            raise ValueError(f"unsupported ACCG variant: {self.variant!r}")
        self.v21.validate()
        self.constraint.validate()
        expected = {
            "accg_joint": ("joint", "real", "least_violation"),
            "accg_coordinate": ("coordinate", "real", "least_violation"),
            "accg_shuffled_graph": ("joint", "shuffled", "least_violation"),
            "accg_marginal_only": ("coordinate", "marginal", "least_violation"),
            "accg_joint_abstain": ("joint", "real", "abstain"),
        }[self.variant]
        observed = (
            self.constraint.selector_mode,
            self.constraint.graph_control,
            self.constraint.infeasible_fallback,
        )
        if observed != expected:
            raise ValueError(f"variant {self.variant!r} requires selector/graph/fallback={expected}, got {observed}")
        if self.v21.warmup_epochs <= 0:
            raise ValueError("ACCG requires a non-empty shared V21 warmup branch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ACCGConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    values = dict(payload)
    if overrides:
        values.update(overrides)
    v21 = E1Config(**_mapping(values.pop("v21", None), "v21"))
    constraint = FeatureConstraintConfig(**_mapping(values.pop("constraint", None), "constraint"))
    config = ACCGConfig(v21=v21, constraint=constraint, **values)
    config.validate()
    return config
