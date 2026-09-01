"""Configuration for the unified TopoGate V0 model.

V0 is the model-identity refactor of the two historical scVICAR runners.  The
backbone and training objective are shared; ``parameterization`` selects the
corruption operator:

``fixed``
    Historical ``NeighborMix_scMAE``/scVICAR-F semantics.  The graph provides
    sampling probabilities and every sample uses ``1 - alpha`` mixing.

``topology``
    Historical ``RG_NeighborMix_scMAE``/scVICAR-T semantics.  Edge reliability
    changes the sampling probabilities and a topology-derived node gate controls
    both the input mixture and the pseudo-view loss weight.

The dataclass is deliberately independent from the legacy runners so a V0
experiment cannot silently inherit a mutable historical default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


# 历史命令行/论文记号统一映射到两个 canonical 参数化，避免 F/T 分支复制。
PARAMETERIZATION_ALIASES = {
    "f": "fixed",
    "-f": "fixed",
    "fixed": "fixed",
    "scvicar-f": "fixed",
    "t": "topology",
    "-t": "topology",
    "topology": "topology",
    "reliability": "topology",
    "scvicar-t": "topology",
}
PARAMETERIZATIONS = frozenset({"fixed", "topology"})
NEIGHBOR_ESTIMATORS = frozenset({"current", "uniform_sample", "full"})
EDGE_RELIABILITY_MODES = frozenset(
    {"none", "sim", "sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}
)


def normalize_parameterization(value: str) -> str:
    """Resolve the documented F/T aliases to one canonical value."""

    # 配置文件和 CLI 都经过这里；之后内部代码只比较 fixed/topology。
    key = str(value).strip().lower()
    try:
        return PARAMETERIZATION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "parameterization must be one of fixed/topology (aliases: F/T)"
        ) from exc


@dataclass(frozen=True)
class V0Config:
    """Validated, serializable configuration for TopoGate V0."""

    protocol_id: str = "topogate_v0_scvicar_v1"
    parameterization: str = "fixed"

    # 共享的 scMAE backbone 与重构/mask 目标；F/T 不在这些字段上分叉。
    hidden_size: int = 128
    dropout: float = 0.0
    masked_data_weight: float = 0.75
    mask_loss_weight: float = 0.70
    mask_ratio: float = 0.40

    # 优化器和 DataLoader 的公共协议。
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    num_workers: int = 0
    drop_last: bool = False

    # 邻域 pseudo-view corruption 的公共参数。
    use_pseudo: bool = True
    pseudo_weight: float = 0.30
    alpha: float = 0.90
    neighbor_k: int = 5
    mix_neighbors: int = 4
    neighbor_estimator: str = "current"
    knn_pca_dim: int = 50
    tau: float = 0.20

    # T 的 topology 参数。即使运行 F 也保留这些字段，保证 resolved config
    # 能完整描述同一个 V0 operator，便于跨参数化审计。
    edge_reliability_mode: str = "sim_mutual_snn_distance"
    gamma_sim: float = 1.0
    gamma_mutual: float = 1.0
    gamma_snn: float = 1.0
    gamma_distance: float = 1.0
    gate_min: float = 0.0
    gate_max: float = 0.15
    beta_mutual: float = 1.0
    beta_snn: float = 1.0
    beta_perturb: float = 2.0
    beta_uncertainty: float = 1.0

    # CLI 预处理和最终 KMeans readout；它们不改变 backbone 的定义。
    kmeans_n_init: int = 20
    n_top_features: int = 1000
    target_sum: float = 10_000.0
    input_mode: str = "auto"
    scale_input: bool = True
    evaluate_unsupervised: bool = False

    def __post_init__(self) -> None:
        # dataclass 创建时立即校验，避免非法配置在训练数小时后才暴露。
        canonical = normalize_parameterization(self.parameterization)
        object.__setattr__(self, "parameterization", canonical)
        if not str(self.protocol_id).strip():
            raise ValueError("protocol_id must be non-empty")

        positive = {
            "hidden_size": self.hidden_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "neighbor_k": self.neighbor_k,
            "mix_neighbors": self.mix_neighbors,
            "knn_pca_dim": self.knn_pca_dim,
            "kmeans_n_init": self.kmeans_n_init,
        }
        if any(int(value) <= 0 for value in positive.values()):
            raise ValueError("model, graph, preprocessing, and training sizes must be positive")
        if int(self.n_top_features) < 0:
            raise ValueError("n_top_features must be non-negative (0 disables selection)")
        if int(self.num_workers) < 0:
            raise ValueError("num_workers must be non-negative")

        bounded = {
            "dropout": self.dropout,
            "masked_data_weight": self.masked_data_weight,
            "mask_loss_weight": self.mask_loss_weight,
            "mask_ratio": self.mask_ratio,
            "alpha": self.alpha,
            "gate_min": self.gate_min,
            "gate_max": self.gate_max,
        }
        if any(not 0.0 <= float(value) <= 1.0 for value in bounded.values()):
            raise ValueError("dropout, mask, alpha, and gate values must be in [0, 1]")
        if float(self.dropout) >= 1.0:
            raise ValueError("dropout must be less than 1")
        if float(self.gate_min) > float(self.gate_max):
            raise ValueError("gate_min must be <= gate_max")
        if float(self.pseudo_weight) < 0.0:
            raise ValueError("pseudo_weight must be non-negative")
        for name in (
            "gamma_sim",
            "gamma_mutual",
            "gamma_snn",
            "gamma_distance",
            "beta_mutual",
            "beta_snn",
            "beta_perturb",
            "beta_uncertainty",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.lr) <= 0.0 or float(self.tau) <= 0.0 or float(self.target_sum) <= 0.0:
            raise ValueError("lr, tau, and target_sum must be positive")
        if self.neighbor_estimator not in NEIGHBOR_ESTIMATORS:
            raise ValueError(f"neighbor_estimator must be one of {sorted(NEIGHBOR_ESTIMATORS)}")
        if self.edge_reliability_mode not in EDGE_RELIABILITY_MODES:
            raise ValueError(
                f"edge_reliability_mode must be one of {sorted(EDGE_RELIABILITY_MODES)}"
            )
        if self.input_mode not in {"auto", "raw", "log1p"}:
            raise ValueError("input_mode must be auto, raw, or log1p")

    @property
    def graph_enabled(self) -> bool:
        # pseudo 分支只有在开关、权重、邻居数都有效时才建图；关闭时 trainer
        # 返回空图并保持诊断形状，避免引入隐式的 fallback 算法。
        return bool(
            self.use_pseudo
            and self.pseudo_weight > 0.0
            and self.neighbor_k > 0
            and self.mix_neighbors > 0
        )

    @property
    def mix_mode(self) -> str:
        # 这是审计字段：fixed 对应 F 的基础概率，reliability 对应 T 的边重加权。
        return "reliability" if self.parameterization == "topology" else "fixed"

    @property
    def gate_mode(self) -> str:
        # F 使用全体共享 gate，T 使用由拓扑统计量解析得到的 node gate。
        return "topology" if self.parameterization == "topology" else "fixed"

    def for_parameterization(self, value: str) -> "V0Config":
        """Return the same protocol with a canonical F/T parameterization."""

        # replace 保留所有共享协议，只替换参数化；返回的新对象仍会触发校验。
        return replace(self, parameterization=normalize_parameterization(value))

    def resolved_dict(self) -> dict[str, Any]:
        """Return the config plus effective operator fields for audit logs."""

        # 除原始字段外写入实际生效的 mix/gate/reliability 语义，避免只看默认值
        # 无法判断某次 F/T 运行究竟走了哪条计算路径。
        payload = asdict(self)
        payload.update(
            {
                "parameterization": self.parameterization,
                "parameterization_aliases": [
                    "F" if self.parameterization == "fixed" else "T"
                ],
                "graph_enabled": self.graph_enabled,
                "mix_mode": self.mix_mode,
                "gate_mode": self.gate_mode,
                "effective_mix_strength": (
                    1.0 - float(self.alpha)
                    if self.parameterization == "fixed"
                    else float(self.gate_max)
                ),
                "effective_edge_reliability_mode": (
                    self.edge_reliability_mode
                    if self.parameterization == "topology"
                    else "base_probability"
                ),
                "effective_pseudo_loss_weight": (
                    float(self.pseudo_weight) if self.graph_enabled else 0.0
                ),
            }
        )
        return payload


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> V0Config:
    """Load a YAML mapping and apply explicit non-``None`` overrides."""

    # YAML 提供可复现实验协议，显式 CLI override 拥有更高优先级；variant 是
    # 兼容历史配置的参数化别名，最终仍由 V0Config 统一归一化和校验。
    payload: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"V0 config must be a mapping: {path}")
        payload.update(loaded)
    if overrides:
        payload.update({str(key): value for key, value in overrides.items() if value is not None})
    if "variant" in payload and "parameterization" not in payload:
        payload["parameterization"] = payload.pop("variant")
    unknown = sorted(set(payload) - set(V0Config.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown V0 config keys: {unknown}")
    return V0Config(**payload)


__all__ = [
    "EDGE_RELIABILITY_MODES",
    "NEIGHBOR_ESTIMATORS",
    "PARAMETERIZATIONS",
    "V0Config",
    "load_config",
    "normalize_parameterization",
]
