# TopoGate V9 完整代码快照

> 本文件是从源仓库当前工作树生成的 V9 代码归档，便于单文件阅读、审阅和引用。代码块中的文件内容保持源文件原样；本文件本身不是可直接导入的 Python 模块。

- 生成日期：2026-08-07
- 源代码根目录：`methods/TopoGate/learnable_gate/`
- 纳入文件：23 个，合计 3157 行、141452 字节
- 版本边界：该目录是项目定义的 V9 legacy 主线；V10、V11、V12 独立实现位于其他目录。

## 范围与边界

- 纳入 `learnable_gate/` 下当前运行树中的 Python 源码、YAML 配置、README 和 `v5_components/` 归档组件。
- `configs/learnable_gate_v10_nomix_init.yaml`、`learnable_gate_v11_nomix_warmup.yaml` 和 `learnable_gate_v12_risk_adaptive.yaml` 仍由 V9 的 `run_npz.py` 读取，因此保留在本快照中；文件名不代表它们属于独立的 V10/V11/V12 实现。
- 排除 `_backup_v3_20260726_004309/` 下的 `.bak` 历史备份、`__pycache__/`、`.pyc`、数据、结果、日志和模型权重。
- 直接运行依赖不在本文件中重复展开：[`methods/NeighborMix_scMAE/model.py`](methods/NeighborMix_scMAE/model.py) 和 [`methods/shared_utils.py`](methods/shared_utils.py)。

## 入口与相关文档

- 命令行入口：`methods/TopoGate/learnable_gate/run_npz.py:main`
- 程序化入口：`methods/TopoGate/learnable_gate/run_npz.py:run_topogate`
- 版本边界与输出契约：[`methods/TopoGate/CORE_CODE_INDEX.md`](methods/TopoGate/CORE_CODE_INDEX.md)
- 相关 V9 报告：[`reports/analysis/V9_AHDPC_feature_profile_2026-08-03.md`](reports/analysis/V9_AHDPC_feature_profile_2026-08-03.md)、[`reports/analysis/V9_AHDPC_advantage_deep_analysis_2026-08-03.md`](reports/analysis/V9_AHDPC_advantage_deep_analysis_2026-08-03.md)

## 文件清单与校验

| 文件 | 类型 | 行数 | 字节数 | SHA-256 |
|---|---:|---:|---:|---|
| `methods/TopoGate/learnable_gate/README.md` | Markdown | 63 | 2680 | `518b588d2307eb945eb46a7c79befd729919b427b9e536bf7effc953ba5884bf` |
| `methods/TopoGate/learnable_gate/__init__.py` | Python | 39 | 1243 | `30ebcae2a13114b52c98f253af15a0f8f4e8f783f0e099afafbceb56240665bf` |
| `methods/TopoGate/learnable_gate/binary_router.py` | Python | 153 | 6881 | `d89cc98ecd71fb33bb95f1c8b281a807d58b810b0e632e8985aa8dc771272cb6` |
| `methods/TopoGate/learnable_gate/configs/binary_router.yaml` | YAML | 21 | 501 | `2263a0e7cdd75869dd9a5fd733e422db8687b353d49316a0c36673bcbca7e955` |
| `methods/TopoGate/learnable_gate/configs/learnable_gate_hvf_adaptive.yaml` | YAML | 24 | 987 | `c953c96fe34d2d1eb4fc549d077245f16b31324fefbb781d084b4e61ab4df81c` |
| `methods/TopoGate/learnable_gate/configs/learnable_gate_sched.yaml` | YAML | 12 | 273 | `c7ab634e5fd2709be43bc85b8b09f66c434008116525280ac94a3123dad76d83` |
| `methods/TopoGate/learnable_gate/configs/learnable_gate_sched_v3.yaml` | YAML | 16 | 363 | `513e9a7935b6b3ba75e22e72e0bc16bc41f5d9918e9fc4cbfa15bc19e9697e13` |
| `methods/TopoGate/learnable_gate/configs/learnable_gate_v10_nomix_init.yaml` | YAML | 15 | 342 | `360087b5a0bf82091978c05300d82d86ac885641792711e7a6524a094e8f1c87` |
| `methods/TopoGate/learnable_gate/configs/learnable_gate_v11_nomix_warmup.yaml` | YAML | 16 | 374 | `900d9aad11e7353f197284c81312a2392e0856d69a875610a47e6ea98d1d552d` |
| `methods/TopoGate/learnable_gate/configs/learnable_gate_v12_risk_adaptive.yaml` | YAML | 17 | 399 | `17c9b8dbaed31869b48c4c081dd45499f32ca9212bcfe252441c5c212a4225d7` |
| `methods/TopoGate/learnable_gate/configs/learnable_gate_v9_adaptive.yaml` | YAML | 19 | 693 | `3d50fe6ced15005821afbee5741364fd2b663c9e9389db7e4c8da278194d14ef` |
| `methods/TopoGate/learnable_gate/diagnostics.py` | Python | 66 | 2303 | `1e8484e9e8a887562e4e5341b3164823e3600ab14ad1e4e757ff05131ee8174e` |
| `methods/TopoGate/learnable_gate/learnable_edge_reliability.py` | Python | 153 | 7233 | `52a97771d65398c268dab15d6a2560ca4c2673da3cf5003669225fcacfb1f864` |
| `methods/TopoGate/learnable_gate/learnable_gate.py` | Python | 244 | 11804 | `9f261b5c850d1f92c534afce569632044f9beb3e1b8a2ec49144cbf7e5f53a6f` |
| `methods/TopoGate/learnable_gate/mixing.py` | Python | 300 | 14632 | `a742994e65e5f55335eec75f5d550dd3b64c6212b2da69f0911e1064db929d94` |
| `methods/TopoGate/learnable_gate/model.py` | Python | 75 | 3067 | `512cf1eadc2df1471fbfcbf7bbc083362c362ba1f54ab1840ba59187074f1401` |
| `methods/TopoGate/learnable_gate/neighbor_graph.py` | Python | 291 | 12099 | `d40721759fcf24b343fd94f4b362608833f51fb20897f951a3f6b766dffd013c` |
| `methods/TopoGate/learnable_gate/run_npz.py` | Python | 1138 | 54492 | `e7424f11ff379f7e936a075738baffadc7fc221f1a39f4c4f329b0d394d2f588` |
| `methods/TopoGate/learnable_gate/uncertainty.py` | Python | 57 | 1991 | `c261075ecce1436a95f74b1a6cdb6a2d8d1cf2ea74b6120b81d462e6ea8fcd18` |
| `methods/TopoGate/learnable_gate/v5_components/__init__.py` | Python | 5 | 205 | `f68088f0e4a24a36561f7c65a78cafb6d0b62bdadfec53b9fe6ebe306710c04b` |
| `methods/TopoGate/learnable_gate/v5_components/learnable_edge_reliability_v5.py` | Python | 223 | 9909 | `f315d298b577093e5d4f479448f000211e177e1bfc099dcc6b033d5295628389` |
| `methods/TopoGate/learnable_gate/v5_components/mask_noise_v5.py` | Python | 85 | 3807 | `31b2dc4bf84841b356879cab66d5bb60590a38256d438d01bcacf2d42c467b9e` |
| `methods/TopoGate/learnable_gate/v5_components/per_sample_mask_v5.py` | Python | 125 | 5174 | `2f451933a98479da2e52cbe22c402880b8274230e8605e8580e0a89c6d6e6e7f` |

## 源码与配置

## `methods/TopoGate/learnable_gate/README.md`

````markdown
# LearnableGate — canonical mainline

This directory contains the **current mainline** TopoGate implementation.

It is the V9 legacy path. The independently implemented V10, V11 and V12
variants live in `../v10_reliable_graph/`, `../V11/` and
`../V12_latent_topology/`; similarly named YAML files in this directory still
use this legacy runner and must not be relabelled as those versions.

## What v2 adds over v1

The single key change is that the four topological-gate coefficients
(`β_mutual`, `β_snn`, `β_perturb`, `β_uncertainty`) became learnable
`torch.nn.Parameter` instead of fixed argparse defaults.  They participate
in the MAE loss gradient through a `LearnableGate` module and a per-epoch
schedule that interpolates from the v1 static gate to the learned gate
during warmup.

### New components

- `learnable_gate.py` — `LearnableGate` (4β → per-node gate via sigmoid) and
  `build_gate_stats_tensor` (graph features → 4-D stats tensor)
- `configs/learnable_gate_sched.yaml` — v2 config: `gate_mode=learned`, plus
  `warmup_epochs=20`, `ramp_epochs=10`, `learned_gate_init_mode=zero`

### New CLI flags (in `run_npz.py`)

- `--gate_mode learned`     (else behaves like v1)
- `--warmup_epochs N`       (default 20)
- `--ramp_epochs N`         (default 10)
- `--learned_gate_init_mode {zero, v1_default}`
- `--init_beta_mutual / _snn / _perturb / _uncertainty`
- `--freeze_mae_after_epoch N`   (default 1e9 = disabled; set to e.g. 30 to freeze
                                 MAE after the ramp so β can settle on a stable target)

### Diagnostics

- `summary.json` now contains `learned_gate_final_beta` and
  `learned_gate_beta_history` (per-epoch β values, including `mae_frozen`
  flag) for post-hoc β-curve analysis.

See `../CORE_CODE_INDEX.md` for the complete version map and output contracts.

## How to run

```bash
# Smoke test (3-way compare: v1, v2 schedule=0, v2 schedule=20/10)
python scripts/learnable_gate/run_learnable_gate_sched_smoke.py --gpu 4 --datasets har enron

# Direct Python API
from methods.TopoGate.learnable_gate.run_npz import run_topogate
labels = run_topogate(X, n_clusters=K, gpu=4, seed=42, variant="learnable_gate_sched",
                      epochs=150)
```

## Modifications log

- 2026-07-25: created this directory during the static_gate/learnable_gate split.  Source files
  byte-identical to the pre-split `methods/TopoGate/*.py` (verified by
  diff against `/tmp/topogate_pre_v1v2_split_*.tar.gz`).
- 2026-07-25: config renamed `learnable_gate.yaml` → `learnable_gate_sched.yaml`.
- 2026-07-25: added `--freeze_mae_after_epoch` and per-epoch
  `learned_gate_beta_history` logging for ablation studies.
````

## `methods/TopoGate/learnable_gate/__init__.py`

```python
"""LearnableGate package (formerly "v2").

The current mainline implementation where the 4 gate coefficients (β_mutual,
β_snn, β_perturb, β_uncertainty) are learnable nn.Parameter.
"""
from methods.TopoGate.learnable_gate.model import AutoEncoder
from methods.TopoGate.learnable_gate.mixing import compute_node_gate, make_pseudo_batch
from methods.TopoGate.learnable_gate.neighbor_graph import (
    NeighborGraph,
    build_pca_knn_graph,
    build_random_neighbors,
    build_far_neighbors,
    compute_edge_reliability,
)
from methods.TopoGate.learnable_gate.diagnostics import (
    embedding_geometry,
    mapped_predictions,
    per_cell_type_metrics,
)
from methods.TopoGate.learnable_gate.learnable_gate import LearnableGate, build_gate_stats_tensor
from methods.TopoGate.learnable_gate.run_npz import run_topogate, main as run_npz_main

__all__ = [
    "AutoEncoder",
    "NeighborGraph",
    "build_pca_knn_graph",
    "build_random_neighbors",
    "build_far_neighbors",
    "compute_edge_reliability",
    "compute_node_gate",
    "make_pseudo_batch",
    "embedding_geometry",
    "mapped_predictions",
    "per_cell_type_metrics",
    "LearnableGate",
    "build_gate_stats_tensor",
    "run_topogate",
    "run_npz_main",
]
```

## `methods/TopoGate/learnable_gate/binary_router.py`

```python
"""BinaryRouter: differentiable hard routing between anchor and topology-aware mixed embedding.

Problem with the continuous gate (LearnableGate):
  mixed = (1-g)*anchor + g*neighbor,  g = sigmoid(beta·stats) ∈ (0, gate_max]
  - Even when g→0, the gradient vanishes if anchor≈neighbor
  - enron: full(g=0.075)=0.768 vs nomix=0.875, Δ=0.107 — even minimum mixing hurts
  - The gate can only suppress, never hard-reset

Solution: BinaryRouter
  r = GumbelSoftmax(logit), logit = beta·stats
  r ∈ {0, 1} (hard during inference, differentiable during training)
  x' = r * mixed + (1-r) * anchor
    = anchor                   if r=0  (topology says neighbor is bad)
    = mixed(anchor,neighbor)  if r=1  (topology says neighbor is good)

The beta parameters are shared with LearnableGate for convenience (same topology
features → same coefficients), but the output head is a CLASSIFICATION rather
than a REGRESSION — the model learns which topology patterns imply "use neighbor".

The Gumbel-Softmax uses a temperature schedule:
  - epochs 1..warmup:    temperature = init_temp (high = soft, ≈ v1 behaviour)
  - after ramp:          temperature → 0.01 (hard samples)
During INFERENCE (epoch=args.epochs+1): argmax over logits (pure hard routing).

Why this works when LearnableGate fails:
  1. r is genuinely binary — when r=0, x'=anchor exactly (not anchor*(1-g)+neighbor*g)
  2. Gradient flows through the logit → beta path even when anchor≈neighbor,
     because the routing decision itself is what matters, not the interpolation weight
  3. The model can learn "for this node's topology, the neighbor is always wrong"
     and hard-route to anchor without any residual mixing

The sample_weight in the pseudo-loss is simply r (the routing probability
during soft phase, or 1.0 during hard phase), so nodes that route to anchor
contribute zero pseudo-loss.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryRouter(nn.Module):
    """Differentiable binary router via Gumbel-Softmax.

    Args:
        temperature_init: Initial Gumbel-Softmax temperature (higher = softer).
            Default 5.0 — at init all betas=0, logits≈0, so with temp=5 the
            router is very soft, close to v1's continuous gate.
        temperature_min: Floor temperature after ramp. Default 0.01.
        warmup_epochs: Number of epochs with high temperature (soft routing).
        ramp_epochs: Linear cool-down from temperature_init to temperature_min.
        enhanced_stats: Same as LearnableGate. Default 4, or 6 for degree/cluster.
        init_beta_*: Beta initialisation. Defaults 0.0 (uniform prior).
    """

    def __init__(
        self,
        temperature_init: float = 5.0,
        temperature_min: float = 0.01,
        warmup_epochs: int = 20,
        ramp_epochs: int = 10,
        enhanced_stats: int = 4,
        init_beta_mutual: float = 0.0,
        init_beta_snn: float = 0.0,
        init_beta_perturb: float = 0.0,
        init_beta_uncertainty: float = 0.0,
        init_beta_degree: float = 0.0,
        init_beta_cluster: float = 0.0,
    ) -> None:
        super().__init__()
        self.temperature_init = float(temperature_init)
        self.temperature_min = float(temperature_min)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.enhanced_stats = int(enhanced_stats)

        self.beta_mutual = nn.Parameter(torch.tensor(float(init_beta_mutual)))
        self.beta_snn = nn.Parameter(torch.tensor(float(init_beta_snn)))
        self.beta_perturb = nn.Parameter(torch.tensor(float(init_beta_perturb)))
        self.beta_uncertainty = nn.Parameter(torch.tensor(float(init_beta_uncertainty)))
        if enhanced_stats == 6:
            self.beta_degree = nn.Parameter(torch.tensor(float(init_beta_degree)))
            self.beta_cluster = nn.Parameter(torch.tensor(float(init_beta_cluster)))

    def _compute_logits(self, stats: torch.Tensor) -> torch.Tensor:
        """logits = beta · stats  (higher → route to mixed/USE_NEIGHBOR)."""
        logits = (
            self.beta_mutual * stats[:, 0]
            + self.beta_snn * stats[:, 1]
            - self.beta_perturb * stats[:, 2]
            - self.beta_uncertainty * stats[:, 3]
        )
        if self.enhanced_stats == 6:
            logits = logits + self.beta_degree * stats[:, 4] - self.beta_cluster * stats[:, 5]
        return logits

    def _temperature(self, epoch: int) -> float:
        """Linear schedule from temperature_init → temperature_min."""
        if epoch <= self.warmup_epochs:
            return self.temperature_init
        t = min(1.0, (epoch - self.warmup_epochs) / max(1, self.ramp_epochs))
        return self.temperature_init + t * (self.temperature_min - self.temperature_init)

    def forward(
        self,
        stats: torch.Tensor,
        epoch: int,
        hard: bool = False,
    ) -> torch.Tensor:
        """Sample routing decision.

        Args:
            stats: (batch, 4) or (batch, 6) topology features.
            epoch: Current training epoch (for temperature schedule).
            hard: If True, use argmax (pure hard routing, no Gumbel noise).
                  If False, use Gumbel-Softmax (differentiable).

        Returns:
            (batch,) tensor of routing decisions:
              1.0 = USE_MIXED (neighbor topology is trusted)
              0.0 = USE_ANCHOR (self-reconstruction)
            Values are either {0.0, 1.0} (hard=True or very low temp)
            or soft probabilities in (0, 1) (high temperature).
        """
        logits = self._compute_logits(stats)

        if hard or not self.training:
            # Inference: pure hard argmax
            return (logits > 0).float()

        temperature = self._temperature(epoch)
        # Gumbel-Softmax: sample from Gumbel - log(-log(Uniform))
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        gumbel_logits = (logits + gumbel_noise) / temperature
        return torch.sigmoid(gumbel_logits)

    def routing_probability(self, stats: torch.Tensor) -> torch.Tensor:
        """Pure routing probability (no Gumbel noise). For analysis only."""
        return torch.sigmoid(self._compute_logits(stats))

    def beta_snapshot(self) -> dict:
        snap = {
            "beta_mutual": float(self.beta_mutual.detach().cpu()),
            "beta_snn": float(self.beta_snn.detach().cpu()),
            "beta_perturb": float(self.beta_perturb.detach().cpu()),
            "beta_uncertainty": float(self.beta_uncertainty.detach().cpu()),
        }
        if self.enhanced_stats == 6:
            snap["beta_degree"] = float(self.beta_degree.detach().cpu())
            snap["beta_cluster"] = float(self.beta_cluster.detach().cpu())
        return snap
```

## `methods/TopoGate/learnable_gate/configs/binary_router.yaml`

```yaml
method_name: TopoGate
variant_name: binary_router
mix_mode: reliability
gate_mode: binary
edge_reliability_mode: sim_mutual_snn_distance
neighbor_k: 5
mix_neighbors: 4
pseudo_weight: 0.3
# BinaryRouter specific
router_init_temp: 5.0
router_temp_min: 0.01
router_warmup_epochs: 20
router_ramp_epochs: 10
# Use v1 default init so early epochs ≈ v1 behaviour
init_beta_mutual: 0.0
init_beta_snn: 0.0
init_beta_perturb: 0.0
init_beta_uncertainty: 0.0
enhanced_stats: 4
warmup_epochs: 20
ramp_epochs: 10
```

## `methods/TopoGate/learnable_gate/configs/learnable_gate_hvf_adaptive.yaml`

```yaml
method_name: TopoGate
variant_name: learnable_gate_hvf_adaptive
mix_mode: reliability
gate_mode: learned
edge_reliability_mode: sim_mutual_snn_distance
neighbor_k: 5
mix_neighbors: 4
gate_max: 0.15
pseudo_weight: 0.3
warmup_epochs: 20
ramp_epochs: 10
learned_gate_init_mode: zero
# ── HVF + Adaptive PCA (new) ──────────────────────────────────────────────
# HVF: keep top-N high-variance features before PCA.
#   0 = disabled (use all features). Recommended: 1000-2000 for d > 5000.
#   For datasets with very high d (e.g. scRNA, MNIST), HVF dramatically
#   improves kNN quality by removing noisy dimensions.
n_top_features: 1000
# Adaptive PCA: auto-select dim to retain ≥95% variance.
#   fixed = use knn_pca_dim directly.
#   adaptive = auto-select (capped at knn_pca_dim).
knn_pca_mode: adaptive
# Upper bound for adaptive PCA. Only used when knn_pca_mode=adaptive.
knn_pca_dim: 200
```

## `methods/TopoGate/learnable_gate/configs/learnable_gate_sched.yaml`

```yaml
method_name: TopoGate
variant_name: learnable_gate_sched
mix_mode: reliability
gate_mode: learned
edge_reliability_mode: sim_mutual_snn_distance
neighbor_k: 5
mix_neighbors: 4
gate_max: 0.15
pseudo_weight: 0.3
warmup_epochs: 20
ramp_epochs: 10
learned_gate_init_mode: zero
```

## `methods/TopoGate/learnable_gate/configs/learnable_gate_sched_v3.yaml`

```yaml
method_name: TopoGate
variant_name: learnable_gate_sched_v3
mix_mode: reliability
gate_mode: learned
edge_reliability_mode: sim_mutual_snn_distance
neighbor_k: 5
mix_neighbors: 4
gate_max: 0.15
pseudo_weight: 0.3
warmup_epochs: 20
ramp_epochs: 10
learned_gate_init_mode: zero
learnable_gate_max: true
gate_max_min: 0.05
gate_max_max: 1.0
gate_lr_multiplier: 10.0
```

## `methods/TopoGate/learnable_gate/configs/learnable_gate_v10_nomix_init.yaml`

```yaml
method_name: TopoGate
variant_name: learnable_gate_v10_nomix_init
mix_mode: reliability
gate_mode: learned
edge_reliability_mode: sim_mutual_snn_distance
neighbor_k: 5
mix_neighbors: 4
gate_max: 0.15
pseudo_weight: 0.3
warmup_epochs: 20
ramp_epochs: 10
learned_gate_init_mode: nomix
n_top_features: 0
knn_pca_mode: adaptive
knn_pca_dim: 2000
```

## `methods/TopoGate/learnable_gate/configs/learnable_gate_v11_nomix_warmup.yaml`

```yaml
method_name: TopoGate
variant_name: learnable_gate_v11_nomix_warmup
mix_mode: reliability
gate_mode: learned
edge_reliability_mode: sim_mutual_snn_distance
neighbor_k: 5
mix_neighbors: 4
gate_max: 0.15
pseudo_weight: 0.3
warmup_epochs: 20
ramp_epochs: 10
learned_gate_init_mode: nomix
use_beta_scale_schedule: true
n_top_features: 0
knn_pca_mode: adaptive
knn_pca_dim: 2000
```

## `methods/TopoGate/learnable_gate/configs/learnable_gate_v12_risk_adaptive.yaml`

```yaml
method_name: TopoGate
variant_name: learnable_gate_v12_risk_adaptive
mix_mode: reliability
gate_mode: learned
edge_reliability_mode: sim_mutual_snn_distance
neighbor_k: 5
mix_neighbors: 4
gate_max: 0.15
pseudo_weight: 0.3
warmup_epochs: 20
ramp_epochs: 10
learned_gate_init_mode: zero
n_top_features: 0
knn_pca_mode: adaptive
knn_pca_dim: 2000
risk_adaptive_mix: true
risk_adaptive_temperature: 1.0
```

## `methods/TopoGate/learnable_gate/configs/learnable_gate_v9_adaptive.yaml`

```yaml
method_name: TopoGate
variant_name: learnable_gate_v9_adaptive
mix_mode: reliability
gate_mode: learned
edge_reliability_mode: sim_mutual_snn_distance
neighbor_k: 5
mix_neighbors: 4
gate_max: 0.15
pseudo_weight: 0.3
warmup_epochs: 20
ramp_epochs: 10
learned_gate_init_mode: zero
# ── v9: no HVF + Adaptive PCA (PCA cap raised to 2000) ───────────────────
# v2 baseline (learnable_gate_sched.yaml) uses knn_pca_mode=fixed + dim=50.
# v9 removes the 50-dim cap and lets the adaptive selector keep ≥95% variance.
# HVF is disabled (n_top_features=0); Adaptive PCA alone is the v9 vs v2 delta.
n_top_features: 0
knn_pca_mode: adaptive
knn_pca_dim: 2000
```

## `methods/TopoGate/learnable_gate/diagnostics.py`

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_fscore_support


def embedding_geometry(embedding: np.ndarray, labels: np.ndarray) -> dict:
    emb = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(labels)
    centroids = []
    within = []
    for lab in np.unique(labels):
        block = emb[labels == lab]
        if block.size == 0:
            continue
        c = block.mean(axis=0)
        centroids.append(c)
        within.append(np.linalg.norm(block - c, axis=1).mean())
    if len(centroids) <= 1:
        between = 0.0
    else:
        c = np.vstack(centroids)
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        between = float(d[np.triu_indices_from(d, k=1)].mean())
    within_mean = float(np.mean(within)) if within else 0.0
    return {
        "within_class_distance": within_mean,
        "between_class_distance": between,
        "between_within_ratio": float(between / max(within_mean, 1e-8)),
    }


def mapped_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    n = max(len(true_values), len(pred_values))
    counts = np.zeros((n, n), dtype=np.int64)
    for i, t in enumerate(true_values):
        for j, p in enumerate(pred_values):
            counts[i, j] = int(np.sum((y_true == t) & (y_pred == p)))
    rows, cols = linear_sum_assignment(-counts)
    mapped = np.zeros_like(y_pred, dtype=np.int64)
    for row, col in zip(rows, cols):
        if row < len(true_values) and col < len(pred_values):
            mapped[y_pred == pred_values[col]] = true_values[row]
    return mapped


def per_cell_type_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    mapped = mapped_predictions(y_true, y_pred)
    labels = np.unique(y_true)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, mapped, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "label": labels,
            "n_cells": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "is_rare_lt_50": support < 50,
        }
    )
```

## `methods/TopoGate/learnable_gate/learnable_edge_reliability.py`

```python
"""LearnableEdgeReliability: promote 4 gamma coefficients to nn.Parameter.

The original `compute_edge_reliability` in neighbor_graph.py takes 4 gamma
coefficients as argparse-fixed constants (gamma_sim=1.0, gamma_mutual=1.0,
gamma_snn=1.0, gamma_distance=1.0).  The 90-run multiseed analysis shows that
the ablation with all gammas fixed (gate_only - full = -0.0009) does NOT improve
ARI on 5 datasets — these coefficients are dead parameters from the gradient's
perspective.

This module wraps the 4 gamma into nn.Parameter so they can be learned via the
MAE + pseudo reconstruction loss.  The weights are computed from a 2D embedding
(numpy → torch tensor) so that autograd flows back.

Design choices:
- Parameters are kept raw (not softplus'd) so the initial value 1.0 corresponds
  to the original v1 default.
- To prevent numerical explosion (very large gamma_mutual can make rel = inf),
  we add a soft L2 regularisation term that the training loop accumulates.
  The actual loss term is exposed as `regularization_loss()`.
- The forward() returns torch tensors so the rest of the pipeline (which
  converts to numpy for kNN sampling) gets the gradient-tracking version.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .neighbor_graph import NeighborGraph, summarize_edge_weights


class LearnableEdgeReliability(nn.Module):
    """Per-edge reliability as a learnable affine combination of 4 signals.

    Args:
        mode: 'sim', 'sim_mutual', 'sim_mutual_snn', 'sim_mutual_snn_distance'.
              When 'none' or empty graph, falls back to graph.probs unchanged.
        init_gamma_sim, init_gamma_mutual, init_gamma_snn, init_gamma_distance:
              Initial values for the four learnable gammas.  Default 1.0 matches
              the v1 argparse defaults.
        reg_weight: weight for the soft L2 regularisation loss (default 1e-4).
                    Prevents the gammas from drifting to extremes.
    """

    def __init__(
        self,
        mode: str = "sim_mutual_snn_distance",
        init_gamma_sim: float = 1.0,
        init_gamma_mutual: float = 1.0,
        init_gamma_snn: float = 1.0,
        init_gamma_distance: float = 1.0,
        reg_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.reg_weight = float(reg_weight)
        self.gamma_sim = nn.Parameter(torch.tensor(float(init_gamma_sim)))
        self.gamma_mutual = nn.Parameter(torch.tensor(float(init_gamma_mutual)))
        self.gamma_snn = nn.Parameter(torch.tensor(float(init_gamma_snn)))
        self.gamma_distance = nn.Parameter(torch.tensor(float(init_gamma_distance)))

    def gamma_snapshot(self) -> dict:
        return {
            "gamma_sim": float(self.gamma_sim.detach().cpu()),
            "gamma_mutual": float(self.gamma_mutual.detach().cpu()),
            "gamma_snn": float(self.gamma_snn.detach().cpu()),
            "gamma_distance": float(self.gamma_distance.detach().cpu()),
        }

    def regularization_loss(self) -> torch.Tensor:
        """L2 penalty on gammas to keep them from drifting to extremes."""
        if self.reg_weight <= 0:
            return torch.zeros((), device=self.gamma_sim.device)
        sq = (
            self.gamma_sim ** 2
            + self.gamma_mutual ** 2
            + self.gamma_snn ** 2
            + self.gamma_distance ** 2
        )
        return self.reg_weight * sq

    def forward(self, graph: NeighborGraph) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (reliability, weights) as torch tensors with gradient flow.

        Args:
            graph: NeighborGraph object holding similarity / mutual / snn / distance
                   as numpy arrays.

        Returns:
            (rel, weights): both (n_cells, k) torch tensors.  rel has gradient
            flowing back to the 4 gamma params; weights are the row-normalised
            version.  Both have requires_grad=False numpy equivalents for logging.
        """
        if graph.indices.shape[1] == 0 or self.mode == "none":
            rel = torch.ones_like(graph.probs, dtype=torch.float32)
            weights = rel.clone()
            return rel, weights

        rel = torch.ones(graph.similarity.shape, dtype=torch.float32,
                         device=self.gamma_sim.device)
        sim_t = torch.as_tensor(graph.similarity, dtype=torch.float32,
                                device=self.gamma_sim.device)
        mutual_t = torch.as_tensor(graph.mutual.astype(np.float32), dtype=torch.float32,
                                   device=self.gamma_sim.device)
        snn_t = torch.as_tensor(graph.snn, dtype=torch.float32,
                                device=self.gamma_sim.device)
        distance_t = torch.as_tensor(graph.distance, dtype=torch.float32,
                                     device=self.gamma_sim.device)
        probs_t = torch.as_tensor(graph.probs, dtype=torch.float32,
                                  device=self.gamma_sim.device)
        if self.mode in {"sim", "sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
            rel = rel * torch.exp(self.gamma_sim * sim_t)
        if self.mode in {"sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
            rel = rel * (1.0 + self.gamma_mutual * mutual_t)
        if self.mode in {"sim_mutual_snn", "sim_mutual_snn_distance"}:
            rel = rel * (1.0 + self.gamma_snn * snn_t)
        if self.mode == "sim_mutual_snn_distance":
            rel = rel * torch.exp(-self.gamma_distance * distance_t)
        rel = torch.clamp(rel, min=1e-6, max=1e6)
        weights = probs_t * rel
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
        return rel, weights


# Helper to convert a torch (n, k) tensor back to numpy for downstream
# numpy-only mix_mode code paths.  The caller is responsible for tracking
# gradients BEFORE this conversion (the conversion is only done for paths
# that don't go through `make_pseudo_batch(..., gate_tensor=...)`).
def edge_weights_to_numpy(weights_t: torch.Tensor) -> "np.ndarray":
    import numpy as np
    return weights_t.detach().cpu().numpy().astype(np.float32)


def summarize_edge_weights_torch(weights_t: torch.Tensor) -> dict:
    """Like summarize_edge_weights but for torch tensors (used in summary logging)."""
    if weights_t.numel() == 0:
        return {
            "edge_weight_entropy": 0.0,
            "effective_neighbor_count": 0.0,
            "max_edge_weight_mean": 0.0,
            "max_edge_weight_p95": 0.0,
            "fraction_effective_neighbors_lt_2": 1.0,
        }
    entropy = -torch.sum(weights_t * torch.log(torch.clamp(weights_t, min=1e-12)), dim=1)
    effective = torch.exp(entropy)
    max_w = torch.max(weights_t, dim=1).values
    return {
        "edge_weight_entropy": float(entropy.mean().detach().cpu()),
        "effective_neighbor_count": float(effective.mean().detach().cpu()),
        "max_edge_weight_mean": float(max_w.mean().detach().cpu()),
        "max_edge_weight_p95": float(torch.quantile(max_w, 0.95).detach().cpu()),
        "fraction_effective_neighbors_lt_2": float((effective < 2.0).float().mean().detach().cpu()),
    }
```

## `methods/TopoGate/learnable_gate/learnable_gate.py`

```python
"""LearnableGate: per-node gate as a learnable affine transform on topology stats.

TopoGate static_gate used 4 hand-tuned beta coefficients that were never exposed to the
gradient path.  LearnableGate promotes them to nn.Parameter so the MAE loss can
shape them per-dataset.  Initializing all betas to zero gives:

    sigmoid(0) = 0.5
    gate       = gate_min + (gate_max - gate_min) * 0.5

which lands at half of the (gate_min, gate_max) range and is numerically close
to the v1 mean gate (~0.079 on a 50-dim synthetic).  Schedule t in [0, 1]
interpolates a precomputed v1-style gate (numpy) toward the live LearnableGate
output so the first warmup_epochs reproduce the static behaviour and the
remaining epochs gradually turn the knobs over to the model.

v3 enhancements:
- learnable_gate_max: the gate_max itself is a learnable parameter (initialised
  at the user-supplied gate_max).  This is the v3 upgrade that solves the
  gate-saturation problem.
- enhanced_stats (default 6): when set to 6 the stats tensor includes
  [mutual, snn, perturb, uncertainty, degree_norm, clustering_coeff] and 6
  betas are learned (beta_degree + beta_cluster).  This expands the topology
  features beyond mutual/snn which the ablation showed were nearly useless.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LearnableGate(nn.Module):
    """Compute per-node gate via sigmoid(beta . stats) with optional schedule.

    Args:
        gate_min, gate_max: Output range for the gate.
        init_beta_mutual, init_beta_snn, init_beta_perturb, init_beta_uncertainty:
            Initial values for the four learnable coefficients.  All four default
            to 0, which yields sigmoid(0) = 0.5 -> gate = mid-point of the
            (gate_min, gate_max) range.  Set these to the v1 defaults
            (1.0, 1.0, 2.0, 1.0) if exact v1 reproduction at schedule=0 is
            required.
        learnable_gate_max: if True, the gate_max is also a learnable parameter
            (initialised at the user-supplied gate_max).  This is the v3 upgrade
            that solves the gate-saturation problem (beta grows but actual gate
            output stays below 0.11 when gate_max=0.15 is fixed).  When True,
            the upper bound of the gate range is gate_max_min + softplus(raw),
            where gate_max_min is the smallest legal value.
        gate_max_min: floor for the learnable gate_max (default 0.05; prevents
            the model from collapsing to a zero-mixing regime).
        gate_max_max: ceiling for the learnable gate_max (default 1.0; prevents
            numerical instability from extreme mixing).
        enhanced_stats: int, default 4.  If set to 6 the stats tensor expects
            [mutual, snn, perturb, uncertainty, degree_norm, clustering_coeff]
            and 6 betas are learned.
    """

    def __init__(
        self,
        gate_min: float = 0.0,
        gate_max: float = 0.15,
        init_beta_mutual: float = 0.0,
        init_beta_snn: float = 0.0,
        init_beta_perturb: float = 0.0,
        init_beta_uncertainty: float = 0.0,
        learnable_gate_max: bool = False,
        gate_max_min: float = 0.05,
        gate_max_max: float = 1.0,
        enhanced_stats: int = 4,
        init_beta_degree: float = 0.0,
        init_beta_cluster: float = 0.0,
    ) -> None:
        super().__init__()
        if enhanced_stats not in (4, 6):
            raise ValueError(f"enhanced_stats must be 4 or 6, got {enhanced_stats}")
        self.enhanced_stats = int(enhanced_stats)
        self.gate_min = float(gate_min)
        self.gate_max_initial = float(gate_max)
        self.gate_max_min = float(gate_max_min)
        self.gate_max_max = float(gate_max_max)
        self.learnable_gate_max = bool(learnable_gate_max)
        self.beta_mutual = nn.Parameter(torch.tensor(float(init_beta_mutual)))
        self.beta_snn = nn.Parameter(torch.tensor(float(init_beta_snn)))
        self.beta_perturb = nn.Parameter(torch.tensor(float(init_beta_perturb)))
        self.beta_uncertainty = nn.Parameter(torch.tensor(float(init_beta_uncertainty)))
        if self.enhanced_stats == 6:
            self.beta_degree = nn.Parameter(torch.tensor(float(init_beta_degree)))
            self.beta_cluster = nn.Parameter(torch.tensor(float(init_beta_cluster)))
        if self.learnable_gate_max:
            init_raw = self._inverse_softplus(self.gate_max_initial)
            self.gate_max_raw = nn.Parameter(torch.tensor(float(init_raw)))
        else:
            self.register_buffer("gate_max_raw", torch.tensor(0.0))
        # beta_scale: external scalar (no gradient). Controlled by run_npz.py for
        # the legacy nomix-warmup experiment. beta_scale=0 also blocks the beta
        # gradient from this branch; it is not a learn-while-closed mechanism.
        self.register_buffer("beta_scale", torch.tensor(1.0))

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x)

    def _inverse_softplus(self, value: float) -> float:
        # value is the target effective gate_max value (already in [gate_max_min, gate_max_max]).
        # y = gate_max_min + span * sigmoid(raw)  =>  raw = logit((y - min) / span)
        y = float(value)
        span = max(float(self.gate_max_max - self.gate_max_min), 1e-6)
        p = (y - self.gate_max_min) / span
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        return float(np.log(p / (1.0 - p)))

    def effective_gate_max(self) -> torch.Tensor:
        if self.learnable_gate_max:
            raw = self.gate_max_raw
            span = max(float(self.gate_max_max - self.gate_max_min), 1e-6)
            return self.gate_max_min + span * torch.sigmoid(raw)
        return torch.tensor(self.gate_max_initial)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """Compute per-sample gate.

        Args:
            stats: (batch, 4) or (batch, 6) tensor.
                   4: [mutual, snn, perturb, uncertainty]
                   6: [mutual, snn, perturb, uncertainty, degree_norm, clustering_coeff]

        Returns:
            (batch,) gate tensor in [gate_min, effective_gate_max] with gradients
            flowing back to all betas AND gate_max when learnable_gate_max=True.
        """
        expected = self.enhanced_stats
        if stats.ndim != 2 or stats.shape[1] != expected:
            raise ValueError(f"stats must be (batch, {expected}), got {tuple(stats.shape)}")
        logits = (
            self.beta_mutual * stats[:, 0]
            + self.beta_snn * stats[:, 1]
            - self.beta_perturb * stats[:, 2]
            - self.beta_uncertainty * stats[:, 3]
        )
        if expected == 6:
            logits = logits + self.beta_degree * stats[:, 4] - self.beta_cluster * stats[:, 5]
        sig = torch.sigmoid(logits)
        gate_max_t = self.effective_gate_max()
        # beta_scale=0 yields NoMix and zero beta gradient from this branch;
        # beta_scale=1 restores the normal learned gate.
        return self.gate_min + (gate_max_t - self.gate_min) * sig * self.beta_scale

    def beta_snapshot(self) -> dict:
        snap = {
            "beta_mutual": float(self.beta_mutual.detach().cpu()),
            "beta_snn": float(self.beta_snn.detach().cpu()),
            "beta_perturb": float(self.beta_perturb.detach().cpu()),
            "beta_uncertainty": float(self.beta_uncertainty.detach().cpu()),
            "effective_gate_max": float(self.effective_gate_max().detach().cpu()),
            "beta_scale": float(self.beta_scale.detach().cpu()),
        }
        if self.enhanced_stats == 6:
            snap["beta_degree"] = float(self.beta_degree.detach().cpu())
            snap["beta_cluster"] = float(self.beta_cluster.detach().cpu())
        return snap


def build_gate_stats_tensor(
    graph_indices: "np.ndarray",
    graph_mutual: "np.ndarray",
    graph_snn: "np.ndarray",
    graph_probs: "np.ndarray",
    graph_similarity: "np.ndarray",
    uncertainty: "np.ndarray | None" = None,
    device: "torch.device | str | None" = None,
    enhanced_stats: int = 4,
) -> torch.Tensor:
    """Stack the per-node stats into a (n_cells, n_stats) tensor.

    Args:
        enhanced_stats: 4 (default) or 6.  When 6, appends degree_norm and
            clustering_coeff to the per-node vector.

    Ordering (4 stats): [mutual, snn, perturb, uncertainty] — MUST match
        LearnableGate.forward.
    Ordering (6 stats): [mutual, snn, perturb, uncertainty, degree_norm,
        clustering_coeff].
    """
    import numpy as np
    n_cells = int(graph_indices.shape[0])
    mutual_ratio = np.asarray(graph_mutual.mean(axis=1), dtype=np.float32)
    snn_avg = np.asarray(graph_snn.mean(axis=1), dtype=np.float32)
    perturb = (1.0 - np.sum(graph_probs * graph_similarity, axis=1)).astype(np.float32)
    if uncertainty is None:
        unc = np.zeros(n_cells, dtype=np.float32)
    else:
        unc = np.asarray(uncertainty, dtype=np.float32).reshape(-1)
        if unc.shape != (n_cells,):
            raise ValueError(f"uncertainty must have shape ({n_cells},), got {unc.shape}")
    cols = [mutual_ratio, snn_avg, perturb, unc]
    if enhanced_stats == 6:
        degree_norm = np.full(n_cells, float(graph_indices.shape[1]) / max(float(n_cells), 1.0),
                              dtype=np.float32)
        # Approximate local clustering coefficient for each node = (# edges
        # among node's neighbours) / (k * (k-1)).
        # - n ≤ 5000: exact O(n²) computation.
        # - n > 5000: sampled approximation (up to 2000 nodes), then broadcast
        #   the global mean to all nodes so beta_cluster always receives a valid signal.
        n = n_cells
        k = int(graph_indices.shape[1])
        cluster = np.zeros(n, dtype=np.float32)
        if k >= 2 and n > 0:
            if n <= 5000:
                # Small datasets: exact O(n²) computation
                rows = np.repeat(np.arange(n, dtype=np.int64), k)
                cols_idx = graph_indices.ravel()
                adj = np.zeros((n, n), dtype=bool)
                adj[rows, cols_idx] = True
                np.fill_diagonal(adj, False)
                mat = adj.astype(np.int32)
                mat2 = mat @ mat  # (n, n)
                triangles = (mat * mat2).sum(axis=1)
                local_edges = triangles // 2
                cluster = local_edges.astype(np.float32) / float(k * (k - 1))
            else:
                # Large datasets: sampled approximation
                # Sample up to 2000 nodes, compute their local clustering coefficients,
                # then broadcast the global mean as a constant estimate for all nodes.
                m = min(2000, n)
                sample_idx = np.random.choice(n, size=m, replace=False)
                ratios = np.zeros(m, dtype=np.float32)
                for si, i in enumerate(sample_idx):
                    neighbors_i = set(graph_indices[i].tolist())
                    if len(neighbors_i) < 2:
                        ratios[si] = 0.0
                        continue
                    triangles = 0
                    for j in neighbors_i:
                        neighbors_j = set(graph_indices[j].tolist())
                        triangles += len(neighbors_i & neighbors_j)
                    ratios[si] = triangles / float(len(neighbors_i) * (len(neighbors_i) - 1))
                global_cluster = np.mean(ratios)
                cluster[:] = global_cluster  # fill all nodes with global estimate
        cols.append(degree_norm)
        cols.append(cluster)
    elif enhanced_stats != 4:
        raise ValueError(f"enhanced_stats must be 4 or 6, got {enhanced_stats}")
    stacked = np.stack(cols, axis=1)
    return torch.as_tensor(stacked, device=device)
```

## `methods/TopoGate/learnable_gate/mixing.py`

```python
from __future__ import annotations

import numpy as np
import torch

from methods.TopoGate.learnable_gate.neighbor_graph import NeighborGraph


def compute_node_gate(
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    gate_mode: str,
    gate_min: float,
    gate_max: float,
    beta_mutual: float,
    beta_snn: float,
    beta_perturb: float,
    beta_uncertainty: float,
    uncertainty: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_cells, k = graph.indices.shape
    if gate_mode == "none" or k == 0:
        gate = np.zeros(n_cells, dtype=np.float32)
        perturb = np.zeros(n_cells, dtype=np.float32)
    elif gate_mode == "constant":
        gate = np.full(n_cells, float(gate_max), dtype=np.float32)
        perturb = np.zeros(n_cells, dtype=np.float32)
    else:
        mutual_ratio = graph.mutual.mean(axis=1).astype(np.float32)
        snn_avg = graph.snn.mean(axis=1).astype(np.float32)
        perturb = 1.0 - np.sum(graph.probs * graph.similarity, axis=1)
        unc = np.zeros(n_cells, dtype=np.float32) if uncertainty is None else uncertainty.astype(np.float32)
        logits = (
            float(beta_mutual) * mutual_ratio
            + float(beta_snn) * snn_avg
            - float(beta_perturb) * perturb
            - float(beta_uncertainty) * unc
        )
        sig = 1.0 / (1.0 + np.exp(-logits))
        gate = float(gate_min) + (float(gate_max) - float(gate_min)) * sig
        gate = gate.astype(np.float32)
    sample_weight = np.clip(gate / max(float(gate_max), 1e-8), 0.0, 1.0).astype(np.float32)
    summary = {
        "gate_mode": gate_mode,
        "gate_min": float(gate_min),
        "gate_max": float(gate_max),
        "mean_node_gate": float(np.mean(gate)) if gate.size else 0.0,
        "min_node_gate": float(np.min(gate)) if gate.size else 0.0,
        "max_node_gate": float(np.max(gate)) if gate.size else 0.0,
        "fraction_gate_lt_0p01": float(np.mean(gate < 0.01)) if gate.size else 1.0,
        "fraction_gate_gt_90pct_max": float(np.mean(gate > 0.9 * float(gate_max))) if gate.size else 0.0,
        "uncertainty_enabled": bool(uncertainty is not None),
        "uncertainty_source": "disabled" if uncertainty is None else "unsupervised",
        "mean_perturb_proxy": float(np.mean(perturb)) if perturb.size else 0.0,
    }
    return gate, sample_weight, summary


def make_pseudo_batch_binary(
    data_np: np.ndarray,
    batch_indices: np.ndarray,
    batch_x: torch.Tensor,
    mix_mode: str,
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    mix_neighbors: int,
    rng: np.random.Generator,
    random_neighbors: np.ndarray | None = None,
    far_neighbors: np.ndarray | None = None,
    neighbor_estimator: str = "current",
    router_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Binary routing version of make_pseudo_batch.

    x' = anchor + r * (neighbor_mean - anchor)
       = (1-r)*anchor + r*neighbor_mean

    where r ∈ [0,1] is the BinaryRouter output (soft during training,
    hard {0,1} during inference).

    When r=0: x' = anchor (pure self-reconstruction)
    When r=1: x' = mixed (topology-aware neighbor blending)

    router_tensor: (batch_size,) torch tensor from BinaryRouter.
        If provided, routing decisions are computed in torch with gradient.
        sample_weight = router_tensor (so nodes that route to anchor
        contribute zero pseudo-loss).
    """
    if neighbor_estimator not in {"current", "uniform_sample", "full"}:
        raise ValueError(f"Unknown neighbor_estimator: {neighbor_estimator!r}")
    use_torch_router = router_tensor is not None
    if use_torch_router:
        r_t = router_tensor.to(dtype=batch_x.dtype, device=batch_x.device).reshape(-1)
        if r_t.shape[0] != batch_x.shape[0]:
            raise ValueError(
                f"router_tensor must be (batch_size,), got {tuple(r_t.shape)}"
            )

    if mix_mode == "none" or graph.indices.shape[1] == 0 or int(mix_neighbors) <= 0:
        zeros = torch.zeros(batch_x.shape[0], dtype=batch_x.dtype, device=batch_x.device)
        return batch_x.detach(), zeros, {"mean_router": 0.0, "mean_perturb_norm": 0.0}

    bsz = int(batch_indices.shape[0])
    k = int(graph.indices.shape[1])
    m = max(1, min(int(mix_neighbors), k))

    if mix_mode in {"random", "far", "fixed"}:
        neighbor_mean = np.empty((bsz, data_np.shape[1]), dtype=np.float32)
        for pos, cell in enumerate(batch_indices):
            if mix_mode == "random" and random_neighbors is not None:
                row = random_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "far" and far_neighbors is not None:
                row = far_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "mutual":
                mask = graph.mutual[cell]
                row = graph.indices[cell][mask] if np.any(mask) else graph.indices[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            else:
                row = graph.indices[cell]
                probs = edge_weights[cell] if mix_mode == "reliability" else graph.probs[cell]
            normalized = probs / np.clip(probs.sum(), 1e-12, None)
            neighbor_mean[pos] = np.sum(data_np[row] * normalized[:, None], axis=0).astype(np.float32)
    else:
        sampled = np.empty((bsz, m), dtype=np.int64)
        weights = np.empty((bsz, m), dtype=np.float32)
        for pos, cell in enumerate(batch_indices):
            if mix_mode == "random" and random_neighbors is not None:
                row = random_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "far" and far_neighbors is not None:
                row = far_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "mutual":
                mask = graph.mutual[cell]
                row = graph.indices[cell][mask] if np.any(mask) else graph.indices[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            else:
                row = graph.indices[cell]
                probs = edge_weights[cell] if mix_mode == "reliability" else graph.probs[cell]
            choices = rng.choice(row.shape[0], size=m, replace=True, p=probs / np.clip(probs.sum(), 1e-12, None))
            sampled[pos] = row[choices]
            picked = probs[choices].astype(np.float32, copy=False)
            if neighbor_estimator == "current":
                weights[pos] = picked / max(float(picked.sum()), 1e-12)
            else:
                weights[pos] = 1.0 / float(m)
        neighbor_expr = data_np[sampled]
        neighbor_mean = np.sum(neighbor_expr * weights[:, :, None], axis=1).astype(np.float32)

    anchor_np = data_np[batch_indices]

    if use_torch_router:
        anchor_t = torch.as_tensor(anchor_np, dtype=batch_x.dtype, device=batch_x.device)
        neighbor_mean_t = torch.as_tensor(neighbor_mean, dtype=batch_x.dtype, device=batch_x.device)
        # x' = anchor + r * (neighbor - anchor) = (1-r)*anchor + r*neighbor
        mixed_t = anchor_t + r_t.unsqueeze(1) * (neighbor_mean_t - anchor_t)
        r_used_np = r_t.detach().cpu().float().numpy()
    else:
        # Fallback: all nodes route to mixed (no routing decision)
        mixed_np = neighbor_mean.astype(np.float32)
        mixed_t = torch.as_tensor(mixed_np, dtype=batch_x.dtype, device=batch_x.device)
        r_used_np = np.ones(bsz, dtype=np.float32)

    perturb = np.linalg.norm(neighbor_mean - anchor_np, axis=1) / (
        np.linalg.norm(anchor_np, axis=1) + 1e-6
    )

    # sample_weight = routing probability; nodes that route to anchor contribute 0
    sample_weight = torch.as_tensor(
        np.clip(r_used_np, 0.0, 1.0),
        dtype=batch_x.dtype,
        device=batch_x.device,
    )

    info = {
        "mean_router": float(np.mean(r_used_np)),
        "mean_perturb_norm": float(np.mean(perturb)),
        "fraction_routed_to_anchor": float(np.mean(r_used_np <= 0.0)),
        "fraction_routed_to_mixed": float(np.mean(r_used_np > 0.0)),
    }
    return mixed_t, sample_weight, info


def make_pseudo_batch(
    data_np: np.ndarray,
    batch_indices: np.ndarray,
    batch_x: torch.Tensor,
    mix_mode: str,
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    node_gate: np.ndarray,
    mix_neighbors: int,
    rng: np.random.Generator,
    random_neighbors: np.ndarray | None = None,
    far_neighbors: np.ndarray | None = None,
    neighbor_estimator: str = "current",
    gate_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build pseudo-mixed batch.

    node_gate: full per-cell (n_cells,) gate array (used for `max` references).
    gate_tensor: optional (batch_size,) torch tensor (with grad) gate values.
        If provided, the mix `(1-g)*anchor + g*neighbor` is computed in torch so
        that gradients flow back through the gate tensor.  The numpy `node_gate`
        is still used for `mean` reference when mix_mode is 'random/far/fixed'.
    """
    if neighbor_estimator not in {"current", "uniform_sample", "full"}:
        raise ValueError(f"Unknown neighbor_estimator: {neighbor_estimator!r}")
    use_torch_gate = gate_tensor is not None
    if use_torch_gate:
        gate_t = gate_tensor.to(dtype=batch_x.dtype, device=batch_x.device).reshape(-1)
        if gate_t.shape[0] != batch_x.shape[0]:
            raise ValueError(
                f"gate_tensor must be (batch_size,), got {tuple(gate_t.shape)}"
            )
    if mix_mode == "none" or graph.indices.shape[1] == 0 or int(mix_neighbors) <= 0:
        zeros = torch.zeros(batch_x.shape[0], dtype=batch_x.dtype, device=batch_x.device)
        return batch_x.detach(), zeros, {"mean_node_gate": 0.0, "mean_perturb_norm": 0.0}

    bsz = int(batch_indices.shape[0])
    k = int(graph.indices.shape[1])
    m = max(1, min(int(mix_neighbors), k))
    gate = np.asarray(node_gate[batch_indices], dtype=np.float32)
    if mix_mode in {"random", "far", "fixed"}:
        gate = np.maximum(gate, float(np.mean(node_gate)) if node_gate.size else 0.1).astype(np.float32)
        neighbor_mean = np.empty((bsz, data_np.shape[1]), dtype=np.float32)
        for pos, cell in enumerate(batch_indices):
            if mix_mode == "random" and random_neighbors is not None:
                row = random_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "far" and far_neighbors is not None:
                row = far_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "mutual":
                mask = graph.mutual[cell]
                row = graph.indices[cell][mask] if np.any(mask) else graph.indices[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            else:
                row = graph.indices[cell]
                probs = edge_weights[cell] if mix_mode == "reliability" else graph.probs[cell]
            normalized = probs / np.clip(probs.sum(), 1e-12, None)
            neighbor_mean[pos] = np.sum(data_np[row] * normalized[:, None], axis=0).astype(np.float32)
    else:
        sampled = np.empty((bsz, m), dtype=np.int64)
        weights = np.empty((bsz, m), dtype=np.float32)
        for pos, cell in enumerate(batch_indices):
            if mix_mode == "random" and random_neighbors is not None:
                row = random_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "far" and far_neighbors is not None:
                row = far_neighbors[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            elif mix_mode == "mutual":
                mask = graph.mutual[cell]
                row = graph.indices[cell][mask] if np.any(mask) else graph.indices[cell]
                probs = np.full(row.shape[0], 1.0 / row.shape[0], dtype=np.float32)
            else:
                row = graph.indices[cell]
                probs = edge_weights[cell] if mix_mode == "reliability" else graph.probs[cell]
            choices = rng.choice(row.shape[0], size=m, replace=True, p=probs / np.clip(probs.sum(), 1e-12, None))
            sampled[pos] = row[choices]
            picked = probs[choices].astype(np.float32, copy=False)
            if neighbor_estimator == "current":
                weights[pos] = picked / max(float(picked.sum()), 1e-12)
            else:
                weights[pos] = 1.0 / float(m)

        neighbor_expr = data_np[sampled]
        neighbor_mean = np.sum(neighbor_expr * weights[:, :, None], axis=1).astype(np.float32)
    if mix_mode in {"random", "far", "fixed"} and not use_torch_gate:
        gate = np.maximum(gate, float(np.mean(node_gate)) if node_gate.size else 0.1).astype(np.float32)
    anchor_np = data_np[batch_indices]
    if use_torch_gate:
        anchor_t = torch.as_tensor(anchor_np, dtype=batch_x.dtype, device=batch_x.device)
        neighbor_mean_t = torch.as_tensor(neighbor_mean, dtype=batch_x.dtype, device=batch_x.device)
        if mix_mode in {"random", "far", "fixed"}:
            mean_full = float(np.mean(node_gate)) if node_gate.size else 0.1
            gate_t = torch.maximum(gate_t, torch.tensor(mean_full, dtype=gate_t.dtype, device=gate_t.device))
        mixed_t = (1.0 - gate_t).unsqueeze(1) * anchor_t + gate_t.unsqueeze(1) * neighbor_mean_t
        gate_used_np = gate_t.detach().cpu().float().numpy()
    else:
        mixed = (1.0 - gate[:, None]) * anchor_np + gate[:, None] * neighbor_mean
        mixed_t = torch.as_tensor(mixed, dtype=batch_x.dtype, device=batch_x.device)
        gate_used_np = gate
    anchor_for_perturb = anchor_np
    perturb = np.linalg.norm(neighbor_mean - anchor_for_perturb, axis=1) / (np.linalg.norm(anchor_for_perturb, axis=1) + 1e-6)
    x_prime = mixed_t
    sample_weight = torch.as_tensor(
        np.clip(gate_used_np / max(float(np.max(node_gate)) if node_gate.size else 1.0, 1e-8), 0, 1),
        dtype=batch_x.dtype,
        device=batch_x.device,
    )
    info = {
        "mean_node_gate": float(np.mean(gate_used_np)),
        "mean_perturb_norm": float(np.mean(perturb)),
        "fraction_zero_gate": float(np.mean(gate_used_np <= 0.0)),
    }
    return x_prime, sample_weight, info
```

## `methods/TopoGate/learnable_gate/model.py`

```python
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.NeighborMix_scMAE.model import AutoEncoder as _BaseAutoEncoder


class AutoEncoder(_BaseAutoEncoder):
    """scMAE AutoEncoder with gate-weighted per-sample loss and optional contrastive head."""

    def __init__(self, *args, contrast_projection_dim: int = 0, **kwargs):
        dropout_rate = float(kwargs.get("dropout", 0.0))
        super().__init__(*args, **kwargs)
        projection_dim = int(contrast_projection_dim or 0)
        if projection_dim > 0:
            self.contrast_projector = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_size, projection_dim),
            )
        else:
            self.contrast_projector = None

    def contrast_projection(self, latent: torch.Tensor) -> torch.Tensor:
        if self.contrast_projector is None:
            return latent
        return self.contrast_projector(latent)

    def loss_mask_weighted(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        mask_loss_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        self._check_expression_shape(x, "x")
        self._check_expression_shape(y, "y")
        self._check_expression_shape(mask, "mask")
        if x.shape != y.shape or x.shape != mask.shape:
            raise ValueError("x, y, and mask must have identical shapes.")

        mask = mask.to(dtype=x.dtype, device=x.device)
        y = y.to(dtype=x.dtype, device=x.device)
        latent, mask_logits, reconstruction = self.forward_mask(x)
        raw_mse = F.mse_loss(reconstruction, y, reduction="none")
        weights = mask * self.masked_data_weight + (1.0 - mask) * (1.0 - self.masked_data_weight)
        weighted_mse = weights * raw_mse
        if self.normalize_reconstruction_by_weight:
            rec_per = weighted_mse.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
        else:
            rec_per = weighted_mse.mean(dim=1)
        rec_per = (1.0 - self.mask_loss_weight) * rec_per
        mask_per = F.binary_cross_entropy_with_logits(mask_logits, mask, reduction="none").mean(dim=1)
        mask_per = self.mask_loss_weight * mask_per
        total_per = rec_per + float(mask_loss_scale) * mask_per

        if sample_weight is None:
            loss = total_per.mean()
        else:
            w = sample_weight.to(dtype=x.dtype, device=x.device).view(-1)
            loss = (total_per * w).sum() / w.sum().clamp_min(1e-8)
        parts = {
            "reconstruction_loss": rec_per.mean().detach(),
            "mask_loss": mask_per.mean().detach(),
            "total_loss": loss.detach(),
            "mask_positive_rate": mask.mean().detach(),
            "per_sample_loss": total_per.detach(),
        }
        return latent, loss, parts
```

## `methods/TopoGate/learnable_gate/neighbor_graph.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


@dataclass
class NeighborGraph:
    indices: np.ndarray
    probs: np.ndarray
    similarity: np.ndarray
    distance: np.ndarray
    embedding: np.ndarray
    mutual: np.ndarray
    snn: np.ndarray
    profile: dict


def build_pca_knn_graph(
    data_np: np.ndarray,
    k: int,
    pca_dim: int,
    tau: float,
    seed: int,
    labels: np.ndarray | None = None,
    stress_bad_edge_ratio: float = 0.0,
    n_top_features: int = 0,
) -> NeighborGraph:
    stress_ratio = float(stress_bad_edge_ratio)
    if not 0.0 <= stress_ratio <= 1.0:
        raise ValueError("stress_bad_edge_ratio must be between 0 and 1")
    data = np.asarray(data_np, dtype=np.float32)
    n_cells, n_genes = data.shape

    if k <= 0 or n_cells <= 1:
        if stress_ratio > 0.0:
            raise ValueError("Cross-label edge stress requires at least one graph edge")
        empty_i = np.zeros((n_cells, 0), dtype=np.int64)
        empty_f = np.zeros((n_cells, 0), dtype=np.float32)
        return NeighborGraph(
            empty_i,
            empty_f,
            empty_f,
            empty_f,
            empty_f,
            empty_f.astype(bool),
            empty_f,
            {
                "neighbor_k": 0,
                "stress_bad_edge_ratio": stress_ratio,
                "stress_bad_edge_ratio_realized": 0.0,
                "label_leakage_diagnostic": False,
            },
        )
    dim = max(1, min(int(pca_dim), n_genes, n_cells - 1))
    emb = PCA(n_components=dim, random_state=seed).fit_transform(data) if dim < min(data.shape) else data
    emb = normalize(np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0), axis=1).astype(np.float32)
    k_eff = min(int(k), n_cells - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
    nn.fit(emb)
    distances, indices = nn.kneighbors(emb)
    indices = indices[:, 1 : k_eff + 1].astype(np.int64, copy=False)
    distances = distances[:, 1 : k_eff + 1].astype(np.float32, copy=False)
    similarity = (1.0 - distances).astype(np.float32)
    scaled = similarity / max(float(tau), 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled).astype(np.float32)
    probs = exp_scaled / np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-12, None)

    neighbor_sets = [set(row.tolist()) for row in indices]
    mutual = np.zeros_like(indices, dtype=bool)
    snn = np.zeros_like(similarity, dtype=np.float32)
    for i in range(n_cells):
        set_i = neighbor_sets[i]
        for pos, j in enumerate(indices[i]):
            mutual[i, pos] = i in neighbor_sets[j]
            union = set_i.union(neighbor_sets[j])
            snn[i, pos] = len(set_i.intersection(neighbor_sets[j])) / float(max(1, len(union)))

    profile = {
        "neighbor_k": int(k_eff),
        "tau": float(tau),
        "knn_pca_dim": int(dim),
        "hvf_n_top_features": int(n_top_features),  # 0 = HVF was applied before this call
        "mean_neighbor_similarity": float(np.mean(similarity)),
        "mean_mutual_ratio": float(np.mean(mutual)),
        "mean_snn": float(np.mean(snn)),
        "mean_max_neighbor_prob": float(np.mean(np.max(probs, axis=1))),
        "stress_bad_edge_ratio": stress_ratio,
        "stress_bad_edge_ratio_realized": 0.0,
        "label_leakage_diagnostic": False,
    }
    graph = NeighborGraph(indices, probs.astype(np.float32), similarity, distances, emb, mutual, snn, profile)
    if stress_ratio == 0.0:
        return graph
    if labels is None:
        raise ValueError("labels are required only when stress_bad_edge_ratio is greater than zero")
    return inject_cross_label_edges(graph, labels, stress_ratio, tau=tau, seed=seed)


def _recompute_graph_edges(indices: np.ndarray, embedding: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_cells, _ = indices.shape
    similarity = np.einsum("ij,ikj->ik", embedding, embedding[indices]).astype(np.float32)
    distance = (1.0 - similarity).astype(np.float32)
    scaled = similarity / max(float(tau), 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled).astype(np.float32)
    probs = exp_scaled / np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-12, None)

    neighbor_sets = [set(row.tolist()) for row in indices]
    mutual = np.zeros_like(indices, dtype=bool)
    snn = np.zeros_like(similarity, dtype=np.float32)
    for i in range(n_cells):
        set_i = neighbor_sets[i]
        for pos, j in enumerate(indices[i]):
            mutual[i, pos] = i in neighbor_sets[j]
            union = set_i.union(neighbor_sets[j])
            snn[i, pos] = len(set_i.intersection(neighbor_sets[j])) / float(max(1, len(union)))
    return probs.astype(np.float32), similarity, distance, mutual, snn


def inject_cross_label_edges(
    graph: NeighborGraph,
    labels: np.ndarray,
    ratio: float,
    tau: float,
    seed: int,
) -> NeighborGraph:
    """Replace a fixed share of each row with cross-label diagnostic edges.

    Labels are consulted only in this explicitly non-default stress path.  A
    separate RNG keeps the clean training RNG stream unchanged.
    """

    stress_ratio = float(ratio)
    if not 0.0 <= stress_ratio <= 1.0:
        raise ValueError("stress_bad_edge_ratio must be between 0 and 1")
    if stress_ratio == 0.0:
        return graph
    n_cells, k = graph.indices.shape
    if k == 0:
        raise ValueError("Cross-label edge stress requires at least one graph edge")
    label_values = np.asarray(labels).reshape(-1)
    if label_values.shape != (n_cells,):
        raise ValueError(f"labels must have shape ({n_cells},), got {label_values.shape}")

    rng = np.random.default_rng(int(seed) + 104729)
    stressed_indices = graph.indices.copy()
    all_cells = np.arange(n_cells, dtype=np.int64)
    total_edges = n_cells * k
    total_replacements = int(np.rint(stress_ratio * total_edges))
    selected_flat = rng.choice(total_edges, size=total_replacements, replace=False)
    replacement_mask = np.zeros(total_edges, dtype=bool)
    replacement_mask[selected_flat] = True
    replacement_mask = replacement_mask.reshape(n_cells, k)
    for cell in range(n_cells):
        positions = np.flatnonzero(replacement_mask[cell])
        replacements_per_row = int(positions.size)
        if replacements_per_row == 0:
            continue
        cross_label = all_cells[label_values != label_values[cell]]
        if cross_label.size == 0:
            raise ValueError(f"Cell {cell} has no cross-label candidate for diagnostic edge stress")
        keep_positions = np.ones(k, dtype=bool)
        keep_positions[positions] = False
        preserved = stressed_indices[cell, keep_positions]
        # Prefer genuinely new cross-label endpoints so the requested stress
        # fraction is also the realized changed-edge fraction.
        candidates = np.setdiff1d(cross_label, graph.indices[cell], assume_unique=False)
        if candidates.size == 0:
            candidates = np.setdiff1d(cross_label, preserved, assume_unique=False)
        if candidates.size == 0:
            candidates = cross_label
        stressed_indices[cell, positions] = rng.choice(
            candidates,
            size=replacements_per_row,
            replace=candidates.size < replacements_per_row,
        )

    probs, similarity, distance, mutual, snn = _recompute_graph_edges(stressed_indices, graph.embedding, tau)
    profile = dict(graph.profile)
    per_row = replacement_mask.sum(axis=1)
    profile.update(
        {
            "stress_bad_edge_ratio": stress_ratio,
            "stress_bad_edge_ratio_realized": float(total_replacements / total_edges),
            "stress_bad_edges_per_row_mean": float(per_row.mean()),
            "stress_bad_edges_per_row_min": int(per_row.min()),
            "stress_bad_edges_per_row_max": int(per_row.max()),
            "stress_cross_label_edge_fraction": float(
                np.mean(label_values[stressed_indices] != label_values[:, None])
            ),
            "label_leakage_diagnostic": True,
            "label_usage": "cross-label edge stress diagnostic only",
            "mean_neighbor_similarity": float(np.mean(similarity)),
            "mean_mutual_ratio": float(np.mean(mutual)),
            "mean_snn": float(np.mean(snn)),
            "mean_max_neighbor_prob": float(np.mean(np.max(probs, axis=1))),
        }
    )
    return NeighborGraph(
        stressed_indices,
        probs,
        similarity,
        distance,
        graph.embedding,
        mutual,
        snn,
        profile,
    )


def compute_edge_reliability(
    graph: NeighborGraph,
    mode: str,
    gamma_sim: float,
    gamma_mutual: float,
    gamma_snn: float,
    gamma_distance: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if graph.indices.shape[1] == 0 or mode == "none":
        weights = graph.probs.copy()
        rel = np.ones_like(weights, dtype=np.float32)
        return rel, weights, summarize_edge_weights(weights)

    rel = np.ones_like(graph.similarity, dtype=np.float32)
    if mode in {"sim", "sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
        rel *= np.exp(float(gamma_sim) * graph.similarity).astype(np.float32)
    if mode in {"sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
        rel *= (1.0 + float(gamma_mutual) * graph.mutual.astype(np.float32))
    if mode in {"sim_mutual_snn", "sim_mutual_snn_distance"}:
        rel *= (1.0 + float(gamma_snn) * graph.snn)
    if mode == "sim_mutual_snn_distance":
        rel *= np.exp(-float(gamma_distance) * graph.distance).astype(np.float32)
    rel = np.clip(rel, 1e-6, 1e6).astype(np.float32)
    weights = graph.probs * rel
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    return rel, weights.astype(np.float32), summarize_edge_weights(weights)


def summarize_edge_weights(weights: np.ndarray) -> dict:
    if weights.size == 0:
        return {
            "edge_weight_entropy": 0.0,
            "effective_neighbor_count": 0.0,
            "max_edge_weight_mean": 0.0,
            "max_edge_weight_p95": 0.0,
            "fraction_effective_neighbors_lt_2": 1.0,
        }
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, None)), axis=1)
    effective = np.exp(entropy)
    max_w = np.max(weights, axis=1)
    return {
        "edge_weight_entropy": float(np.mean(entropy)),
        "effective_neighbor_count": float(np.mean(effective)),
        "max_edge_weight_mean": float(np.mean(max_w)),
        "max_edge_weight_p95": float(np.percentile(max_w, 95)),
        "fraction_effective_neighbors_lt_2": float(np.mean(effective < 2.0)),
    }


def build_random_neighbors(n_cells: int, k: int, rng: np.random.Generator, exclude: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros((n_cells, k), dtype=np.int64)
    all_idx = np.arange(n_cells)
    for i in range(n_cells):
        banned = {i}
        if exclude is not None:
            banned.update(exclude[i].tolist())
        candidates = np.setdiff1d(all_idx, np.fromiter(banned, dtype=np.int64), assume_unique=False)
        if candidates.size == 0:
            candidates = all_idx[all_idx != i]
        out[i] = rng.choice(candidates, size=k, replace=candidates.size < k)
    return out


def build_far_neighbors(embedding: np.ndarray, k: int, rng: np.random.Generator, candidate_pool: int = 96) -> np.ndarray:
    n_cells = int(embedding.shape[0])
    out = np.zeros((n_cells, k), dtype=np.int64)
    all_idx = np.arange(n_cells)
    for i in range(n_cells):
        candidates = rng.choice(all_idx[all_idx != i], size=min(candidate_pool, n_cells - 1), replace=False)
        sim = embedding[candidates] @ embedding[i]
        far = candidates[np.argsort(sim)[:k]]
        if far.size < k:
            far = rng.choice(candidates, size=k, replace=True)
        out[i] = far[:k]
    return out
```

## `methods/TopoGate/learnable_gate/run_npz.py`

```python
#!/usr/bin/env python
"""
TopoGate runner for generic tabular / npz datasets.
=================================================

Accepts .npz files (X, y) or the compressed binary format used by ToPoGate's
dataset/reader.py.  Produces the same output layout as TopoGate/run.py so that
all downstream paper-evaluation code works unchanged.

Outputs (same layout as TopoGate/run.py):
    embedding_final.npy  — learned embedding (n_samples, hidden_size)
    labels.npy           — integer-encoded ground-truth labels
    metrics.json         — ACC/NMI/ARI/F1/... after Hungarian-aligned KMeans
    summary.json         — runtime + method metadata
    <plus all TopoGate training diagnostics>

Usage:
    python run_npz.py --data_path <path/to/data.npz> --save_dir <out_dir> --seed 42 --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Block BLAS / MKL / OpenMP thread leakage before any heavy import
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = next(
    p for p in [CURRENT_DIR, *CURRENT_DIR.parents]
    if (p / "methods" / "DeepLearning" / "scMAE_family.py").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.learnable_gate.model import AutoEncoder
from methods.TopoGate.learnable_gate.neighbor_graph import (
    build_pca_knn_graph,
    build_random_neighbors,
    build_far_neighbors,
    compute_edge_reliability,
)
from methods.TopoGate.learnable_gate.mixing import compute_node_gate, make_pseudo_batch, make_pseudo_batch_binary
from methods.TopoGate.learnable_gate.learnable_gate import LearnableGate, build_gate_stats_tensor
from methods.TopoGate.learnable_gate.learnable_edge_reliability import (
    LearnableEdgeReliability,
    edge_weights_to_numpy,
)
from methods.TopoGate.learnable_gate.binary_router import BinaryRouter
from methods.TopoGate.learnable_gate.uncertainty import compute_mc_dropout_uncertainty
from methods.shared_utils import ensure_dir


# ──────────────────────────────────────────────
#  Adaptive PCA helper
# ──────────────────────────────────────────────

def select_adaptive_pca_dim(X: np.ndarray, max_dim: int = 200,
                             var_threshold: float = 0.95,
                             min_dim: int = 10,
                             seed: int = 0) -> int:
    """Auto-select PCA dim to retain at least var_threshold (default 95%) variance.

    Args:
        X: (n, d) raw feature matrix (should be scaled before calling).
        max_dim: Upper bound on the selected dim.
        var_threshold: Fraction of cumulative variance to retain.
        min_dim: Lower bound on the selected dim.
        seed: Random seed for PCA.
    Returns:
        int: selected PCA dimension.
    """
    actual_max = min(max_dim, X.shape[0] - 1, X.shape[1])
    if actual_max < min_dim:
        return actual_max
    pca = PCA(n_components=actual_max, random_state=seed)
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_needed = int(np.searchsorted(cumvar, var_threshold)) + 1
    n_needed = max(min_dim, min(n_needed, max_dim, X.shape[1], X.shape[0] - 1))
    return n_needed


# ──────────────────────────────────────────────
#  Data loading helpers
# ──────────────────────────────────────────────

def load_npz(path: str):
    """Load X, y from .npz file."""
    data = np.load(path)
    X = data.get("X", data.get("x", data.get("data")))
    y = data.get("y", data.get("labels", data.get("label", None)))
    if X is None:
        raise ValueError(f"npz at {path!r} must contain 'X'/'x'/'data' key.")
    X = np.asarray(X, dtype=np.float64)
    if y is not None:
        y = np.asarray(y).ravel()
    return X, y


def load_compressed(path: str):
    """Load from ToPoGate's compressed binary reader path."""
    import zlib
    with open(os.path.join(path, "data.bin"), "rb") as f:
        data = np.array(json.loads(zlib.decompress(f.read()).decode("utf8")))
    label_path = os.path.join(path, "label.bin")
    if os.path.exists(label_path):
        with open(label_path, "rb") as f:
            labels = np.array(json.loads(zlib.decompress(f.read()).decode("utf8")))
    else:
        labels = None
    return data.astype(np.float64), labels


def load_data(path: str):
    """Auto-detect format and load (X, y)."""
    path = Path(path)
    if path.suffix == ".npz":
        return load_npz(str(path))
    elif path.is_dir():
        return load_compressed(str(path))
    else:
        raise ValueError(f"Unsupported data path: {path!r}")


class TensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None):
        self.X = torch.as_tensor(X.astype(np.float32))
        self.y = None if y is None else torch.as_tensor(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        y_val = self.y[idx] if self.y is not None else torch.tensor(0, dtype=torch.int64)
        return int(idx), self.X[idx], y_val


# ──────────────────────────────────────────────
#  Argument parsing
# ──────────────────────────────────────────────

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got {v!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="TopoGate (npz / tabular)")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--method_name", default="TopoGate")
    parser.add_argument("--variant_name", default="topogate_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Number of clusters (auto-detected from y if not given)")
    parser.add_argument("--gpu", type=int, default=1)

    # Preprocessing
    parser.add_argument("--n_top_features", type=int, default=0,
        help="HVF: number of top high-variance features to keep before PCA. "
             "0=disabled (use all features). Recommended: 1000-2000 for d>5000.")
    parser.add_argument("--knn_pca_mode", type=str, default="fixed",
        choices=["fixed", "adaptive"],
        help="fixed: use knn_pca_dim directly. "
             "adaptive: auto-select dim to retain at least 95 percent variance "
             "(capped at knn_pca_dim).")
    parser.add_argument("--input_mode", default="raw", choices=["raw", "log1p"])
    parser.add_argument("--scale_input", type=str2bool, default=True)

    # Model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--masked_data_weight", type=float, default=0.75)
    parser.add_argument("--mask_loss_weight", type=float, default=0.7)

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    # TopoGate topology
    parser.add_argument("--neighbor_k", type=int, default=10)
    parser.add_argument("--mix_neighbors", type=int, default=4)
    parser.add_argument("--mix_mode", default="reliability",
                        choices=["none", "fixed", "mutual", "reliability", "random", "far"])
    parser.add_argument("--risk_adaptive_mix", type=str2bool, default=False,
                        help="V12: scale each node's learned topology gate by an unsupervised local-risk proxy.")
    parser.add_argument("--risk_adaptive_temperature", type=float, default=1.0,
                        help="V12 temperature for the local-risk attenuation; must be positive.")
    parser.add_argument("--neighbor_estimator", default="current",
                        choices=["current", "uniform_sample", "full"])
    parser.add_argument("--gate_mode", default="topology",
                        choices=["none", "constant", "topology", "learned", "binary"])
    parser.add_argument("--gate_max", type=float, default=0.15)
    parser.add_argument("--gate_min", type=float, default=0.0)
    parser.add_argument("--pseudo_weight", type=float, default=0.3)
    parser.add_argument("--edge_reliability_mode", default="sim_mutual_snn_distance")
    parser.add_argument("--knn_pca_dim", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--gamma_sim", type=float, default=1.0)
    parser.add_argument("--gamma_mutual", type=float, default=1.0)
    parser.add_argument("--gamma_snn", type=float, default=1.0)
    parser.add_argument("--gamma_distance", type=float, default=1.0)
    parser.add_argument("--beta_mutual", type=float, default=1.0)
    parser.add_argument("--beta_snn", type=float, default=1.0)
    parser.add_argument("--beta_perturb", type=float, default=2.0)
    parser.add_argument("--beta_uncertainty", type=float, default=1.0)

    # learnable_gate learnable gate (only used when gate_mode == 'learned')
    parser.add_argument("--init_beta_mutual", type=float, default=0.0)
    parser.add_argument("--init_beta_snn", type=float, default=0.0)
    parser.add_argument("--init_beta_perturb", type=float, default=0.0)
    parser.add_argument("--init_beta_uncertainty", type=float, default=0.0)
    parser.add_argument("--learned_gate_init_mode", type=str, default="zero",
                        choices=["zero", "v1_default", "nomix"])
    parser.add_argument("--learnable_gate_max", type=str2bool, default=False,
                        help="v3: promote gate_max to a learnable parameter "
                             "(initialised at --gate_max, range [0.05, 1.0]).")
    parser.add_argument("--gate_max_min", type=float, default=0.05,
                        help="Floor for learnable gate_max.")
    parser.add_argument("--gate_max_max", type=float, default=1.0,
                        help="Ceiling for learnable gate_max.")
    parser.add_argument("--gate_lr_multiplier", type=float, default=10.0,
                        help="v3: multiplier for the gate parameter group's lr. "
                             "Defaults to 10x. This decouples the gate learning rate "
                             "from the small pseudo-loss channel (pseudo_weight times "
                             "gate_max), so the gate can compete with the MAE loss.")
    # Direction B: BinaryRouter
    parser.add_argument("--router_init_temp", type=float, default=5.0,
                        help="Direction B: initial Gumbel-Softmax temperature. "
                             "Higher = softer routing. Default 5.0.")
    parser.add_argument("--router_temp_min", type=float, default=0.01,
                        help="Direction B: minimum temperature after cool-down. Default 0.01.")
    parser.add_argument("--router_warmup_epochs", type=int, default=20,
                        help="Direction B: warmup epochs before temperature cools. Default 20.")
    parser.add_argument("--router_ramp_epochs", type=int, default=10,
                        help="Direction B: ramp epochs for temperature cool-down. Default 10.")
    # v3: LearnableEdgeReliability (promote the 4 gamma to nn.Parameter)
    parser.add_argument("--learnable_gamma", type=str2bool, default=False,
                        help="v3: promote the 4 gamma coefficients to learnable nn.Parameter. "
                             "Only effective when edge_reliability_mode != 'none'.")
    parser.add_argument("--gamma_reg_weight", type=float, default=1e-4,
                        help="v3: L2 regularisation weight on the 4 gamma.")
    # v3: EnhancedTopologyFeatures (extend stats from 4 → 6 with degree/clustering)
    parser.add_argument("--enhanced_stats", type=int, default=4, choices=[4, 6],
                        help="v3: number of per-node stats passed to the gate. 6 "
                             "adds degree_norm and clustering_coeff on top of "
                             "mutual/snn/perturb/uncertainty.")
    # v3: AdaptiveMaskRatio (promote mask_ratio to a learnable parameter)
    parser.add_argument("--learnable_mask_ratio", type=str2bool, default=False,
                        help="v3: make mask_ratio a learnable parameter initialised "
                             "at --mask_ratio and clamped to [0.1, 0.6].")
    parser.add_argument("--mask_ratio_min", type=float, default=0.1,
                        help="v3: floor for the learnable mask_ratio.")
    parser.add_argument("--mask_ratio_max", type=float, default=0.6,
                        help="v3: ceiling for the learnable mask_ratio.")
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--ramp_epochs", type=int, default=10)
    parser.add_argument(
        "--use_beta_scale_schedule",
        type=str2bool,
        default=False,
        help="Legacy experiment switch. False preserves the V9 gate equation; "
             "True enables the later nomix beta-scale curriculum.",
    )
    # Freeze the MAE encoder after this epoch number so that LearnableGate β
    # can settle without chasing a moving target.  Set to a large value
    # (e.g. 10**9) to disable freezing entirely.  Only effective when
    # gate_mode == 'learned'.
    parser.add_argument("--freeze_mae_after_epoch", type=int, default=10**9,
                        help="Freeze the MAE encoder+decoder after this epoch "
                             "(β keeps updating).  Default 1e9 = disabled.")

    # I/O
    parser.add_argument("--lightweight_outputs", action="store_true")
    parser.add_argument(
        "--legacy_labels_output",
        type=str2bool,
        default=True,
        help="Keep the historical overloaded labels.npy output. New protocol "
             "runners should set false and use predictions.npy/labels_true.npy.",
    )
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


# ──────────────────────────────────────────────
#  Device
# ──────────────────────────────────────────────

def get_device(gpu: int, no_cuda: bool) -> torch.device:
    if no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    forbidden = {0, 7}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        ids = [i.strip() for i in visible.split(",") if i.strip()]
        if any(i in forbidden for i in ids):
            raise ValueError("CUDA_VISIBLE_DEVICES includes forbidden GPU 0 or 7.")
        if len(ids) == 1:
            return torch.device("cuda:0")
        if str(gpu) in ids:
            return torch.device(f"cuda:{ids.index(str(gpu))}")
        if 0 <= gpu < len(ids):
            return torch.device(f"cuda:{gpu}")
        raise ValueError(f"GPU {gpu} not in CUDA_VISIBLE_DEVICES={visible!r}.")
    if gpu in forbidden:
        raise ValueError("Physical GPU 0 and GPU 7 are forbidden. Use 1-6.")
    return torch.device(f"cuda:{gpu}")


# ──────────────────────────────────────────────
#  Masked noise (same as scMAE_family.py)
# ──────────────────────────────────────────────

def apply_mask_noise(x: torch.Tensor, mask_ratio) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply row-shuffle noise to x with given mask_ratio.

    mask_ratio can be a float (constant for the call) or a torch.Tensor scalar
    (e.g. a learnable parameter).  The function uses .item() to extract the
    Python float so torch.bernoulli receives a Python scalar.
    """
    if isinstance(mask_ratio, torch.Tensor):
        ratio_val = float(mask_ratio.detach().cpu())
    else:
        ratio_val = float(mask_ratio)
    should_swap = torch.bernoulli(ratio_val * torch.ones_like(x))
    if x.shape[0] <= 1:
        replacement = x
    else:
        replacement = x[torch.randperm(x.shape[0], device=x.device)]
    corrupted = torch.where(should_swap.bool(), replacement, x)
    mask = (corrupted != x).float()
    return corrupted, mask


# ──────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────

def linear_assignment(cost_matrix):
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.column_stack([row_ind, col_ind])


def align_labels(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(np.unique(y_pred)) == 1:
        return y_pred
    cm = confusion_matrix(y_true, y_pred)
    aligned = np.zeros_like(y_pred)
    for r, c in linear_assignment(-cm):
        aligned[y_pred == c] = r
    return aligned


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_aligned = align_labels(y_true, y_pred)
    return {
        "acc":           float(accuracy_score(y_true, y_aligned)),
        "nmi":           float(normalized_mutual_info_score(y_true, y_pred)),
        "ari":           float(adjusted_rand_score(y_true, y_pred)),
        "f1_macro":      float(f1_score(y_true, y_aligned, average="macro", zero_division=0)),
        "fmi":           float(fowlkes_mallows_score(y_true, y_pred)),
        "v_measure":     float(v_measure_score(y_true, y_pred)),
        "homogeneity":   float(homogeneity_score(y_true, y_pred)),
        "completeness":  float(completeness_score(y_true, y_pred)),
    }


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, default=conv)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # Seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    X_raw, y_raw = load_data(args.data_path)
    dataset_name = args.dataset_name or Path(args.data_path).stem

    # Preprocess
    if args.input_mode == "log1p" and np.nanmax(X_raw) <= 30:
        X_raw = np.log1p(X_raw)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Input scaling is an explicit protocol switch.  The historical V9
    # configs default to True; paper-matched AHDPC comparisons can disable it
    # for rows whose published preprocessing is raw.
    if args.scale_input:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_np = scaler.fit_transform(X_raw).astype(np.float32)
    else:
        X_np = X_raw.astype(np.float32, copy=False)

    # Labels
    if y_raw is None:
        # Fully unsupervised path: TopoGate training is self-supervised and does not
        # need labels. n_clusters must be provided via --n_clusters.
        y_np = None
        if args.n_clusters is None:
            raise ValueError(
                f"{args.data_path!r} has no labels and --n_clusters was not provided. "
                "TopoGate is unsupervised; either provide labels in the .npz or pass "
                "--n_clusters explicitly."
            )
        n_clusters = args.n_clusters
    else:
        le = LabelEncoder()
        y_np = le.fit_transform(np.asarray(y_raw).ravel()).astype(np.int64)
        n_clusters = args.n_clusters if args.n_clusters is not None else len(np.unique(y_np))

    save_dir = Path(ensure_dir(args.save_dir))
    save_json(vars(args), str(save_dir / "args.json"))

    # ── HVF feature selection ────────────────────────────────────────────────
    # Remove low-variance / noisy features before PCA to improve kNN quality.
    # HVF must happen BEFORE adaptive PCA dimension selection.
    if args.n_top_features > 0 and args.n_top_features < X_np.shape[1]:
        original_d = X_np.shape[1]
        var = np.var(X_np, axis=0)
        hvf_idx = np.argsort(var)[-args.n_top_features:]
        X_np = X_np[:, hvf_idx]
        # Re-scale after feature selection (mean=0, std=1 per selected feature)
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_np = scaler.fit_transform(X_np).astype(np.float32)
        print(f"[HVF] Reduced from {original_d} to {X_np.shape[1]} dims "
              f"(top {args.n_top_features} high-variance features)")
    # ───────────────────────────────────────────────────────────────────────

    device = get_device(args.gpu, args.no_cuda)
    print(f"[{dataset_name}] device={device}  n={X_np.shape[0]}  d={X_np.shape[1]}  K={n_clusters}", flush=True)

    # ── Determine kNN PCA dim ────────────────────────────────────────────────
    knn_pca_dim = min(args.knn_pca_dim, X_np.shape[1])
    if args.knn_pca_mode == "adaptive":
        actual_dim = select_adaptive_pca_dim(X_np, max_dim=knn_pca_dim, seed=args.seed)
        print(f"[Adaptive PCA] Selected dim={actual_dim} (retains ≥95% variance, "
              f"upper_bound={knn_pca_dim})")
        knn_pca_dim = actual_dim
    else:
        print(f"[Fixed PCA] Using knn_pca_dim={knn_pca_dim}")

    # Build topology graph
    graph = build_pca_knn_graph(
        X_np, k=args.neighbor_k,
        pca_dim=knn_pca_dim,
        tau=args.tau, seed=args.seed,
        n_top_features=0,  # HVF already done above
    )
    # v3: LearnableEdgeReliability (optional).  When enabled, replaces the
    # argparse-fixed 4 gammas with nn.Parameter.
    learnable_edge_module = None
    if args.learnable_gamma and args.edge_reliability_mode != "none":
        learnable_edge_module = LearnableEdgeReliability(
            mode=args.edge_reliability_mode,
            init_gamma_sim=args.gamma_sim,
            init_gamma_mutual=args.gamma_mutual,
            init_gamma_snn=args.gamma_snn,
            init_gamma_distance=args.gamma_distance,
            reg_weight=args.gamma_reg_weight,
        ).to(device)
        edge_rel_t, edge_weights_t = learnable_edge_module(graph)
        edge_weights = edge_weights_to_numpy(edge_weights_t)
        from methods.TopoGate.learnable_gate.learnable_edge_reliability import summarize_edge_weights_torch
        edge_summary = summarize_edge_weights_torch(edge_weights_t)
    else:
        _, edge_weights, edge_summary = compute_edge_reliability(
            graph, mode=args.edge_reliability_mode,
            gamma_sim=args.gamma_sim, gamma_mutual=args.gamma_mutual,
            gamma_snn=args.gamma_snn, gamma_distance=args.gamma_distance,
        )
    node_gate, _, gate_summary = compute_node_gate(
        graph, edge_weights=edge_weights,
        gate_mode=args.gate_mode if args.gate_mode != "learned" else "none",
        gate_min=args.gate_min, gate_max=args.gate_max,
        beta_mutual=args.beta_mutual, beta_snn=args.beta_snn,
        beta_perturb=args.beta_perturb, beta_uncertainty=args.beta_uncertainty,
        uncertainty=None,
    )
    # Gate module placeholders — will be initialised after model + uncertainty are ready
    learned_gate_module = None
    learned_gate_stats = None
    learned_gate_static = None
    binary_router_module = None
    binary_router_stats = None
    rng = np.random.default_rng(args.seed + 3089)
    random_neighbors = build_random_neighbors(
        X_np.shape[0], max(1, min(args.mix_neighbors, X_np.shape[0] - 1)),
        rng, graph.indices,
    )
    far_neighbors = build_far_neighbors(
        graph.embedding,
        max(1, min(args.mix_neighbors, X_np.shape[0] - 1)),
        rng,
    )

    # V12: compute an unsupervised local-risk attenuation from the frozen graph.
    # The proxy does not inspect y: it combines disagreement in mutual-neighbor
    # support with the existing perturbation statistic.  High-risk nodes retain
    # the self/feature branch and receive less topology mixing; low-risk nodes
    # keep the original V9 gate.  This is deliberately configuration-gated so
    # V9 remains bit-for-bit reachable through risk_adaptive_mix=false.
    risk_scale_np = np.ones(X_np.shape[0], dtype=np.float32)
    risk_summary = {
        "enabled": bool(args.risk_adaptive_mix),
        "temperature": float(args.risk_adaptive_temperature),
        "mean_local_risk": 0.0,
        "mean_topology_trust": 1.0,
        "p10_topology_trust": 1.0,
        "p90_topology_trust": 1.0,
    }
    if args.risk_adaptive_temperature <= 0:
        raise ValueError("risk_adaptive_temperature must be positive")
    if args.risk_adaptive_mix:
        mutual_support = np.asarray(graph.mutual.mean(axis=1), dtype=np.float32)
        perturb_proxy = (1.0 - np.sum(graph.probs * graph.similarity, axis=1)).astype(np.float32)
        local_risk = 0.5 * (1.0 - np.clip(mutual_support, 0.0, 1.0)) + 0.5 * np.clip(perturb_proxy, 0.0, 1.0)
        risk_scale_np = np.exp(-local_risk / float(args.risk_adaptive_temperature)).astype(np.float32)
        risk_summary = {
            "enabled": True,
            "temperature": float(args.risk_adaptive_temperature),
            "mean_local_risk": float(np.mean(local_risk)),
            "mean_topology_trust": float(np.mean(risk_scale_np)),
            "p10_topology_trust": float(np.percentile(risk_scale_np, 10)),
            "p90_topology_trust": float(np.percentile(risk_scale_np, 90)),
        }

    # DataLoaders
    dataset = TensorDataset(X_np, y_np)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, generator=gen)
    eval_loader  = DataLoader(dataset, batch_size=max(args.batch_size * 4, 512), shuffle=False)

    # Model
    model = AutoEncoder(
        num_genes=X_np.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        masked_data_weight=args.masked_data_weight,
        mask_loss_weight=args.mask_loss_weight,
    ).to(device)

    # ── MC Dropout uncertainty ───────────────────────────────────────────────
    # Compute structural instability per node using the untrained encoder's latent
    # variance.  This replaces the previously hardcoded uncertainty=None, making
    # the 4th topology stat meaningful for the LearnableGate.
    uncertainty_np = None
    if args.gate_mode == "learned" and args.enhanced_stats >= 4:
        print(f"[Uncertainty] Computing MC dropout uncertainty (n_passes=5)...")
        model.eval()
        with torch.no_grad():
            all_unc = []
            for idx_batch, x_batch, _ in eval_loader:
                x_t = x_batch.to(device)
                unc = compute_mc_dropout_uncertainty(model, x_t, n_passes=5, device=device)
                all_unc.append(unc)
        uncertainty_np = np.concatenate(all_unc)  # (n,)
        u_mean = float(np.mean(uncertainty_np))
        u_std = float(np.std(uncertainty_np))
        print(f"[Uncertainty] done. mean={u_mean:.4f} std={u_std:.4f} "
              f"range=[{float(np.min(uncertainty_np)):.4f}, {float(np.max(uncertainty_np)):.4f}]")
        model.train()
    # ───────────────────────────────────────────────────────────────────────

    # ── LearnableGate initialisation (requires model + uncertainty) ──────────
    if args.gate_mode == "learned":
        if args.learned_gate_init_mode == "v1_default":
            init_m, init_s, init_p, init_u = (
                args.beta_mutual, args.beta_snn, args.beta_perturb, args.beta_uncertainty
            )
        elif args.learned_gate_init_mode == "nomix":
            # nomix_init: all betas start at -1.5 → gate ≈ 0.018 (≈ NoMix)
            # gate = gate_min + (gate_max - gate_min) * sigmoid(betas * stats)
            # with stats≈1: sigmoid(-1.5) ≈ 0.018 → gate ≈ 0.15 * 0.018 ≈ 0.0027
            # This is NOT exactly zero, but close to NoMix. Gradient is strong:
            # sigmoid'(-1.5) ≈ 0.018 (vs 3e-7 at -5.0 — 50,000x improvement).
            init_m = init_s = init_p = init_u = -1.5
        else:
            init_m = init_s = init_p = init_u = 0.0
        learned_gate_module = LearnableGate(
            gate_min=args.gate_min, gate_max=args.gate_max,
            init_beta_mutual=init_m, init_beta_snn=init_s,
            init_beta_perturb=init_p, init_beta_uncertainty=init_u,
            learnable_gate_max=bool(args.learnable_gate_max),
            gate_max_min=args.gate_max_min,
            gate_max_max=args.gate_max_max,
            enhanced_stats=int(args.enhanced_stats),
        ).to(device)
        learned_gate_stats = build_gate_stats_tensor(
            graph.indices, graph.mutual, graph.snn, graph.probs, graph.similarity,
            uncertainty=uncertainty_np, device=device,
            enhanced_stats=int(args.enhanced_stats),
        )
        learned_gate_static = node_gate.copy()
        gate_summary = {
            **gate_summary,
            "learned_gate_init_mode": args.learned_gate_init_mode,
            "warmup_epochs": int(args.warmup_epochs),
            "ramp_epochs": int(args.ramp_epochs),
            "init_beta_mutual": float(init_m),
            "init_beta_snn": float(init_s),
            "init_beta_perturb": float(init_p),
            "init_beta_uncertainty": float(init_u),
            "uncertainty_computed": uncertainty_np is not None,
            "use_beta_scale_schedule": bool(args.use_beta_scale_schedule),
        }
    # ── BinaryRouter initialisation (requires model + uncertainty) ──────────
    if args.gate_mode == "binary":
        binary_router_module = BinaryRouter(
            temperature_init=float(args.router_init_temp),
            temperature_min=float(args.router_temp_min),
            warmup_epochs=int(args.router_warmup_epochs),
            ramp_epochs=int(args.router_ramp_epochs),
            enhanced_stats=int(args.enhanced_stats),
            init_beta_mutual=float(args.init_beta_mutual),
            init_beta_snn=float(args.init_beta_snn),
            init_beta_perturb=float(args.init_beta_perturb),
            init_beta_uncertainty=float(args.init_beta_uncertainty),
        ).to(device)
        binary_router_stats = build_gate_stats_tensor(
            graph.indices, graph.mutual, graph.snn, graph.probs, graph.similarity,
            uncertainty=uncertainty_np, device=device,
            enhanced_stats=int(args.enhanced_stats),
        )
        gate_summary = {
            **gate_summary,
            "gate_mode": "binary",
            "router_temperature_init": float(args.router_init_temp),
            "router_temperature_min": float(args.router_temp_min),
            "router_warmup_epochs": int(args.router_warmup_epochs),
            "router_ramp_epochs": int(args.router_ramp_epochs),
        }
    # ───────────────────────────────────────────────────────────────────────

    # Two param-groups so we can freeze the MAE encoder/decoder without
    # touching the LearnableGate β params.  Group 0 = MAE, Group 1 = gates.
    mae_params = list(model.parameters())
    gate_params = list(learned_gate_module.parameters()) if learned_gate_module is not None else []
    binary_router_params = list(binary_router_module.parameters()) if binary_router_module is not None else []
    edge_params = list(learnable_edge_module.parameters()) if learnable_edge_module is not None else []
    gate_params = gate_params + binary_router_params  # both get the amplified lr
    # v3: AdaptiveMaskRatio — promote mask_ratio to a learnable scalar parameter
    # initialised at args.mask_ratio, clamped to [mask_ratio_min, mask_ratio_max].
    learnable_mask_ratio_module = None
    mask_ratio_param = None
    mask_params = []
    if args.learnable_mask_ratio:
        import torch.nn as nn
        learnable_mask_ratio_module = nn.Module()
        # initial value: logit((init - min) / (max - min))
        with torch.no_grad():
            span = max(args.mask_ratio_max - args.mask_ratio_min, 1e-6)
            p0 = max(min((args.mask_ratio - args.mask_ratio_min) / span, 1.0 - 1e-4), 1e-4)
            raw0 = float(np.log(p0 / (1.0 - p0)))
        learnable_mask_ratio_module.mask_ratio_raw = nn.Parameter(torch.tensor(raw0))
        mask_params = list(learnable_mask_ratio_module.parameters())
        def mask_ratio_value():
            span = max(args.mask_ratio_max - args.mask_ratio_min, 1e-6)
            return args.mask_ratio_min + span * torch.sigmoid(
                learnable_mask_ratio_module.mask_ratio_raw
            )
        mask_ratio_param = mask_ratio_value()
    optimizer = torch.optim.Adam([
        {"params": mae_params, "lr": args.lr},
        # v3: amplify gate group lr (default 10x).  The β/gate_max gradient
        # travels through pseudo_weight(0.3) * gate(<0.15) ≈ 4.5% of the loss
        # signal, so without amplification it cannot compete with MAE loss.
        {"params": gate_params, "lr": args.lr * float(args.gate_lr_multiplier)},
        # v3: LearnableEdgeReliability params share the MAE lr (they affect
        # the pseudo branch via edge_weights, which gets ~30% gradient).
        {"params": edge_params, "lr": args.lr},
        # v3: AdaptiveMaskRatio shares the MAE lr.  Its gradient flows through
        # both real_loss and pseudo_loss (every apply_mask_noise call).
        {"params": mask_params, "lr": args.lr},
    ])
    # Easier approach: just track which params belong to MAE so we can zero out
    # their gradients after freeze_mae_after_epoch.  Using requires_grad toggle
    # also works but PyTorch prints a redundant lr=0 warning.
    mae_param_ids = {id(p) for p in mae_params}

    pseudo_enabled = args.mix_mode != "none" and float(args.pseudo_weight) > 0

    # Beta history for post-hoc analysis.  Recorded every epoch.  Only
    # active when gate_mode == 'learned'.
    beta_history = []

    # Training loop
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if learned_gate_module is not None:
            t = max(0.0, min(1.0, (epoch - args.warmup_epochs) / max(1, args.ramp_epochs)))
            # Keep V9 reproducible by leaving beta_scale=1 unless the legacy
            # nomix-warmup experiment explicitly opts into the later schedule.
            # Note: beta_scale=0 also makes d(gate)/d(beta)=0; beta parameters do
            # not learn from the pseudo branch during that interval.
            if not args.use_beta_scale_schedule:
                beta_scale_val = 1.0
            elif epoch < args.warmup_epochs:
                beta_scale_val = 0.0
            elif epoch < args.warmup_epochs + args.ramp_epochs:
                beta_scale_val = (epoch - args.warmup_epochs) / max(1, args.ramp_epochs)
            else:
                beta_scale_val = 1.0
            learned_gate_module.beta_scale.fill_(beta_scale_val)
        else:
            t = 1.0
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        # ── MAE freezing ────────────────────────────────────────────────────
        # After freeze_mae_after_epoch, the encoder/decoder no longer update.
        # Implemented by zeroing out gradients of MAE params after loss.backward()
        # but BEFORE optimizer.step().  We track this with `_mae_frozen`.
        # BinaryRouter is also frozen at this point (like learned_gate_module).
        mae_frozen = (
            (learned_gate_module is not None or binary_router_module is not None)
            and args.freeze_mae_after_epoch >= 0
            and epoch > args.freeze_mae_after_epoch
        )
        for idx_t, x_cpu, _ in train_loader:
            idx_np = idx_t.numpy().astype(np.int64)
            x = x_cpu.to(device)

            # Real branch
            x_corrupt, real_mask = apply_mask_noise(x, mask_ratio_param if mask_ratio_param is not None else args.mask_ratio)
            _, real_loss, _ = model.loss_mask_weighted(x_corrupt, x, real_mask)
            loss = real_loss

            # Pseudo branch
            pseudo_loss = torch.zeros((), dtype=real_loss.dtype, device=device)
            if pseudo_enabled:
                router_tensor_pass = None
                if learned_gate_module is not None:
                    batch_stats = learned_gate_stats[idx_t]
                    gate_dyn = learned_gate_module(batch_stats)
                    if t < 1.0:
                        gate_static_t = torch.as_tensor(
                            learned_gate_static[idx_np],
                            dtype=gate_dyn.dtype, device=gate_dyn.device,
                        )
                        gate_eff = (1.0 - t) * gate_static_t + t * gate_dyn
                    else:
                        gate_eff = gate_dyn
                    if args.risk_adaptive_mix:
                        gate_eff = gate_eff * torch.as_tensor(
                            risk_scale_np[idx_np], dtype=gate_eff.dtype, device=gate_eff.device
                        )
                    gate_tensor_pass = gate_eff
                    x_prime, sample_weight, _ = make_pseudo_batch(
                        data_np=X_np, batch_indices=idx_np, batch_x=x,
                        mix_mode=args.mix_mode, graph=graph,
                        edge_weights=edge_weights, node_gate=node_gate,
                        mix_neighbors=args.mix_neighbors, rng=rng,
                        random_neighbors=random_neighbors,
                        far_neighbors=far_neighbors,
                        neighbor_estimator=args.neighbor_estimator,
                        gate_tensor=gate_tensor_pass,
                    )
                elif binary_router_module is not None:
                    # Direction B: BinaryRouter — hard routing between anchor and mixed
                    batch_stats = binary_router_stats[idx_t]
                    # Use Gumbel-Softmax during training; argmax during inference
                    router_tensor_pass = binary_router_module(batch_stats, epoch=epoch, hard=False)
                    x_prime, sample_weight, _ = make_pseudo_batch_binary(
                        data_np=X_np, batch_indices=idx_np, batch_x=x,
                        mix_mode=args.mix_mode, graph=graph,
                        edge_weights=edge_weights,
                        mix_neighbors=args.mix_neighbors, rng=rng,
                        random_neighbors=random_neighbors,
                        far_neighbors=far_neighbors,
                        neighbor_estimator=args.neighbor_estimator,
                        router_tensor=router_tensor_pass,
                    )
                else:
                    # static gate path (gate_mode != learned && != binary)
                    x_prime, sample_weight, _ = make_pseudo_batch(
                        data_np=X_np, batch_indices=idx_np, batch_x=x,
                        mix_mode=args.mix_mode, graph=graph,
                        edge_weights=edge_weights, node_gate=node_gate,
                        mix_neighbors=args.mix_neighbors, rng=rng,
                        random_neighbors=random_neighbors,
                        far_neighbors=far_neighbors,
                        neighbor_estimator=args.neighbor_estimator,
                        gate_tensor=None,
                    )
                xp_corrupt, pseudo_mask = apply_mask_noise(x_prime, mask_ratio_param if mask_ratio_param is not None else args.mask_ratio)
                _, pseudo_loss, _ = model.loss_mask_weighted(
                    xp_corrupt, x, pseudo_mask,
                    sample_weight=sample_weight,
                )
                loss = loss + float(args.pseudo_weight) * pseudo_loss
            # v3: LearnableEdgeReliability L2 regularisation.  Keeps the 4
            # gammas from drifting to extreme values that break neighbour sampling.
            if learnable_edge_module is not None:
                edge_reg = learnable_edge_module.regularization_loss()
                loss = loss + edge_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if mae_frozen:
                # MAE is frozen at this epoch.  Zero out the MAE params'
                # gradients so they don't update.  The gate params keep theirs.
                for pg in optimizer.param_groups:
                    if not pg["params"]:
                        continue
                    for p in pg["params"]:
                        if p.grad is not None and id(p) in mae_param_ids:
                            p.grad = None
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        # ── β history logging ─────────────────────────────────────────────
        if learned_gate_module is not None:
            beta_history.append({
                "epoch": epoch,
                "schedule_t": float(t),
                "mae_frozen": bool(mae_frozen),
                **learned_gate_module.beta_snapshot(),
            })
        elif binary_router_module is not None:
            beta_history.append({
                "epoch": epoch,
                "temperature": float(binary_router_module._temperature(epoch)),
                "mae_frozen": bool(mae_frozen),
                **binary_router_module.beta_snapshot(),
            })

        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            sched = f"  sched_t={t:.2f}" if learned_gate_module is not None else ""
            fz = "  [MAE-frozen]" if mae_frozen else ""
            print(f"  [{dataset_name}] epoch {epoch:03d}/{args.epochs}  loss={epoch_loss/max(1,n_batches):.4f}{sched}{fz}", flush=True)

    train_time = time.time() - t0

    # Extract embedding
    model.eval()
    embeddings, labels_out = [], []
    with torch.no_grad():
        for _, x, y in eval_loader:
            z = model.feature(x.to(device))
            embeddings.append(z.detach().cpu().numpy())
            labels_out.append(y.numpy())
    embedding = np.concatenate(embeddings, axis=0).astype(np.float32)
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

    # KMeans + metrics
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=args.seed)
    pred = km.fit_predict(embedding)
    if y_np is not None:
        metrics = compute_metrics(y_np, pred)
        metrics_str = f"ACC={metrics['acc']:.4f}  NMI={metrics['nmi']:.4f}  ARI={metrics['ari']:.4f}"
    else:
        metrics = {}
        metrics_str = "ACC=N/A  NMI=N/A  ARI=N/A"

    # Save
    np.save(save_dir / "embedding_final.npy", embedding)
    np.save(save_dir / "predictions.npy", pred.astype(np.int64))
    if y_np is not None:
        np.save(save_dir / "labels_true.npy", y_np.astype(np.int64))
    if args.legacy_labels_output:
        # Preserve the historical contract for old callers. The manifest-driven
        # protocol sets this flag false so labels and predictions cannot be
        # confused by downstream analysis.
        np.save(save_dir / "labels.npy", y_np.astype(np.int64) if y_np is not None else pred.astype(np.int64))
    if not args.lightweight_outputs:
        np.save(save_dir / "embeddings_base.npy", embedding)
    save_json(metrics, str(save_dir / "metrics.json"))
    # Metrics are saved to metrics.json separately.  Copy them into the
    # top-level json so callers get everything in one file.
    metrics_path = save_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            _metrics = json.load(f)
    else:
        _metrics = {}
    save_json({
        "dataset": dataset_name,
        "method": args.method_name,
        "variant": args.variant_name,
        "seed": int(args.seed),
        "n_samples": int(X_np.shape[0]),
        "n_features": int(X_np.shape[1]),
        "n_clusters": int(n_clusters),
        "labels_used_during_fit": False,
        "k_source": "explicit_n_clusters",
        "legacy_labels_output": bool(args.legacy_labels_output),
        "mix_mode": args.mix_mode,
        "pseudo_weight": float(args.pseudo_weight),
        "gate_mode": args.gate_mode,
        "train_seconds": float(train_time),
        "freeze_mae_after_epoch": int(args.freeze_mae_after_epoch),
        "metrics": _metrics,
        "edge_reliability_summary": edge_summary,
        "gate_summary": gate_summary,
        "learned_gate_final_beta": (
            learned_gate_module.beta_snapshot() if learned_gate_module is not None else None
        ),
        "binary_router_final_beta": (
            binary_router_module.beta_snapshot() if binary_router_module is not None else None
        ),
        "learned_edge_final_gamma": (
            learnable_edge_module.gamma_snapshot() if learnable_edge_module is not None else None
        ),
        "learned_mask_ratio": (
            float(mask_ratio_param.detach().cpu()) if mask_ratio_param is not None else None
        ),
        "learned_gate_beta_history": beta_history if learned_gate_module is not None else None,
        "binary_router_beta_history": beta_history if binary_router_module is not None else None,
        "risk_adaptive_mix": bool(args.risk_adaptive_mix),
        "risk_adaptive_temperature": float(args.risk_adaptive_temperature),
        "risk_summary": risk_summary,
    }, str(save_dir / "summary.json"))

    _f1 = metrics.get('f1_macro')
    f1_str = f"{_f1:.4f}" if _f1 is not None else "N/A"
    print(
        f"[{dataset_name}] done  "
        f"{metrics_str}  "
        f"F1={f1_str}  time={train_time:.1f}s",
        flush=True,
    )


def _load_variant_config(variant_name: str, config_dir: str | Path | None = None) -> dict:
    """Load <variant>.yaml from a configs directory and return flat dict of overrides.

    Args:
        variant_name: e.g. 'learnable_gate_sched' or 'static_gate_full'.  The `.yaml` suffix is appended.
        config_dir:   directory containing the YAML.  Defaults to
                      `methods/TopoGate/learnable_gate/configs/` (the learnable_gate default location).

    The YAML only contains a handful of keys (mix_mode / gate_mode / …).
    Anything not in the YAML falls back to argparse defaults defined in parse_args().
    """
    import yaml as _yaml
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent / "configs"
    cfg_path = Path(config_dir) / f"{variant_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"variant config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f) or {}
    return cfg


def run_topogate(X, n_clusters, y=None, gpu=4, variant="topogate_learnable_gate", save_dir=None,
                 seed=42, return_metrics=False, config_dir=None, **overrides):
    """Public entry point that wraps run_npz.main() with dataset-name + save_dir handling.

    Args:
        X: (n_samples, n_features) feature matrix (np.ndarray / torch.Tensor).
        n_clusters: int, number of clusters (K).
        gpu: physical GPU id (forbidden: 0, 7).
        variant: e.g. 'learnable_gate_sched' (main) or 'static_gate_full'.
                 The '.yaml' is appended; set config_dir to point at the right location.
        save_dir: optional output dir (defaults to a tmp dir under /tmp/topogate_<ts>).
        seed: random seed.
        return_metrics: if True, returns (labels, runtime, metrics_dict).
        config_dir: directory containing the variant YAML.  Defaults to
                    `methods/TopoGate/learnable_gate/configs/` (learnable_gate default).  Pass
                    `methods/TopoGate/static_gate/configs/` for v1 ablation variants.
        **overrides: any CLI arg of run_npz.parse_args() can be overridden by camelCase or snake_case key.

    Returns:
        (labels, runtime) by default; (labels, runtime, metrics_dict) if return_metrics=True.

    Strategy: drive main() via injected sys.argv. We build a complete argv from
    YAML + overrides, inject it, then call main() (which calls parse_args() again
    on the injected argv — argparse is idempotent). Algorithm code in main() is
    NEVER modified; this function is purely a wrapper.
    """
    import argparse
    import tempfile
    import shutil
    import io
    import contextlib

    yaml_cfg = _load_variant_config(variant, config_dir=config_dir)

    # These are wrapper-level metadata controls rather than model parameters.
    # Keeping them explicit prevents the protocol runner from inheriting the
    # legacy ``adhoc`` dataset name and overloaded labels.npy output.
    dataset_name_override = overrides.pop("dataset_name", None)

    # Merge: YAML < explicit overrides (overrides win)
    cli_args = dict(yaml_cfg)
    for k, v in overrides.items():
        snake = ''.join('_' + c.lower() if c.isupper() else c for c in k).lstrip('_')
        cli_args[snake] = v

    # Decide data_path / save_dir (use placeholders for the build phase)
    # Persist X (and optional y) to a temp .npz (main() needs ground-truth for metrics)
    tmp_npz = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp_npz.close()
    save_kwargs = {"X": np.asarray(X, dtype=np.float64)}
    if y is not None:
        save_kwargs["y"] = np.asarray(y).ravel()
    np.savez(tmp_npz.name, **save_kwargs)

    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix=f"topogate_{variant}_")
        _cleanup_tmp = True
    else:
        save_dir = str(save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _cleanup_tmp = False

    # Build the final argv (used by main() → parse_args())
    full_argv = ["run_npz.py"]
    action_flags = {"lightweight_outputs", "no_cuda"}
    for k, v in cli_args.items():
        if k in action_flags:
            if bool(v):
                full_argv.append(f"--{k}")
            continue
        full_argv += [f"--{k}", str(v)]
    full_argv += [
        "--data_path", tmp_npz.name,
        "--save_dir", save_dir,
        "--dataset_name", str(dataset_name_override or "adhoc"),
        "--variant_name", variant,
        "--method_name", "TopoGate",
        "--n_clusters", str(int(n_clusters)),
        "--seed", str(int(seed)),
        "--gpu", str(int(gpu)),
    ]

    # Inject argv, run main(), restore argv
    saved_argv = sys.argv
    sys.argv = full_argv
    buf = io.StringIO()
    try:
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            main()
        elapsed = time.time() - t0
    finally:
        sys.argv = saved_argv

    # Read metrics.json written by main()
    metrics_path = Path(save_dir) / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # main() saves embedding_final.npy but not pred_labels — re-run KMeans to recover them
    emb_path = Path(save_dir) / "embedding_final.npy"
    if emb_path.exists():
        embedding = np.load(emb_path)
        km = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(seed))
        pred_labels = km.fit_predict(embedding)
    else:
        pred_labels = np.array([], dtype=np.int64)

    # Cleanup tmp artefacts
    try:
        os.unlink(tmp_npz.name)
    except OSError:
        pass
    if _cleanup_tmp:
        shutil.rmtree(save_dir, ignore_errors=True)

    if return_metrics:
        return pred_labels, elapsed, metrics
    return pred_labels, elapsed


if __name__ == "__main__":
    raise SystemExit(main())
```

## `methods/TopoGate/learnable_gate/uncertainty.py`

```python
"""MC Dropout uncertainty estimation for TopoGate LearnableGate.

Uncertainty is one of the four topology stats (mutual, snn, perturb, uncertainty)
passed to the LearnableGate.  Previously this was always None, making the
4th stat a zero vector and beta_uncertainty a dead parameter.

This module computes a per-node "structural instability" score via Monte Carlo
Dropout: we run the encoder forward n_passes times with dropout enabled, then
measure the standard deviation of the latent representations across passes.

Nodes whose latent embedding is unstable under dropout are structurally ambiguous —
they sit near decision boundaries — so a higher uncertainty score justifies
mixing more aggressively with their neighbors.
"""
from __future__ import annotations

import torch
import numpy as np


def compute_mc_dropout_uncertainty(
    model: torch.nn.Module,
    X_tensor: torch.Tensor,
    n_passes: int = 5,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Monte Carlo Dropout uncertainty via encoder latent variance.

    Args:
        model: AutoEncoder model. Must have an `encoder` attribute.
        X_tensor: (n, d) input tensor.
        n_passes: Number of MC forward passes.
        device: Device to run on.

    Returns:
        uncertainty: (n,) per-sample uncertainty scores in [0, 1].
    """
    model.train()  # keep dropout active
    X = X_tensor.to(device)

    with torch.no_grad():
        preds = []
        for _ in range(n_passes):
            latent = model.encoder(X)  # (n, hidden_size)
            preds.append(latent.float())

    preds = torch.stack(preds, dim=0)  # (n_passes, n, hidden)
    # Variance across MC passes, then average over latent dims
    uncertainty = preds.std(dim=0).mean(dim=1)  # (n,)

    # Normalise to [0, 1] per batch (min-max)
    u_min = uncertainty.min()
    u_max = uncertainty.max()
    if u_max > u_min:
        uncertainty = (uncertainty - u_min) / (u_max - u_min)

    return uncertainty.cpu().numpy()
```

## `methods/TopoGate/learnable_gate/v5_components/__init__.py`

```python
"""v5 component subpackage — contains v5 extensions to TopoGate learnable_gate.

These are SEPARATE FROM v3 code. v5 runner should import from this package
only when --learnable_gate_v5 flag is set.
"""
```

## `methods/TopoGate/learnable_gate/v5_components/learnable_edge_reliability_v5.py`

```python
"""v5 Edge reliability: simplified single-gamma learnable version.

v3 LearnableEdgeReliability had 4 learnable gammas (sim/mutual/snn/distance).
Phase 2.1 diagnosis showed: across 5 datasets × 3 seeds, all 4 gammas
converged to EXACTLY the same value (std=0.000000). This happens because:

  d(loss)/d(γ_k) ∝ feature_k(d/d_edge_weight)
where each feature_k has the same scale ~[0,1] and roughly same distribution,
so the 4 gradients are similar, and the 4 gammas drift together.

This v5 module fixes the issue in two ways (controlled by mode):

- mode='all_params_4f': 4 learnable γ (legacy v3 behaviour, kept for ablation)
- mode='one_param_scalar': a single learnable γ scalar used for all 4 signals
  (since they correlate, this single-γ form is mathematically equivalent and
  more stable).
- mode='one_fixed_one_learnable': γ_sim fixed=1.0, all others = single γ
  (different initialisation per feature)
- mode='one_param_per_learnable_lr': 4 γ, each with its own lr multiplier
  (so gradient magnitudes are no longer determined solely by feature scale).

The component is isolated from the rest of the pipeline by replicating the
public surface of `LearnableEdgeReliability` from v3.  To enable: set the
v5 runner flag --learnable_edge_reliability_v5 and --v5_gamma_mode <mode>.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from learnable_gate.neighbor_graph import NeighborGraph




VALID_MODES = (
    "all_params_4f",        # legacy v3: 4 separate gammas
    "one_param_scalar",     # v5 default: single scalar γ for all 4 features
    "one_fixed_one_learnable",  # γ_sim=1.0 fixed, others share 1 γ
    "one_param_per_learnable_lr",  # 4 γ, each with own lr multiplier
)


class LearnableEdgeReliabilityV5(nn.Module):
    """v5 edge reliability: simplified single-γ learnable variant.

    Modes:
      - 'all_params_4f': legacy v3 behaviour.
      - 'one_param_scalar': a single γ used as: rel = exp(γ*sim) * (1+γ*mutual)
        * (1+γ*snn) * exp(-γ*distance).  Mathematically equivalent to v3 when
        all four gammas collapse to the same value (which they always do).
      - 'one_fixed_one_learnable': γ_sim = 1.0 fixed, rest = γ (single learnable).
        Tests whether freezing sim helps learning.
      - 'one_param_per_learnable_lr': 4 γ, each gradient multiplied by a
        'learnable lr' per-feature (allows rebalancing gradient magnitudes).
    """

    def __init__(
        self,
        mode: str = "one_param_scalar",
        init_gamma: float = 1.0,
        # v3-compat kwargs (ignored if not applicable to mode)
        init_gamma_sim: float | None = None,
        init_gamma_mutual: float | None = None,
        init_gamma_snn: float | None = None,
        init_gamma_distance: float | None = None,
        # per-γ lr multipliers (mode='one_param_per_learnable_lr')
        init_lr_mul_sim: float = 1.0,
        init_lr_mul_mutual: float = 1.0,
        init_lr_mul_snn: float = 1.0,
        init_lr_mul_distance: float = 1.0,
        reg_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        # v3 legacy modes translated to v5 equivalent
        _MODE_TRANSLATION = {
            "sim": "all_params_4f",
            "sim_mutual": "all_params_4f",
            "sim_mutual_snn": "all_params_4f",
            "sim_mutual_snn_distance": "all_params_4f",
        }
        mode = _MODE_TRANSLATION.get(mode, mode)
        if mode not in VALID_MODES:
            raise ValueError(f"mode {mode!r} not in {VALID_MODES}")
        self.mode = mode
        self.reg_weight = float(reg_weight)

        # resolve init values, prefer v3-style if provided
        if init_gamma_sim is not None:
            init_gamma = init_gamma_sim

        if mode == "all_params_4f":
            self.gamma_sim = nn.Parameter(torch.tensor(float(init_gamma_sim or init_gamma)))
            self.gamma_mutual = nn.Parameter(torch.tensor(float(init_gamma_mutual or init_gamma)))
            self.gamma_snn = nn.Parameter(torch.tensor(float(init_gamma_snn or init_gamma)))
            self.gamma_distance = nn.Parameter(torch.tensor(float(init_gamma_distance or init_gamma)))
        elif mode == "one_param_scalar":
            self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        elif mode == "one_fixed_one_learnable":
            self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        elif mode == "one_param_per_learnable_lr":
            self.gamma_sim = nn.Parameter(torch.tensor(float(init_lr_mul_sim)))
            self.gamma_mutual = nn.Parameter(torch.tensor(float(init_lr_mul_mutual)))
            self.gamma_snn = nn.Parameter(torch.tensor(float(init_lr_mul_snn)))
            self.gamma_distance = nn.Parameter(torch.tensor(float(init_lr_mul_distance)))

    def effective_gammas(self) -> tuple[float, float, float, float]:
        """Return (sim, mutual, snn, distance) gammas for logging."""
        if self.mode == "all_params_4f":
            return (
                float(self.gamma_sim.detach().cpu()),
                float(self.gamma_mutual.detach().cpu()),
                float(self.gamma_snn.detach().cpu()),
                float(self.gamma_distance.detach().cpu()),
            )
        if self.mode == "one_param_scalar":
            g = float(self.gamma.detach().cpu())
            return (g, g, g, g)
        if self.mode == "one_fixed_one_learnable":
            g = float(self.gamma.detach().cpu())
            return (1.0, g, g, g)
        # one_param_per_learnable_lr
        return (
            float(self.gamma_sim.detach().cpu()),
            float(self.gamma_mutual.detach().cpu()),
            float(self.gamma_snn.detach().cpu()),
            float(self.gamma_distance.detach().cpu()),
        )

    def gamma_snapshot(self) -> dict:
        sim, mutual, snn, dist = self.effective_gammas()
        return {
            "gamma_sim": sim,
            "gamma_mutual": mutual,
            "gamma_snn": snn,
            "gamma_distance": dist,
        }

    def regularization_loss(self) -> torch.Tensor:
        if self.reg_weight <= 0:
            return torch.zeros((), device=next(self.parameters()).device)
        if self.mode == "all_params_4f":
            sq = (
                self.gamma_sim ** 2
                + self.gamma_mutual ** 2
                + self.gamma_snn ** 2
                + self.gamma_distance ** 2
            )
        elif self.mode == "one_param_scalar":
            sq = self.gamma ** 2 * 4
        elif self.mode == "one_fixed_one_learnable":
            sq = self.gamma ** 2 * 3
        else:
            sq = (
                self.gamma_sim ** 2
                + self.gamma_mutual ** 2
                + self.gamma_snn ** 2
                + self.gamma_distance ** 2
            )
        return self.reg_weight * sq

    def forward(self, graph: NeighborGraph) -> tuple[torch.Tensor, torch.Tensor]:
        if graph.indices.shape[1] == 0 or self.mode == "none":
            rel = torch.ones_like(graph.probs, dtype=torch.float32)
            return rel, rel.clone()

        device = next(self.parameters()).device
        sim_t = torch.as_tensor(graph.similarity, dtype=torch.float32, device=device)
        mutual_t = torch.as_tensor(graph.mutual.astype(np.float32), dtype=torch.float32, device=device)
        snn_t = torch.as_tensor(graph.snn, dtype=torch.float32, device=device)
        distance_t = torch.as_tensor(graph.distance, dtype=torch.float32, device=device)
        probs_t = torch.as_tensor(graph.probs, dtype=torch.float32, device=device)

        rel = torch.ones(graph.similarity.shape, dtype=torch.float32, device=device)
        if self.mode == "all_params_4f":
            rel = rel * torch.exp(self.gamma_sim * sim_t)
            rel = rel * (1.0 + self.gamma_mutual * mutual_t)
            rel = rel * (1.0 + self.gamma_snn * snn_t)
            rel = rel * torch.exp(-self.gamma_distance * distance_t)
        elif self.mode == "one_param_scalar":
            g = self.gamma
            rel = rel * torch.exp(g * sim_t)
            rel = rel * (1.0 + g * mutual_t)
            rel = rel * (1.0 + g * snn_t)
            rel = rel * torch.exp(-g * distance_t)
        elif self.mode == "one_fixed_one_learnable":
            g = self.gamma
            rel = rel * torch.exp(1.0 * sim_t)  # γ_sim fixed at 1.0
            rel = rel * (1.0 + g * mutual_t)
            rel = rel * (1.0 + g * snn_t)
            rel = rel * torch.exp(-g * distance_t)
        elif self.mode == "one_param_per_learnable_lr":
            rel = rel * torch.exp(self.gamma_sim * sim_t)
            rel = rel * (1.0 + self.gamma_mutual * mutual_t)
            rel = rel * (1.0 + self.gamma_snn * snn_t)
            rel = rel * torch.exp(-self.gamma_distance * distance_t)

        rel = torch.clamp(rel, min=1e-6, max=1e6)
        weights = probs_t * rel
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
        return rel, weights


def summarize_edge_weights_torch(weights_t: torch.Tensor) -> dict:
    if weights_t.numel() == 0:
        return {
            "edge_weight_entropy": 0.0,
            "effective_neighbor_count": 0.0,
            "max_edge_weight_mean": 0.0,
            "max_edge_weight_p95": 0.0,
            "fraction_effective_neighbors_lt_2": 1.0,
        }
    entropy = -torch.sum(weights_t * torch.log(torch.clamp(weights_t, min=1e-12)), dim=1)
    effective = torch.exp(entropy)
    max_w = torch.max(weights_t, dim=1).values
    return {
        "edge_weight_entropy": float(entropy.mean().detach().cpu()),
        "effective_neighbor_count": float(effective.mean().detach().cpu()),
        "max_edge_weight_mean": float(max_w.mean().detach().cpu()),
        "max_edge_weight_p95": float(torch.quantile(max_w, 0.95).detach().cpu()),
        "fraction_effective_neighbors_lt_2": float((effective < 2.0).float().mean().detach().cpu()),
    }
```

## `methods/TopoGate/learnable_gate/v5_components/mask_noise_v5.py`

```python
"""v5 mask_noise: Gumbel-Sigmoid + Straight-Through Estimator with proxy gradient.

v3 apply_mask_noise (in run_npz.py) used `torch.bernoulli(p)` to sample masks,
which makes the operation non-differentiable wrt the probability (p). As a
result, `mask_ratio` as a learnable parameter never moved during training
(Phase 2.1: 30/50 epochs, mask_ratio remained exactly 0.300).

This v5 fix replaces the bernoulli sample with a Gumbel-Sigmoid + straight-
through estimator AND adds a `mask_ratio_reg` function that the training
loop adds to the loss.  The reg is `mean(y_soft) - mask_ratio`, with gradient
flowing into mask_ratio directly via the Gumbel output.

The repair drops in like apply_mask_noise_v3_legacy.  For Phase 2.2, the
runner is expected to *call* apply_mask_noise_v5_ste AND *accumulate*
mask_ratio_reg to the loss sum.
"""
from __future__ import annotations

import torch


def apply_mask_noise_v3_legacy(x: torch.Tensor, mask_ratio) -> tuple[torch.Tensor, torch.Tensor]:
    """v3 legacy version - kept for ablation.  Uses bernoulli (non-differentiable)."""
    if isinstance(mask_ratio, torch.Tensor):
        ratio_val = float(mask_ratio.detach().cpu())
    else:
        ratio_val = float(mask_ratio)
    should_swap = torch.bernoulli(ratio_val * torch.ones_like(x))
    noisy_x = torch.where(should_swap > 0.5, x[torch.randperm(x.size(0))].to(x.device), x)
    return noisy_x, should_swap


def apply_mask_noise_v5_ste(x: torch.Tensor, mask_ratio,
                            temperature: float = 1.0,
                            generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """v5 Gumbel-Sigmoid + STE.  Returns (noisy_x, mask_hard, y_soft).

    Args:
        x: input tensor (B, D).
        mask_ratio: scalar or 0-d tensor (if Parameter, will receive grad).
        temperature: Gumbel temperature.

    Returns:
        (noisy_x, mask_hard, y_soft):
          - noisy_x uses hard mask (forward only).
          - mask_hard is the binary mask (for the loss function).
          - y_soft is the soft mask, used to compute mask_ratio_reg via STE.
    """
    if isinstance(mask_ratio, torch.Tensor):
        ratio = mask_ratio
    else:
        ratio = torch.tensor(float(mask_ratio), device=x.device, dtype=x.dtype)

    p = torch.clamp(ratio, min=1e-5, max=1 - 1e-5)
    logit = torch.log(p / (1.0 - p))

    if generator is not None:
        u = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    else:
        u = torch.rand(x.shape, device=x.device, dtype=x.dtype)
    gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logit + gumbel) / temperature)
    y_hard = (y_soft >= 0.5).to(x.dtype)

    # Forward path uses y_hard only (no gradient flow through x).
    # Gradient flows via y_soft to mask_ratio (it's a leaf in computation graph).
    noisy_x = torch.where(y_hard > 0.5, x[torch.randperm(x.size(0))].to(x.device), x)
    return noisy_x, y_hard, y_soft


def mask_ratio_alignment_loss(y_soft: torch.Tensor, mask_ratio: torch.Tensor,
                              target_ratio: float = 0.3,
                              weight: float = 1.0) -> torch.Tensor:
    """Auxiliary loss that aligns the *expected* mask ratio with the target.

    The expected mask ratio is `mean(y_soft)`.  When mask_ratio == target_ratio,
    the expected value is ~target_ratio.  This auxiliary loss provides
    gradient signal to mask_ratio through the differentiable y_soft.

    Returns a scalar tensor (mean squared discrepancy).
    """
    if not isinstance(mask_ratio, torch.Tensor):
        return torch.zeros((), device=y_soft.device if isinstance(y_soft, torch.Tensor) else "cpu")
    expected_ratio = y_soft.mean()
    return weight * (expected_ratio - target_ratio).pow(2)
```

## `methods/TopoGate/learnable_gate/v5_components/per_sample_mask_v5.py`

```python
"""v5 per-sample adaptive mask ratio (SBAM-style).

v5 mask_noise_v5 uses a single global `mask_ratio` for all samples in a batch.
SBAM [arXiv:2404.08327, 2024] shows that per-sample adaptive mask ratio
significantly improves masked image modeling by tailoring the mask density to
the per-sample complexity (token salience).

For tabular TopoGate, the analogous notion is **per-sample feature complexity**:
  salience_i = mean(cosine distance to k nearest neighbors)

A high-salience sample has unstable neighbors and benefits from aggressive
masking (the model is forced to learn the hard neighbourhood). A low-salience
sample is in a homogeneous neighbourhood and benefits from light masking
(less information loss from the swap).

The implementation:
  - Compute salience_i from the kNN graph (CPU, once per epoch).
  - Per-row mask ratio = mask_base + mask_scale * salience_i (clipped to
    [mask_ratio_min, mask_ratio_max]).
  - Apply Gumbel-Sigmoid + STE per-row (broadcast logit across feature dim).

The two new learnable parameters (mask_base, mask_scale) get gradient signal
through the per-row y_soft mask via mask_ratio_reg_loss (computed below).

The fix is intended to be a drop-in replacement for apply_mask_noise_v5_ste
in scripts/learnable_gate/run_v5_separate.py: callers should pass the
(B,) tensor of mask ratios per row instead of a scalar.
"""
from __future__ import annotations

import torch


def compute_sample_salience(
    x: torch.Tensor,
    precomputed: "torch.Tensor | None" = None,
    k: int = 10,
) -> torch.Tensor:
    """Per-sample salience score in [0, 1].

    Args:
        x: (B, D) feature tensor on any device.
        precomputed: optional (B,) pre-computed salience (e.g. from the kNN
            graph used to build edge reliability).  If None, falls back to
            a quick kNN in feature space.

    Returns:
        (B,) tensor of salience in [0, 1].
    """
    if precomputed is not None:
        sal = precomputed.detach().clone()
    else:
        # Quick kNN-based salience: mean distance to k nearest neighbours.
        # We use cosine distance which is consistent with the build_pca_knn_graph
        # metric.
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        x_np = x.detach().cpu().numpy()
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(x_np)
        d, _ = nn.kneighbors(x_np)
        # Skip self (column 0 = 0 distance)
        d = d[:, 1:].mean(axis=1)
        sal = torch.as_tensor(d, device=x.device, dtype=x.dtype)
    s_min = sal.min()
    s_max = sal.max()
    span = (s_max - s_min).clamp(min=1e-8)
    return ((sal - s_min) / span).detach()


def apply_mask_noise_v5_per_sample(
    x: torch.Tensor,
    mask_ratio_per_sample: torch.Tensor,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """v5 Gumbel-Sigmoid + STE with per-row mask ratio.

    Args:
        x: (B, D) input tensor.
        mask_ratio_per_sample: (B,) tensor with one mask ratio per row.  Can
            be a leaf tensor with gradient (mask_base + mask_scale * salience).
        temperature: Gumbel temperature.

    Returns:
        (noisy_x, mask_hard, y_soft):
          - noisy_x uses hard mask (forward only).
          - mask_hard is the binary mask (B, D).
          - y_soft is the soft mask (B, D), used for mask_ratio_reg_loss.
    """
    assert mask_ratio_per_sample.ndim == 1, \
        f"mask_ratio_per_sample must be (B,), got {tuple(mask_ratio_per_sample.shape)}"
    assert mask_ratio_per_sample.shape[0] == x.shape[0], \
        f"mask_ratio_per_sample batch dim {mask_ratio_per_sample.shape[0]} != x batch dim {x.shape[0]}"
    p = torch.clamp(mask_ratio_per_sample, min=1e-5, max=1.0 - 1e-5)
    logit = torch.log(p / (1.0 - p))  # (B,)
    logit_b = logit.unsqueeze(1).expand(x.shape[0], x.shape[1])  # (B, D)
    if generator is not None:
        u = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    else:
        u = torch.rand(x.shape, device=x.device, dtype=x.dtype)
    gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logit_b + gumbel) / temperature)
    y_hard = (y_soft >= 0.5).to(x.dtype)
    # Forward path uses y_hard only.
    noisy_x = torch.where(y_hard > 0.5, x[torch.randperm(x.size(0))].to(x.device), x)
    return noisy_x, y_hard, y_soft


def per_sample_mask_ratio_reg_loss(
    y_soft: torch.Tensor,
    mask_ratio_per_sample: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Auxiliary loss that aligns per-row y_soft means with the row mask ratios.

    For each row i: mean(y_soft[i, :]) should track mask_ratio_per_sample[i].
    This provides gradient signal to mask_base and mask_scale.

    Returns scalar (mean squared discrepancy across rows).
    """
    if not isinstance(mask_ratio_per_sample, torch.Tensor):
        return torch.zeros((), device=y_soft.device if isinstance(y_soft, torch.Tensor) else "cpu")
    expected_per_row = y_soft.mean(dim=1)  # (B,)
    return weight * (expected_per_row - mask_ratio_per_sample.detach()).pow(2).mean()
```
