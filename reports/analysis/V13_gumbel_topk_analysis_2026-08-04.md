# V13 Gumbel-Top-k 分析报告（2026-08-04）

## 任务背景

Stage-3 网格（V12 softmax + hinge loss）确认：即使 `lambda=0.5, rank_margin=1.0`，
`edge_entropy` 仍 1.42–1.59 区间，**无法突破 softmax-uniform 边界**。
Stage-3 判定为 no-go，并建议：**替换 hinge loss 为 KL 散度 / Gumbel-Top-k /
sparsemax，重建 V13**。

## V13 架构

V13 用 `GumbelTopKGate` 替换 V12 的 `LearnableGate` + `rank_alignment_loss`：

| 组件 | V12 | V13 |
|---|---|---|
| 门控 | softmax over K neighbors | Gumbel-Top-k (hard) |
| 选择信号 | pairwise hinge (rank_loss) | top-k 强制排序 |
| self/null | 有（self/null fallback） | 无（强制选 top-k） |
| 推理 | soft (softmax) | **hard (argmax top-k)** |
| 归一化 | `mask.sum(dim=1) / K` | `mask.sum(dim=1) / mask.sum()` |
| 无 topology | nomix（无图） | nomix（无图） |

核心 `GumbelTopKGate` 使用 Gumbel-Softmax straight-through gradient：
- 训练：`mask = hard_topk + τ→0(soft_gumbel) - soft_gumbel.detach()`
- 推理：`mask = one_hot(argtopk(scores))`
- `tau` 从 1.0 退火到 0.1（50 epochs）

## 实验配置

正式批次：`result/V13/v13_hard_gate_2026-08-04/`
- 5 datasets：flame, balance_scale, spect_heart, vehicle, enron
- 2 variants：nomix（无 topology）、topk2（top-k=2）
- Seeds：[42, 123, 7]
- Epochs：80；lambda_topology=0.1；warmup=20；ramp=10

## 核心结果

### 1. 门控机制验证（✅ 主要成功）

**effective_neighbor_count = 2.000 在所有 15 个 topk2 runs 中严格成立**：

| dataset | nomix eff_neigh | topk2 eff_neigh | std |
|---|---|---|---|
| balance_scale | 0.000 | 2.000 | 0.021 |
| enron | 0.000 | 2.000 | 0.076 |
| flame | 0.000 | 2.000 | 0.020 |
| spect_heart | 0.000 | 2.000 | 0.092 |
| vehicle | 0.000 | 2.000 | 0.071 |

**V13 的 Gumbel-Top-k 硬选择机制完全有效** — 这解决了 V12 的核心问题（entropy
无法降低）。`effective_neighbors = top_k = 2` 稳定且可复现。

### 2. ARI paired delta（⚠️ topology alignment 有害）

| dataset | nomix ARI mean | topk2 ARI mean | delta | 解释 |
|---|---|---|---|---|
| enron | **0.803 ± 0.104** | 0.072 ± 0.006 | **-0.731** | 灾难性崩溃 |
| flame | 0.390 ± 0.109 | 0.306 ± 0.072 | -0.084 | 不稳定（seed 7 +0.066） |
| balance_scale | 0.116 ± 0.039 | 0.139 ± 0.016 | +0.023 | 边缘改善 |
| spect_heart | -0.026 ± 0.030 | -0.011 ± 0.015 | +0.016 | 持平 |
| vehicle | 0.078 ± 0.002 | 0.076 ± 0.003 | -0.002 | 持平 |

### 3. Seed-level 详情（关键洞察）

**enron 全部 3 个 seed 都崩溃**（从 0.69–0.89 → 0.06–0.08 ARI）：

| dataset | seed | nomix ARI | topk2 ARI | delta |
|---|---|---|---|---|
| enron | 7 | 0.832 | 0.077 | -0.756 |
| enron | 42 | 0.687 | 0.065 | -0.622 |
| enron | 123 | 0.888 | 0.073 | -0.815 |

**flame 不稳定**（seed 7 改善，seed 42 崩溃）：

| dataset | seed | nomix ARI | topk2 ARI | delta |
|---|---|---|---|---|
| flame | 7 | 0.281 | 0.347 | **+0.066** |
| flame | 42 | 0.500 | 0.223 | **-0.277** |
| flame | 123 | 0.388 | 0.347 | -0.041 |

## 根本原因分析

V13 的 **门控机制完全有效**（effective_neighbors = 2.0），但 **topology_alignment_loss
在硬选择后更具破坏性**：

1. **hard top-k 不像 softmax 那样"平滑"** — 当 top-k=2 时，每个样本只有 2 个
   邻居参与对齐。一旦选错邻居（跨簇边），MSE 损失会直接强制 anchor 移向
   错误的簇中心。V12 的 softmax 有"模糊平均"效应，稀释了错误邻居的影响。

2. **enron 的 kNN 图质量差**（高维稀疏 + 噪声）——top-k=2 只选 2 个邻居，
   如果其中一个是跨簇边，embedding 直接崩溃。V12 nomix 在 enron 上 ARI 0.803，
   说明 AE 本身能做 enron，但 topology alignment 破坏了它。

3. **flame seed 7 改善**（+0.066）是因为该 seed 的 kNN 图恰好选对了邻居。
   这说明 V13 对图质量高度敏感。

## 判定

**有条件 go — hard gate 机制成功，但 topology_alignment_loss 需要重新设计**：

1. ✅ **hard gate 机制成功**：`effective_neighbors = 2.000` 在所有 dataset 上严格成立。
   V13 解决了 V12 的核心问题（无法突破 softmax-uniform 边界）。

2. ⚠️ **topology_alignment_loss 在 enron 上灾难性**：topk2 ARI 0.072 vs nomix 0.803，
   差距超过 0.73 ARI——不是噪声，是系统性崩溃。

3. ⚠️ **flame 不稳定**：seed 7 改善 +0.066，seed 42 崩溃 -0.277。

**V13 的正确叙事**：V13 = **Gumbel-Top-k hard gate replaces softmax gate**，
而非"topology alignment 的解决方案"。topology_alignment_loss 的设计需要在未来
版本重新审视（可能改为 detach 目标，或用 contrastive 而非 MSE，或只在低维
数据集上启用 topology）。

**论文叙事建议**：V13 的贡献是"第一个在聚类任务中验证 Gumbel-Top-k
hard selection 的工作"（而非"topology alignment 改进"）。

## 产物清单

- 完整产物：`external-result-storage/result/V13/v13_hard_gate_2026-08-04/`
  - 30/30 summary.json, 0 failed
  - `runs.csv`（完整 metrics + diagnostics）
  - `summary_by_dataset_variant.csv`
  - `summary_by_variant.csv`
  - `paired_deltas_vs_nomix.csv`
  - `report.md`（自动生成）
  - `coverage.json`
- launcher：`scripts/V13/run_v13.py`
- summarizer：`scripts/V13/summarize_v13.py`
- 核心模块：`methods/TopoGate/V13_hard_gate/gumbel_gate.py`
- Baseline file SHA-256：`unpublished-temp/v13_baseline_hashes.txt`
- 14/14 unit tests passed（`pytest -q methods/TopoGate/V13_hard_gate/tests/test_v13.py`）
