# V12 stage-3 拓扑信号强化搜索分析（2026-08-04）

## 任务背景

V12 edge-rank stage-2 (`v12_edge_rank_stage2_2026-08-04`) 诊断显示：rank
mechanism 已实现（rank_loss 单调下降、gate 梯度非零、reliability 行内
非退化），但 `rank_loss_weight=0.1, rank_margin=0.1` 不足以让
`edge_entropy` 显著低于 `log(5) ≈ 1.6094`（4 AHDPC 上仍 1.45–1.60，
effective_neighbors 4.3–5.0）。Stage-2 结论是 restricted go，需要
hyperparameter search 放大拓扑信号验证是否能让 gate 真正塌缩。

## 搜索空间

| 参数 | 取值 | 触发条件 |
|---|---|---|
| `lambda_topology` | 0.3, 0.5 | 提升拓扑损失占比（默认 0.1） |
| `rank_margin` | 0.5, 1.0 | 锐化 hinge loss（默认 0.1） |
| `self_init_weight` | 0.3, 0.5 | 仅 self_null，降低自身初始保留（默认 0.8） |
| `topology_mode` | self_null, edge_only | 两个核心 variant |
| `rank_loss_weight` | 0.1（恒定） | 通过 `rank_margin` 调节强度 |
| `mask_loss_weight` | 0.1 | 沿用 Stage-2 弱化设置 |

总计 **12 configs**（self_null 8 + edge_only 4）× 4 datasets × 3 seeds
= **144 runs**，CPU 3-worker 并发约 5 分钟。

## 数据集与评估协议

- **flame / balance_scale / spect_heart / vehicle**（沿用 Stage-2 4 AHDPC）
- **K 协议**：`K = int(unique(y).size)`，Benchmark K 来源：labels_unique
- **labels_used_during_fit**：`none`（runner 仅将 `y` 用于 KMeans 评估）
- **Seeds**：42, 123, 7

## 核心指标结果

### 1. edge_entropy（headline metric）

Stage-3 关键观察：**48 个 (dataset, config) cell 都 < log(5) ≈ 1.6094，
但没有任何 cell 触及 1.0 目标。**

| dataset | entropy 区间 | effective_neighbors 区间 | 结论 |
|---|---|---|---|
| flame | 1.586 – 1.591 | 4.889 – 4.911 | 仍接近 log(5)=1.6094，**远未达到 < 1.0** |
| balance_scale | 1.398 – 1.481 | 4.100 – 4.418 | 已低于 log(5)，但未达 < 1.0 |
| spect_heart | 1.459 – 1.531 | 4.339 – 4.634 | 同上 |
| vehicle | 1.196 – 1.324 | 3.416 – 3.821 | 最接近 < 1.0 但仍未触及 |

**rank_loss 行为**：随 `rank_margin` 增大从 ~0.21（margin=0.5）提升到
~0.49（margin=1.0），证实 rank signal 在工作。但 entropy 变化极小
（< 0.1），说明 hinge signal 已达饱和：softmax 饱和 + 残差
edge-information（high-dim features）使 gate 难以进一步塌缩。

### 2. ARI 与 paired delta vs stage-2 self_null baseline

跨 config 的 ARI mean 在 **0.1833–0.1885** 之间（≈ 0.005 区间），
**所有 config 之间 ARI 差异不超过 0.005**。这与 stage-2 观察一致：
KMeans(k=2 or 3) 对 topology 分支 0.04–0.13 ARI 差异不敏感，4 AHDPC
embedding 主要由 AE 主成分决定。

| dataset | paired delta vs stage-2 mean | 解释 |
|---|---|---|
| flame | -0.012 ~ -0.016 | 退化（落在 0.03 容差内） |
| balance_scale | **+0.039 ~ +0.043** | **真实增益**（> 0.03 容差） |
| spect_heart | -0.001 ~ +0.008 | 持平 |
| vehicle | +0.009 ~ +0.027 | 边缘增益 |

### 3. edge_only vs self_null 差异化

跨 12 个 config 的 self_null vs edge_only ARI mean 差值 < 0.001。
**stage-3 没有让 self_mass 发挥可观测作用**：self_mass
(si=0.3, 0.5) 启动值在不同 config 下衰减到 0.40–0.64，仍处于
gate gradient 上行通道；但对最终 ARI 几乎无影响——KMeans 评估不敏感。

## 判定

按 Stage-3 plan 的成功/失败标准：

**失败（命中失败条件）**：即使 lambda=0.5, margin=1.0, self_init=0.3
等最强组合，`edge_entropy` 仍处于 1.42–1.59 区间（**没有任一 cell < 1.0**），
effective_neighbors 仍处于 3.4–4.9 区间。Hinge loss 架构无法突破
softmax-uniform 边界——结论与 stage-2 诊断一致：当前
`rank_alignment_loss` (log-space pairwise hinge) 的梯度强度已达饱和，
再提升 `rank_margin` 和 `lambda_topology` 都不再继续塌缩 edge
distribution。**触发 plan 中"hinge loss 架构需要彻底替换"的兜底结论**。

**部分 ARI 信号**：balance_scale +0.04 ARI 跨 config 稳定，paired
delta > 0.03 容差，但 flame -0.012 反向退化。**不宣称"已修复选择"**——
edge_entropy 未达 1.0 = 选择机制未真正塌缩，但 balance_scale 上的 ARI
提升可能是 lambda=0.3/0.5 共同带来，与 edge collapse 无关。

## 决策

V12 latent_topology 在 stage-3 网格内 **no-go**：hinge loss 信号强度
已饱和。继续提高 `rank_loss_weight`/`lambda_topology`/`rank_margin`
不再产生 edge collapse 增益，ARI 跨 config 差异落在噪声带内。

**下一步建议（已记录于 plan 失败条件）**：

1. **替换 hinge loss 为 KL 散度 / Gumbel-top-k / sparsemax**
 - 当前架构 `softmax + pairwise hinge` 的理论约束：softmax 极值下
   rank loss 对应 logit 梯度恒为 1，但 parameter-scale 已经被 softmax
   内化。KL 散度对分布形状更敏感。
 - sparsemax 提供真正的 sparsity 保证（projection to probability simplex）。
2. **重建 V13**：把 LearnableGate 改为 top-k gating（K=2 选出可信邻居），
  而非 softmax over K。
3. **重写 reliability target**：当前 `(1/(1+dist) + mutual + snn) →
  row-std` 是 row-positive 但 row-spread 极小（balance_scale 上仅
  1.398–1.481 的熵区间；flame 更大），可能需要 source-path entropy
  或 BILBO-style 多视图一致性。

**当前结论（写入 RESULTS_SUMMARY / CHANGELOG / CORE_CODE_INDEX）**：
V12 latent_topology 在 4 AHDPC × 12-config 网格内无显著增益；
edge_entropy 仍接近 log(5)；不更新 V12 论文叙事为 "topology selection
works"；保持 "rank mechanism + flame 部分证据" 的 stage-2 结论。

## 产物清单

- 完整产物：`external-result-storage/result/V12/v12_topology_search_stage3_2026-08-04/`
 - 144/144 summary.json, 0 failed
 - `runs.csv`：144 行完整 metrics + diagnostics
 - `summary_by_config.csv`：12 configs × 4 datasets aggregate
 - `summary_by_dataset.csv`：4 datasets × 12 configs aggregate
 - `summary_by_dataset_config.csv`：48 cells aggregate
 - `entropy_diagnostic.csv`：edge_entropy/eff_neigh/rank_loss 表
 - `paired_deltas_vs_stage2.csv`：seed-matched 对比
 - `coverage.json`：覆盖率 + 来源审计
 - `report.md`：自动生成的 markdown 报告
- launcher：`scripts/V12/run_stage3.py`
- summarizer：`scripts/V12/summarize_stage3.py`
- baseline file：`unpublished-temp/v12_stage3_pre_hashes.txt`
