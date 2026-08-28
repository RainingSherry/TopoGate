# TopoGate 核心代码索引

本文件是 `methods/TopoGate/` 的代码导航和版本边界说明。它用于回答三个问题：

1. 某个版本的真正训练入口在哪里；
2. 图、门控、混合、损失和模型分别由哪些文件负责；
3. 结果目录中的 variant 名称对应哪一套源码。

它不是新的运行入口，也不改变任何算法。源码、配置、测试和结果产物仍以磁盘上的当前文件为准。

## 版本总览

| 版本/目录 | 状态 | 核心语义 | 训练入口 | 是否当前主线 |
|---|---|---|---|---|
| `learnable_gate/` | V9 legacy | 输入空间邻居伪混合，节点级 gate，历史 NumPy 采样路径 | `run_npz.py:main`、`run_topogate` | 是 V9 对照主线，不是 V10/V11/V12 |
| `v10_reliable_graph/` | V10 | EMA latent 动态图、逐边可靠性、可导聚合和聚类正则 | `run.py:run_v10`、`train_v10` | 独立 V10 |
| `V11/` | V11 | 动态候选图、Student-t cluster head、teacher/graph posterior | `run.py:run_v11`、`fit_v11` | 独立 V11 |
| `V12_latent_topology/` | V12 | 纯净输入、masked AE、self/null + edge latent topology alignment | `run_npz.py:main`、`train_and_evaluate` | 独立 V12 实验分支 |
| `V17_topology_native/` | V17-reference | 同一稀疏 `C` 定义自表达、精确零 gate、affinity 与谱读出 | `run.py:fit_v17`、`model.py:fit_topology` | 当前机制 reference，尚非性能主线 |
| `static_gate/` | V1/static | 冻结的静态 gate 与历史消融 | `run.py:main` | 只作历史对照 |
| `v6_latent_mix/` | V6 prototype | latent-space mix | `v6_runner.py:run_v6` | 已废弃的研究原型 |
| `v7_cross_attn/` | V7 prototype | cross-attention latent mix | `v7_runner.py:run_v7` | NO-GO 历史原型 |

### 重要的命名边界

`learnable_gate/configs/` 中的 `learnable_gate_v10_nomix_init.yaml`、
`learnable_gate_v11_nomix_warmup.yaml` 和 `learnable_gate_v12_risk_adaptive.yaml`
仍由 `learnable_gate/run_npz.py` 读取。它们是 V9 legacy 的风险/消融配置，
不是 `v10_reliable_graph/`、`V11/` 或 `V12_latent_topology/` 的实现。

同理，`V11/configs/` 中的 `topogate_v13_*` 和 `topogate_v14_*` 是 V11
实验配置名；它们不会自动生成新的 `methods/TopoGate/V13/` 或 `V14/` 代码目录。

## 当前代码地图

### V9 legacy：`learnable_gate/`

**入口与训练编排**

- `run_npz.py`：NPZ 加载、预处理、训练循环、KMeans 评估和结果落盘。
- `run_npz.py:run_topogate`：供 CLUBench wrapper/脚本调用的程序化入口。
- `configs/*.yaml`：V9 主线、schedule、risk-adaptive、NoMix 和历史消融配置。

**模型与门控**

- `model.py:AutoEncoder`：masked autoencoder、mask predictor、重构损失。
- `learnable_gate.py:LearnableGate`：V9 节点级 gate 和 `build_gate_stats_tensor`。
- `learnable_edge_reliability.py:LearnableEdgeReliability`：逐边可靠性实验模块及 Torch 诊断。
- `binary_router.py:BinaryRouter`：二值路由实验模块。
- `uncertainty.py`：不确定性/风险特征。

**图与输入混合**

- `neighbor_graph.py:NeighborGraph`、`build_pca_knn_graph`：固定 PCA-kNN 图及边特征。
- `mixing.py:compute_node_gate`：节点 gate 计算。
- `mixing.py:make_pseudo_batch_binary`、`make_pseudo_batch`：输入空间伪混合和历史采样路径。

**诊断与辅助**

- `diagnostics.py`：embedding 几何、映射指标和 cell-type 诊断。
- `v5_components/`：V5 组件归档，不是 V9 当前默认路径。
- `_backup_v3_20260726_004309/`：源码备份，不参与运行。

**V9 输出契约**

`embedding_final.npy`、`metrics.json`、`summary.json` 是稳定输出；旧 runner 还可能写
`labels.npy` 和 `embeddings_base.npy`。`labels.npy` 在历史产物中语义不统一，新的汇总器
不得把它当作标准真值文件；新代码应使用 `predictions.npy` 和 `labels_true.npy`。

### V10：`v10_reliable_graph/`

训练路径：

```text
run.py:run_v10/train_v10
  -> graph.py:build_knn_graph/build_consensus_graph
  -> gate.py:EdgeGate
  -> mixing.py:mix_with_reliable_neighbors/aggregate_neighbors
  -> losses.py:V10Objective
  -> model.py:V10AutoEncoder
  -> summary.json + graph/history artifacts
```

- `graph.py`：输入/latent kNN、候选 union 图、cosine/mutual/SNN/density/recurrence 特征。
- `gate.py:EdgeGate`：逐边可导可靠性。
- `mixing.py`：Torch gather、邻居聚合和 assignment gather。
- `losses.py`：重构、view consistency、assignment JS、entropy、gate budget/temporal loss。
- `model.py:V10AutoEncoder`：低秩 masked autoencoder。
- `run.py`：训练、EMA、图刷新、KMeans 主读出和 output contract。
- `configs/topogate_v10_reliable_graph.yaml`：完整动态 V10。
- `configs/topogate_v10_fixed_graph.yaml`：固定图对照。
- `configs/topogate_v10_feature_only.yaml`：无图/无 prototype 的 feature-only 对照。
- `scripts/v10_reliable_graph/run_v10_multiseed.py`：多 seed 编排，GPU 池 `[1, 4, 5]`。
- `tests/v10_reliable_graph/`：V10 单元和 runner 契约测试。

主要输出：`embedding_final.npy`、`predictions.npy`、可选
`cluster_probabilities.npy`、`final_graph_edges.npz`、`history.json`、
`graph_history.json`、`config_resolved.json`、`summary.json`。

### V11：`V11/`

训练路径：

```text
run.py:run_v11/fit_v11
  -> config.py:V11Config
  -> graph.py:build_candidate_graph
  -> trainer.py:V11Trainer
       -> model.py:TopoGateV11/TopologyMixture/StudentTMixtureHead
       -> optional tda.py H0 prior
  -> predictions.npy + embedding/graph histories
```

- `config.py`：配置 dataclass、YAML 读取和 override。
- `graph.py`：raw-kNN 与 EMA-latent-kNN 候选 union、SNN、recurrence。
- `model.py`：masked AE、Student-t mixture head、topology mixture、EMA teacher。
- `trainer.py`：warmup、teacher/student、counterfactual target、graph posterior 和联合优化。
- `tda.py`：固定稀疏 kNN 1-skeleton 上的 H0 union-find pilot；不实现 dense VR/H1 persistence。
- `run.py`：NPZ/程序化入口、K 协议、指标和结果落盘。
- `tests/test_v11.py`：V11 图、损失、TDA、标签隔离和 runner 契约测试。
- `v9_reference_manifest.json`：V9 比较路径的 hash manifest，不是原始源码快照。

主要输出：`embedding_final.npy`、`cluster_probabilities.npy`、`predictions.npy`、可选
`labels_true.npy`/`label_mapping.json`、`metrics.json`、`args.json`、`summary.json`。

### V12：`V12_latent_topology/`

训练路径：

```text
run_npz.py:train_and_evaluate
  -> label-free PCA-kNN graph (topology enabled only)
  -> learnable_gate.py:build_gate_stats_tensor
       + edge_reliability_full (row-standardized distance/mutual/snn)
  -> learnable_gate.py:LearnableGate(self/null + edge weights)
  -> model.py:AutoEncoder.forward_mask/loss_mask_weighted
  -> learnable_gate.py:topology_alignment_loss
  -> learnable_gate.py:rank_alignment_loss (log-space pairwise hinge)
  -> KMeans evaluation + summary.json
```

- `model.py:AutoEncoder`：默认 `legacy_mask_conditioned` decoder；默认 additive
  `reconstruction + 0.1 * mask`；`latent_only` 和 `legacy_weighted` 仅作显式消融。
- `learnable_gate.py`：Torch-only `[N,K,4]` edge stats、`[B] self_weight`、`[B,K]`
  `edge_weights`、self/null fallback、`topology_alignment_loss` 和
  `rank_alignment_loss`（rank/trust signal，与 phase-1 兼容）。
- `run_npz.py`：纯净输入；只做 mask corruption，不调用 V9 `make_pseudo_batch`。
  全量 clean latent 通过 Torch gather 提供 detached topology targets。rank 损失
  与 topology 损失共享 `ramp` schedule；warmup 期间 gate 仍 no_grad。CLI
  新增 `--rank_loss_weight` / `--rank_margin`。
- `configs/topogate_v12_self_null.yaml`：默认修复版本（rank_loss_weight=0.1）。
- `configs/topogate_v12_edge_only.yaml`：旧 edge-only topology 对照（rank=0.1）。
- `configs/topogate_v12_nomix.yaml`：严格 NoMix，不构图（rank=0 显式关闭）。
- `configs/topogate_v12_self_null_lambda001.yaml`、`...lambda003.yaml`、`...lambda01.yaml`：预注册 topology 强度。
- `tests/test_v12.py`：权重归一化、edge stats 形状、梯度、loss、decoder、NoMix 不构图
  和 3 个 rank 单元测试（rewards_top_similarity_edge、detach_reliability、
  is_zero_when_reliability_is_constant）。
- `scripts/V12/run_stage1.py`：flame/enron × 5 variants × 3 seeds 编排（stage-1 旧证据保留）。
- `scripts/V12/run_stage2.py`：5 AHDPC × 3 variants × 3 seeds 编排，DATASETS 包含 flame、
  enron、balance_scale、spect_heart、vehicle；VARIANTS 暴露 `rank_loss_weight`；允许 `--variants` 子集。
- `scripts/V12/summarize_stage1.py` / `summarize_stage2.py`：配对表、均值/标准差、覆盖审计和 rank 字段。

主要输出：`embedding_final.npy`、`predictions.npy`、可选
`labels_true.npy`/`label_mapping.json`、`final_graph_edges.npz`、`history.json`、
`resolved_args.json`、`summary.json`。

**当前结果边界**：
- 阶段一权威目录为 `result/V12/v12_self_null_stage1_2026-08-03_warmup_fix/`，30/30 完成，
  restricted no-go。
- 阶段二 edge-rank 修复产物在 `result/V12/v12_edge_rank_stage2_2026-08-04/`，4 AHDPC
  × 3 variants × 3 seeds = 36/36 完成。判定为 restricted go：
  - 边缘选择机制已建立（rank_loss 单调下降、gate 梯度非零、reliability 行内非退化）；
  - flame ARI 0.5154 > NoMix 0.3897（+0.126），但 seed 7 是单点提升；
  - 当前 `rank_loss_weight=0.1` 不足以让 edge entropy 显著低于 log(5)
    （4 AHDPC 上仍 1.45–1.60，effective_neighbors 4.3–5.0）；
  - balance_scale / vehicle 退化 0.003–0.015（落在 0.03 ARI 容差内）；
  - 不宣称"已修复选择"——仅宣称"已实现选择机制 + flame 部分证据"。
  详细配对诊断见 `result/analysis/V12_edge_rank_stage2_2026-08-04.md`。
- 下一阶段（day-2 task）建议：`rank_loss_weight=0.3, rank_margin=0.2` 在
  5 datasets × 3 variants × 3 seeds 重跑；entropy 显著低于 log(5) 是升为正式 go 的条件。
- 阶段三拓扑信号强化网格产物在
  `result/V12/v12_topology_search_stage3_2026-08-04/`，12 configs
  (8 self_null × 4 edge_only) × 4 AHDPC × 3 seeds = **144/144 完成, 0 failed**。
  新增 `scripts/V12/run_stage3.py`（grid launcher）和
  `scripts/V12/summarize_stage3.py`（entropy diagnostic + paired vs
  stage-2 self_null_lambda01 baseline）。判定为 **no-go**：
  - 全部 48 (dataset, config) cell **edge_entropy 都 < log(5) 但 0/48 < 1.0**；
  - effective_neighbors 仍 3.4–4.9；rank_loss 0.21→0.49 随 margin 增大但 entropy
    降幅 < 0.1 → **hinge loss 架构饱和确认**；
  - paired delta vs stage-2：balance_scale **+0.04 ARI 真实增益**（lambda
    0.3/0.5 放大），flame -0.012，spect_heart/vehicle 持平；edge_only vs
    self_null ARI 差 < 0.001；
  - 当前 V12_latent_topology **不进入论文 main-result 表**。
  详细报告见 `result/analysis/V12_topology_signal_amplification_stage3_2026-08-04.md`。
  下一步按 plan 失败条件：替换 hinge loss 为 KL/Gumbel-top-k/sparsemax，
  或重建 V13 top-k gating，或重写 reliability target。

### V13：`V13_hard_gate/`

训练路径：

```text
run_npz.py:train_and_evaluate
  -> label-free PCA-kNN graph (topology enabled only)
  -> gumbel_gate.py:build_gate_stats_tensor
  -> gumbel_gate.py:GumbelTopKGate
       + top-k selection (K=2 default)
       + Gumbel-Softmax straight-through gradient during training
       + hard top-k argmax at evaluation
  -> model.py:AutoEncoder.forward_mask/loss_mask_weighted
  -> gumbel_gate.py:hard_topk_alignment_loss
       (mask_sum normalisation, not K)
  -> KMeans evaluation + summary.json
```

- `gumbel_gate.py`：`GumbelTopKGate` 使用 Gumbel-Softmax straight-through
  梯度；推理时 `hard=True` 使用 deterministic top-k argmax。输出
  `GumbelTopKGateOutput(mask, gumbel_probs, scores)`。无 rank_loss，
  无 self/null fallback。`hard_topk_alignment_loss` 用 mask_sum 归一化
  而非 K，保证 top-k=2 时目标均值始终是 2 个选中邻居的平均。
- `model.py`：V12 AutoEncoder 副本，decoder_mode=legacy_mask_conditioned
  默认。
- `run_npz.py`：V12 runner 改造版，移除 rank_loss，新增
  `--gumbel_tau`/`--gumbel_tau_min`/`--gumbel_tau_anneal_epochs`，
  tau 从 1.0 退火到 0.1（50 epochs）。推理时 `hard=True` 获取 binary mask。
- `configs/topogate_v13_topk2.yaml`：lambda=0.1，top_k=2，warmup=20，ramp=10。
- `configs/topogate_v13_nomix.yaml`：topology_enabled=false。
- `tests/test_v13.py`：14 passed。
- `scripts/V13/run_v13.py`：5 datasets × nomix/topk2 × 3 seeds 编排。
- `scripts/V13/summarize_v13.py`：pairwise vs nomix 诊断。

主要输出：`embedding_final.npy`、`predictions.npy`、可选
`labels_true.npy`/`label_mapping.json`、`final_graph_edges.npz`（含 mask）、
`history.json`、`resolved_args.json`、`summary.json`。

**当前结果边界**：
- 正式批次 `result/V13/v13_hard_gate_2026-08-04/`：**30/30 completed, 0 failed**。
  5 datasets × 2 variants × 3 seeds。
- **判定：有条件 go**：
  - ✅ `effective_neighbor_count = 2.000` 在所有 15 个 topk2 runs 严格成立
    — **hard gate 机制完全有效**，解决了 V12 的核心问题；
  - ⚠️ enron topk2 vs nomix：**-0.73 ARI**（灾难性崩溃，从 0.803 → 0.072），
    topology_alignment_loss 在 hard 选择后更具破坏性；
  - ⚠️ flame topk2 vs nomix：**-0.084**（seed 不稳定，seed 7 +0.066, seed 42 -0.277）；
  - balance_scale +0.023, spect_heart +0.016, vehicle -0.002（持平）。
- **论文叙事**：V13 的贡献 = "第一个在聚类任务中验证 Gumbel-Top-k hard
  selection 的工作"，而非"topology alignment 改进"。topology_alignment_loss
  在 hard 选择后的破坏性是新发现，需要未来版本重新设计。
  详细报告见 `result/analysis/V13_gumbel_topk_analysis_2026-08-04.md`。

### V17-reference：`V17_topology_native/`

唯一数据流：

```text
sparse X
  -> input_adapter.py:prepare_input/build_projection_views
  -> candidate.py:build_candidate_union
  -> relation.py:solve_candidate_self_expression
  -> C exact-zero support gate
  -> relation.py:affinity_from_coefficients = abs(C)+abs(C.T)
  -> spectral.py:normalized_spectral_readout
```

- `model.py:fit_topology` 不接收 `K` 或 `y`；输入、候选、`C` 与 affinity 完全在
  readout 之前冻结。
- `run.py:fit_v17` 只在最终 normalized spectral readout 使用 `K`，标签只计算后验
  ARI/NMI/AMI。
- degree-zero 节点输出 `-1` abstention，不调用第二个 feature-space 聚类器。
- `THEORY.md` 明确当前是非深度 reference solver；spectral feedback 和 learnable
  unrolling 尚未实现，也没有真实数据性能证据。
- focused tests 位于 `V17_topology_native/tests/`；脚本入口为
  `scripts/V17/run_reference.py`。

### StaticGate：`static_gate/`

这是冻结的 V1 风格历史消融，文件结构与 V9 legacy 有意重复以避免耦合：

- `run.py`：CSV/CLUBench 时代的静态 runner。
- `model.py`、`neighbor_graph.py`、`mixing.py`、`diagnostics.py`：冻结模块副本。
- `configs/static_gate_*.yaml`：8 个历史消融。
- `scripts/static_gate/run_topogate_ablation.py`：静态消融编排。

它不应被当作当前 V9/V10/V11/V12 主实现，也不应为了对比结果修改。

### 历史原型：`v6_latent_mix/`、`v7_cross_attn/`

- V6：`v6_runner.py`、`latent_mixer.py`、`micro_encoder.py`。
- V7：`v7_runner.py`、`cross_attn_mixer.py`。
- 对应 smoke 编排在 `scripts/v6_latent_mix/` 和 `scripts/v7_cross_attn/`。
- 两者复用部分 V9 helper，但不改变 V9；研究日志和 `EXPERIMENT_PHASES.md` 将它们标为废弃/NO-GO。

它们保留用于历史复盘，不进入当前主方法或论文性能主表。

## 统一数据与标签边界

- 图构建、gate、loss、variant 选择不读取真值 `y`。
- benchmark 可用 `K = int(np.unique(y).size)`，仅用于评估协议；无标签运行必须显式传入 `n_clusters`。
- 正式输出使用 `predictions.npy`（预测）和 `labels_true.npy`（真值）；不要新增含义不明的 `labels.npy`。
- 训练结果必须写入 `result/` 软链接目标；短 smoke 写 `/tmp`，核验后清理。
- GPU 0 和 GPU 7 禁止使用；当前 V10/V12 编排使用 `[1, 4, 5]`，V11 多 seed runner 按其源码声明的池执行。

## 推荐核验命令

从仓库根目录执行：

```bash
python -m compileall -q methods/TopoGate/learnable_gate methods/TopoGate/v10_reliable_graph methods/TopoGate/V11 methods/TopoGate/V12_latent_topology scripts/V12
python -m pytest -q tests/v10_reliable_graph
python -m pytest -q methods/TopoGate/V11/tests/test_v11.py
python -m pytest -q methods/TopoGate/V12_latent_topology/tests
python -m pytest -q methods/TopoGate/V17_topology_native/tests
```

`__pycache__/` 是解释器/pytest 生成的缓存，不属于核心源码；
`learnable_gate/_backup_v3_20260726_004309/` 是已记录的备份，不参与导入和运行。
整理时保留二者的现状，不把它们列入核心模块，也不把删除缓存误报为算法变更。
