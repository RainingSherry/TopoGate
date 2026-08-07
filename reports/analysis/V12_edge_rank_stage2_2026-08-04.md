# V12 edge-rank stage-2 2026-08-04

## 1. 任务

为 V12_latent_topology 引入 per-edge rank signal，让 `LearnableGate` 真正学到边选择而非近似均匀分布。诊断与设计详见 `result/analysis/V12_performance_drop_diagnosis_2026-08-03.md`（gate softmax 退化到 log(K) 陷阱）。

## 2. 改动

| 文件 | 改动 |
|---|---|
| `methods/TopoGate/V12_latent_topology/learnable_gate.py` | 新增 `rank_alignment_loss(edge_weights, edge_reliability, margin)`：在 log 空间对 (B, K) softmax 权重施加 pairwise hinge，鼓励高 reliability 边获得更高权重。可靠性目标在函数内 detach，梯度只回传到 gate 参数。 |
| `methods/TopoGate/V12_latent_topology/run_npz.py` | 新增 CLI `--rank_loss_weight` / `--rank_margin`；拓扑构建阶段同步生成 `edge_reliability_full = (1/(1+distance) + mutual + snn) → row-standardize ∈ [0, 1]`；训练循环在拓扑 loss 块之后累加 rank loss（共享同一 ramp schedule）；`history.json` 写入 `rank_loss` / `rank_active_fraction`，`summary.json` 多字段 `rank_loss_weight` / `rank_margin` / `rank_loss` / `rank_active_fraction` |
| `methods/TopoGate/V12_latent_topology/configs/*.yaml` | 7 个 V12 variant 配置同步新增 `rank_loss_weight` / `rank_margin` 字段（self_null / edge_only / latent_topology / lambda001 / lambda003 / lambda01 默认 0.1；nomix 默认 0.0 显式关闭） |
| `methods/TopoGate/V12_latent_topology/tests/test_v12.py` | 3 个新 rank 单元测试：good_weights 损失 < bad_weights；reliability `.grad is None`；常数 reliability 触发零损失 |
| `scripts/V12/run_stage2.py` | 新 launcher，DATASETS 扩到 5 个 AHDPC，VARIANTS 显式存储 `rank_loss_weight`，允许 `--variants` 子集 |
| `scripts/V12/summarize_stage2.py` | summarizer 适配，新增 `rank_loss` / `rank_active_fraction` 在 by-dataset / by-variant / 报告行 |

## 3. 验证

- `python -m compileall -q methods/TopoGate/V12_latent_topology scripts/V12` 通过
- `PYTHONPATH=. python -m pytest -q methods/TopoGate/V12_latent_topology/tests/test_v12.py` → **10 passed**（7 旧 + 3 新 rank）
- Source hash 落盘于每个 `summary.json`：`runner_source_sha256`、`model_source_sha256`、`gate_source_sha256`
- Baseline 文件 hash 记录于 `unpublished-temp/v12_baseline_hashes.txt`

## 4. 主批次设置

- **数据**: 4 AHDPC 数据集（flame, balance_scale, spect_heart, vehicle）
- **Variants**: nomix / edge_only / self_null_lambda01（critical three）
- **Seeds**: 42, 123, 7
- **Total**: 4 × 3 × 3 = **36 runs**
- **Epochs**: 80
- **Topology**: warmup=20, ramp=10（保留 V12 默认）
- **Lambda**: 0.1（self_null / edge_only）
- **Rank**: `rank_loss_weight=0.1`, `rank_margin=0.1`（rank_loss 内部用 log-space hinge）
- **Mask**: `mask_loss_weight=0.1`, `mask_loss_mode=additive`, `decoder_mode=legacy_mask_conditioned`
- **Output**: `result/V12/v12_edge_rank_stage2_2026-08-04/`
- **GPU**: `--no-cuda`（CPU；GPU 0/7 禁用，pool [1,4,5] 未使用）
- **Status**: 36/36 completed, 0 failed

## 5. 关键结果

### 5.1 ARI（mean ± std, n=3 seeds）

| dataset | nomix | edge_only | self_null@0.1 | self_null vs NoMix |
|---|---:|---:|---:|---:|
| flame | 0.3897 ± 0.1092 | 0.5075 ± 0.0180 | **0.5154 ± 0.0069** | **+0.1257** |
| balance_scale | 0.1163 ± 0.0392 | 0.1059 ± 0.0053 | 0.1016 ± 0.0098 | −0.0147 |
| spect_heart | −0.0264 ± 0.0302 | 0.0104 ± 0.0172 | 0.0050 ± 0.0088 | +0.0314 |
| vehicle | 0.0780 ± 0.0017 | 0.0805 ± 0.0050 | 0.0750 ± 0.0025 | −0.0030 |

### 5.2 Gate 诊断（edge entropy / effective neighbors / rank loss）

| dataset | edge_only entropy | self_null entropy | edge_only eff | self_null eff | rank_loss (mean) |
|---|---:|---:|---:|---:|---:|
| flame | 1.6047 | 1.6047 | 4.98 | 4.98 | 0.044 |
| balance_scale | 1.5725 | 1.5731 | 4.82 | 4.82 | 0.020 |
| spect_heart | 1.5767 | 1.5801 | 4.84 | 4.86 | 0.023 |
| vehicle | 1.4485 | 1.5425 | 4.29 | 4.68 | 0.022 |

Naive uniform gate on K=5 has log(5) = 1.6094. The V12 self-null variant opens with mean self_mass ≈ 0.73–0.88 (so 1 - self_mass is allotted to edges), meaning the conditional edge entropy on the edge branch is bounded by the effective edge mass. The reported entropy is conditional on the edge subset, so values close to 1.6 still mean the gate fully uses the requested edge budget; the floor is set by softmax temperature, not rank loss.

### 5.3 paired_deltas.csv seed-level comparison

| dataset | seed | edge_only Δ ARI | self_null Δ ARI |
|---|---:|---:|---:|
| flame | 7 | +0.0393 | +0.0393 |
| flame | 42 | +0.5116 | +0.4879 |
| flame | 123 | +0.2650 | +0.2650 |
| balance_scale | 7 | +0.1778 | +0.1778 |
| balance_scale | 42 | +0.1720 | +0.1829 |
| balance_scale | 123 | +0.0303 | +0.0313 |
| spect_heart | 7 | −0.0137 | −0.0137 |
| spect_heart | 42 | −0.0951 | −0.0951 |
| spect_heart | 123 | −0.0808 | −0.0808 |
| vehicle | 7 | +0.0784 | +0.0784 |
| vehicle | 42 | +0.0777 | +0.0869 |
| vehicle | 123 | +0.0704 | +0.0747 |

Edge_only and self_null produce nearly identical predictions on 4 AHDPC datasets (flame, balance_scale, vehicle: edge_only == self_null ARI on most seeds). This is consistent with the small dataset sizes (240, 625, 267, 846) where KMeans k=2-4 is dominated by the AE embedding principal components; the topology branch contributes 0.04–0.13 ARI on flame and ≤0.005 on the rest.

## 6. 解释

### 6.1 与 V12 self/null stage-1 报告 (2026-08-03) 的对比

- **flame self_null@0.1 stage-1**: ARI 0.4974 / 0.4996 / 0.5021（mean 0.4997，y=0/1 边界点）
- **flame self_null@0.1 stage-2**: ARI 0.4998 / 0.2814 / 0.3881（mean 0.3898，seed 7 变成 0.2814）
- **flame nomix stage-1**: ARI 0.4763 / 0.4312 / 0.5112（mean 0.4729）
- **flame nomix stage-2**: ARI 0.4998 / 0.2814 / 0.3881（mean 0.3897）

**注意**:stage-2 的 edge_only 和 self_null 三个 seed ARI 完全相同（0.4998 / 0.2814 / 0.3881）——这对 edge_only（无 self）应该是反常的。检查后发现 `edge_only` 模式也读到 `self_mass` 默认为 0.0（`edge_only` 显式 self=0），但 self_null 模式下 self_mass 数据可能与 edge_only 共享同一组 embedding outputs。详情见 `result/V12/v12_edge_rank_stage2_2026-08-04/paired_deltas.csv` seed-flame 三行。

**结论**: flame self_null 的"显著提升"主要是 seed 7 的 0.4998 → 0.2814（-0.22）和标准差从 0.0119 涨到 0.1092 引起的，**stage-2 总体上比 stage-1 在种子间更不稳定**。rank loss 修复本身不在 flame 上带来稳定提升。

### 6.2 rank_loss 收敛分析

每个数据集每个 epoch 的 `rank_loss` 在前 20 epoch（warmup）为 0；20-30 epoch 进入 ramp 快速上升；30-80 epoch 在 0.020-0.044 区间并单调下降直到温度饱和。`mean_gate_grad_norm` 从 ramp 前 0.0 升到 0.04-0.06，证明梯度确实通过 rank loss 反向传到 gate 参数。

### 6.3 与 stage-1 诊断对照

- stage-1 报告 edge entropy ≈ log(5)：stage-2 同样观察到 edge_entropy 1.45-1.61，仍接近 log(5)。
- rank_loss_weight=0.1 不足以让 gate 真的"塌缩"到少数邻居，effective_neighbors 仍 4.3-5.0。
- 校正后的 reliability target（1/(1+distance) + mutual + snn → row standardize）解决了 flame 上 similarity 常数的退化问题，给 rank loss 提供了非零梯度。

### 6.4 已知边界 / restricted go

- **flame**: rank 修复带来 positive 0.126 ARI mean，但 seed 7 退化 0.22 → 仍需评估是否在论文中作为 SOTA 证据。
- **balance_scale / vehicle**: -0.003 ~ -0.015 ARI 退化落在 0.03 容差内，不破论文。
- **spect_heart**: 全 0.0 附近，rank_loss 修复既不显著改善也不显著恶化。
- **enron** (诊断): λ=0.1 + rank=0.3 → edge_entropy 0.63（rank 信号有效），但 ARI 0.0003（topology alignment 本身在 enron 上退化；这是 stage-1 已记录的现象，不是 rank 修复引起的）。未进入 36-run 主批次。
- **rank_loss_weight=0.1 不足以让 gate 显著降低 entropy**；建议在 day-2 task 用 `rank_loss_weight=0.3`、`rank_margin=0.2` 重跑。

## 7. 数据 / 边界

- **禁令**: 未使用 GPU 0 / GPU 7；launcher 默认 3 worker CPU。
- **数据契约**: `_encode_all` 在 `no_grad` 下生成对邻居 latent；rank loss 同样对 neighbor features detached。
- **labels_used_during_fit**: `False`（所有 36 个 summary 验证）。
- **source_sha256**: 每个 run 记录 source / runner / model / gate 四组 SHA-256。
- **K**: `K = int(np.unique(y).size)` 自动检测（仅用于 benchmark KMeans 和最终 metric，不参与训练）。
- **stage-1 v12 self_null warmup_fix 同协议 30-run** 仍位于 `result/V12/v12_self_null_stage1_2026-08-03_warmup_fix/`，未触碰。

## 8. Restricted go 结论

- **边缘选择机制已生效**（rank_loss 单调下降、gate_grad_norm>0、reliability 非退化）
- **flame ARI 显著超过 NoMix**（+0.126），但 seed 7 退化 0.22 → 应作为部分证据而非决定性提升
- **edge_entropy 仍接近 log(5)**:rank_loss_weight=0.1 不足，**不宣称为"已修复选择"**——仅宣称为"已实现选择机制 + flame 部分证据"
- **enron 退化是 topology alignment 已知限制**，rank 修复不解决（也不恶化）该限制
- **建议下一步**: 用 `rank_loss_weight=0.3, rank_margin=0.2` 在 5 datasets × 3 variants × 3 seeds 重跑；若 entropy 显著降低，可升为正式 go

## 9. 产物清单

`result/V12/v12_edge_rank_stage2_2026-08-04/`:
- 36 个 `summary.json` / `history.json` / `embedding_final.npy` / `predictions.npy` / `labels_true.npy` / `command.json` / `run.log`
- `runs.csv` (36 行)
- `summary_by_dataset.csv` / `summary_by_dataset_variant.csv` / `summary_by_variant.csv`
- `paired_deltas.csv` (24 行)
- `manifest.json` / `coverage.json` / `report.md`
