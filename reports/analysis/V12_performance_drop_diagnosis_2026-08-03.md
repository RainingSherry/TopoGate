# TopoGate V12 性能下降诊断报告

**日期**：2026-08-03
**目的**：解释“重构后性能大幅下降”的来源，区分真正的 `V12_latent_topology` 与历史结果目录中的 legacy runner，给出可复核的根因排序和后续验证门槛。

## 结论先行

当前看到的下降不是一个单点故障，而是两个相互叠加的回归：

1. **自编码器 decoder 曾发生非必要的接口变化**。将原来的 `[latent, mask_logits] -> Linear` 改为 `latent -> MLP` 会改变 mask 预测与重构的联合优化问题。在当前源码的同协议 flame 复核中，`mask_loss_weight=0.1` 时，兼容 decoder 的 V12 NoMix ARI 为 `0.4998`，`latent_only` decoder 的 NoMix ARI 只有 `0.1843`。
2. **latent topology 路径没有 abstention/self 分支**。每行 K 个边权被 softmax 强制归一化，哪怕所有邻居都不可靠，也必须把 anchor 拉向邻居均值。当前 flame 的 K=5 边权熵为 `1.60888`，而 `log(5)=1.60944`；最大边权均值仅 `0.20807`，说明 gate 基本是均匀平均，不是逐边选择。兼容 decoder 上叠加该对齐项后 ARI 从 `0.4998` 降到 `0.1844`。

因此，**“mask loss 从 0.7 降到 0.1”不是当前大跌的主因**。它是需要保留的可逆消融变量，但当前证据首先指向 decoder 回归和无条件邻居均值对齐。

## 证据边界与 provenance

### 1. `v12_results_2026-08-03_advantage` 不是实际 V12

`result/v12_results_2026-08-03_advantage/runs.csv` 有 144 条完成记录，表面上包含 `v12_full/v12_nomix`，但其运行摘要和参数来自 legacy `methods/TopoGate/learnable_gate/run_npz.py`：

- 示例 `banknote_authentication__v12_full__seed42/args.json` 的 `variant_name` 是 `learnable_gate_v12_risk_adaptive`，`mix_mode=reliability`，`mask_loss_weight=0.7`，并没有 `lambda_topology`、`decoder_mode` 或 latent alignment 字段；
- 同一目录的 `summary.json` 使用旧的 `gate_summary`/`risk_summary` 契约，且 `dataset` 写为 `adhoc`；
- 真正 V12 的入口是 `methods/TopoGate/V12_latent_topology/run_npz.py`，其 summary 应包含 `topology_enabled`、`lambda_topology`、`mean_final_edge_entropy` 和 `mean_gate_grad_norm`。

所以该目录不能作为真正 V12 的性能证据。它的同批 Full-NoMix 统计（36 个 dataset-seed 配对）是：平均 `ΔARI=-0.001244`，胜/平/负为 `17/1/18`，Wilcoxon `p=0.8828`。这说明该 legacy 批次没有“大幅下降”，只是被错误命名为 V12。跨版本汇总中的 V12 行也必须按此边界解读。

### 2. 当前源码的工程复核

为定位真正回归，使用 `datasets/AHDPC/processed/flame.npz`、seed=42、CPU、80 epochs、batch=256、hidden=128、mask ratio=0.3、StandardScaler、K=5 graph，临时输出写入 `/tmp`，完成核验后清理。所有数值都是单 seed 工程诊断，不是论文级结论。

| 路径 | decoder | topology | mask loss | ARI | 相对当前 V9 NoMix(0.1) |
|---|---|---:|---:|---:|---:|
| legacy V9 NoMix | `[latent, mask_logits] -> Linear` | 关闭 | 0.1 | 0.4764 | 0 |
| V12 NoMix | 兼容 decoder | 关闭 | 0.1 | 0.4998 | +0.0234 |
| V12 NoMix | `latent -> MLP` | 关闭 | 0.1 | 0.1843 | -0.2921 |
| V12 Full | 兼容 decoder | latent alignment | 0.1 | 0.1844 | -0.2920 |
| V12 Full | `latent -> MLP` | latent alignment | 0.1 | 0.0747 | -0.4017 |

当前复核使用的源码 SHA-256 为：

- `methods/TopoGate/V12_latent_topology/model.py`: `44576442f3ef75a6e6d15bc35ee2f501e88c88e8d36982290615087b1958811a`
- `methods/TopoGate/V12_latent_topology/learnable_gate.py`: `84f4498c3cbfb0f1f97054fd4fcfe99c6b2583435db80333e06b7c9f42e05336`
- `methods/TopoGate/V12_latent_topology/run_npz.py`: `ea62509920a40a84414ece74e07fb0d5459f47d6294ba0651d1a0fd364ec6804`

此前记录的历史 smoke 使用过另一份现场源码，因此其中 `V9 NoMix=0.4764/0.4649`、恢复 decoder 后 `0.4534` 等数字不能与本次当前源码逐项混合；两次诊断的因果排序一致：latent-only decoder 会造成大跌，强制邻居对齐会造成额外过平滑。

## 代码级根因

### A. decoder 接口改变了优化问题

当前 V12 `AutoEncoder` 在 `methods/TopoGate/V12_latent_topology/model.py:58-72, 86-97` 提供两个模式：

- `legacy_mask_conditioned`：重建输入是 `cat([latent, mask_logits])`，与原 scMAE 契约一致；
- `latent_only`：重建只接收 latent，并额外使用一个 MLP decoder。

这不是“把拓扑从输入层移到隐层”的必要改动，而是同时改变了 decoder 容量、mask logits 到重构的梯度路径和参数化方式。当前配置文件默认已经恢复为兼容模式（`configs/topogate_v12_latent_topology.yaml`），但任何显式 `--decoder_mode latent_only` 或旧现场代码都会复现 `0.1843` 的 NoMix 回归。

**判断**：高置信度、已由 NoMix 对照隔离。V12 的 decoder 默认必须固定为兼容接口；latent-only 只能作为单独 ablation。

### B. softmax gate 没有“拒绝邻居”的语义

`methods/TopoGate/V12_latent_topology/learnable_gate.py:87-104` 对每行 K 条边执行 softmax，权重和恒为 1。`topology_alignment_loss`（同文件 `:113-128`）计算：

\[
  L_{topo}=\left\|z_i-\sum_{j=1}^{K}w_{ij}\,\operatorname{stopgrad}(z_j)\right\|_2^2.
\]

这里没有 self/null expert，也没有节点级幅度 `alpha_i`。因此：

- 所有候选边都不可靠时，仍然要选出一个均值；
- 边权的均值 `gate_mean` 只是 `1/K`，不能表示拓扑路径是否打开；
- 对 flame 边界点，跨簇邻居的均值会把两个簇的 latent 拉到中间；
- `detach_neighbors=True` 防止邻居 encoder 被当前 batch 拖动，但不会防止 anchor encoder 被错误均值拉平。

当前 full smoke 的诊断为：

| 量 | 兼容 decoder | latent-only decoder |
|---|---:|---:|
| final edge entropy | 1.6088766 | 1.6088055 |
| `log(5)` | 1.6094379 | 1.6094379 |
| mean max edge weight | 0.2080687 | 0.2088275 |
| mean gate gradient norm | 1.5324e-5 | 2.1123e-5 |

梯度非零只能证明“可导”，不能证明“学到了选择”。这里的梯度非常小，且最终权重仍接近均匀，说明 gate 没有形成可靠的边级判别。

### C. topology 对齐启动过早且没有独立 ramp

`run_npz.py:289-309` 的 ramp 是：

```text
ramp = clamp((epoch - topology_warmup_epochs) / topology_warmup_epochs, 0, 1)
```

默认 warmup=5，约在第 10 个 epoch 已达到完整 topology 权重；代码没有 V9 使用的独立 `warmup=20, ramp=10` 结构。此时 encoder 仍在学习 masked reconstruction，拓扑项已经持续改变 latent 几何，容易把早期随机邻域误差固化到表示中。

### D. graph 特征在 flame 上缺乏足够的边区分度

当前 K=5 图的原始逐边特征统计为：

| feature | mean | std | range |
|---|---:|---:|---:|
| similarity | 0.998748 | 0.001725 | 0.981074--1.000000 |
| mutual | 0.856667 | 0.350412 | 0--1 |
| snn | 0.405807 | 0.188477 | 0--1 |
| distance | 0.001252 | 0.001725 | 0--0.018926 |

similarity/distance 在 flame 上几乎是常量；逐边 MLP 实际主要依赖 mutual/SNN 的少数离散模式。再加上 `LearnableGate` 的 `LayerNorm`（`learnable_gate.py:60-65`）是对每条边的 4 维向量独立归一化，绝对相似度和距离尺度会进一步被削弱。该项是中等置信度的机制风险，不能单独解释全部跌幅，但与均匀 edge entropy 相互印证。

### E. K 与旧 V9 的默认协议并不一致

V9 论文匹配配置通常使用 `neighbor_k=5`、`mix_neighbors=4`；V12 CLI 默认 `neighbor_k=10`（`run_npz.py:97`）。V12 每条边都参与 latent mean，因此 K 从 5 增至 10 会扩大均值平滑范围。当前复核为保证可比性显式使用 K=5；若直接使用 V12 默认参数，不能把结果与 K=5 的 V9 对照混写。

## 哪些解释目前不成立

1. **不能说是“V12 逐边 gate 学不会”这么简单**：当前 gate 有非零梯度，但其权重熵表明它学到的是近似均匀分布；问题是缺少 abstention 和可靠的训练目标，而不是 NumPy 采样仍在主路径中。
2. **不能把 `v12_results_2026-08-03_advantage` 的 -0.001244 当成真正 V12 的退化**：该批次是 legacy V9 runner 改名。
3. **不能把单个 flame 的 -0.3154 或 -0.4017 写成跨数据集性能结论**：这是单 seed、工程 smoke，且 decoder/topology 变量刚被隔离。
4. **不能把 mask loss=0.1 视为唯一病因**：兼容 decoder 的 V12 NoMix 在同一复核中达到 0.4998；必须把 mask 权重作为独立三水平消融（0、0.1、0.7）验证，而不是继续同时改 decoder 和 topology。

## 修复优先级与可回退实验

### P0：先恢复可比的 reconstruction backbone

- V12 默认永久使用 `legacy_mask_conditioned`；保留 `latent_only` 但标注为 decoder ablation。
- 增加 NoMix 回归测试：固定 seed、flame、3--5 epoch，只检查兼容 decoder 与 V9 的 loss/embedding 量级不发生数量级变化。
- 在正式实验前固定 `scale_input`、`mask_ratio`、K、hidden、batch、epoch，并把 resolved config 和 source hash 写入每个 run。

### P1：给 topology 路径增加 abstention

推荐的最小形式是 self/null 专家：

\[
  \tilde z_i=\alpha_i z_i+(1-\alpha_i)\sum_j w_{ij}z_j,
  \qquad \alpha_i\in[0,1].
\]

`alpha` 可以由节点风险或一个独立节点 gate 产生；当可靠边质量不足时必须允许 `alpha` 接近 1。等价实现是给 gate 增加 self/null logit，再对 `K+1` 个候选做 softmax。必须报告 self mass、edge entropy、effective neighbor count，而不是只报告 `weights.mean()`。

### P1：延迟并减弱 latent alignment

- 先复现 V9 的 20 epoch warmup + 10 epoch ramp，再测试 `lambda_topology ∈ {0.01, 0.03, 0.1}`；
- 训练中固定 clean-neighbour target 的 stop-gradient，但不要让 topology 在 encoder 尚未稳定时主导几何；
- 先做 `lambda=0`、`topology_enabled=false`、`legacy decoder` 的三路 sanity check，再增加 self/null。

### P2：修正 edge feature 归一化和目标形式

- 对全数据集逐列标准化 edge features，避免逐边 LayerNorm 抹掉绝对可靠性；
- 用带 self fallback 的 pairwise/trust-weighted residual，替代无条件 MSE-to-mean；
- 保留 random-neighbor、fixed-filtration 和严格 NoMix 控制，确认收益不是“任何额外正则”带来的。

## 正式验证门槛

当前真正 V12 仍只有工程诊断，没有论文级性能批次。下一次正式运行至少应满足：

1. 五个代表数据集（至少包含 `flame`、`enron`、`balance_scale`、`spect_heart`、`vehicle`）；
2. seeds `[42, 123, 7]`；
3. 同一数据集/seed 配对：V9 或 V12-compatible NoMix、V12 legacy decoder NoMix、V12 legacy decoder Full、self/null 版本；
4. 同时报告 head/KMeans ARI、NMI、silhouette、edge entropy、self mass、effective neighbor count、topology loss 与 reconstruction/mask loss；
5. `labels_used_during_fit=false`、source SHA-256、K 来源和 decoder mode 全部落盘；
6. 以配对 ΔARI 和置信区间/非参数检验决定 go/no-go，不以单个 flame 的最佳 seed 选主配置。

## 当前判定

V12 的设计目标仍有价值：输入层伪混合已被移除，逐边统计量和 Torch 梯度路径已经实现，且 V12 单测当前为 `4 passed`。但当前实现的 topology loss 不是“选择性去噪”，而是“没有拒绝选项的邻居均值约束”；这正是低维 flame 边界被抹平的直接机制。短期应冻结兼容 decoder、加入 self/null abstention、延后 topology ramp，并重新完成正式多 seed 对照。在这些修复和验证完成前，V12 只能标记为架构诊断分支，不能宣称性能改进。
