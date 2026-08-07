# TopoGate 实验阶段记录（2026-07-27 更新）

> **2026-07-27 重要澄清**：这些实验探索是为了寻找一个合理的机制和搭配来解决 mix 分支消融效果不稳定的问题。**最终只有一份机制会进入论文**，其余全部删除。此文档是过程记录，不是论文叙事。

> **2026-08-04 当前研究定位**：TopoGate 的原型和主要 backbone 是 `scMAE`，总目标是在高维、特征噪声强、同时具有天然稀疏性的单视图数据中获得可靠的聚类效果。V1--V 系列都是围绕同一原型的探索性改良、诊断和消融；版本号只用于追溯，不代表不同应用场景、永久模型边界或已确定的论文主方法。最终论文只从全部探索结果中选择一代作为对外的 TopoGate。数据集选择和 CLM 分层参考 `hj-n/labeled-datasets`、`hj-n/clm` 及 `papers/参考资料/Measuring_the_Validity_of_Clustering_Validation_Datasets.md`，未核验的外部 CLM 映射不得写成正式证据。

---

### V16.1 expanded-count continuation（2026-08-07）

PBMC3K 新增为固定 count 候选。其 H5AD `raw.X` 是可逆 `log1p(count)`，转换为
CSR 后通过 Stage-0（`2638×13714`、high_sparse_bonus、candidate recurrence
`0.285671`、稳定边率 `0.582272`）。clean/compound 在 GPU5/6 并行完成三 seed、
五路 paired readout；clean/compound mean Delta ARI 均为 `0`，固定状态为
`empirical_not_supported`。该结果只扩展候选筛选记录，不改变 gate/support 或
其他数据集的协议。

`Bach` 与 `PBMC_68K` 在同一固定批次中也已完成三 seed、clean/compound、五路
readout，clean Delta ARI 均为 `0.000000`，记录为 `empirical_not_supported`。该段保留
的是早期运行快照；`Shekhar` 与 `PRJNA895163` 后续已经完成，不以中间状态形成论文
结论。

`Shekhar` 已完成完整 30/30 paired 矩阵，clean/compound Delta ARI 均为 `0.000000`，
固定标记为 `empirical_not_supported`；fixed graph 的提升没有被 predictive gate
复现。`PRJNA895163` 与 `hrvatin_geo_maintype_counts` 后续已完成并按固定规则标记；
NormanWeissman 的 Stage-0 已按搜索上限停止。

此前跨四个结果根目录去重后有 33 个完整 paired 数据集，全部为
`empirical_not_supported`。合并后续完成的 `PRJNA895163` 与 `hrvatin_geo_maintype_counts`，
当前临时快照为 35 个完整数据集，文件为
`unpublished-temp/v16_1_global_dedup_summary_current_20260807.json`，`candidate_positive=0`；
Norman 未完成任务不计入统计。

继续执行固定三 seed paired 协议，不根据结果调节门控。`Norman_perturb_e_distance`
和 `Quake_Smart-seq2_Lung` 已完成 30/30 summaries，均为
`empirical_not_supported`，clean Delta ARI 分别为 `-0.000017`、`-0.000094`。
新增的 `subsample_2k`（原始整数 count、2000x53678）通过 Stage 0 后进入 GPU 5；
`hrvatin_geo_maintype_counts` 通过 Stage 0 后已完成 GPU 2 的固定 Stage-1。早期快照中 `Bach`、
`PRJNA895163`、`Quake_10x_Spleen` 的矩阵状态曾分别为 10/30、10/30 和 30/30；Quake 10x 的
clean Delta ARI 为 `+0.000064`、compound Delta ARI 为 `-0.000069`，按固定规则记为
`empirical_not_supported`。`subsample_2k` 随后完成 30/30，clean/compound Delta
ARI 为 `-0.000060`/`-0.000082`，同样标记 `empirical_not_supported`。后续完整结果已在
顶部固定判定记录；当前 V16.1 仍没有 candidate_positive，不改变 gate 或 support 定义。

## 一、实验探索记录

### V16.1 expanded-count Stage-1 执行中（2026-08-06）

已完成 `Arabidopsis_Stereo_seq_leaf`、`CRA002977_1`、`HCA_subsampled_20k`、
`TabulaSapiens_Pancreas`、`tr45.wc` 的完整固定 paired 矩阵。五者均为
`empirical_not_supported`，当前正例数为 0；不因这些负例修改模型机制。`SRP224648`
因单卡峰值内存不足记为 `stage1_incomplete_compute`。`Baron Human`、`Campbell`、
`Human_Pancreas_3`、`Macosko`、`SRP182008`、`Tosches` 仍按同一协议运行，完成后才可
统一更新晋级表。带点号的 word-count 名称解析已修复，`tr45.wc` 的有效重跑通过 22 个
focused tests 所覆盖的输入契约；无训练的首轮状态不作为性能结果。

### V16.1 expanded-count Stage-0/Stage-1 confirmation（2026-08-06）

在已有 210 个 summaries 基础上，继续按同一固定协议并行测试两个未完成候选：
`TabulaSapiens_Pancreas` 与 `CRA002977_1`。输出暂存于
`unpublished-temp/v16_1_stage1_parallel_20260806/`，GPU 3/4、三 seed、clean/compound、五路
readout 均不改变；当前仅部分 clean seed 已落盘，仍不能据此写入
`candidate_positive`，待完整 paired 矩阵结束后统一运行 `summarize_stage1.py`。

随后完成延长 Stage-0 的 `Shekhar`、`Tosches`，并新增本地 HCA、Paul15 和
`Arabidopsis_Stereo_seq_leaf` count 源。Shekhar/Paul15 support 全负；Tosches、HCA 和
Arabidopsis support 非退化，已分别把 Tosches/HCA 放入 GPU4/GPU5 的固定 paired 测试，
Arabidopsis 的三 seed clean/compound 已完成但为 `empirical_not_supported`。Tabula 的
clean 三 seed已完成，compound 仍在运行；所有新输出仍暂存 `unpublished-temp`，未写入正例表。

扩展输入策略已登记 10 个本地 H5AD 原始 count 源并生成 CSR bundle。固定 Stage-0
完成 9 个新增源的无标签审计；`Human_Pancreas_1`、`Bone_Marrow`、`Blood_BoneMarrow`
和 `TabulaSapiens_Pancreas` 的 support 结构非退化，但不将其当作性能预测。

首批正式 Stage-1 在 3 个数据集上使用同一 Stage-A 表征配对五路 readout：
`Blood_BoneMarrow`、`Bone_Marrow`、`Human_Pancreas_1` 均标记
`empirical_not_supported`。GPU 1 外部占用造成的 OOM 被隔离；GPU 2--6 的 90 个正式
readout 已完成。下一阶段继续对固定候选池中尚未完成 Stage-0/Stage-1 的数据进行筛选，
不修改 V16.1 的 gate、support、temperature、thinning 或 `k=20`。

### V16.1 Sparse Count Predictive Gate（2026-08-06，Stage-0 机制审查未进入训练）

V16.1 是冻结 V16 后的独立修复路径，目录为
`methods/TopoGate/V16_1_predictive_graph_gate/` 和 `scripts/V16_1/`。它不再
寻找不可识别的 ARI utility，而是使用 cross-fitted predictive graph support：
view A 建 sparse cosine `k=20` 候选图，view B 评价 per-token log-likelihood
ratio，三次 split 后只保留至少出现两次的边。Stage A 是 topology-disabled
scMAE-compatible sparse count MAE，prototype head 在 KMeans 初始化后冻结；
拓扑只在 assignment readout 中传播。

null/self 是同一个 abstention 分支；全负 support 时 `q_out == q_self`。不使用
learned utility scorer、forced Top-k、edge entropy loss 或 latent mixing。输出语义
固定为 `cluster_probabilities.npy=q_out`、`embedding_final.npy=z_self`。
paired runner 默认 seeds 为 `[42,123,7]`，正式输出根目录为
`result/V16_1/v16_1_paired`；Stage-0 固定 `k=20` 和三次 split。

静态验证 compileall 通过，focused tests **21 passed**。每个 split 同时评分 A→B 与
B→A，逐边 median；Stage 0 已完成 `Campbell`、`Mouse_retina`、`Baron Human`、
`tr45.wc`、`fbis.wc`、`Quake_Smart-seq2_Lung`、`hrvatin` 和
`hrvatin_filtered`：前五者通过理论域证书，但 support 正值率仅
`0.0034%`/`0.0054%`/`0.0253%`/`0.0169%`/`0.0856%`；后三者因 dense member
或 count encoding 无法恢复，记录为 `theory_domain_not_supported`。Campbell/Mouse_retina
的延长窗口产物分别为 `unpublished-temp/v16_1_stage0_campbell_exchange.json` 和
`unpublished-temp/v16_1_stage0_mouse_exchange.json`；初次 360 秒超时只作为计算成本事件保留。
依据预注册边界，V16.1 Stage 1 训练尚未启动，也不为单个数据集调 gate 或 support。

扩展输入域只放宽四项分层指标的硬门槛，真正的硬条件仍是可核验 count 语义、CSR/分块
读取、held-out split 可观测以及样本/标签长度一致。新增 registry 登记了本地 scCluBench
scRNA count 源；目前 `Melanoma_5K` 和 `Guo` 的 Stage-0 recurrence/support 结构最有
希望，`Limb_Muscle`、`worm_neuron_cell` 的 support 仍全负。所有新增数据均保持
`stage0_candidate` 或计算未完成状态，不能据此宣称正例。GPU 不可用时不启动 CPU
Stage-1；等候可用物理 GPU 1--6 后，按同一表征一次、五路 paired readout、三 seed 和
固定 compound stress 运行。

### V16 Predictive Graph Gate（2026-08-06，Stage-1 锚点 restricted no-go）

V16 是围绕 Campbell/Mouse_retina 历史正向线索建立的独立计数域分支。它以
held-out count thinning 估计 predictive graph support，只在 assignment
readout 中做拓扑传播；理论域外数据标记为 `theory_domain_not_supported`，
理论证书通过但 paired 结果或消融失败的数据标记为
`empirical_not_supported`，不通过重新调 gate 挽救。

代码、launcher 和理论边界已完成协议修正：门控直接使用 raw predictive
support，NPZ 支持分块 memmap→CSR，assignment 输出与 embedding 输出分离，
paired runner 固定五路 readout，并冻结 clean/compound 记录契约。最小
compile/test 为 **12/12**。此前 paired runner 曾把四个非主 readout 写到缺少
condition 的目录，导致 clean/compound 覆盖；该错误已修复，错误批次未纳入证据。

Stage 0 对 `Campbell` 和 `Mouse_retina` 均通过计数域证书，但 candidate recurrence
仅为 `0.472`/`0.267`，support 正值率仅为 `0.153%`/`0.063%`。修正后的 Stage 1
按 `[42,123,7]`、五路 readout 和固定 compound stress 完成 60 个 run，产物暂存
`unpublished-temp/v16_stage1_anchors_20260806_fixed/`。V16 clean paired delta 分别为
`-0.000607` 和 `-0.000033`，compound delta 分别为 `0.000000` 和 `+0.000961`；
fixed graph 虽高于 self-only，但 V16 的 support 几乎全部 abstain，且
`shuffled_support` 没有被破坏出机制增益。两个锚点均标记
`empirical_not_supported`，因此按计划暂停候选池确认和正式五数据集扩展，不重新
调 gate、temperature、k、thinning 或 support 定义。此前 fbis exploratory
仍按原规则为 `empirical_not_supported`，不被本次结果改写。

### V15 Counterfactual Gate（2026-08-04，Stage-1 restricted no-go）

V15 是独立的探索路径，研究问题是：在稀疏、高维、特征噪声和图污染同时
存在时，能否用 detached single-edge counterfactual utility 学习 topology
abstention。实现目录为
`methods/TopoGate/V15_counterfactual_gate/`，launcher/审计脚本位于
`scripts/V15/`；V2--V13 和外部 baseline 保持冻结。

当前实现与验证边界：

| 项目 | 当前状态 | 证据 |
|---|---|---|
| sparse MAE、union graph、detached utility、null sparsemax | 已实现 | V15 focused tests 48 passed；cnae9 smoke 完整产物 |
| self/uniform/exact/local-consensus/learned/forced-topk/shuffled/output-disabled 对照 | 已实现 | `scripts/V15/run_formal.py` dry-run 与小矩阵 |
| Stage-0 固定数据 manifest | exploratory 已完成 | `unpublished-temp/v15_manifest_fixed16.json`，CLM 未核验时统一 `CLM-unranked` |
| Stage-1 utility/candidate 门槛 | 未通过短 panel | 当前源码重跑 utility AUROC 达标 2/6；candidate recall 中位数约 0.70 |
| 边界/低密度/离群拒绝 | 未证实 | 受控集 null-AUROC 均为 0.5 |
| graph pollution abstention | 局部、非严格单调 | cnae9 0/0.5/1.0 的 null mass 为 0.885/0.884/1.000，端点上升但中间点略降 |
| 三证书独立审计 | teacher/utility 证据不完整 | `scripts/V15/audit_stage1b_certificates.py`：7/7 graph 后验可算，7/7 utility 仅 in-sample 可算；teacher、held-out utility、independent gain 均 0/7 |
| 正式 Stage-3 多 seed | 暂停 | 按计划 Stage-1 失败即暂停，不写性能主结论 |

修复 readout、local-consensus dispatch 和 YAML 漂移后，最小 paired matrix 仍显示：
clean 条件下 exact/local readout 只有数据集局部增益；compound 条件下 coherent
错误 donor 会让 local-consensus 继续高 edge mass，learned scorer 的 null mass
可为 0，且 held-out utility AUROC 约 0.50--0.54。因此 V15 当前仍是“机制实现 +
可证伪失败边界”的探索分支，不能进入正式多种子矩阵。下一步必须先处理
coherent graph pollution 的外生证书和 scorer 的零边界校准，再考虑 `[42,123,7]`
正式配对运行；不再重复全量 Stage-0/1 审计。

Stage-1B 进一步说明，当前不能把三件事混为一件事：EMA teacher 的存在不等于
teacher 正确；候选图的 post-hoc label purity 不等于训练时可获得的图证书；同一
run 的 `utility_hat` 与 `utility_target` 的 AUROC 不等于 held-out utility 或
独立聚类收益。下一阶段必须先扩展产物契约，保存 teacher 的跨视图/时间诊断和
逐边反事实 downstream gain，再决定是否修改 utility 或进入正式 benchmark。

### 当前所有实验的定位

下表保留 2026-07-27 的历史判断。“进入论文”等文字是当时的探索记录，受上面的 2026-08-04 当前定位覆盖；最终论文归属尚未预先指定为 v2、v9 或其他某一代。

| 代号 | 内容 | 状态 | 2026-07-27 当时记录（非当前最终归属） |
|------|------|------|---------|
| **v2 (LearnableGate)** | 将 4 个 β 从静态超参改为可学习参数，加入 warmup/ramp schedule | **当时最优** | → 当时拟进入论文 |
| v3 | v2 + learnable_gate_max（让 gate_max 也可学） | 效果不稳定，已废弃 | → 删除 |
| v3_tune | v3 的 lr 和 gate_max 超参搜索 | 为 v3 服务的调优 | → 删除 |
| v3_best | v3 + enhanced topology features | 跟随 v3 废弃 | → 删除 |
| v5_components | Gumbel-Sigmoid STE mask + 1-γ edge reliability |写了未部署 | → 删除 |
| v5_main | v5_components 的独立实验脚本 | 跟随 v5_components | → 删除 |
| v6_latent_mix | 在 latent space 做 neighbor mixing | smoke 失败 | → 删除 |
| v7_cross_attn | cross-attention latent mixing | smoke NO-GO | → 删除 |
| Direction B (BinaryRouter) | Gumbel-Softmax 二元硬路由 | smoke 彻底失败 | → 删除 |

### 核心问题：mix 分支消融效果不稳定

**问题描述**：StaticGate（v1）的 8-variant ablation 显示 neighbor mixing 的贡献**完全数据集依赖**：
- enron / iris 上 nomix >> full（+0.108 / +0.132）→ mixing 有害
- Mouse_retina / har 上 full >> nomix（+0.004 / +0.099）→ mixing 有益

**这说明静态的 mixing 超参（neighbor_k=5, mix_mode=topology）对不同数据集是反的。**

v2 (LearnableGate) 的解决思路：用可学习的 β 让模型自己决定"什么时候用邻居，什么时候不用"。

---

## 二、历史最优方案记录：v2 (LearnableGate，2026-07-27 判断)

### 2.1 核心机制

将 StaticGate 中固定的 β = (1.0, 1.0, 2.0, 1.0) 改为 `torch.nn.Parameter`（4 个可学习标量），通过 warmup + ramp schedule 逐渐开启：

```python
# epoch <= warmup: gate = static_gate (v1 行为)
# epoch > warmup: gate = (1-t) * static_gate + t * learnable_gate
#   其中 t = max(0, (epoch - warmup) / ramp)
```

### 2.2 Multi-seed 验证结果（15 datasets × 3 seeds × 2 variants = 90 runs）

历史来源：`result/learnable_gate_smoke/multiseed/comparison.csv`（产物已按 2026-08-03 smoke 生命周期规则清理；本文件仅保留过程记录，不是当前可复核证据）

|| 数据集 | StaticGate | LearnableGate | Δ | 解读 |
|---|---|---:|---:|---:|---|
| ✅ | enron | 0.7236±0.034 | 0.7681±0.063 | **+0.044** | 自适应有效 |
| ✅ | har | 0.4985±0.043 | 0.5268±0.037 | **+0.028** | 自适应有效 |
| ✅ | Campbell | 0.0855±0.044 | 0.1214±0.067 | **+0.036** | 自适应有效 |
| ✅ | Mouse_retina | 0.9270±0.019 | 0.9374±0.003 | **+0.011** | 自适应有效 |
| ✅ | cnae9 | 0.2980±0.014 | 0.3003±0.016 | **+0.002** | 持平 |
| ✅ | reuters | 0.2007±0.008 | 0.2012±0.013 | **+0.001** | 持平 |
| ✅ | Quake | 0.1891±0.078 | 0.1906±0.006 | **+0.002** | 持平 |
| ≈ | breast_cancer | 0.8854±0.012 | 0.8854±0.008 | **0.000** | 持平 |
| ≈ | iris | 0.6530±0.017 | 0.6530±0.017 | **0.000** | 持平 |
| ≈ | mammographic_mass | 0.3651±0.009 | 0.3651±0.006 | **0.000** | 持平 |
| ≈ | spambase | 0.6400±0.018 | 0.6317±0.027 | **-0.008** | 持平 |
| ❌ | sms_spam_collection | 0.8247±0.013 | 0.8082±0.026 | **-0.017** | 略输 |
| ❌ | ISOLET | 0.5167±0.035 | 0.5070±0.007 | **-0.010** | 略输 |
| ❌ | hrvatin_filtered | 0.3838±0.160 | 0.3439±0.095 | **-0.040** | 退化 |
| ≈ | first-order-theorem-proving | 0.0242±0.003 | 0.0195±0.004 | **-0.005** | 持平 |
| | **OVERALL** | **0.4810** | **0.4840** | **+0.003** | 整体略胜 |

### 2.3 β 自适应证据

|| 数据集 | β_mutual | β_snn | β_perturb | β_uncertainty | 学到的模式 |
|---|---:|---:|---:|---:|---|---|
| Mouse_retina | +1.245 | +2.351 | -1.561 | 0.0 | mutual/snn 主导 |
| enron | -2.720 | -3.718 | +4.103 | 0.0 | **perturb 主导**（与 v1 默认 2.0 同向更强） |
| sms_spam | +0.790 | +0.775 | -0.794 | 0.0 | 对称 mutual/snn |
| har | -0.154 | -0.036 | +0.019 | 0.0 | **β 接近 0**（几乎不用邻居） |
| breast_cancer | +0.551 | +0.529 | -0.560 | 0.0 | 对称 mutual/snn |

**关键观察**：5 个数据集学到 5 种完全不同的 β 模式，证明 LearnableGate 真的在自适应。

---

## 三、为什么其他方案被废弃

| 方案 | 废弃原因 |
|------|---------|
| v3 (lgm) | learnable_gate_max 让 gate 飞向 0.98 饱和，enron 退化 -0.10 |
| v5_components | Gumbel-STE 写了但未部署；即使部署，1-seed avg Δ=-0.068 |
| v6 latent_mix | har 上 Δ=-0.046（负向），latent mixing 在 core 数据集上无效 |
| v7 cross_attn | NO-GO：enron -0.128, cnae9 -0.063，2/6 数据集严重退化 |

---

## 四、下一步关系图

```
问题：mix 分支消融效果不稳定（有时有益，有时有害）

诊断 → v2 LearnableGate：让 β 可学习，模型自己决定何时用邻居
    │
    ├── 机制确认 ✅：15 ds × 3 seeds Δ=+0.003，7/15 正向，4/15 略输
    │
    ├── β 自适应确认 ✅：5 种完全不同的 β 模式
    │
    └── 根本问题：
            "连续 gate 永远无法精确关断"
            - 即使 gate→0，mixed = (1-g)·anchor + g·neighbor 中 g>0
            - enron 上 full(g=0.075) = 0.768 vs nomix = 0.875，差 0.107
            - gate 只能压制，不能关断

方向 B：Binary Router（2026-07-27 实施）
    │
    ├── 核心思想：用 Gumbel-Softmax 做离散硬路由
    │   r = GumbelSoftmax(logits), logits = β · stats
    │   x' = r * mixed + (1-r) * anchor
    │   → r=0: x'=anchor（精确等价于 nomix）
    │   → r=1: x'=mixed（拓扑感知的邻居混合）
    │
    ├── 温度调度：soft(epoch 1-20) → hard(epoch 31+)
    │   - 高温：soft gradient ≈ v1 行为
    │   - 低温：hard sample {0,1}
    │
    └── 预期：
            - 在 nomix>full 的数据集（enron, iris），router 学到 r≈0
            - 在 full>nomix 的数据集（har, Campbell），router 学到 r≈1
            - 在 Mouse_retina 上，两种策略接近（ARI 差异仅 0.004）

v2 vs Direction B 对照：
    - v2 LearnableGate：gate ∈ [0, 0.15]，始终有 residual mixing
    - Binary Router：r ∈ {0,1}，硬关断或硬开启，无残留
```

---

## 五、待解决的核心问题

> 如何让消融实验能说通？
>
> StaticGate 8-variant ablation 证明了 neighbor mixing 在某些数据集上有效（Mouse_retina +0.004, har +0.099），在某些数据集上有害（enron -0.108, iris -0.132）。v2 的 LearnableGate 让模型自适应选择，但整体 Δ 只有 +0.003。
>
> **Direction B 的叙事**：不是"v2 替换了 v1"，而是"连续 gate 有结构性缺陷——无法精确关断。Binary Router 用离散路由解决了这个问题，让模型在每一层都能做最优选择。"

---

## 六、方向 B 实验记录（2026-07-27）

### 实验编号：Direction B
### 核心机制：BinaryRouter + Gumbel-Softmax

| 文件 | 内容 |
|------|------|
| `methods/TopoGate/learnable_gate/binary_router.py` | BinaryRouter 模块 |
| `methods/TopoGate/learnable_gate/configs/binary_router.yaml` | variant 配置 |
| `methods/TopoGate/learnable_gate/mixing.py` | 新增 `make_pseudo_batch_binary()` |
| `methods/TopoGate/learnable_gate/run_npz.py` | 集成 binary router 到训练循环 |
| `scripts/run_binary_router_smoke.py` | smoke test 脚本 |

### Smoke test 设计

3 datasets × 3 variants × 1 seed = 9 runs

| 变体 | gate_mode | mix_mode | 预期 |
|------|-----------|----------|------|
| binary_router | binary | reliability | 新机制 |
| learnable_gate_sched | learned | reliability | v2 baseline |
| nomix | learned | none | 无邻居混合 |

### 关键指标

- enron: nomix=0.875 >> full=0.768 → BinaryRouter 应该 ≈ nomix（学到关断）
- har: nomix=0.458 << full=0.558 → BinaryRouter 应该 ≈ full（学到开启）
- Mouse_retina: nomix≈full（边界情况）

### 最终归属

> Direction B **已失败**（2026-07-27）：BinaryRouter 在所有 3 个数据集上崩溃。根本原因：二元硬决策产生离群梯度，破坏 MAE encoder 训练动态。不适合此任务，已删除所有相关代码。

### 失败数据（2026-07-27）

| Dataset | binary_router | v2(learnable_gate) | nomix | Binary vs nomix |
|---------|-------------|---------------------|-------|-----------------|
| enron | **0.052** | 0.897 | 0.878 | -0.826 (!!!) |
| har | 0.335 | 0.353 | 0.425 | -0.090 |
| Mouse_retina | 0.681 | 0.933 | 0.926 | -0.245 |

---

## 2026-08-07 V16.1 expanded-count confirmation: hrvatin

`hrvatin_geo_maintype_counts` 是满足四项 high-sparse bonus 的 raw-count 数据集。
固定协议完成 clean/compound、三 seed、五路 readout 共 `30/30` 个产物。候选图的
后验 purity/recall 很高，但 held-out predictive support 几乎全为负，V16.1 gate 的
平均 null mass 为 `0.999118`，clean Delta ARI 相对 self-only 为 `-0.000309`，而
fixed graph 明显更高。因此按预注册规则标为 `empirical_not_supported`，该结果用于
证明“候选边召回”和“可分离的 predictive support”不是同一条件；不通过调整 gate
挽救该数据集。正式汇总位于
`result/V16_1/expanded_count_stage1_20260807/promotion/hrvatin_geo_maintype_counts.json`。

Norman Stage-0（`111445 x 33694`，约 `361582621` 个 CSR 非零项）在约 4 小时 45 分钟
内未完成固定 sparse-cosine 审计。由于 35 个完整候选均未产生正例，已达到预注册搜索
上限，任务停止并记为 `stage0_incomplete_compute`；不将其作为理论域或模型性能结论，
V16.1 数据扩充到此关闭。

---

## 2026-08-07 V 系列失败复盘后的研究方向冻结

完整复盘见 [`V_SERIES_FAILURE_RETROSPECTIVE.md`](V_SERIES_FAILURE_RETROSPECTIVE.md)。
当前不再把 scMAE 视为 TopoGate 的默认论文主干：scMAE 的主目标是 masked anchor
reconstruction，而拓扑门控需要优化可解释的 co-assignment/graph objective，二者在
V1--V16 中长期通过弱 pseudo branch 耦合，造成目标错位、自证循环和大量不可归因的
协议变化。

下一阶段采用以下顺序：

1. 先固定 sparse count/topic-mixture 或 noisy union-of-subspaces 的生成命题；
2. 静态审查候选主干是否让 topology、gate 和 final assignment 共享同一对象；
3. 优先设计 candidate-restricted robust sparse self-expression，令稀疏系数同时
   承担边权、精确零门控、affinity 和最终 readout；
4. 在最小可证伪原型通过前，不添加 EMA teacher、utility scorer、动态图刷新、
   多距离可靠性、attention 或新的 gate 形式；
5. 若自表达假设不适用，再评估污染图概率混合模型；普通图对比聚类暂列第二候选，
   不能直接因为其名称含 graph/neighborhood 就视为目标一致。

该段是研究计划和停止规则，不构成新模型或性能结果。

---

## V17 topology-native reference phase（2026-08-07）

V17 已建立独立 code path，但当前阶段只完成机制 reference solver，不启动真实数据
benchmark。唯一主变量为 candidate-restricted sparse relation `C`：其支持集是 gate，
`A=|C|+|C.T|` 是唯一 affinity，normalized spectral readout 是唯一输出。scMAE、EMA
teacher、predictive utility/support、forced Top-k、独立 gate probability 和 encoder
latent KMeans 均不在该路径中。

当前阶段已完成：

1. sparse-safe input adapter 与多个 sparse random projection 视图；
2. blockwise multi-projection candidate union，不物化完整 `n x n` 距离矩阵；
3. group-Huber + elastic sparse self-expression 与 exact-zero proximal gate；
4. same-`C` affinity、谱读出、degree-zero abstention 和标签/K 隔离产物契约；
5. compileall、两个 runner `--help` 和 `11` 个 focused tests。

进入最小真实机制验证前必须保持固定顺序：先只读检查 Campbell、
`hrvatin_geo_maintype_counts`、Mouse_retina、enron 和一个 count/text 数据的输入语义；
随后只比较 candidate affinity、single-view self-expression、shuffled candidate/C 和
V17-reference。若 `C` 大面积全零、门控后图纯度不升或同一 affinity 的谱输出不改善，
立即停止 unrolling/encoder 开发；不得通过新增 utility 或逐数据集调 gate 挽救。

当前未实现的 spectral feedback 与 learnable unfolded layers 是后续条件项，不属于已经
完成的 V17，也没有任何 V17 性能结论。
