# ToPoGate 全项目、数学对象与 TDA 边界审计

**日期**：2026-08-03
**范围**：项目规则、目录边界、TopoGate V9/V10/V11 及其可逆实验、外部 baseline 协议、正式结果产物、参考书中的拓扑学/机器学习/数学分析内容，以及结果存储和 smoke 生命周期。
**审计入口**：`AGENTS.md`、`.cursor/rules/*.mdc`、`CHANGELOG_errors.md`、`result/RESULTS_SUMMARY.md`、`papers/references/INDEX.md`、`CHANGELOG_lit.md`、CodeGraph 索引和当前磁盘产物。

> 本报告是研究仓库审计和方法分析，不是论文正文。报告可以出现 V9、V10、V11 及历史变体；论文对外叙事仍应只保留最终确定的方法名称。报告中的结果只有在对应文件当前存在时才属于可复核证据；已经清理的 smoke 数值只保留为历史工程记录。

## 1. 执行摘要

1. Cursor 规则已经有 Codex 对应入口：根目录 `AGENTS.md`。三份 `.cursor/rules/*.mdc` 保留给 Cursor 使用，没有删除或制造第二个互相漂移的事实源。本轮进一步把用户指定的结果存储规则固化到 `AGENTS.md`：逻辑路径为 `source-repository/result`，实际软链接目标为 `external-result-storage/result`，禁止在根目录堆积结果，短 smoke 完成验证后清理并同步文档。
2. 项目不是单一 Python 包，而是以论文证据和可审计实验为中心的研究仓库。`methods/TopoGate/learnable_gate/` 是 legacy LearnableGate/V9，`methods/TopoGate/v10_reliable_graph/` 是独立 V10，`methods/TopoGate/V11/` 是独立 V11；外部 baseline 默认冻结。
3. 当前模型使用的是 topology-inspired graph structure，而不是严格意义上的 persistent homology 或完整 TDA。源码中存在 PCA/kNN、mutual neighbor、SNN、图连通性、图稳定性和边可靠性，但没有完整的 filtration -> simplicial complex -> boundary operator -> homology -> persistence diagram/barcode 管线。
4. V9 的有效事实是数据集依赖的局部相对优势，不是普遍拓扑优势。论文匹配协议下，V9 相对 AHDPC 只有 `spect_heart`、`balance_scale`、`landsat` 三个 ARI 正差值，胜/平/负为 `3/1/20`，平均 Delta ARI 约 `-0.1715`。V9 相对 HDPC 为 `5/1/18`，平均 Delta ARI 约 `-0.1530`。
5. V9 topology mixing 的独立贡献没有显著性证据：Full 宏平均 ARI `0.278302`，NoMix `0.262946`，差值 `+0.015356`，Wilcoxon `p=0.3905`，配对 t 检验 `p=0.1417`。因此不能把 V9 相对外部 baseline 的差异直接归因于 topology mixing。
6. V12、V13、V14 已经给出清晰的 no-go 边界：V12 Full-NoMix 约 `-0.001244`，`p=0.8828`；V13 head ARI 差约 `-0.000238`，`p=0.8314`；V14 差值 `+0.004373`，Wilcoxon `p=0.8139`，target gate 均值约 `0.006276`。V14 的机制可运行，但性能不能晋级为主方法。
7. CLUBench 131 数据集批次是单 seed=42 的工程/对照证据，不是论文级多 seed 结论。V9 相对 AHDPC 的 ARI 胜/平/负为 `105/2/24`，平均 Delta ARI `+0.1396`；这个结果必须与 24 数据集、三 seed、AHDPC 匹配预处理的另一套结论分开叙述。
8. TDA 方向已经完成一个范围受限的 V11 pilot：固定 raw-kNN 稀疏 1-skeleton 上的 H0 union-find persistence 作为 detached prior；它不是 dense VR/H1 实现。五个预注册数据集、五种 variant、三个 seed 共 75/75 完成，H0、fixed-filtration 和 random prior 均未显示独立聚类收益，因此在该固定协议内判定为性能 no-go，仅保留为诊断。

## 2. Cursor 到 Codex 的规则转换

### 2.1 对应关系

| Cursor 规则 | Codex 入口 | 已保留的核心约束 |
|---|---|---|
| `.cursor/rules/project-structure.mdc` | `AGENTS.md` 的目录边界、任务启动、数据/K 协议、实验纪律、GPU、文档维护 | 软链接布局、先读错误日志和事实表、`K=int(np.unique(y).size)`、GPU 0/7 禁用、结果和文献可追溯 |
| `.cursor/rules/model-integrity.mdc` | `AGENTS.md` 的模型完整性、baseline 冻结、版本隔离、变更记录和回退 | 不用简化模型绕过错误，不替换损失/图/门控/聚类头，TopoGate 改动可逆并有验证，外部方法不私改 |
| `.cursor/rules/literature-management.mdc` | `AGENTS.md` 的文献检索、归档和引用 | PDF -> `INDEX.md` -> `CHANGELOG_lit.md` -> 论文引用，未下载文献不写成已使用，外部数值不冒充统一协议结果 |
| 用户本轮新增存储要求 | `AGENTS.md` 的“结果存储与临时产物” | 所有输出进入 `result` 软链接目标，根目录不堆积，smoke 清理后同步事实表 |

### 2.2 规则优先级

- 用户明确要求高于仓库默认规则，但“快速跑通”不能解释为允许改算法、伪造数据或绕过验证。
- 当前源码、配置、实际产物、测试和 hash 高于旧 changelog。`CHANGELOG_errors.md` 是错误线索，不是实验存在性的证明。
- CodeGraph 只负责定位符号和调用关系，不能替代对当前磁盘源码和产物的复读。当前索引状态正常，状态检查显示 4,331 个文件、索引已同步。
- V11 不使用可变 V9 runner；V10 不顺手修改 V1-V9；baseline 不为了比较结果而改变算法。

### 2.3 结果存储执行情况

本轮已完成以下整理：

- 根目录没有独立的 `*results*` 或 `*smoke*` 结果目录；`result` 仍是 `external-result-storage/result` 的软链接。
- 正式 V9、V12、V13、V14 目录已经位于结果盘：`result/v9_results_2026-08-02/`、`result/v9_results_2026-08-02_paper_preprocess/`、`result/v9_results_2026-08-02_advantage_ablation/`、`result/v12_results_2026-08-03_advantage/`、`result/v13_results_2026-08-03_advantage/`、`result/v14_results_2026-08-03_advantage_5ds/`。
- V11 多种子候选批次已经从 `/tmp` 迁入 `result/V11/`；旧多种子探索放入 `result/V11/legacy_2026-08-03/`。
- 已清理明确的 LearnableGate、V6、V7、HVF、AHDPC、V10、V11 smoke 目录，以及 `unpublished-temp/topogate_v11_semantic_*`、V10 smoke/verify 和 V3/V9 临时工程目录。
- 文档中仍保留的旧 smoke 路径均已改成“历史产物已清理”或“历史来源”，不能再被解释成当前权威输出。

## 3. 项目全局结构和版本边界

### 3.1 目录责任

| 目录 | 责任 | 审计结论 |
|---|---|---|
| `methods/TopoGate/learnable_gate/` | legacy LearnableGate/V9、图构建、MAE、KMeans readout | 当前 V9 对照主线，不能因 V10/V11 改动而静默改变 |
| `methods/TopoGate/v10_reliable_graph/` | dynamic reliable graph 独立实现 | 有 consensus graph、edge gate、assignment JS 和动态刷新；保留 feature-only/fixed-graph 控制 |
| `methods/TopoGate/V11/` | self/null expert、动态图、Student-t mixture、EMA teacher | 独立于 V9 runner，具备配置校验和回归测试 |
| `methods/TopoGate/static_gate/` | V1 风格冻结消融 | 用于解释 topology/mixing 组成，不是当前主方法 |
| `baseline/` | AHDPC、HDPC、CLUBench 和其他外部/clean-room 方法 | 外部实现默认冻结，状态以 `PROVENANCE.md`、`STATUS.md`、README 和 registry 为准 |
| `scripts/` | 训练、批量实验、汇总、可视化 | 输出路径、GPU pool、K 协议必须与 runner 一致 |
| `datasets/` | 数据软链接、manifest、处理后 NPZ | 数据源、shape、K、hash 和 unresolved 状态必须可追溯 |
| `result/` | 结果软链接和事实表 | 只放产物、汇总、分析和配置，不把结果复制进源码目录 |
| `papers/` | 论文、图表、参考资料和引用链 | 文献引用必须经过 PDF/INDEX/阅读链 |

### 3.2 证据分级

| 等级 | 可证明内容 | 不可证明内容 |
|---|---|---|
| 编译/import/help | 语法、导入、CLI 契约 | 训练正确性和性能 |
| 单 seed/短 epoch smoke | 梯度、输入、图刷新、输出契约 | 性能提升、泛化和统计显著性 |
| 3 seed 配对消融 | 在明确数据集和协议上的均值、方差和方向 | 跨数据集普遍优越，除非覆盖和统计协议足够 |
| 5+ 数据集、3 seed、预注册控制 | 受限范围的主对照和机制证据 | 所有领域、所有数据分布上的理论保证 |
| 131 数据集单 seed | 大规模工程/对照画像 | 论文级稳定性和多 seed 显著性 |

## 4. TopoGate 源码对象与数学对象映射

### 4.1 V9: 固定图、可靠性统计、节点门控和 MAE

设输入矩阵为 `X in R^(n x d)`。V9 的主路径可以抽象为：

```text
X
 -> scaling / PCA projection
 -> fixed kNN graph G=(V,E)
 -> mutual/SNN/distance/perturbation statistics
 -> edge reliability and node gate
 -> topology-weighted pseudo neighbor view
 -> masked autoencoder reconstruction
 -> embedding
 -> KMeans readout
```

源码中的 `build_pca_knn_graph` 和 `compute_edge_reliability` 位于
`methods/TopoGate/learnable_gate/neighbor_graph.py`。它们构造的是有限样本上的邻接关系和边特征，不是抽象拓扑空间上的开集族。V9 的“拓扑”至少包含四个容易混淆的层次：

1. **图结构**：PCA 后的 kNN 邻接、mutual neighbor、SNN 和连通分量。
2. **边可靠性**：相似度、互惠性、共享邻居、距离/局部统计及扰动风险的组合。
3. **节点门控**：对样本或 pseudo branch 的 topology mass 进行控制。
4. **表征目标**：masked reconstruction 和 neighbor mixing 改变 embedding，最终再由 KMeans 输出标签。

这四层不能在论文中压缩成一个“拓扑损失”。特别是图的可靠性、样本 gate、pseudo reconstruction 权重和最终 KMeans readout 分别属于不同的计算对象。

从机器学习角度，V9 的关键风险是目标错配：重建损失降低并不等价于簇边界改善；一个 gate 通过关闭邻居路径可以减少错误混合，却不代表模型学会了更好的拓扑。`beta` 的跨数据集异号和绝对值变化说明参数确实会适应，但这不是 topology contribution 的独立因果证据。

### 4.2 V10: consensus graph 和 edge-level reliable graph

V10 位于 `methods/TopoGate/v10_reliable_graph/`，是独立实现。其主要对象为：

- 输入图 `G_x`：由原始输入/PCA 表征构建；
- EMA latent 图 `G_z`：由 teacher/EMA 表征周期性构建；
- consensus candidate graph：保留 `G_x union G_z` 的候选边；
- `EdgeGate`：对每条候选边给出 reliability，而不是只给一个全局 node gate；
- assignment consistency：在可信边上约束分配的 Jensen-Shannon 一致性；
- dynamic refresh：图是离散周期刷新，刷新间隔内的权重仍在 Torch autograd 中。

V10 的工程优点是把 input graph、latent graph、edge gate、prototype readout 和 feature-only 控制拆开，并修复了 duplicate-row/tie 情况下“最近邻第一项必为 self”的错误假设。工程上还支持 exact/FAISS HNSW 后端和显式不使用 GPU 0/7 的 runner 约束。

V10 的理论边界仍然存在：图刷新是交替的离散操作，不是端到端连续可微的拓扑优化；EMA graph stability 是经验统计量，不等于迭代收敛定理。V10 需要 full、fixed-graph、feature-only 三组在同一预处理、同一 seed 集合和同一 K 协议下比较。

### 4.3 V11: self/null expert、概率聚类头和语义 gate

V11 的主要类位于 `methods/TopoGate/V11/model.py`，训练器位于 `methods/TopoGate/V11/trainer.py`。抽象流程为：

```text
raw X -> raw PCA graph
      -> encoder f_theta(X)
EMA teacher f_bar(X) -> latent graph
raw graph union EMA-latent graph -> candidate edges

student/null expert + candidate edge experts
      -> topology mixture weights
      -> mixed or assignment-residual path
      -> reconstruction + cluster + graph objectives
      -> Student-t mixture responsibilities
      -> head prediction and optional KMeans diagnostic
```

数学上，V11 用一个 self/null expert 表示“不使用拓扑”，其权重可以写成 `a_i,self`；所有 edge expert 的总质量约为 `1 - a_i,self`。这比 V9 同时维护 node gate、edge gamma、sample weight 更容易解释，但仍需监控 gate 是否退化为“默认关闭”。

`StudentTMixtureHead` 产生 soft responsibility，主 prediction 来自概率聚类头；KMeans embedding readout 是独立诊断，不能用真值标签选择二者。EMA teacher 用于稳定表征和 assignment target，但 EMA 数值稳定不自动提供统计一致性。

`counterfactual_semantic_target` 使用 detached teacher/reference 计算边加入后的 reconstruction/assignment 帮助，并只对正向改善产生 topology target。`minimum` combiner 是保守的双通道 abstention：拓扑开门程度受 reconstruction-help 和 assignment-help 中较弱的一项限制。这可以降低错误开门，但也会让 gate mass 很小，必须同时报告 coverage、target calibration 和 no-topology 对照。

V11 的标签边界是正确方向：训练器和 candidate graph 不读取 `y`；runner 可以从 `np.unique(y)` 得到 benchmark K，并把 `benchmark_oracle_from_y` 写进 summary；无标签运行必须显式给 `n_clusters`。这一区分要保持到所有新 variant。

## 5. 参考书阅读后的数学结论

### 5.1 《基础拓扑学及应用》

本书目录明确区分了：

- 第 1 章拓扑空间与连续映射；
- 第 4 章连通性与紧致性；
- 第 5 章同伦与基本群；
- 第 8、9 章单纯复形、单纯同调及其应用。

这些章节给出的严格数学对象不是“邻居关系”本身，而是空间、映射和在变形下保持的不变量。特别是单纯同调的标准路线是：从单纯复形建立链群和边缘同态，再取同调群并得到 Betti 数等不变量。

与源码对照后得到的边界：

- kNN、mutual kNN、SNN 和图连通分量是有限图上的组合结构，可以称为 topology-inspired locality 或 graph-topological signal。
- PCA、标准化、距离度量和 `k` 改变后，邻接关系可能改变；因此当前图结构通常不是输入空间拓扑不变量。
- 当前代码没有 filtration 参数 `alpha`、Vietoris-Rips/Čech complex、boundary matrix、homology group、persistence diagram 或 barcode。因此不能把当前方法直接写成 persistent homology 方法。
- 图中一个 cycle 也不能自动等价于 H1 特征：若没有 2-simplex 和边界算子，就无法区分“填充的三角形边界”和真正的 1 维洞。

本书对项目最有价值的启示是“定义先于类比”：如果论文使用 topology 一词，必须明确是邻域图结构、拓扑不变量、流形假设，还是 TDA persistence。当前最稳妥的术语是 topology-inspired reliable neighborhood graph，而不是 persistent-topology-aware clustering。

### 5.2 Bishop《Pattern Recognition and Machine Learning》

本书相关章节包括概率分布、模型复杂度和过拟合、混合模型与 EM（第 9 章）、连续潜变量和 probabilistic PCA（第 12 章）、模型组合（第 14 章）。对当前项目的直接映射如下：

| 书中对象 | 项目对象 | 需要保持的边界 |
|---|---|---|
| 无监督学习 | 仅以 `X` 拟合，`y` 只用于 benchmark K/后验指标 | 不能用标签选图、gate、epoch、seed 或 variant |
| 局部邻域和维度灾难 | PCA 后 kNN、HVF、输入尺度和高维 latent | kNN 质量随维度、尺度和预处理改变，不是稳定事实 |
| mixture model / responsibility | V11 Student-t mixture head | responsibility 饱和、温度、尺度和维度归一化必须诊断 |
| EM/潜变量思想 | V11 warmup prototype 与 soft assignment | KMeans 初始化和 soft head 是训练程序，不应写成完整 EM 收敛证明 |
| model selection / reject option | self/null expert 和 NoMix | gate 可以 abstain，但 coverage 与 clustering quality 必须一起报告 |
| PCA / latent variable | V9/V10/V11 的 PCA、encoder 和 EMA latent | latent geometry 改善不等于聚类边界改善 |

V11 的 Student-t head 是合理的稳健重尾建模选择，但当前实现是对角尺度和有限训练目标下的神经 mixture readout。不能直接声称完成了 Bayesian posterior inference，也不能把 soft responsibility 的数值当作经过校准的概率，除非另外做 calibration、coverage、NLL/Brier 或可靠性曲线审计。

### 5.3 《数学分析》

本书的序列极限、连续性、Lipschitz/Hölder 连续、紧致性、Cauchy 准则、多元微分和梯度内容为项目提供了稳定性语言，但不能把工程观察自动升级为定理：

- EMA 是离散递推，只有在有界性、步长、噪声和目标稳定等条件下，才能讨论收敛；当前实验中的 graph refresh fraction 只是经验统计。
- kNN 是不连续的离散算子。数据或 latent 小幅变化可能在边界处导致邻居集合跳变，因此“输入变化小 -> 图变化小”需要 margin、density gap 或概率稳定性条件。
- PCA 和 StandardScaler 的连续线性代数部分不保证 kNN 关系连续；离散排序是额外的敏感性来源。
- 紧致性和有限覆盖可以帮助解释有限数据上的局部覆盖，但当前代码没有对数据分布支持集、流形或采样密度建立证明。
- 重建损失的梯度下降是优化过程；训练 loss 下降、gate 稳定和 cluster ARI 上升是三个不同命题。

因此报告和论文应使用“empirical stability”“observed graph recurrence”“dataset-dependent robustness”等表述，不能写成“拓扑结构在训练中收敛”或“保持拓扑不变量”。

### 5.4 《普林斯顿数学指南》及其他数学资料

《普林斯顿数学指南》从集合、函数、关系、线性映射、分析、几何、拓扑、流形和度量的层次组织数学对象。它帮助确认当前项目横跨的是：

```text
集合/关系 -> 有限图 -> 距离与线性代数 -> 概率模型 -> 优化与统计验证
```

这条链路不应跳过中间对象直接从“邻居图”推断“拓扑不变量”。`近世代数`、`数论基础`、`微分方程数值解法`、`生物信息学`、`信号与系统`等参考资料也已按目录和与当前问题相关的章节进行核对。它们对群作用/不变量、离散结构、数值稳定性、数据域预处理和高维生物数据有背景价值，但目前没有在 TopoGate 源码中形成已验证的代数拓扑或微分方程模块。后续引用这些书只能作为背景，不应制造不存在的代码对应关系。

## 6. 当前正式证据和结果解释

### 6.1 当前结果盘中的主要产物

| 批次 | 当前入口 | 规模 | 证据等级 |
|---|---|---:|---|
| V9 论文预处理匹配 | `result/v9_results_2026-08-02_paper_preprocess/` | 24 datasets x 3 seeds；`v9_runs.csv` 72 行 | 三 seed 配对对照，baseline 为持久化单次参考 |
| V9 相关集消融 | `result/v9_results_2026-08-02_advantage_ablation/` | 7 datasets x 4 variants x 3 seeds；84 行 | 三 seed 消融 |
| V12 | `result/v12_results_2026-08-03_advantage/runs.csv` | 144 completed | 三 seed，机制 no-go |
| V13 | `result/v13_results_2026-08-03_advantage/runs.csv` | 72 completed | 三 seed，机制 no-go |
| V14 | `result/v14_results_2026-08-03_advantage_5ds/runs.csv` | 30 completed | 三 seed，机制可运行但性能 no-go |
| V11 conservative candidates | `result/V11/topogate_v11_minimum_5x3/` 等 | 多个 5 dataset x 3 seed 候选 | 候选/消融，不自动等于主方法 |
| CLUBench | `result/clubench_ahdpc_hdpc_v9_2026-08-02/` | 131 x 3 methods，393/393 | 单 seed 工程/对照证据 |

### 6.2 V9 相对 AHDPC/HDPC

论文匹配输入协议下，V9 的三个 AHDPC 正差值是：

| 数据集 | V9 ARI | AHDPC ARI | Delta ARI |
|---|---:|---:|---:|
| `spect_heart` | `0.245936 +/- 0.023536` | `-0.015367` | `+0.261303` |
| `balance_scale` | `0.190844 +/- 0.073938` | `0.015174` | `+0.175669` |
| `landsat` | `0.537993 +/- 0.004904` | `0.513200` | `+0.024792` |

这三个数据集共同提供了“局部邻域可利用且固定密度/epsilon 假设可能失配”的研究假设，但它们的 mutual-neighbor 比例跨 `0.431-0.829`，所以高 mutual 不是充分条件。相反，`banknote`、`flame`、`asymmetric`、`aggregation`、`unbalance` 等数据上 AHDPC/HDPC 已经很强，V9 邻居混合和 MAE 目标可能造成过度平滑或表征-聚类错配。

这里的正确结论是“相对优势集中在特定几何区间”，不是“TopoGate 的 topology 普遍有效”。V9 相对 AHDPC 的 `3/1/20` 和平均负 Delta 必须在任何论文表格前明确写出。

### 6.3 V9 topology mixing 消融

当前保存的消融摘要给出：

| variant | 宏平均 ARI |
|---|---:|
| Full | `0.278302` |
| NoMix | `0.262946` |
| Static | `0.279565` |
| Random | `0.272016` |

Full-NoMix 为 `+0.015356`，但 Wilcoxon `p=0.3905`、配对 t `p=0.1417`。这说明 Full 的平均值高于 NoMix 不能独立证明 mixing 是稳定机制；数据集方向相反本身就是结果。论文应把 topology mixing 写成候选机制或数据集依赖模块，除非后续有更强的预注册证据。

### 6.4 V12-V14 no-go

- **V12**：12 datasets x 4 variants x 3 seeds，144/144 完成；Full-NoMix 约 `-0.001244`，Wilcoxon `p=0.8828`。
- **V13**：12 datasets x Full/NoMix x 3 seeds，72/72 完成；head ARI Full-NoMix 约 `-0.000238`，Wilcoxon `p=0.8314`；KMeans embedding 差约 `+0.003657`，没有显著性证据。
- **V14**：5 datasets x Full/NoMix x 3 seeds，30/30 完成；Full ARI `0.133629`，NoMix `0.129256`，差 `+0.004373`；Wilcoxon `p=0.8139`，配对 t `p=0.6597`；平均 target gate `0.006276`。这更像“保守目标使 gate 极少开门”的诊断，而不是有效性能机制。

### 6.5 V11 candidate batch

当前已迁入结果盘的 `result/V11/topogate_v11_minimum_5x3/` 包含 5 datasets x Full/NoMix x 3 seeds 的 summary CSV。按三个 GPU 分片的当前 CSV 聚合，head ARI 的均值约为：Full `0.653593`，NoMix `0.654067`，差约 `-0.000475`。这支持“minimum combiner 的 head ARI 没有形成稳定净增益”的保守解释。V11 的其他 candidate（harmonic、semantic_metric、reweighted、no-edge）应作为候选消融分别报告，不能凭单个分支挑选主配置。

### 6.6 CLUBench 131 单 seed

CLUBench 统一 `load_data` z-score、K 由 `np.unique(y)` 得到、训练不传 `y` 的单 seed=42 批次为：AHDPC mean ARI `0.1830`，HDPC `0.1614`，V9 `0.3227`；V9 相对 AHDPC `105/2/24`，平均 Delta `+0.1396`。这是很有价值的工程画像，但与 24 数据集三 seed 的论文匹配协议不同，不能混合成一个“普遍提升”结论。

## 7. 当前模型的理论和工程风险

1. **拓扑术语风险**：当前是 metric-dependent finite graph，不是 topological invariant，也不是 persistent homology。
2. **预处理风险**：StandardScaler、PCA 维数、HVF 和 kNN metric 会改变边集合。Olivetti 的 baseline 使用 t-SNE、V9 使用 raw 4096-D，不能作为同输入比较。
3. **目标错配风险**：MAE reconstruction、assignment consistency 和最终 KMeans/head prediction 可能优化不同几何目标。必须同时保存 loss、gate、embedding 和多个 readout。
4. **gate 退化风险**：self/null expert 可能通过关闭 topology 获得较低训练风险。需要报告 gate mass、target mass、edge coverage、abstention rate 和 topology-on/off paired risk。
5. **概率校准风险**：高维对角 Student-t product responsibility 可能饱和，confidence filtering 可能只是在确认早期伪标签。需要温度、尺度 floor、责任度熵和 calibration 诊断。
6. **动态图离散风险**：kNN refresh 是非连续的离散排序；EMA 只平滑表征，不保证图集合或训练目标收敛。
7. **标签协议风险**：benchmark oracle K 可以合法存在，但必须写明来源；不能让 `y` 进入图构建、gate target、超参选择或 variant selection。
8. **基线公平风险**：AHDPC 论文公式和表格复现存在冲突，必须保留 literal/reported/table-reproduction 模式；unresolved 数据不能用相似数据替代。
9. **存储复核风险**：历史文档路径不等于当前产物。任何汇总脚本都应在生成表格时检查 summary、预测数组、配置和 source hash 是否存在。
10. **统计风险**：single seed、短 epoch 和小批次只能作工程证据；跨数据集平均值必须给出配对关系、seed、预处理、K 来源和显著性方法。

## 8. TDA/persistent homology 的可逆引入方案与正式结果

### 8.1 目标和限制

目标不是把 kNN 改名成 TDA，而是验证持久性统计能否提供比 mutual/SNN 更稳定的无标签边/节点先验。第一版必须满足：

- 不读取 `y`，不用 label 选择 filtration、阈值、最佳 epoch 或 variant；
- 不替换 V11 的主模型、loss、Student-t head 或 self/null expert；
- TDA 特征 detached，只作为 edge ranking/prior 或诊断输入；
- 对 H0 的固定稀疏 1-skeleton，union-find 是精确且依赖无关的计算；H1、dense VR 或 persistence image 仍需先确认成熟 TDA 库（例如 GUDHI/ripser）及复杂度上限，不能手写未经验证的替代实现；
- 保留 raw-only、graph-only、TDA-prior、NoMix 四类控制；
- 所有产物写入 `result/V11/` 或新版本结果目录，并保存 filtration、库版本、参数、diagram hash 和 source hash。

### 8.2 数学对象

给定无标签表征 `z_i` 和边权 `w_ij`，先定义归一化距离或相似度 filtration。以 Vietoris-Rips 为例：

```text
K_alpha = {sigma : max_{i,j in sigma} w_ij <= alpha}
```

随着 `alpha` 增大，得到嵌套复形：

```text
K_alpha1 subseteq K_alpha2 subseteq ...
```

对每个复形构造链群和 boundary operator，计算 H0/H1 的 birth-death pairs，再形成 persistence diagram 或 barcode。可派生：

- H0 component lifetime：局部连通结构和合并尺度；
- H1 cycle lifetime：可能的环结构，需确保使用真正的 simplicial complex；
- edge/vertex local persistence score：把全局 diagram 转成边排序先验；
- persistence entropy、total lifetime、diagram distance：只作无标签诊断，不直接当作标签监督。

### 8.3 第一阶段实现

1. 当前 pilot 在 raw PCA 上固定候选 skeleton，记录单位行 Euclidean chord filtration、k、scale 和 resolved config；EMA latent 只用于原有候选 union，不被标成 TDA skeleton。
2. H0 由 `methods/TopoGate/V11/tda.py` 的 union-find 精确计算；H1/dense VR 仍等待成熟库核验。大规模数据若进入第二阶段，必须先固定 landmark/subsample 或 graph-induced complex 近似。
3. 将 persistence feature `q_ij` detach 后输入 edge-prior/ranking 分支，而不是直接当 target label；当前实现已遵守该边界。
4. 不在第一版加入 TDA message passing，不把 persistence diagram 直接拼进 latent，不同时改动 reconstruction、assignment 和 gate target。
5. 已加入 permutation-independent deterministic controls、metric/scale sanity checks 和 MST merge-edge tests；正式批次已完成 5 datasets x 5 variants x 3 seeds，共 75/75 completed、0 errors。H0 相对 V11 Full 的 head ARI 为 `+0.000010`、KMeans ARI 为 `-0.000726`；fixed-filtration 为 `+0.000002/-0.000665`，random 为 `+0.000018/-0.000274`。这些结果只支持固定五数据集协议内的 no-go 判定。

### 8.4 正式预注册比较

正式批次使用 `balance_scale`、`spect_heart`、`banknote`、`flame` 和 `vehicle` 五个数据集，固定 seeds `[42, 123, 7]`，比较：

| 变体 | 目的 |
|---|---|
| NoMix | 无拓扑路径下界 |
| V11 Full | 当前 topology baseline |
| V11 + detached TDA prior | 测试持久统计的独立帮助 |
| V11 + random/topology-shuffled prior | 检查是否只是额外参数/噪声 |
| V11 fixed filtration | 隔离动态刷新影响 |

主指标同时包括 head ARI、KMeans ARI、NMI、silhouette、edge coverage、gate calibration、assignment risk、reconstruction risk、graph recurrence 和计算成本。全部预注册数据集和 seed 已完成；15 个 dataset-seed 配对显示三种 prior 的 head ARI 大多为 ties，且 KMeans ARI 相对 Full 均为负或近零。所有运行均记录 `benchmark_oracle_from_y` 作为 K 来源，但训练器和图构建不读取 `y`；没有逐数据集用 `y` 选择 filtration 或主变体。

### 8.5 Go/no-go 标准

- **Go**：TDA prior 在至少多数预注册数据集上不损害 NoMix/Full 的主指标，且 edge/gate 诊断和 persistence stability 有一致方向；额外计算成本可接受，结果可由当前 `result/` 产物复核。
- **No-go**：只在一个数据集提升、只提升 KMeans 而 head 下降、gate mass 仍接近 0、对尺度/PCA 极度敏感、或与 random prior 无差别。此时保留为分析特征，不进入主方法。

**本批次判定**：No-go。H0、fixed-filtration 和 random prior 相对 V11 Full 的 head ARI 均近零，KMeans ARI 分别为负或近零，且 H0 与 random prior 没有可辨认的独立聚类收益。该判定只适用于上述固定稀疏 skeleton、输入预处理、五个数据集和三个 seed，不否定完整 H1/dense VR TDA 在其他协议中的可能性。

## 9. 后续工作边界

1. 不把本批次 TDA prior 写入论文主方法，也不再以新的 smoke 重复支持已完成的 no-go 判定；保留 `result/V11/tda_h0_pilot_2026-08-03/` 作为可审计诊断。
2. 继续对 V11 做 gate calibration 和 representation-clustering mismatch 诊断，重点观察 self/null mass、target mass、edge coverage、responsibility entropy 和 risk difference。
3. 使用 `result/V11/topogate_v11_minimum_5x3/`、V12-V14 和 TDA 正式批次建立统一 evidence table，明确 head/KMeans、seed、K 来源、输入协议和 metadata gap。
4. 若未来重新推进 TDA，应先引入经过核验的成熟库和新的预注册协议，单独评估 H1/dense VR 或 graph-induced complex；不得把当前 H0 稀疏 pilot 扩写成完整 persistent homology。
5. 当前轮已更新 `RESULTS_SUMMARY.md`、`CHANGELOG_data.md`、`CHANGELOG.md`、`CHANGELOG_errors.md`；没有新增未经核验的 TDA 文献，因此不修改 `CHANGELOG_lit.md` 或 `papers/references/INDEX.md`。

## 10. 已知限制

- 本次是全项目结构和关键算法路径审计，不把第三方 baseline 的每个二进制/权重文件冒充已逐行阅读；外部方法的事实仍以其 provenance/status 和实际 runner 为准。
- 参考书的阅读深度按任务相关性分层：拓扑学、PRML、数学分析和数学指南进行了与模型直接相关的章节阅读；近世代数、数论、微分方程数值解法、生物信息学和信号与系统资料完成目录及相关主题核对。它们提供背景，不自动成为项目引用。
- V9 对 AHDPC 的 baseline 是持久化单次参考，V9 多 seed 和 baseline 并非完全对称重跑；差值必须保留这个限制。
- 非 TDA 的 V11 候选批次虽然已迁入结果盘，但只有配置、summary、CSV、预测数组和 source hash 全部核验后才能进入论文主表；TDA 正式批次已完成该级别的产物核验，但其性能判定仍为 no-go。
- 当前没有证据证明 TopoGate 在所有数据分布上优于密度峰、图聚类或 NoMix；新增 pilot 只实现固定稀疏 1-skeleton 上的 H0 persistence，不能写成完整 persistent homology 方法。

## 11. 当前可引用的最小结论

> 在固定输入和明确 K 协议下，TopoGate 的局部邻域图与门控路径可以稳定运行并生成可审计产物；但其性能和 topology mixing 贡献具有明显数据集依赖性。现有源码属于 topology-inspired reliable graph learning，不等价于 persistent homology。V12-V14 及 V11 sparse H0 TDA prior 的配对证据均未建立稳定、显著的 topology 净增益；当前最稳妥的方向是继续验证 gate calibration 和表征-聚类目标一致性，而不是扩大未经验证的性能宣称。

## 12. 本轮最终核验

- `PYTHONPATH=source-repository python -m pytest -q methods/TopoGate/V11/tests/test_v11.py`：`19 passed`，仅有 3 条 FAISS/SWIG 第三方弃用警告。
- `PYTHONPATH=source-repository python -m pytest -q tests/v10_reliable_graph`：`14 passed`，仅有 3 条 FAISS/SWIG 第三方弃用警告。
- `python -m compileall -q methods/TopoGate/V11 methods/TopoGate/v10_reliable_graph scripts/V11 scripts/v10_reliable_graph scripts/v9_learnable_gate`：通过。
- V11 和 V10 的 `--help`：通过；V10 CLI 的 GPU 选项仍为 `[1,4,5]`。
- `readlink -f source-repository/result`：`external-result-storage/result`；正式 V9/V12/V13/V14 目录均存在。
- 显式跟随软链接检查 `smoke`/`debug` 目录、`/tmp` 中的 V11/V10 临时目录和仓库根目录结果目录：均无匹配项。结果盘中的 `smoker_condition` 是数据集名称，不是 smoke 产物。
