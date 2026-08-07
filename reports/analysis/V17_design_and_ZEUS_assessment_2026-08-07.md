# TopoGate V17：统一目标、ZEUS 评估与拓扑原生主干

更新时间：2026-08-07

## 1. 本轮问题与证据边界

本轮只回答四个研究问题：

1. ZEUS 是否适合作为 TopoGate 的前置编码器？
2. 如何避免 encoder、topology、gate 和最终 clustering readout 再次目标错位？
3. V17 应当优化哪个可识别对象，才能保留“拓扑 + 门控”而不再发明 ARI utility？
4. 应当如何划定可证伪的理论适用域和最小实验命题？

本轮没有修改 V1--V16.1，没有运行训练或 benchmark，没有重新计算数据或源码哈希。项目性能事实来自当前 `result/RESULTS_SUMMARY.md`、`V_SERIES_FAILURE_RETROSPECTIVE.md` 和仍可读取的产物；文献机制来自已归档全文。MCP 新命中的未归档论文只作为检索候选，不进入正式方法论结论。

## 2. 从 V1--V16.1 得到的硬约束

截至 2026-08-07，V16.1 expanded-count 固定协议约 35 个候选没有产生 `candidate_positive`。这不是因为 null/self 写错，而是因为 predictive support 与传播收益不等价。

最关键的反例是 `hrvatin_geo_maintype_counts`：candidate purity 约 `0.9968`、candidate recall 约 `0.9971`，fixed graph ARI 约 `0.8504`，但 predictive support 几乎全负、null mass 约 `0.9991`，V16.1 退化为 self-only。它同时否定了以下三个未经证明的等价关系：

\[
\text{候选边很纯}
\not\Rightarrow
\text{单 donor 预测风险为正}
\not\Rightarrow
\text{assignment 传播有益}.
\]

跨版本还给出六条必须继承到 V17 的约束：

1. 距离近、边纯、recurrence 高和 reconstruction 好都不是聚类收益的同义词。
2. softmax 会强制分配质量，forced Top-k 会强制使用错误边；门控必须允许精确零边。
3. teacher、graph、target 和 gate 由同一模型产生时，容易形成自证循环。
4. scMAE 的主目标是 anchor reconstruction；独立 topology branch 只是附加扰动，最终 KMeans 又读取另一个对象。
5. 逐边 utility、可靠性 MLP、距离堆叠和额外 entropy loss 都没有解决可识别性。
6. 真正需要统一的不是损失项数量，而是被所有模块共同优化和读取的数学对象。

因此 V17 不应继续修 V15/V16 的 utility/support，也不应回到 V2/V9 增加 gate 形式。

## 3. ZEUS 的真实目标

当前项目中的 ZEUS 路径为：

```text
X
-> dense PCA(30)
-> MinMaxScaler(-1, 1)
-> frozen ZEUS Transformer
-> 512-d embedding
-> MinMaxScaler
-> KMeans(K)
```

ZEUS 不是在目标数据上无监督学习 topology。它在合成数据上使用已知合成标签预训练，通过球形 GMM 风格的 cluster assignment likelihood、簇内收缩和簇中心分离学习表示；目标论文还明确把协方差固定为单位阵，促使 encoder 形成近似圆形/球形簇。推理时目标数据标签不进入模型，但最终仍由 KMeans 或 GMM 读取 embedding。

这意味着 ZEUS 的归纳偏置是：

\[
\text{synthetic mixture prior}
\rightarrow
\text{spherical cluster embedding}
\rightarrow
\text{centroid readout}.
\]

而候选 V17 的归纳偏置是：

\[
\text{union of local subspaces / sparse self-expression}
\rightarrow
\text{relation matrix } C
\rightarrow
\text{graph partition}.
\]

二者不是同一个生成假设。

### 3.1 ZEUS 可能带来的收益

- 对接近混合分布、能被 30 维 PCA 保留的表格数据，ZEUS 可能提供比随机初始化或普通 PCA 更强的初始表示。
- 当前本地归档点估计中，`Mouse_retina` 和 `hrvatin_filtered` 的 ZEUS 结果相对较强，说明它值得保留为强 baseline 或初始化候选。
- frozen inference 不需要在每个数据集上训练大模型，适合评估“强表示是否足以解释拓扑收益”。

### 3.2 把 ZEUS 直接作为 V17 主干的风险

1. **目标错位仍然存在**：ZEUS 优化球形簇，TopoGate 优化边/子空间，最终若仍由 KMeans 读取 ZEUS embedding，gate 只是附属模块。
2. **输入几何可能被提前破坏**：固定 dense PCA 30 维可能删除稀疏 feature support、稀有方向和局部子空间结构。
3. **稀疏内存协议冲突**：当前 wrapper 在入口执行 `np.asarray(X)`，随后使用 dense PCA，不能作为极高维 CSR 输入的主路径。
4. **冻结后不可校正**：错误的 mixture prior 无法被 topology objective 反向纠正。
5. **微调后不再是 ZEUS 原命题**：若端到端微调，zero-shot 语义消失，且需要重新证明 30 维预处理和 set-context inference 的合理性。
6. **归因混乱**：若结果上升，难以区分来自 ZEUS 预训练、PCA、KMeans，还是 topology gate。

### 3.3 对 ZEUS 的正式定位

结论不是“ZEUS 无效”，而是：

> ZEUS 适合作为强 frozen baseline、可选初始化和表示诊断，不适合作为 V17 默认主干。

建议以后只保留三种有清晰归因的对照：

1. `ZEUS -> KMeans`：原生外部 baseline。
2. `ZEUS frozen -> same self-expression C -> spectral readout`：检验强表示是否改善同一 topology objective。
3. `V17 native input -> C -> spectral readout`：论文主方法。

只有第 2 项稳定优于第 1 项且拓扑消融成立时，才能说 TopoGate 在 ZEUS 表示上产生额外收益。`ZEUS + 独立 gate + KMeans` 不进入候选架构。

## 4. 文献给出的可用闭环与边界

### 4.1 稀疏自表达提供可识别的关系对象

SSC 将同一子空间中的样本表示为其他样本的稀疏线性组合，系数矩阵 `C` 直接生成 affinity。Robust/Noisy SSC 进一步给出噪声条件和 subspace-preserving 边的理论边界。这里没有“边是否提高 ARI”的不可观测标签；边存在的语义是“该系数是否参与解释 anchor 的结构”。

### 4.2 稀疏不自动保证连通

Graph Connectivity in Noisy SSC 明确区分：

- subspace-preserving：没有跨子空间边；
- within-subspace connectivity：同一子空间内的图保持连通；
- exact clustering：前两者及额外条件共同成立。

因此 V17 不能只追求越稀疏越好。`L1` 负责拒绝跨簇边，轻量 elastic/connectivity 项负责避免把同簇图切碎；二者是必要张力，不是多余模块。

### 4.3 降维不是任意 encoder

Noisy SSC on Dimensionality-Reduced Data 和 Sketched SSC 支持在满足 subspace embedding/JL 条件的投影后恢复自表达结构。它们不支持“任何强 embedding 都保持 topology”。因此 V17 的默认前端应优先选择有结构保持解释的 sparse random projection、TruncatedSVD 或输入分布匹配的统计残差，而不是直接使用合成 GMM 预训练表示。

### 4.4 深度自表达已有大量先例

Deep Subspace Clustering Networks 已把 `C` 作为 self-expressive layer，并从 `|C|+|C^T|` 构造 affinity。Structured Graph Learning 已联合关系矩阵和 spectral structure；MCP 还命中 Deep Sparse Subspace Clustering、Deep Closed-Form Subspace Clustering、Learning Self-Expression Metrics 和 unfolded ISTA 类工作。

这给 V17 一个重要的否定性结论：

> “编码器 + 自表达层 + spectral clustering”本身不是足够的新贡献。

V17 的新增轴必须明确为：高维稀疏输入下，多种结构保持视图共享同一个鲁棒稀疏 `C`；`C` 同时承担 exact-zero edge gate、affinity 和最终分区，并专门评估 feature corruption 与 candidate edge contamination。

### 4.5 count 数据需要分布匹配的输入适配

Townes 等人的 multinomial scRNA 分析表明，log-normalization、伪计数和普通 PCA 会把 total count/zero fraction 变成主要变化来源；raw UMI 更适合 multinomial/Poisson likelihood、GLM-PCA 或 deviance residual。Mixture of Multinomial PCA 和 model-based multinomial clustering也说明 count 数据不应强行套 Gaussian MSE。

因此“统一模型”不意味着所有输入都用同一个预处理。可以按可核验的输入统计类型选择 adapter，但 adapter 后必须统一到同一个关系变量 `C`。

## 5. 建议的 V17 主命题

V17 的论文目标应收敛为：

> 对经过结构保持统计变换后近似满足局部 union-of-subspaces/self-expression 假设的高维稀疏单视图数据，TopoGate 学习一个在特征扰动下稳定、可精确拒绝候选边的稀疏关系矩阵；该矩阵同时定义自表达、拓扑门控、affinity 和最终图分区。

它不声称：

- 适用于所有高维数据；
- 估计某条边的 ARI utility；
- 识别任意类型的离群点；
- 在候选集没有同结构邻居时仍能恢复聚类；
- 固定 KNN、ZEUS 或任意 encoder 天然正确。

### 5.1 理论适用域

主张只在以下条件下成立：

1. 经过预注册 input adapter 后，样本近似位于若干低维线性/仿射子空间或局部可自表达区域。
2. 同结构样本能以少量其他样本重建，跨结构样本的最优残差存在正间隔。
3. candidate set 对每个非孤立样本召回足够的同结构点，但 candidate 本身可以含污染边。
4. feature noise/outlier contamination 低于 robust residual 的承受范围。
5. 稀疏化后每个真实簇仍满足最小连通条件。
6. count 输入使用 raw-count-compatible 变换；连续输入使用 sparse-safe、结构保持的投影。

不满足这些条件的数据是 `theory_domain_not_supported`，不是通过修改 gate 挽救的对象。

## 6. V17 网络：Topology-Native Robust Self-Expression Gate

### 6.1 总体数据流

```text
raw sparse X
  -> input-type adapter T(X)
  -> V 个结构保持投影视图 H^(1)...H^(V)
  -> candidate union E0（只限制计算支持集）
  -> unrolled robust sparse self-expression
  -> shared coefficient matrix C with exact zeros
  -> A = |C| + |C^T|
  -> normalized spectral readout from A
  -> cluster assignment
```

这里没有独立 utility scorer、EMA teacher、pseudo-neighbor reconstruction、forced Top-k 或输出端的第二个 KMeans embedding。

### 6.2 输入适配器

输入适配器由数据来源和统计语义决定，不由标签、ARI 或逐数据集性能选择。

- raw UMI / document count：multinomial deviance residual 或 Poisson/GLM-PCA 的 sparse approximation；保留 library-size offset。
- 非负连续稀疏矩阵：row normalization 后的 sparse random projection / TruncatedSVD。
- 一般连续高维矩阵：robust centering/scaling 后的 subspace embedding。

不建议默认使用 HVF、全维 L2 KNN、log-CPM PCA 或 ZEUS 的 dense PCA(30)。

### 6.3 多投影 candidate recall

对 `V` 个结构保持视图分别构造小候选集，然后取 union：

\[
\mathcal E_0 = \bigcup_{v=1}^{V}\mathcal E^{(v)}.
\]

候选图只控制复杂度，不决定边有效性，不要求每条边必须使用。多投影 union 的目的不是产生多套 reliability，而是降低单个高维 KNN 排序错误造成的召回失败。

### 6.4 同一个 `C` 完成门控和表示关系

共享系数矩阵满足：

\[
\operatorname{diag}(C)=0,
\qquad
\operatorname{supp}(C)\subseteq \mathcal E_0.
\]

基础目标为：

\[
\min_{C,\{O^{(v)}\},F}
\sum_{v=1}^{V}
\rho\!\left(H^{(v)}-H^{(v)}C-O^{(v)}\right)
+\lambda_1\lVert C\rVert_1
+\lambda_2\lVert C\rVert_F^2
+\lambda_o\sum_v\lVert O^{(v)}\rVert_{2,1}
+\lambda_g\operatorname{Tr}\!\left(F^\top L(A(C))F\right),
\]

其中：

- robust residual `rho` 抑制 feature noise；
- `L1` 产生 exact-zero edge gate；
- 小的 `L2`/elastic 项缓解同簇图断裂；
- `O` 是样本级异常残差，不被冒充为 edge utility；
- `A(C)=|C|+|C^T|`；
- `F` 是同一 `A` 的 spectral variable，满足 `F^T F=I`。

第一版可以采用交替优化：proximal update `C`，group shrinkage 更新 `O`，spectral step 更新 `F`。若将若干轮迭代展开成网络，每层的 soft-threshold 就是可解释 gate：

\[
C^{t+1}
=
\mathcal S_{\eta_t\lambda_1}
\left(C^t-\eta_t\nabla_C\mathcal L_{\mathrm{relation}}\right)
\odot M_{\mathcal E_0}.
\]

`C_ij=0` 即拒绝边；没有第二套 gate probability。

### 6.5 输出语义

最终 affinity 唯一地定义为：

\[
A=|C|+|C^\top|.
\]

最终预测来自 `A` 的 normalized spectral readout。谱嵌入后的 KMeans 只是标准 graph partition 的离散化步骤，不读取另一个 encoder latent；其语义不同于“训练 topology，最后却对 `z_self` 做 KMeans”。

K 在 benchmark 中可以由 `unique(y)` 提供，但必须记录为 `benchmark_oracle_from_y=true`，且不得进入 input adapter、candidate graph、`C` 的超参数选择或训练早停。

## 7. 各模块是否目标一致

| 模块 | 优化对象 | 对最终输出的作用 | 是否存在独立代理 |
|---|---|---|---|
| input adapter | 保留可自表达结构 | 决定 `C` 的可估计几何 | 否 |
| candidate union | 提高可用边召回、限制计算 | 只限定 `supp(C)` | 否 |
| robust self-expression | 估计 `C` | 定义关系强度 | 否 |
| exact-zero gate | `C` 的稀疏支持 | 直接删除 affinity 边 | 否 |
| connectivity/spectral term | 同一 `A(C)` 的分区结构 | 防止同簇图被过度切碎 | 否 |
| final readout | `A(C)` | 输出 labels | 否 |

目标闭环是：

\[
\boxed{
\text{input geometry}
\rightarrow C
\rightarrow \operatorname{supp}(C)\text{ as gate}
\rightarrow A(C)
\rightarrow \text{partition}
}
\]

这比“encoder 重建 + gate utility + KMeans”更一致，但仍需实验证明 `C` 在目标数据上可估计。

## 8. 相对现有工作的新增轴

仅实现标准 DSC-Net 会触发“无新意”拒稿。V17 必须同时保留以下差异，且每一点都要有消融：

1. **shared multi-projection coefficient**：多个结构保持投影视图共享同一个 `C`，专门处理高维 feature corruption，而不是在一个 latent 上训练全连接 `n x n` 层。
2. **candidate-restricted exact abstention**：candidate edge 可以全部归零；不是 row-stochastic graph，也不是 forced neighbor。
3. **robust residual + connectivity balance**：明确同时验证 subspace preservation 与 within-cluster connectivity，避免“越稀疏越好”。
4. **same-object readout**：gate、affinity 和最终 clustering 都读取 `C`，不外挂独立 utility 或另一个 latent KMeans。
5. **input-statistics contract**：count 与连续数据使用预注册的不同 adapter，但下游关系目标完全相同。

若这些差异在最近邻全文核验后仍被已有方法完整覆盖，V17 必须缩小贡献或转向实证/应用型期刊，不应虚构 novelty。

## 9. 最小消融与决定性实验

第一轮不是大规模找正例，而是回答统一命题是否成立。

### 9.1 固定方法对照

1. `input_only_spectral`：adapter 后固定相似图。
2. `ungated_candidate`：candidate union 直接作为 affinity。
3. `single_view_self_expression`：去掉多投影共享。
4. `no_connectivity`：去掉 elastic/spectral connectivity 项。
5. `V17_full`。
6. `shuffled_C`：保持稀疏度和度分布但打乱系数位置。
7. `output_disabled`：最终 readout 不读取 `C`。
8. `ZEUS_KMeans` 与 `ZEUS_frozen_C`：只用于 encoder 归因。

### 9.2 固定发现集

- topology-positive anchors：`Campbell`、`hrvatin_geo_maintype_counts`，因为历史 fixed graph 明显强于 self-only。
- boundary：`Mouse_retina`。
- bad-edge stress：`enron`，因为 forced Top-k 曾灾难性崩溃。
- count/text representative：`fbis.wc` 或同源 raw-count 文本集。

正例锚点不是最终证据。它们只用于判断：现有 topology 信息能否被 `C` 保留，以及 gate 是否优于 ungated graph。

### 9.3 机制门槛

进入全量实验前至少满足：

- candidate union 明显提高稳定同结构边的召回，而不是只增加度数；
- `C` 的跨投影 recurrence 高于 shuffled control；
- feature corruption 增强时跨投影不稳定边的归零率上升；
- V17 优于 ungated candidate 和 single-view self-expression；
- `shuffled_C` 与 `output_disabled` 消除主要收益；
- 稀疏化没有把真实簇大规模切成多个小连通分量。

## 10. Supervisor-Skills 评估

### 10.1 Idea-evaluator 判定

**ZEUS 作为默认主干 + 独立 TopoGate**：`Reject and Pivot`。

理由是其核心目标仍然错位，且改善很可能无法归因给 topology。它不是代码不可行，而是科学命题不统一。

**V17 shared-C topology-native 方案**：`Accept with Revisions`，等待最小决定性实验。

| 维度 | 评分 | 依据 |
|---|---:|---|
| Higher | 7 | 机制上把历史 fixed-graph 正例转为可学习关系，但尚无 V17 数据 |
| Stronger | 8 | 直接面向 feature corruption、candidate contamination 和 exact rejection；机制分，未实证 |
| Broader | 7 | count/continuous adapter 后共享关系主干，但不能声称所有 tabular 通用 |
| Faster | 5 | candidate restriction/展开优化有可扩展路径，尚无复杂度实测 |
| Cheaper | 5 | 无标签、无需外部 teacher，但优化和谱分解成本仍存在 |

主要 fatal-flaw 风险：

1. **F1，MAJOR**：SSC、DSC-Net、SGL、metric self-expression 和 unfolded SSC 与该方向高度接近。必须用“多投影共享 + 污染边拒绝 + same-object readout”建立明确差异。
2. **F6，MAJOR**：目前只有机制论证，尚未证明该 `C` 在历史 topology-positive 数据上优于标准 SSC/EnSC/SGL。

### 10.2 Tech-paper-template 逻辑链

论文类型：Technique Paper。

| 环节 | 内容 |
|---|---|
| Research background | 高维稀疏单视图聚类中，先建图再学习表示会把 feature noise 转化为 edge contamination |
| Limitation 1 | kNN/固定图把候选召回和边有效性混为一体 |
| Limitation 2 | reconstruction/teacher utility 与最终 partition 不共享可识别对象 |
| Limitation 3 | 强稀疏门控可能消除坏边，也可能破坏同簇连通 |
| Key Idea | 用一个跨结构保持视图共享的鲁棒稀疏系数矩阵，同时定义自表达、exact-zero gate、affinity 和最终分区 |
| Challenge 1 | 高维稀疏输入的候选召回和几何保持 |
| Challenge 2 | 特征噪声下区分可重复关系与污染边 |
| Challenge 3 | 在跨簇拒绝与簇内连通之间取得可验证平衡 |
| Module A | 统计匹配 input adapter + multi-projection candidate union |
| Module B | unrolled robust sparse self-expression with shared `C` |
| Module C | connectivity-aware affinity + same-`C` spectral readout |
| Contribution 1 | 定义 topology-native 统一关系目标 |
| Contribution 2 | 提出面向 feature/edge corruption 的 shared-`C` exact gate |
| Contribution 3 | 在预注册适用域内给出机制消融和失败边界 |

四项一致性检查均通过：limitations 对应 key idea，challenges 由 key idea 自然产生，三个模块一一解决三个 challenge，三项贡献分别覆盖目标、方法和验证。外部 novelty 风险仍是 MAJOR，不因内部逻辑一致而消失。

## 11. 下一步顺序

1. **先完成最近邻全文审查**：重点对比 DSC-Net、SGL、Learning Self-Expression Metrics、Deep Closed-Form、Deep Sparse SC 和 unfolded ISTA；形成逐项差异表。
2. **冻结数学命题**：明确 `C` 的生成假设、candidate recall 条件、noise bound、connectivity 条件和失败域。
3. **先实现非深度参考解**：candidate-restricted robust SSC/EnSC + same affinity readout，判断核心关系对象是否在锚点成立。
4. **参考解成立后再展开成网络**：只学习 proximal step/threshold，不加入 teacher、utility MLP、attention 或动态图。
5. **ZEUS 保持外部对照**：只有 native V17 成立后，才增加 `ZEUS_frozen_C` 诊断，不让它决定主架构。
6. **最小机制通过后扩展数据**：寻找满足同一 self-expression 适用域的数据，不为负例逐集修改 gate。

## 12. 最终决策

更好的网络不是“更强 encoder + 再叠一个 gate”，而是让 topology 本身成为主干：

\[
\boxed{
C\text{ 是关系}
=C\text{ 是门控}
=A(C)\text{ 是 affinity}
=A(C)\text{ 决定最终 partition}
}
\]

ZEUS 可以提高某些数据的基础表示，但它解决的是 synthetic mixture prior 下的 zero-shot embedding，不是 noisy topology recovery。把它直接放在最前面会重新产生 V1--V16 的目标错位。V17 应以统计匹配、结构保持的轻量输入适配器作为前端，以鲁棒稀疏自表达 `C` 作为唯一核心，再把 ZEUS 放到可归因的强对照位置。

## 13. 本轮正式使用的本地全文

- `../references/pdf/65_zeus_arxiv2025.pdf`
- `../references/pdf/66_elhamifar_ssc_tpami2013.pdf`
- `../references/pdf/67_soltanolkotabi_robust_sc_aos2014.pdf`
- `../references/pdf/68_wang_noisy_ssc_jmlr2013.pdf`
- `../references/pdf/69_wang_graph_connectivity_noisy_ssc2016.pdf`
- `../references/pdf/70_wang_dimreduced_noisy_ssc_tit2018.pdf`
- `../references/pdf/71_you_ensc_active_set_cvpr2016.pdf`
- `../references/pdf/72_you_ssc_omp_cvpr2016.pdf`
- `../references/pdf/74_traganitis_sketched_ssc_tsp2018.pdf`
- `../references/pdf/75_kang_structured_graph_learning_tcyb2021.pdf`
- `../references/pdf/76_ji_deep_subspace_clustering_nips2017.pdf`
- `../references/pdf/77_you_provable_outlier_detection_union_subspaces2017.pdf`
- `../references/pdf/80_soltanolkotabi_geometric_outliers_aos2012.pdf`
- `../references/pdf/82_papastamoulis_multinomial_count_modelbased2023.pdf`
- `../references/pdf/83_jouvin_multinomial_pca_2020.pdf`
- `../references/pdf/86_townes_multinomial_scRNA_2019.pdf`

MCP 本轮还命中 Neural Normalized Cut、Deep Closed-Form Subspace Clustering、Learning Self-Expression Metrics、Deep Sparse Subspace Clustering 和 unfolded ISTA for Deep SSC。未完成全文归档的条目只作为下一轮 novelty 审查候选，不作为本报告的正式方法证据。
