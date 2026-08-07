# V9 相对 AHDPC/HDPC 的数据特征再分析

## 结论先行

基于 `result/v9_results_2026-08-02_paper_preprocess/comparison_by_dataset.csv` 和 `V9_vs_AHDPC_HDPC.md` 的 24 个数据集、每个 3 个 seed，V9 的画像不是“普遍优于 AHDPC”，而是**在少量 UCI 数据上相对占优、在合成几何基准上明显落后、在某些高维输入上绝对表现很好但未超过已饱和的密度峰基线**。

- 相对 AHDPC：ARI 胜/平/负 = **3/1/20**，平均 ΔARI=`−0.1715`，中位数=`−0.0433`。正差值只有 `spect_heart`、`balance_scale`、`landsat`。
- 相对 HDPC：ARI 胜/平/负 = **5/1/18**，平均 ΔARI=`−0.1530`，中位数=`−0.1483`。除上述三个外，`vehicle`、`vertebral_column` 只相对 HDPC 小幅为正；`website_phishing` 是相对 HDPC 的明显优势，但相对 AHDPC 基本持平。
- V9 的优势是**相对优势**，不是所有情况下的高绝对质量：`spect_heart` 的 V9 ARI 只有 `0.2459`，但 AHDPC/HDPC 分别为 `−0.0154/−0.0274`；`balance_scale` 的 V9 ARI=`0.1908`，仍低于 HDPC=`0.3389`。
- 现有 NoMix 消融只覆盖 7 个 UCI 数据集：21 个配对的 ARI 平均 `full−NoMix=+0.0154`，但 Wilcoxon `p=0.3905`、配对 t 检验 `p=0.1417`；因此不能把 V9 的总体优势归因于拓扑混合。
- 所有训练都不使用真值标签；K 由 `unique(y)` 仅用于 benchmark 和评估。AHDPC/HDPC 是持久化单次参考，V9 是 seeds `[42,123,7]` 的均值±标准差，因此差值不是三方同等多种子置信区间。

## 1. 优势数据集

### 1.1 相对 AHDPC 的三个正差值

| 数据集 | n | d | K | V9 ARI | AHDPC ARI | HDPC ARI | ΔARI(V9−AHDPC) | 5-NN mutual | 连通分量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `spect_heart` | 267 | 22 | 2 | 0.2459±0.0235 | −0.0154 | −0.0274 | **+0.2613** | 0.406 | 1 |
| `balance_scale` | 625 | 4 | 3 | 0.1908±0.0739 | 0.0152 | 0.3389 | **+0.1757** | 0.829 | 1 |
| `landsat` | 4435 | 36 | 6 | 0.5380±0.0049 | 0.5132 | 0.4698 | **+0.0248** | 0.512 | 1 |

三者的共同点是单视图数值矩阵、d=4/22/36、5-NN 图单连通；但 mutual 从 0.406 到 0.829，差异很大。因此“高 mutual 邻居是 V9 获胜条件”不成立。更稳妥的机制假设是：**局部邻域可用，同时固定密度/ε 归纳偏置在该数据上不匹配；V9 的 MAE 表征与邻居混合提供了另一种局部结构偏置。**

### 1.2 相对 HDPC 的补充优势

- `website_phishing`：V9 ARI=`0.2929`，HDPC=`0.0712`，Δ=`+0.2217`；六项指标均高于 HDPC，但相对 AHDPC ARI=`−0.0034`，说明优势主要来自 HDPC 的具体规则，而非对 AHDPC 的普遍突破。
- `vehicle`、`vertebral_column` 相对 HDPC 仅分别为 `+0.0063`、`+0.0158`，只能算接近持平。

### 1.3 相对优势和绝对质量要分开

- `spect_heart`：六项指标均相对 AHDPC、HDPC 为正，这是最完整的相对优势，但 V9 ARI/NMI 仍仅 `0.2459/0.1482`。
- `landsat`：相对 HDPC 六项指标均为正；相对 AHDPC 只有 FMI 略负（`−0.0058`），属于稳定但很小的改进。
- `balance_scale`：相对 AHDPC 的 ARI/NMI/AMI/RI 为正，但 FMI=`−0.0898`；相对 HDPC 六项指标全为负。

### 1.4 与 NoMix 的配对消融

消融产物位于 `result/v9_results_2026-08-02_advantage_ablation/`，覆盖 7 个数据集、
3 个 seed（`42,123,7`）。`v9_full` 使用 `mix_mode=reliability`、
`pseudo_weight=0.3`；严格 NoMix 使用 `mix_mode=none`、`pseudo_weight=0`，并在摘要中
记录 `mean_node_gate=0`。因此这里的差值是当前“拓扑混合 + 伪一致性”配置包相对 NoMix
的效果，不能狭义解释为只改变邻居混合而其他路径完全不变。

| 数据集 | Full ARI | NoMix ARI | ΔARI (Full−NoMix) | 正向 seed | ΔNMI | 对 AHDPC 的 ΔARI | 判读 |
|---|---:|---:|---:|---:|---:|---:|---|
| `balance_scale` | 0.1900±0.0892 | 0.1091±0.1033 | **+0.0809** | 3/3 | +0.0815 | +0.1757 | 最清晰的拓扑正例；同时超过 AHDPC |
| `glass` | 0.3904±0.0098 | 0.3497±0.0689 | +0.0407 | 2/3 | +0.0240 | −0.0191 | 相对 NoMix 改善，但仍略输 AHDPC |
| `image_segment` | 0.2728±0.0399 | 0.2515±0.0140 | +0.0214 | 2/3 | +0.0001 | −0.1484 | 小幅且不稳定，未转化为基线优势 |
| `landsat` | 0.5379±0.0064 | 0.5324±0.0099 | +0.0055 | 3/3 | +0.0002 | +0.0248 | 稳定但很小；`v9_random` 均值反而为 0.5392 |
| `spect_heart` | 0.2481±0.0253 | **0.2606±0.0169** | −0.0125 | 1/3 | −0.0080 | +0.2613 | V9 相对 AHDPC 的优势在 NoMix 下仍存在，不能归因于拓扑 |
| `vehicle` | 0.0749±0.0099 | **0.0863±0.0274** | −0.0114 | 1/3 | −0.0110 | −0.0120 | NoMix 更好，拓扑包可能有害 |
| `vertebral_column` | 0.2340±0.0072 | **0.2511±0.0130** | −0.0171 | 0/3 | −0.0062 | −0.0230 | NoMix 更好，拓扑包可能有害 |

这组消融的关键信息不是“Full 平均略高”，而是**数据集依赖性**：
`balance_scale` 的 3/3 seed 正向且效应最大；`landsat` 虽 3/3 正向但效应只有
`+0.0055`，并且随机混合略高，不能据此证明 reliability 权重本身最优；
`spect_heart` 的 AHDPC 相对优势在 NoMix 下反而更高，说明该优势来自表征/聚类路径或
基线失配，而不是拓扑；`vehicle` 与 `vertebral_column` 则出现一致的 NoMix 倾向。
因此，在当前 7 个数据集上，拓扑混合最多只能被描述为**局部有效、总体未显著**。

## 2. 劣势数据集

### 2.1 最大退化：简单、紧致、低维合成几何

| 数据集 | n | d | K | V9 ARI | AHDPC ARI | HDPC ARI | ΔARI(V9−AHDPC) | 典型特征 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `banknote` | 1372 | 4 | 2 | 0.0116±0.0056 | 0.9624 | 0.6239 | **−0.9508** | 低维、近均衡，5-NN mutual 较高，但 V9 几乎没有形成可分簇表示。 |
| `flame` | 240 | 2 | 2 | 0.1324±0.0029 | 0.9666 | 0.7620 | **−0.8342** | 经典二维密度峰/非凸几何，AHDPC 的密度峰归纳偏置高度匹配。 |
| `asymmetric` | 1000 | 2 | 5 | 0.5324±0.0561 | 0.9723 | 0.9700 | **−0.4400** | 二维强几何结构，AHDPC/HDPC 接近饱和；邻居混合可能破坏边界。 |
| `unbalance` | 6500 | 2 | 8 | 0.5750±0.0001 | 1.0000 | 1.0000 | **−0.4250** | 极端类别不均衡（最大/最小类约 20 倍），固定 k=5 混合不适合不同密度尺度。 |
| `aggregation` | 788 | 2 | 7 | 0.5797±0.0065 | 1.0000 | 0.7689 | **−0.4203** | 二维紧致、mutual 约 0.845，但高 mutual 局部图没有转化为 V9 的可分表示。 |

这组结果反驳了“邻域越可靠 V9 越强”：`banknote`、`flame`、`aggregation` 的 5-NN mutual 都约 0.72–0.85，却是最严重退化。更可能是**AHDPC 的密度峰/距离排序已经解决问题，而 V9 的邻居混合与 MAE 重建引入了过度平滑或优化噪声**。

### 2.2 高维/高簇数不是充分解释

- `olivetti_faces`：V9=`0.1103`，AHDPC=`0.4999`，HDPC=`0.5698`；这是协议例外，AHDPC/HDPC 用 t-SNE(2D)，V9 用原始 4096-D，不能作为同输入算法结论。
- `libras_movement`（d=90,K=15）和 `student_evaluation`（d=33,K=3）较弱，但 `dim064/dim512`（K=16）接近 1.0，说明高维或 K 大本身不是充分解释。
- `2d_20c_no0`、`2d_4c_no9` 的 V9 ARI 仍为 0.7631/0.8076，只是相对已近饱和的 AHDPC/HDPC 不够强；应与 `flame/asymmetric` 的严重失败区分。

### 2.3 中等退化和不稳定

- `rice`：ΔARI=`−0.2694`，V9 seed std=`0.0474`；`smile`：`−0.1859`；`image_segment`：`−0.1484`。
- `2d_4c_no4` 相对 AHDPC 仅 `−0.0229`，但相对 HDPC 为 `−0.3133`，说明基线本身的规则差异会改变“劣势”判断。
- `glass`、`vehicle`、`vertebral_column` 的 ARI 差值约 −0.023 到 +0.016，属于接近持平，不宜归入严重失败。

## 3. 探索性特征关系

基于协议输入的无标签 5-NN 检查（24 个数据集）：

| 特征 | Spearman ρ 与 ΔARI(V9−AHDPC) | Spearman ρ 与 ΔARI(V9−HDPC) | 解释 |
|---|---:|---:|---|
| d | +0.508 (p≈0.011) | +0.511 (p≈0.011) | 中高维相对更可能接近基线；不等于“V9 适合高维”。 |
| 5-NN mutual | −0.501 (p≈0.013) | −0.598 (p≈0.002) | 高 mutual 更多出现在 AHDPC/HDPC 已很强的简单几何上，不是 V9 充分条件。 |
| n | +0.010 | ≈0 | 样本数没有可见单调关系。 |
| K | −0.022 | −0.125 | 簇数没有稳定单调关系；d=64/512、K=16 是反例。 |
| 类别不均衡比 | +0.034 | −0.035 | 不能从 24 个数据集推出不均衡直接决定差值；`unbalance` 是明确风险案例。 |
| 5-NN 连通分量 | −0.279 | −0.323 | 可能有影响，但小样本不显著，也不是单变量解释。 |

这些相关性是小样本描述性统计，不是因果检验；也不能把同一批 24 个数据同时作为探索和验证证据。

## 4. 分组画像

| 组别 | 数量 | V9 平均 ARI | 平均 ΔARI vs AHDPC | 平均 ΔARI vs HDPC |
|---|---:|---:|---:|---:|
| 合成 | 11 | 0.6750 | −0.2439 | −0.2344 |
| UCI | 12 | 0.2279 | −0.0868 | −0.0528 |
| Olivetti | 1 | 0.1103 | −0.3896 | −0.4595 |

正差值全部来自 UCI；但 UCI 平均仍为负，主要受 `banknote` 极端退化拉低。去掉 `banknote` 后 UCI 平均 ΔARI 约为 −0.0083（AHDPC）和 −0.0019（HDPC），只是描述性敏感性分析，不是无偏估计。

## 5. 对 V9 机制的解释

1. 拓扑路径不是自动增益：7 个数据集的 full−NoMix 平均 ARI 仅 `+0.0154` 且不显著；`balance_scale` 是最清晰的正例，而 `spect_heart`、`vehicle`、`vertebral_column` 的 NoMix 更好。后续 V12–V14 也没有显著、稳定的 topology 增益。
2. 成功区间更像“基线密度假设失配区”：`spect_heart` 和 `balance_scale` 的 AHDPC 本身接近随机，这比“图必须高 mutual”更能解释优势。
3. 失败区间是“密度峰已足够 + 混合会破坏边界”：`flame/asymmetric/aggregation/unbalance/banknote` 的 AHDPC 很强且局部图不差，问题更可能是无监督 MAE/邻居混合目标与最终簇边界不一致。
4. V9 以 MAE/伪重建为主，最终使用 embedding 上的 KMeans；这允许重建目标改善而聚类边界变差。这里是架构解释，不是 CSV 单独可证明的因果结论。
5. Olivetti 的 t-SNE vs raw 4096-D 必须单独控制；AHDPC 的 epsilon、归一化和 table-reproduction 模式保持冻结，不能为追求 V9 优势调参。

## 6. 可用于论文/报告的表述

> 在与 AHDPC 发布预处理相匹配的 24 个数据集上，TopoGate V9 仅在 `spect_heart`、`balance_scale` 和 `landsat` 的 ARI 上超过 AHDPC；相对 HDPC 另在 `vehicle`、`vertebral_column` 和 `website_phishing` 上取得正差值。总体平均 ΔARI 仍为负（分别为 −0.1715 和 −0.1530）。7 个数据集的 full−NoMix 消融显示平均 ARI 仅 `+0.0154` 且不显著：`balance_scale` 的拓扑包有清晰正向作用，`landsat` 只有微小增益，而 `spect_heart` 的基线相对优势在 NoMix 下仍保留，`vehicle`/`vertebral_column` 则偏向 NoMix。因此优势更接近局部邻域可利用但密度峰假设失配的场景，而不是由高 mutual-neighbor 比例、样本数或簇数单独决定；在 `flame`、`asymmetric`、`aggregation`、`unbalance` 和 `banknote` 等 AHDPC/HDPC 已能稳定恢复结构的数据上，V9 的邻居混合与 MAE 目标可能引入过度平滑或表征—聚类错配，导致明显退化。上述结论基于单个固定 V9 配置、7 个数据集的三 seed NoMix 消融和持久化 baseline 参考；仍需在代表性优势/劣势数据上做对称多 seed baseline 重跑、门控/embedding 诊断和预注册统计检验。

## 可复核入口

- `result/v9_results_2026-08-02_paper_preprocess/comparison_by_dataset.csv`
- `result/v9_results_2026-08-02_paper_preprocess/V9_vs_AHDPC_HDPC.md`
- `result/v9_results_2026-08-02_advantage_ablation/ablation_runs.csv`
- `result/v9_results_2026-08-02_advantage_ablation/summary_by_dataset.csv`
- `result/analysis/V9_AHDPC_advantage_deep_analysis_2026-08-03.md`
- `result/v9_results_2026-08-02/geometry_features_no_label.csv`（历史标准化协议的几何补充，不与论文匹配差值混用）
