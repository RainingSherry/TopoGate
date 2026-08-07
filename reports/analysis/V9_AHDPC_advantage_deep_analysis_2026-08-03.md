# V9 相对 AHDPC 的优势数据深度分析

## 1. 分析边界

本报告区分两种不可混用的输入协议。正式产物均位于结果软链接目标下，本文
使用项目根路径 `result/...` 表示它们；只有当前存在的数组、CSV、summary 和
配置才是可复核证据。

- 论文预处理匹配协议：`result/v9_results_2026-08-02_paper_preprocess/`，24 个数据集、3 seeds；V9 相对 AHDPC 为 3 胜、1 平、20 负。
- 历史标准化协议：`result/v9_results_2026-08-02/`，同样是 24 个数据集、3 seeds；该协议只用于补充几何假设，不能替代论文匹配协议。

所有训练路径均不读取真值标签；`K=int(unique(y).size)` 只用于 benchmark K 和事后指标。AHDPC/HDPC 是持久化的单次参考输出，因此差值不是统一多种子重新运行的 baseline 置信区间。

## 2. 论文匹配协议下的真实优势

| 数据集 | V9 ARI mean±std | AHDPC ARI | V9−AHDPC | 无标签几何摘要 |
|---|---:|---:|---:|---|
| `spect_heart` | 0.245936±0.023536 | −0.015367 | **+0.261303** | n=267, d=22, mean-kNN=3.229, CV=0.518, mutual=0.431, 1 component |
| `balance_scale` | 0.190844±0.073938 | 0.015174 | **+0.175669** | n=625, d=4, mean-kNN=0.709, CV=0.030, mutual=0.829, 1 component |
| `landsat` | 0.537993±0.004904 | 0.513200 | **+0.024792** | n=4435, d=36, mean-kNN=1.429, CV=0.356, mutual=0.512, 1 component |

因此“V9 在优势数据上更强”只能作为这 3 个数据集的局部事实，不能扩展成总体优于 AHDPC。

## 3. 最大共性：可用的局部邻域 + AHDPC 密度假设不稳定

无标签几何特征与 `V9−AHDPC ARI` 的 Spearman 相关（24 个数据集，探索性统计）如下：

| 特征 | Spearman ρ | p 值 | 解释边界 |
|---|---:|---:|---|
| 特征维数 `d` | +0.549 | 0.0055 | 优势更常见于中/高维，但不是充分条件 |
| 平均 kNN 距离 | +0.504 | 0.0120 | 局部邻域尺度较大时，V9 的表征/拓扑偏置可能更有帮助 |
| kNN 距离 p95 | +0.535 | 0.0070 | 与上项一致，但仍是小样本探索 |
| `log(n·d)` | −0.507 | 0.0114 | 规模/维度联合变化，不能单独解释因果 |
| mutual-neighbor 比例 | −0.298 | 0.157 | 高 mutual 不是共同充分条件 |
| 连通分量数 | −0.121 | 0.574 | 单连通不是统计上唯一决定因素 |

论文匹配协议的 3 个优势数据共同满足：

1. 输入是单视图数值矩阵，维数低到中等（4、22、36）。
2. 5-NN 图存在可利用的局部结构；三者在图几何检查中均为单连通分量。
3. `mutual` 跨数据集差异很大（0.431–0.829），所以“高 mutual”不能作为核心解释。
4. 相比 AHDPC，V9 的 kNN/MAE 归纳偏置在这些数据上提供了不同于固定密度峰/固定 ε 的局部结构假设。

最稳妥的一句话是：**V9 的最大共性不是某个单一图统计量，而是“局部邻域仍可利用、但固定密度/ε 假设不稳定”的几何区间。** 这是无标签几何证据支持的研究假设，不是因果证明。

## 4. 相关数据集与 V9/nomix 消融

相关数据集目录：`datasets/AHDPC_related_advantage/MANIFEST.json`。目录内 12 个文件为已有真实数据的软链接，未伪造或复制数据；文件哈希、shape、K 和来源路径均在 manifest 中。

V9 相关集消融产物：`result/v9_results_2026-08-02_advantage_ablation/`，7 个数据集 × 4 个主要 variant × 3 seeds，84 个主要运行。`v9_full` 使用 reliability mix 与 `pseudo_weight=0.3`；严格 `v9_nomix` 使用 `mix_mode=none`、`pseudo_weight=0`，所以 full−nomix 是当前拓扑混合/伪一致性配置包的差值，不是只改变邻居混合的单因素实验。宏平均 ARI：

| variant | ARI |
|---|---:|
| V9 full | 0.278302 |
| V9 nomix | 0.262946 |
| V9 static | 0.279565 |
| V9 random | 0.272016 |

`full−nomix=+0.015356`，Wilcoxon `p=0.3905`，配对 t 检验 `p=0.1417`。按数据集，`balance_scale` 的 full−nomix 为 +0.080941（3/3 seeds 正向），但 `spect_heart`、`vehicle`、`vertebral_column` 的均值差为负；因此 topology mixing 是数据集几何敏感的双刃剑，不能宣称稳定有效。

逐数据集的 ARI/NMI 与 AHDPC 联表如下。`正向 seed` 是 full 的 ARI 高于 NoMix 的 seed 数，不是统计显著性检验。

| 数据集 | Full ARI | NoMix ARI | ΔARI | 正向 seed | ΔNMI | ΔARI vs AHDPC | 分类 |
|---|---:|---:|---:|---:|---:|---:|---|
| `balance_scale` | 0.1900±0.0892 | 0.1091±0.1033 | **+0.0809** | 3/3 | +0.0815 | +0.1757 | 拓扑正例且基线胜 |
| `glass` | 0.3904±0.0098 | 0.3497±0.0689 | +0.0407 | 2/3 | +0.0240 | −0.0191 | 相对 NoMix 改善但基线负 |
| `image_segment` | 0.2728±0.0399 | 0.2515±0.0140 | +0.0214 | 2/3 | +0.0001 | −0.1484 | 小幅改善但基线负 |
| `landsat` | 0.5379±0.0064 | 0.5324±0.0099 | +0.0055 | 3/3 | +0.0002 | +0.0248 | 微小正例，random 均值更高 |
| `spect_heart` | 0.2481±0.0253 | **0.2606±0.0169** | −0.0125 | 1/3 | −0.0080 | +0.2613 | 基线优势不依赖拓扑 |
| `vehicle` | 0.0749±0.0099 | **0.0863±0.0274** | −0.0114 | 1/3 | −0.0110 | −0.0120 | NoMix 更好且基线负 |
| `vertebral_column` | 0.2340±0.0072 | **0.2511±0.0130** | −0.0171 | 0/3 | −0.0062 | −0.0230 | NoMix 更好且基线负 |

因此，`spect_heart` 的 V9−AHDPC 正差值在 NoMix 下仍保留，不能把该优势写成拓扑带来的收益；`balance_scale` 才是当前最完整的“Full > NoMix 且 Full > AHDPC”案例。`landsat` 的正差很小，且 `v9_random` ARI 均值为 0.5392、高于 Full 的 0.5379，不能据此证明 reliability 权重最优。该消融只有 7 个数据集，不能外推为论文预处理 24 个数据集的普遍拓扑结论。

## 5. V12/V13/V14 迭代边界

- V12 risk-adaptive：12 个相关数据集、4 variants、3 seeds，共 144/144 成功；宏平均 V12 full−nomix = −0.001244，V12 full−V9 full = −0.000087，Wilcoxon full vs nomix `p=0.8828`。未改变行为。
- V13 assignment-residual：12 个相关数据集、full/nomix、3 seeds，共 72/72 成功；head ARI full−nomix = −0.000238，Wilcoxon `p=0.8314`。KMeans embedding 差值约 +0.003657，但没有显著性证据。
- V14 strict minimum：5 个代表性数据集、full/nomix、3 seeds，共 30/30 成功；宏平均 full ARI=0.133629，nomix ARI=0.129256，差值 +0.004373；Wilcoxon `p=0.8139`，配对 t 检验 `p=0.6597`。full 分支平均 target gate 约 0.006276，说明拓扑被调用但监督质量/强度仍不足。V14 标记为 **机制可运行、性能 no-go**，不提升为论文主方法。

V14 配置与 runner：

- `methods/TopoGate/V11/configs/topogate_v14_advantage_minimum.yaml`
- `scripts/v9_learnable_gate/run_v14_advantage_smoke.py`
- `result/v14_results_2026-08-03_advantage_5ds/runs.csv`

## 6. 研究结论与下一步

当前可以发表/报告的结论是：V9 对 `spect_heart`、`balance_scale`、`landsat` 存在协议特定的局部优势；优势与维数和邻域尺度相关，但不能由 mutual-neighbor 比例单独预测。nomix 消融不稳定，V12–V14 尚未产生显著、稳定的 topology 增益。后续若继续，应先研究更强的无标签拓扑目标校准与 gate-target 强度，而不是继续扩大未证实版本的 benchmark。
