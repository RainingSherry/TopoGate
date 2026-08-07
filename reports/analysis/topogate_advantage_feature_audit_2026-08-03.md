# TopoGate 优势/劣势数据集与真正 TDA 特征审计

生成时间：2026-08-03。特征计算只读取 NPZ 中的 `x`；脚本没有加载 `y`，也没有用标签选择图、阈值、尺度或 variant。结果列在特征提取后从持久化 CSV 连接，仅用于事后解释。

## 研究边界

本报告把现有 `PCA/kNN`、mutual/SNN 和动态图称为有限图结构，不把它们称为 persistent homology。`tda_h0_*` 是固定稀疏 kNN Vietoris–Rips 1-skeleton 上的 0 维 component persistence 摘要；`cycle_rank_*` 是阈值图 1-skeleton 的循环秩。它们是可审计的 TDA/拓扑诊断候选，但不是完整 dense VR complex 的 H1 persistence diagram。

## 版本结果边界

- V9 paper-preprocess：相对 AHDPC 只有 `spect_heart`、`balance_scale`、`landsat` 正差值；Full-NoMix 仅在 7 个相关数据集上配对。
- V11 minimum：5 datasets × Full/NoMix × 3 seeds；宏观 head ARI 差值接近零且为负。
- V12/V13：同一批 12 个扩展数据集；两者 Full-NoMix 均未形成正向稳定证据。
- V14：5 个代表性数据集；机制路径可运行，但配对 ARI 增益未显著。
- StaticGate：15 个数据集的历史消融只作机制方向参考，不与 V9/V11 输入协议混合。

## 特征计算覆盖与协议

本轮共有 49 个结果相关数据集；47 个完成无标签特征计算，2 个因矩阵元素上限跳过。计算协议为标准化后 PCA 上限 50、graph `k=5`、TDA skeleton `k=15`；超过 4000 个样本或 512 个特征时使用固定随机子集。`analysis_sampled` 和 `analysis_feature_sampled` 在 CSV 中显式记录。

`Campbell` 与 `hrvatin_filtered` 被跳过；V11 的 `Mouse_retina`、`enron` 和 StaticGate 的部分高维数据使用采样结果。采样特征只用于生成候选假设，不足以替代完整矩阵的预注册验证。

## 如何解读优势与劣势

当前最完整的拓扑正例是 `balance_scale`：V9 Full 同时高于 NoMix 和 AHDPC，且 3/3 seeds 方向一致。`spect_heart` 的 V9 相对 AHDPC 优势在 NoMix 下仍存在，因此不能归因于 topology mixing；`landsat` 差值很小且 random 邻居均值略高，也不能证明 reliability 权重有效。

## 各版本正负集合

下列集合按同一批次内的 Full−NoMix 或 V9−baseline 结果列出；`positive/negative` 只描述方向，不代表显著性。

- **V9 vs AHDPC** (`n=24`)：正向 `spect_heart` (+0.2613), `balance_scale` (+0.1757), `landsat` (+0.0248)；负向 `banknote` (-0.9508), `flame` (-0.8342), `asymmetric` (-0.4400), `unbalance` (-0.4250), `aggregation` (-0.4203), `olivetti_faces` (-0.3896), `rice` (-0.2694), `smile` (-0.1859), `2d_20c_no0` (-0.1826), `2d_4c_no9` (-0.1540), `image_segment` (-0.1484), `libras_movement` (-0.0637), `vertebral_column` (-0.0230), `2d_4c_no4` (-0.0229), `glass` (-0.0191), `twodiamonds` (-0.0149), `student_evaluation` (-0.0140), `vehicle` (-0.0120), `dim064` (-0.0035), `website_phishing` (-0.0034)。
- **V9 vs HDPC** (`n=24`)：正向 `spect_heart` (+0.2734), `website_phishing` (+0.2217), `landsat` (+0.0682), `vertebral_column` (+0.0158), `vehicle` (+0.0063)；负向 `flame` (-0.6295), `banknote` (-0.6123), `olivetti_faces` (-0.4595), `asymmetric` (-0.4376), `unbalance` (-0.4250), `2d_4c_no4` (-0.3133), `2d_20c_no0` (-0.2250), `aggregation` (-0.1892), `smile` (-0.1859), `rice` (-0.1792), `2d_4c_no9` (-0.1540), `image_segment` (-0.1486), `balance_scale` (-0.1481), `libras_movement` (-0.0566), `glass` (-0.0491), `student_evaluation` (-0.0253), `twodiamonds` (-0.0149), `dim064` (-0.0035)。
- **V9 Full−NoMix** (`n=7`)：正向 `balance_scale` (+0.0809), `glass` (+0.0407), `image_segment` (+0.0214), `landsat` (+0.0055)；负向 `vertebral_column` (-0.0171), `spect_heart` (-0.0125), `vehicle` (-0.0114)。
- **V11 Full−NoMix** (`n=5`)：正向 `spambase` (+0.0047), `Mouse_retina` (+0.0006), `har` (+0.0001)；负向 `enron` (-0.0078), `breast_cancer_wisconsin_original` (-0.0000)。
- **V12 Full−NoMix** (`n=12`)：正向 `spectf_heart` (+0.0333), `banknote_authentication` (+0.0192), `statlog_image_segmentation` (+0.0135), `glass_identification` (+0.0051)；负向 `rice_dataset_cammeo_and_osmancik` (-0.0397), `wine` (-0.0133), `seeds` (-0.0098), `ionosphere` (-0.0079), `image_segmentation` (-0.0046), `vehicle` (-0.0038), `vertebral_column` (-0.0038), `satellite_image` (-0.0032)。
- **V13 Full−NoMix** (`n=12`)：正向 `statlog_image_segmentation` (+0.0087), `vehicle` (+0.0058), `banknote_authentication` (+0.0014), `wine` (+0.0008)；负向 `seeds` (-0.0116), `ionosphere` (-0.0029), `satellite_image` (-0.0019), `vertebral_column` (-0.0019), `image_segmentation` (-0.0013)。
- **V14 Full−NoMix** (`n=5`)：正向 `landsat` (+0.0316), `balance_scale` (+0.0028), `vehicle` (+0.0012)；负向 `vertebral_column` (-0.0138)。
- **StaticGate Full−NoMix** (`n=15`, 单 seed 历史表)：正向 `har` (+0.0997), `Campbell` (+0.0305), `spambase` (+0.0265), `cnae9` (+0.0158), `breast_cancer_wisconsin_original` (+0.0111)；负向 `iris` (-0.1318), `enron` (-0.1076), `ISOLET` (-0.0776), `Quake_Smart-seq2_Lung` (-0.0434), `sms_spam_collection` (-0.0243), `mammographic_mass` (-0.0176), `reuters` (-0.0046), `Mouse_retina` (-0.0040), `first-order-theorem-proving` (-0.0027)；`hrvatin_filtered` 为近中性 (+0.0005)。该表只描述方向，不提供多 seed 稳定性。

V12/V13/V14 的结果说明继续增加 risk、assignment residual 或 strict minimum 约束并没有把局部拓扑信号稳定转化为聚类收益。随后完成的 TDA H0/fixed/random 正式 pilot 也没有在固定五数据集协议内显示独立聚类收益，因此当前保留为 no-go 诊断，不进入训练梯度路径。

## 特征与版本增益的统计边界

下面的 Spearman 结果使用本报告的标准化/PCA/采样协议，是小样本探索性关联，不是选择配置或证明因果；不得以其 p 值直接支持论文性能结论。若某版本只有 5 个数据集，相关系数只用于生成下一轮预注册候选，不用于宣称普遍规律。它不能直接替换既有 `geometry_features_no_label.csv` 的历史标准化协议。

| outcome | strongest exploratory feature | n | rho | p |
|---|---|---:|---:|---:|
| `v9_full_nomix_ari` | `cv_knn_cosine_distance` | 7 | 0.607 | 0.1482 |
| `v11_full_nomix_ari` | `sparse_graph_components` | 5 | 0.707 | 0.1817 |
| `v12_full_nomix_ari` | `mean_mutual_ratio` | 12 | 0.182 | 0.5717 |
| `v13_full_nomix_ari` | `tda_h0_q50_death_norm` | 12 | 0.310 | 0.3270 |
| `v14_full_nomix_ari` | `p95_knn_cosine_distance` | 5 | -0.800 | 0.1041 |
| `staticgate_full_nomix_ari` | `cycle_rank_at_1nn_scale` | 13 | 0.368 | 0.2159 |

## TDA pilot 结论与下一候选

1. 固定 raw kNN 稀疏 1-skeleton 的 H0 pilot 已完成 5 datasets × 5 variants × 3 seeds，共 75/75；H0、fixed-filtration 和 random 相对 `V11_full` 的 head/KMeans 差值分别为 `+0.000010/-0.000726`、`+0.000002/-0.000665` 和 `+0.000018/-0.000274`。
2. 该结果满足固定协议内 no-go：prior 改变了 graph target/gate mass，但没有形成稳定的独立聚类收益。完整逐数据集表见 `result/analysis/topogate_cross_version_landscape_2026-08-03_tda.csv`。
3. 下一候选只能是默认关闭、可回退的 detached prior：检查“早合并稳定边增强、晚合并 bridge 边抑制”的相反语义；先做 toy graph 单元测试，再决定是否运行新的多种子对照。该候选不是已验证方法。
4. H1/persistence image/Mapper 只有在引入经过验证的 TDA 库、固定 subsampling 和复杂度上限后才进入第二阶段；不能用当前 graph cycle-rank proxy 替代 H1 persistence。

## 可复核产物

- `result/analysis/topogate_dataset_features_2026-08-03.csv`：标签隔离的特征与事后结果连接表。
- `result/analysis/topogate_feature_version_correlations_2026-08-03.csv`：探索性 Spearman 表。
- `scripts/analysis/build_topogate_dataset_feature_audit.py`：可重跑脚本。
