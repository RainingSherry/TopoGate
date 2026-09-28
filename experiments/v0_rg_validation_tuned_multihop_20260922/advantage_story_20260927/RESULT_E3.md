# E3：全队列的聚类关系变化

59个匹配数据集均完成；对每个H1与锁定multihop embedding使用10个KMeans states，先求每训练seed均值，再汇总3个训练seed。未重新选参，未改变encoder权重。44个平均ΔARI正、10个负、5个平；22个三个训练seed均正；30个同时减少同类拆分和异类合并。

| 案例 | H1 ARI | multihop ARI | ΔARI | 同类拆分减少 | 异类合并减少 |
|---|---:|---:|---:|---:|---:|
| Baron | .771040 | .837703 | +.066663 | 7.415 pp | .539 pp |
| Quake Lung | .706870 | .738201 | +.031330 | 3.617 pp | .040 pp |
| student_grade | .328167 | .367869 | +.039701 | 2.196 pp | 1.782 pp |

三个案例均为3/3训练seed正向。两种错误的分母不同，不能把两个百分点相加或当作ARI的可加分解；这里用于定位关系变化。Baron、Lung主要呈现同类拆分减少，student_grade同时改善两种关系，为模型叙事提供互补而可衔接的观察。

历史E1保存的46个正值与本轮44个正值不同，差异集中在极小效果处：Limb_Muscle与dry_bean发生接近零的翻号。三主案例与冻结历史结果完全一致。最终表图用本轮统一口径；历史逐数据差异另存，不择优采用两次读出。

产物：outputs/E3_cohort/summary.csv、paired_per_seed.csv、partition_gain_plane.pdf/svg/png、historical_readout_comparison.csv、input_hashes.json。图展示全部59个数据集，仅给三个预先固定案例标注名称。

可用正文：Multi-hop training improves mean ARI on 44 of the 59 matched datasets, with positive changes across all three training seeds on 22 datasets. The partition-level analysis shows that these gains are not confined to a single error pattern: Baron and Quake Lung primarily improve same-class aggregation, whereas student_grade improves both aggregation and separation. This motivates examining which additional relations become available beyond first-order neighborhoods.
