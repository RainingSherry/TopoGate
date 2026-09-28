# E1：匹配队列收益背景（复用核验）

本步骤将“多跳值得研究”落实为跨数据的观察，而不是用单一案例代替效果背景。已逐文件核验现有 figure provenance 的 SHA256，重新从 CSV 计数；未运行 encoder 训练。

59 个有效配置匹配数据集：46 个平均 ΔARI 为正、8 负、5 持平；22 个三个训练 seed 全部为正。全队列平均 ΔARI=0.0056568483。三案例按此前固定规则保留，不根据本轮结果替换。

数据见 outputs/E1/summary.json、discovery59_raw.csv；既有收益图已复制为 outputs/E1/figure2_discovery_bridge.pdf/svg/png。主编辑已检查 PNG：完整正负分布、三个案例及每个 seed 均可辨识。

解释：在已具备一阶优势的研究队列中，改变训练来源范围仍反复产生额外收益。这为进一步研究“新增关系如何改变聚合”提供背景；E2/E3/E4 将分别回答有用范围、关系类型与具体分布中的改善位置。队列是历史性能选择后的研究队列，评价为 Full-X，设置中统一说明。

可用正文：Across the 59 configuration-matched datasets, expanding training-source reach yields positive mean ARI changes in 46 cases, with consistent improvements across all three training seeds in 22 cases. These recurring gains motivate a closer examination of which additional relations enter the recovery task and how they affect the resulting partitions.

状态：多跳背景复用完成；131 CLUBench / 11 biological 的 benchmark 表继续引用原文档，未在本步骤重新审核所有模型的评估可比性，也不新增 SOTA 判定。
# 本轮重算口径更新

本文件原有46/59为冻结历史readout背景。本轮对相同锁定embedding统一重算10个KMeans读出后为44正、10负、5平，平均ΔARI=0.0056647191，22个数据集三个训练seed均正。Limb_Muscle与dry_bean两个接近零的差值翻号，分别为−0.000132、−0.000073；三个主案例结果不变。保留历史图表作来源归档，最终主文应统一使用本轮44/59及outputs/E3_cohort结果，不能把历史46/59与本轮误差分解混写。已保存逐数据集对照historical_readout_comparison.csv；尚未将数值差异归因到某个具体软件或线程因素。

