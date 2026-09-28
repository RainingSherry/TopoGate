# V0_RG 多跳来源组织实验（2026-09-27）

本目录保存 ToPoGate V0_RG 多跳实验的**冻结执行代码、结果汇总、论文分析和图表**。`../v0rg_tunable_multihop/` 是实际使用的多跳模型源码；`../source_coverage_20260925/` 与 `../multihop_mechanism_20260924/` 提供本轮训练所依赖的来源构造代码。源码与服务器运行版本的 SHA-256 对照见 `../frozen_source_manifest_20260927.json`。

## 主要结果

在此前按排名选出的 59 个数据集中，统一读出后 44 个数据集的平均多跳增益为正、10 个为负、5 个持平；这个比例描述筛选队列，不能外推为随机数据集的成功率。三个预先锁定的深入案例均在 3 个训练 seed 上改善：

| 数据集与锁定深度 | H1 ARI | 多跳 ARI | 等候选数 Kwide ARI |
| --- | ---: | ---: | ---: |
| Baron，H8 | 0.771040 | 0.837703 | 0.783289 |
| Quake_Smart-seq2_Lung，H8 | 0.706870 | 0.738201 | 0.703739 |
| student_grade，H4 | 0.328167 | 0.367869 | 0.350063 |

每个训练 seed 先平均 10 次 KMeans 读出，再汇总 3 个训练 seed。Kwide 逐样本匹配候选数量；它没有匹配来源权重、扰动方向或实际位移，因此比较检验的是**整套来源组织方式**。三案例的跨图社区同类共同分配率分别提高 11.95、4.64、2.82 个百分点。Baron 的 5 个独立 seed 真实采样轨迹中，38.93% 的邻域混合内部权重来自 H1 未覆盖的图社区；其中占**总混合权重**21.45 个百分点的是同类来源。图社区不直接等于生物学状态。

## 文件导航

- `FINAL_ANALYSIS.md`：中文完整解释、方法与实验的衔接，以及证据边界。
- `RESULT_E*.md`：各步骤的表、图和结论；`EXECUTION_SPEC.md`：冻结的研究设计。
- `outputs/E3_cohort/`：59 数据集统一读出与同类拆分、异类合并结果。
- `outputs/E2_E6/`：三案例深度曲线、等数量 Kwide 比较和饱和检查。
- `outputs/E4_layouts/case_layouts.pdf`：三个案例在**固定输入布局**上的跨社区同类共同分配率；颜色并非 embedding 位移。
- `outputs/E5_trace/`、`outputs/E5_shared_partition/`：Baron 真实来源角色与对齐图社区后的逐类结果。
- `outputs/E5_response/`：固定 checkpoint 的方向响应探针，作为辅助结果。
- `../weight_index_20260927.json`：45 个模型单元、90 个权重和 embedding 文件的 SHA-256 与字节数。
- `PUBLIC_OUTPUT_MANIFEST.json`：本目录公开结果文件的 SHA-256。

本轮新增 21 次训练、复用 24 个历史模型单元，新增训练均成功。模型拟合、预处理、建图和参数选择不读取测试标签；benchmark 的类别数 K 由真值给定并明确记录，标签用于事后指标和机制分析。运行仅使用服务器 GPU 2、4。历史复用的 5 个 Baron 来源 seed 与 3 个读出 seed 属于独立集合，逐类对应是描述性证据。student_grade 的 H4/H8 候选分布和 embedding 逐元素相同，实际可达阶数在 H4 饱和。

## 复现边界

本目录保留实验当时运行的脚本和锁定参数；执行脚本中的服务器绝对路径记录了原始环境。其他机器需要将实验根目录、数据与历史 checkpoint 路径映射到本地，并重新记录修改后的脚本哈希。多跳核心依赖仓库的 `methods/TopoGate/V0_RG/` 与 `methods/NeighborMix_scMAE/`。数据 NPZ、逐样本真值与预测、完整采样轨迹、45 组 checkpoint/embedding 不进入 Git；它们在研究服务器保留，公开的 `weight_index_20260927.json` 可用于核验文件。

本公开快照保留汇总 CSV、逐 seed 汇总、可编辑矢量图及预览图。原始运行日志与大数组不在此处。代码静态校验和结果一致性审查记录见 `AUTO_REVIEW.md`、`DEPTH_WORKER_REVIEW_LOG.md` 与 `MONITOR_FINAL_AUDIT.md`。结果支持对三个选定案例的来源组织与群体整合论述；未据此宣称广泛的生物语义恢复、严格因果归因或统一协议 SOTA。
