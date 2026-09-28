# E4：三案例的改善位置（冻结结果后处理）

研究问题：多跳改善发生在已有局部片段内部，还是同类片段之间？利用同一输入图的社区，将同类样本对分为社区内与跨社区两组；对每个训练seed的10个读出取均值，再对3个seed取均值。下表按实际样本对数量汇总，未按结果挑选群体。

| 数据集 | 社区内 H1 → multi | 跨社区 H1 → multi | 跨社区增量 |
|---|---:|---:|---:|
| Baron | 94.32% → 95.54% | 57.65% → 69.60% | +11.95 pp |
| Quake Lung | 99.16% → 99.42% | 57.01% → 61.65% | +4.64 pp |
| student_grade | 89.05% → 89.04% | 59.08% → 61.90% | +2.82 pp |

发现：三个案例的共同现象是，同类片段内部在 H1 下已有较高一致性，而多跳之后跨片段共同分配增加更明显。student_grade 的内部共同分配基本不变，跨片段提升依然存在，因此无须把三个案例说成所有局部关系都改善。

模型联系：这些冻结结果把“聚合收益”定位在同类局部片段之间，为扩大训练来源范围提供具体任务价值。E5需要将新增来源的结构角色与这些关系对齐，E6进一步比较路径来源和直接宽kNN来源。这里不是逐路径因果归因，也不把图社区自动当成生物状态。

证据：outputs/E4_summary.json；每例 e4_community_pair_per_seed.csv、e4_community_pair_coassignment.csv；e4_community_matrix.pdf/svg/png。矩阵按群体大小选18个群体展示，完整表包含全部同类社区对和内部对角线。Baron PNG 已人工检查；主文最终图仍应合并三例并统一排版。

可用正文：Across the three cases, first-order embeddings already support comparatively coherent within-community class groups, while multi-hop training primarily improves their integration across communities. Cross-community same-class coassignment increases by 11.95, 4.64, and 2.82 percentage points in Baron, Quake Lung, and student_grade, respectively. This locates the additional benefit in relations that extend beyond already coherent local groups.

状态：三例确定性后处理已完成；仍需最终Sol/Claude审阅、E2/E5/E6结果合并。未新增训练、未改变锁定参数。
