# Parent inspection of E4 preliminary matrices

2026-09-27: inspected all three exported PNGs after the palette correction. H1 and multihop panels use the same 0–100% scale. Difference panels explicitly use percentage points. Gray cells are excluded different-class relations, not zero coassignment. Ordered class/community labels and sample counts are visible; no overlap hides data.

Observed figure structure: Baron class 2 contains strong cross-community positive changes; Lung class 7 has several positive off-diagonal changes while class 2 is mixed; student_grade class 1 is mixed with net positive cross-community change and class 0 changes little. These are descriptive observations of displayed size-selected groups, not a selection rule for further training.

Before manuscript assembly, combine these with the frozen input layouts, provide verified class-name mappings where available, reduce excess whitespace, and explain excluded gray cells in the caption. Current files are analysis figures, not a claim of final publication layout. Keep all-class and all-community-pair tables available; do not omit negative cells.

## 最终三案例统一输入布局图
Claude提案经父编辑修正：逐anchor O(n²)分母改为联合计数；恢复所有类别样本显示；固定同一布局和跨案例共享色标；补输入/代码哈希和类别名称映射。三个随机小样本的逐点暴力计算与向量算法一致；三案例按有效样本对数量加权的总体共同分配率与E4一致（≤1e-12）。父编辑已查看PNG，无遮挡，灰色点代表不存在跨社区同类对。图不表示embedding位移。

GPT-6 Sol Max 独立复核：每案例随机20个有效anchor ×3seed×读出0/3/9逐点重算误差0；CSV与3×10数组在1e-14内一致；分母加权总体与E4最大差1.11e-16；9输入及代码哈希一致。已补output_hashes.json。
