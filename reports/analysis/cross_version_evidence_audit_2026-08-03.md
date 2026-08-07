# TopoGate 跨版本证据与 provenance 审计

生成时间：2026-08-03。该报告只读取当前 `result/` 软链接目标中的 CSV/JSON；不重新训练、不读取标签做选择，也不改写历史产物。

## 输出

- `cross_version_evidence_2026-08-03.csv`：按数据集和 variant 聚合的多种子指标、Full/NoMix 差值和 gate/risk 诊断。
- `paired_version_deltas_2026-08-03.csv`：同一批次内的配对 ARI 差值，不能跨协议合并。
- `provenance_audit_2026-08-03.csv`：summary/CSV 中 source hash、K 来源、标签字段和数据集身份的覆盖情况。

## 解释规则

1. `V9 paper_preprocess` 的 AHDPC/HDPC 是持久化单次参考；它的 `ari_vs_ahdpc`/`ari_vs_hdpc` 不是对称多种子差值。
2. V9、V12、V13、V14 和 V11 的 Full/NoMix 差值只在同一结果批次内配对；同名数据集如果 source hash 或预处理不同，不能拼成跨版本纵向结论。
3. `provenance_status` 不是性能评价。`partial_*` 表示指标仍可由 CSV/run_record 和 source hash 追溯，但不能把缺失字段写成已记录。

## 关键发现

- V9 论文匹配批次必须保留 `3/1/20` 的 AHDPC 胜/平/负边界；三个正差值为 `spect_heart`、`balance_scale`、`landsat`。
- V9 优势消融的 `summary.json` 有 `dataset=adhoc`，真实身份在 `ablation_runs.csv`/`run_record.json` 中；该批次被标为 `partial_summary_dataset_adhoc`。
- V12 的 summary 同样保留 `adhoc`，且 source path、K protocol、显式 label flag 不在 summary 中；使用 runs.csv 的 source/hash 和 runner 源码审计作为补充。
- V11 minimum 5x3 具备 source hash、`benchmark_oracle_from_y` 和语义分离的 prediction/labels_true 输出，但 summary 未显式写 `labels_used_during_fit=false`；这是文档契约缺口，不是标签泄漏证据。
- V14 的 gate/risk 诊断可以证明路径被调用，但 paired ARI 增益仍不足以晋级主方法；应继续报告 gate coverage、target gate 和 readout 分歧。

## 书籍与数学边界

当前源码中的 kNN、mutual/SNN、动态图刷新和边可靠性属于依赖度量的有限图结构；它们没有 filtration、simplicial complex、boundary operator、homology 或 persistence diagram。因此本表不会把这些量标成 persistent homology。真正 TDA 的第一版应作为 detached edge prior/诊断，并保留 NoMix、原 V11、random prior 和 fixed filtration 控制。

参考书映射见 `TopoGate_whole_project_math_TDA_audit_2026-08-03.md`：拓扑学用于区分邻域图与同调不变量，数学分析用于约束 kNN 离散跳变和 EMA 稳定性表述，Bishop/PRML 用于解释 mixture responsibility、目标错配和无监督 K 协议。
