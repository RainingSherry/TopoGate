# S0 Decision — Protocol and Adapter Audit

正式 S0 replay 状态：`adapter_not_estimable`；`S1_opportunity_only=allowed`；S2 confirmation
仍是 conditional；S3/S4/S5/S6 permanently locked。可复核工件位于
`result/representation_consumer_probe/S0_freeze/`。

S0 不是性能实验。它只验证公共 H0/SVD、positive-cosine candidate pool、row-specific budget、
graph/loss numerical contract、真正的 Spectral apparatus 和标签隔离，并审计现有 V21/V25 代码
是否能从 feature-side gate 直接产生 sample-edge membership。

## Adapter boundary

现有 FeatureGate 输出 feature-coordinate scores，当前代码没有语义忠实的
feature-mask→sample-edge membership adapter。若补 adapter 需要新 aggregation、threshold、
trainable mapping 或 human rule，就会改变研究对象；因此 `adapter_not_estimable` 在本项目中
是 terminal state，不是继续添加 selector 的中间状态。

## Budget boundary

S0 使用 `budget_cap=8`，不是 exact-8 eligibility filter。每行保存：

```text
b_i = min(8, positive_count_i)
```

R/O_pool/O_full 必须共享这个 vector；保存 effective-budget hash、均值、最小值、低于 cap 的
比例和零预算比例。六个 stress datasets 均保留在 manifest 中；positive shortfall 不再整集
排除。正式 replay 的 zero-budget row 数为：`cnae9=1`、`sms_spam_collection=40`、
`hate_speech=135`；其余三个 stress datasets 的 zero-budget row 数为 `0`。

## Synthetic boundary

`synthetic_apparatus.json` 分开记录：

- `graph_numerical_sanity`：L/L_sym/loss finite；
- `spectral_recovery_sanity`：clean block `ARI>=0.95`、clean 优于 contaminated、isolate rows
  为零且不进入 active eigenstructure。

它们只是测量装置 contract，不是数据集性能证据。

## 解锁

正式 S0 artifact（含 `artifact_hashes.json`）写入并通过 source/hash/contract 后，只能运行不含 T 的
`S1 opportunity-only`；必要时运行 S2 SimpleCut confirmation。当前项目终点是 Decision：

```text
opportunity_absent_under_frozen_relation_family
candidate_family_requires_review
opportunity_positive_diagnostic
```

其中第一项只适用于 frozen H0/cosine/budget/tested consumers，不能扩展为“topology 没有价值”。
holdout manifest 状态为 `dormant_due_to_adapter_not_estimable`，不执行、不进入 generalization
claim。
