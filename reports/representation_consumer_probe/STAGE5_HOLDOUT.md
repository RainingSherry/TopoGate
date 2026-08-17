# S5 — Dormant Holdout Manifest

本文件和机器可读 manifest 保留既有 outcome-independent holdout freeze，作为审计历史；在当前
`representation_consumer_probe` 中永久 dormant，原因是 S0 的 `adapter_not_estimable` 关闭了
T-related causal chain，且本项目不再执行 S3/S4/S5/S6。

```text
status = dormant_due_to_adapter_not_estimable
```

因此这里的 12 个 dataset IDs、五个 seeds 和 source manifest 不进入当前 S1/S2 denominator，
也不支持 generalization claim。未来若要研究 sample-edge selector 或新的 consumer，必须创建
新的独立项目（建议 `relation_selection_probe`），重新冻结输入、预算、adapter 和 holdout；
不得把本 manifest 作为新项目的隐式授权。

manifest 仍保留原始 selection basis、label-free coverage features、universe、exclusion reasons
和 freeze-time source hashes，便于证明它没有按本项目 outcome 选样。
