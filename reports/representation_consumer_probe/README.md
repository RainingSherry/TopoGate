# Representation-Consumer Probe

这是一个独立的 mechanism study，不叫 V26，也不继续扩展 Gate。它只回答：在冻结的公共
`H0 → candidate relation pool → row-specific budget → graph consumer` family 中，是否存在可被
理想 relation membership 选择利用的 clustering opportunity。

当前 V21/V25 的 feature-side Gate 没有可审计的 sample-edge adapter，S0 因此把
`adapter_not_estimable` 定义为本项目 T-related causal chain 的 terminal state。项目执行链收缩为：

```text
S0 Freeze + adapter audit
    → S1 opportunity-only (F/U/R/O_pool/O_full + Spectral)
    → S2 opportunity confirmation (SimpleCut, if needed)
    → Decision
```

S3 objective isolation、S4 strong backbone、S5 holdout、S6 paper-scale expansion、TopoCut 和
任何新 selector 在本项目中永久锁定。未来 sample-edge selector 研究必须新建
`relation_selection_probe`。

## 文件

- `PROTOCOL.md`：科学边界与标签隔离；
- `S0_FREEZE.md`：数值、预算、solver 和 artifact 合同；
- `EXECUTION_PLAN.md`：唯一授权的执行顺序；
- `PRE_REGISTRATION.md`：S1/S2 opportunity estimands 与 terminal rules；
- `STAGE1_ORACLE.md`：row-specific budget、R/O_pool/O_full 与 Spectral；
- `S1_RESULTS.md`：S1 v2 dataset-level `H_pool/H_full/C` 原始表与条件 S2 决策；
- `STAGE2_CONSUMER.md`：SimpleCut confirmation；
- `S2_RESULTS.md`：S2 18-run SimpleCut opportunity confirmation、审计边界与 terminal decision；
- `DECISION.md` / `S0_DECISION.md`：当前终局和执行边界；
- `STAGE5_HOLDOUT_MANIFEST.json`：保留但 dormant 的历史冻结 holdout。

带时间戳的 `*_20260817_133751.md` 文件是上一轮审查快照，仅用于 provenance，不是当前协议事实
源；当前事实源是上面列出的 canonical 文件和 formal S0 artifacts。

正式 S0 工件只能写入 `result/representation_consumer_probe/S0_freeze/`。临时审计、smoke、资源
失败和外部 review 失败不能冒充性能结果。GPU 资源充足不改变 S1 的 CPU-first 语义；需要 GPU
时只使用 `[1,2,3,4,5,6]`，物理 GPU 0/7 永远禁用。

S1 正式结果使用 `result/representation_consumer_probe/S1_oracle_v2/`；旧的
`S1_oracle/` 保留为 `invalid_design` provenance，原因见其 `INVALID_DESIGN.md`。
