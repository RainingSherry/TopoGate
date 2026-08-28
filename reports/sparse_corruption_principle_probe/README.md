# Sparse Corruption Principle Probe

这是一个独立的机制研究项目，不属于任何 V 系列。它承接已经关闭的
`adaptive_corruption_probe` B1 现象，但不把 B1 重新解释成 confirmation，也不修改旧
`adaptive_corruption_probe` 或 `learned_relation_rule_probe`。

## 有限问题

> 在 naturally sparse high-dimensional data 上，corruption 带来的 clustering 变化主要来自
> support、value、difficulty，还是它们改变了 cell-cell geometry？是否存在一个足够简单且可
> 转移的 static principle？

当前项目已经完成：

```text
C0 Freeze
  -> C1 Mechanism Localization (no new model fit)
  -> C2 finite static corruption library + toy contract
  -> C2 54-run GPU matrix + independent integrity audit
```

C2 矩阵已经按显式 `sparse_corruption_principle_probe_c2_v1` overlay 完成；C3 holdout runs、
adaptive policy、MLP selector、GAN 或 learned generator 仍未授权。GPU 资源只用于加速已冻结
矩阵，不改变阶段门控。

## 研究三角

| development dataset | frozen role | question |
|---|---|---|
| `Mouse_retina` | corruption-presence case | 为什么 random corruption 已经有帮助？ |
| `Baron Human` | support-sensitive case | SupportOnly 的大幅变化是否与 support semantics 有关？ |
| `Campbell` | difficulty-sensitive case | StaticHard 的变化是否来自 coordinate difficulty？ |

三者是 mechanism/development panel，不是独立 generalization denominator。

## 证据边界

- C1 的 B1 指标只从关闭项目的 compact post-fit summary 读取；结构指标由固定 S0 H0 的
  label-free replay 重新计算，`fit_runs=0`。
- `H0` 的 support 是 B1 已冻结的阈值 support proxy，不等同于原始 count matrix 的零模式；
  这一限制会在报告中保留，不能被写成“已证明生物学零语义”。
- toy S/V/M 只验证 apparatus 能区分 support/value 设计，不是 clustering 结果。
- 所有正式 performance 的 K/ARI/NMI/ACC 都是 fit 后外层 benchmark readout；标签不得进入
  corruption、preprocessing、loss、optimizer 或 principle selection。
- 发布层仅包含 protocol、manifest、compact summaries、audits 和重要图表；原始数据、标签、
  arrays、embeddings、predictions、weights、checkpoints、logs、cache 不进入 GitHub。

## 当前状态

以 `scripts/sparse_corruption_principle_probe/protocol.py` 为 machine-readable authority。
`result/sparse_corruption_principle_probe/` 只保存本项目的 compact/timestamped artifacts。
所有 GPU 运行必须显式使用 `[1,2,3,4,5,6]`，物理 GPU `0` 与 `7` 禁止使用。

当前 C2 终态为 `simple_static_principle_sufficient`：54/54 runs completed-valid，P2 在三个
development datasets 上达到 `Delta_ARI >= 0.03`，P3/P4 各在两个数据集上达到该描述性 margin。
这只是 tested static library evidence，不是 oracle 上界，也不是 raw-X sparse-support 结论。
