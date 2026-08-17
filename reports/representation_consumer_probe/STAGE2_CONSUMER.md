# S2 — SimpleCut Opportunity Confirmation

S2 只用于防止把 Spectral relaxation 的阴性误判为 frozen relation family 没有 opportunity。
它不是新 backbone，也不是 TopoGate selector 实验；当前项目仍没有 `T_adapter`。

## SimpleCut probe

输入为同一冻结 `H0` 和同一 R/O graph artifacts。encoder 固定为 `128→64→32`，不使用 GCN、
GAT、Transformer、OT、ZINB、DEC、GAN 或 contrastive branch。对 `W_sym` 使用 full-graph
normalized graph energy、degree-weighted orthogonality 和 variance penalty：

```text
E_cut = Tr(Z.T @ L @ Z) / max(Tr(Z.T @ D @ Z), eps)
L     = E_cut + lambda_orth * L_orth + lambda_var * L_var
```

optimizer、epochs、initialization、lambda 和 readout 必须来自 S0 resolved config；不能因结果调参。
SimpleCut 的 K 不进入 representation，只进入最终 known-K KMeans readout。

## Opportunity confirmation

S2 仍只报告：

```text
H_pool = ARI(O_pool) - ARI(R)
H_full = ARI(O_full) - ARI(R)
C      = H_full - H_pool
```

它不计算 `S_graph`、不声称 selector utility、不创建 S3/S4 代码。结果分类为：

- `opportunity_absent_under_frozen_relation_family`：tested consumers 上两类 H 都小；
- `candidate_family_requires_review`：`H_full` 明显高于 `H_pool`；
- `opportunity_positive_diagnostic`：冻结 relation family 存在上界，但当前项目仍无法定位
  selector 或 representation consumer causal gap。

## Stop conditions

任何 labels-after-fit violation、arm 间 budget/weight/readout mismatch、GPU/resource failure、
incomplete compute 或单一 consumer 阴性都不得被包装成科学 No-Go；它们写入 audit 并停止当前
job。S2 完成后项目终点是 Decision，不进入 objective isolation、TopoCut 或 holdout。
