# S0 — Protocol Freeze Contract

本文件冻结当前 `representation_consumer_probe` 的 opportunity-only 合同。它不授权训练、新
selector、新 backbone 或 holdout。

## Numeric contract

| field | frozen value |
|---|---|
| `project_id` | `representation_consumer_probe` |
| `protocol_id` | `representation_consumer_probe_s0_v1` |
| common stem | `TruncatedSVD`, `d0=128`, `random_state=0`, one H0 per dataset |
| candidate neighbors | `k=20`, positive cosine only |
| budget | `budget_cap=8`; `b_i=min(8, positive_count_i)` |
| sensitivity | nominal caps `[4,8,12]`, only after current opportunity decision |
| edge weights | positive H0 cosine, float32 |
| symmetrization | `(W+W.T)/2`, then remove self-loops |
| normalized Laplacian | `L_sym=I-D^-1/2 W D^-1/2`, zero inverse for zero degree |
| spectral solver | `scipy.sparse.linalg.eigsh`, `which="SM"`, `tol=1e-6`, `maxiter=max(1000,10*n_active)` |
| spectral start | `v0=ones/sqrt(n_active)`; `drop_first=false` |
| spectral postprocess | active induced subgraph only, row-L2 normalization, isolated rows zero |
| readout | `KMeans(n_clusters=K,n_init=20,random_state=seed)` |
| seeds | S1/S2 pilot `[42,123,7]` |
| labels in fit | `false` |

## Effective budget audit

每个 candidate pool 必须保存：

```text
effective_budget = min(8, positive_count)
effective_budget_hash
effective_budget_mean / min / max
fraction_budget_below_cap
fraction_budget_zero
```

R/O_pool/O_full 的 row counts 必须逐行等于这个 vector。`positive_count<8` 不再导致整个
dataset 从 stress panel 删除；`b_i=0` 的行保留为空行并进入 graph/isolate diagnostics。

## Consumer-level K contract

```text
Spectral:  K_used_in_representation=true,  K_used_in_readout=true
SimpleCut: K_used_in_representation=false, K_used_in_readout=true
F:         K_used_in_representation=false, K_used_in_readout=true
```

所有 consumer 的 `labels_vector_used_in_fit=false`。K 可以来自 benchmark-known outer protocol，
但完整标签向量不得进入 H0、candidate、R、loss、spectral fit 或 KMeans fit。

## Synthetic acceptance

S0 必须分别记录：

1. `graph_numerical_sanity`：L/L_sym/NCut/loss finite；
2. `spectral_recovery_sanity`：clean block ARI≥0.95、clean 优于 contaminated、no-opportunity
   pipeline 无 NaN/异常、isolated rows 全零且 active eigenstructure 不含 isolates。

synthetic 只验证测量装置，不构成真实数据性能结果。

## Terminal status

- `adapter_valid`：只有真实 sample-edge membership adapter 通过 semantic audit 才可能出现；
- `adapter_not_estimable`：当前预期，也是本项目 T-related causal chain 的 terminal state；
- `protocol_mismatch`：source/hash/shape/contract 不一致。

当状态为 `adapter_not_estimable` 时，只能解锁 S1 opportunity-only（必要时 S2 SimpleCut）和
最终 Decision；S3/S4/S5/S6、TopoCut、holdout 和任何新 selector 永久锁定。
