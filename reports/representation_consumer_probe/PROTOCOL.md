# Representation-Consumer Probe — Protocol

这是独立 mechanism study，不是 V 系列升级。研究对象是固定的 candidate-relation family，而非
继续修改 TopoGate Gate。S0 已将当前 V21/V25 的 feature-coordinate Gate 到 sample-edge
membership 的映射判定为 `adapter_not_estimable`；该状态关闭本项目的 T-related causal chain。

## 研究边界

```text
X → common H0 → positive-cosine candidate rows → R/O graph membership
  → W_sym → Spectral (或必要时 SimpleCut) → known-K readout
```

当前只允许 `S0 → S1 opportunity-only → S2 opportunity confirmation → Decision`。不实现
`T_adapter`、S3 objective isolation、S4 strong backbone、S5 holdout、S6 expansion 或 TopoCut。
sample-edge selector 若以后需要研究，另建 `relation_selection_probe`。

## Frozen input and graph family

- `H0=TruncatedSVD(X)`，`d0=128`（受样本/特征数限制时取可行最大值），`random_state=0`；
- candidate `k=20`，只保留严格 positive cosine；
- edge weight 是 H0 cosine，任何 oracle 不得改变 weight；
- `budget_cap=8`，每行 `b_i=min(8, positive_count_i)`；R/O_pool/O_full 共享 budget vector 和
  budget hash；不整集排除 positive shortfall；
- 先审计 directed row budget，再统一 `W=(W+W.T)/2` 并删除 self-loops；
- 所有 ordinary consumer fit 接口不接收 `y/labels/ground_truth`。标签只进入独立 oracle
  diagnostic builder 与外层 metrics。

### Arms

- `F`：H0 + known-K KMeans；
- `U`：完整 candidate graph；
- `R`：每行 positive candidate uniform without replacement 选 `b_i`；
- `O_pool`：candidate 内 same-class 优先、按 H0 cosine 排序，异类补足；
- `O_full`：全空间 same-class 优先、按相同 H0 cosine 规则异类补足，仍只选 `b_i`；
- `T_adapter`：本项目不存在，状态为 `not_estimable`。

## Spectral contract

对 `W_sym` 先取 `V+={i:d_i>0}`，仅在 positive-degree induced subgraph 上构造
`L_sym=I-D^{-1/2}WD^{-1/2}`。固定：`eigsh`、`which="SM"`、`tol=1e-6`、
`maxiter=max(1000,10*n_active)`、`v0=ones/sqrt(n_active)`、`drop_first=False`。
取最小 K 个特征向量（保留 zero eigenspace），row-L2 normalize，再运行
`KMeans(n_clusters=K,n_init=20,random_state=seed)`。isolated rows 填零且不参与 eigenstructure；
eigsh 失败写 `incomplete_compute`，不得换 solver。

Spectral artifact：`K_used_in_representation=true`、`K_used_in_readout=true`、
`labels_vector_used_in_fit=false`。SimpleCut/F 的 K 只用于 readout，写在 consumer artifact 而
不是全局字段。

## Estimands and decision

仅报告：

```text
H_pool = ARI(O_pool) - ARI(R)
H_full = ARI(O_full) - ARI(R)
C      = H_full - H_pool
```

不报告 `S_graph`、rho 或 T interaction。`delta=0.03` 只作 descriptive materiality margin：

- tested consumers 上 H_pool/H_full 均小：`opportunity_absent_under_frozen_relation_family`；
- H_full 明显高于 H_pool：`candidate_family_requires_review`；
- H_pool 明显为正：`opportunity_positive_diagnostic`。

第一项不能扩展成“topology 没有价值”，只说明冻结 H0/cosine/budget/consumer family 的
opportunity 未达到预注册阈值。

## Artifacts

S0 formal output 位于 `result/representation_consumer_probe/S0_freeze/`，至少包含 resolved
config、input provenance、dataset manifest、candidate pools、effective-budget manifests、
adapter audit、synthetic apparatus、s0 manifest、s0 decision 和 artifact hash manifest。临时 smoke、资源失败和 review
失败不进入正式性能事实表。
