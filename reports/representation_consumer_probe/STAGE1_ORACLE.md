# S1 — Graph Opportunity Oracle (opportunity-only)

S1 不训练神经网络，也不估计当前 TopoGate 的 selector utility。目标是测量冻结 candidate
relation family 的 opportunity 上界，并区分 candidate-pool limitation 与 tested consumer 的
读出能力。

## Common pool and effective budget

从冻结 `H0` 构造 `k=20` 的 candidate rows，只保留严格 positive cosine。`budget_cap=8` 是上限，
不是 exact row budget：

```text
b_i = min(8, positive_count_i)
```

所有 R、O_pool、O_full 共享完全相同的 `b_i`、budget hash、H0 cosine weights 和
`(W+W.T)/2` symmetrization。S0 保存 effective-budget mean/min、`fraction(b_i<8)` 和
`fraction(b_i=0)`；不能因为某些行不足 8 而删除整个 dataset，也不能用异类边增加预算。

## Graph arms

- `F`：`H0 → known-K KMeans`，作为 feature-only reference；
- `U`：保留全部 candidate rows；
- `R`：每行在 positive candidate slots 中 uniform without replacement 选 `b_i`，不接收标签；
- `O_pool`：candidate pool 内优先同类边；同类不足时按冻结 H0 cosine 从异类 positive edges
  补足；只改变 membership；
- `O_full`：全样本空间分别搜索同类与异类 positive cosine 邻居，仍只选同一个 `b_i`，不因
  candidate recall 更高而增加预算。

当前 V21/V25 adapter 审计为 `adapter_not_estimable`，因此没有 `T_adapter` arm。任何新
sample-edge selector 都必须另建 `relation_selection_probe`。

正式 runner 为 `scripts/representation_consumer_probe/s1_opportunity.py`，当前有效协议为
`representation_consumer_probe_s1_opportunity_spectral_v2`，结果目录为
`result/representation_consumer_probe/S1_oracle_v2/`。`F` 严格使用原始 `H0 → known-K KMeans`；
U/R/O arms 使用同一 Spectral consumer。首版 `S1_oracle/` 因 F 的预处理语义错误保留为
`invalid_design` provenance，不得进入汇总或论文证据。

## Graph diagnostics

每个 graph 保存 candidate recall（以 `b_i` 为 per-node denominator）、edge purity、directed
row-budget fill、symmetrized edge count、components、isolates、giant-component ratio、degree
mean/std/CV、ground-truth NCut、graph hash、budget hash 和权重/symmetrization policy。标签只
进入 post-hoc diagnostics，不进入 graph construction 的公共输入。

## Spectral consumer

对 `W_sym` 先取正度节点 `V+={i:d_i>0}`，仅在 `W[V+,V+]` 上构造 `L_sym`。固定使用
`eigsh(which="SM", tol=1e-6, maxiter=max(1000,10*n_active), v0=ones/sqrt(n_active))`，取最小
K 个特征向量（不丢 zero eigenspace），做 row-L2 normalization，再以
`KMeans(n_clusters=K,n_init=20,random_state=seed)` 读出。isolated rows 重新填零，并且不参与
eigenstructure；eigsh 失败记录 `incomplete_compute`，不得换 solver 重跑。

Spectral 的 `K_used_in_representation=true`；`K_used_in_readout=true`；
`labels_vector_used_in_fit=false`。SimpleCut/F 的 K 语义记录在 consumer artifact，而不是全局
协议字段。

## S1 outputs and boundary

S1 只计算 `H_pool=ARI(O_pool)-ARI(R)`、`H_full=ARI(O_full)-ARI(R)` 和
`C=H_full-H_pool`。不计算 `S_graph`，不做 T-vs-R interaction，不解锁 backbone swap。若
Spectral 结果可能受 relaxation 影响，才按 S2 文档运行 SimpleCut confirmation。
