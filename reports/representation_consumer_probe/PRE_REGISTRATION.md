# Pre-registration: Representation-Consumer Probe

状态：`frozen_for_opportunity_only`。本文件只授权 S0→S1→S2→Decision，不授权新的 selector、
backbone、holdout 或 paper-scale benchmark；S3/S4/S5/S6 在当前项目中永久锁定。

## 研究问题

在固定的 `H0 → positive-cosine candidate pool → row-specific budget → tested consumer` family
中，是否存在可被理想 relation membership 选择利用的 clustering opportunity？当前 V21/V25
代码的 feature-mask → sample-edge adapter 已审计为 `adapter_not_estimable`，因此不存在
可估计的 `T_adapter` 或 `S_graph`。

## Frozen input and budget contract

- `H0=TruncatedSVD(X, n_components=min(128,n-1,d-1), random_state=0)`，每个 dataset 只生成一次。
- candidate rows：cosine kNN，`k=20`，只保留严格 positive cosine。
- `budget_cap=8`；对每个 node 固定 `b_i=min(8, positive_count_i)`。`R/O_pool/O_full` 必须
  使用同一 `b_i`，并保存 budget hash、均值、最小值、`fraction(b_i<8)` 和 `fraction(b_i=0)`。
- edge weights 只能由冻结 H0 cosine 产生；oracle 只能改变 membership；最后统一
  `W=(W+W.T)/2` 并删除 self-loop。
- seeds `[42,123,7]` 是 paired repeat；统计单位是 dataset。

## Estimands

对 dataset `d`、consumer `c` 和 seed `s`，只计算：

```text
H_pool(d,c) = ARI(O_pool,c) - ARI(R,c)
H_full(d,c) = ARI(O_full,c) - ARI(R,c)
C(d,c)      = H_full(d,c) - H_pool(d,c)
```

`S_graph`、opportunity capture ratio 和任何 T-vs-R interaction 在本项目中均为
`not_estimable`，不得用旧 V25 `S_d^{V21}` 代替。`O_pool/O_full` 是 label-derived diagnostic
upper bounds，不是可部署方法。

## Consumers and K semantics

- `Spectral`：active positive-degree induced subgraph → `L_sym` → `eigsh(which="SM", tol=1e-6,
  maxiter=max(1000,10*n_active), v0=ones/sqrt(n_active))` → smallest K eigenvectors → row-L2 →
  `KMeans(n_clusters=K,n_init=20,random_state=seed)`。K 进入 representation 和 readout。
- `SimpleCut`（仅在需要确认时）：固定小 encoder，不在 representation 中使用 K；K 只进入
  readout。它不引入 message passing 或新模型自由度。
- `F`：H0 + known-K KMeans；K 只进入 readout。

任何 consumer 的 fit 都不能接收完整标签向量；标签只进入 oracle builder 和外层指标。
isolated nodes 不参与 spectral eigenstructure，embedding row 固定为零。

## Materiality and decision rules

`delta=0.03` 是 descriptive margin，不是显著性阈值。

S1 结果同时写入两个正交字段：

```text
within_pool_opportunity = absent / present
candidate_gap = absent / present
```

`candidate_gap` 只表示 matched-budget 下 `H_full` 相对 `H_pool` 的额外机会，不能解释为
完整 candidate recall/support-capacity loss；后者由 `fraction(b_i<8)`、`fraction(b_i=0)` 和
candidate diagnostics 单独报告。

1. 如果 Spectral 与 SimpleCut（若执行）上 `H_pool<delta` 且 `H_full<delta` 普遍成立，写入
   `opportunity_absent_under_frozen_relation_family`。这只约束冻结 relation family，不能扩展为
   “topology 没有价值”。
2. 如果 `H_full` 明显高于 `H_pool`，写入 `candidate_family_requires_review`，不更换 backbone。
3. 如果 `H_pool` 明显为正，写入 opportunity-positive diagnostic；不把它升级为 selector 或
   representation-consumer 主张。
4. 如果 `H_pool` 明显为正且 `H_full-H_pool` 未达到 candidate-gap margin，终局标签为
   `opportunity_present_within_frozen_candidate_pool`；这只授权新建独立的
   `relation_selection_probe`，不解锁本项目的 T/S3/S4。

## Falsifiers and prohibited rescue

以下任一情况只能导致 protocol invalid/incomplete，不得 rescue：labels 进入 fit；arm 间 budget、
weights、symmetrization 或 readout 不匹配；oracle 改变 feature/K/weights；eigsh 失败后换 solver；
依据 outcome 调 budget、consumer、epoch、seed 或 dataset；把 isolated node 随意填入 eigenvectors；
创建新 T selector、TopoCut、S3/S4/S5/S6 或 architecture sweep。

## Dormant holdout

`STAGE5_HOLDOUT_MANIFEST.json` 作为 outcome-independent 历史冻结输入保留，但状态固定为
`dormant_due_to_adapter_not_estimable`，不运行、不进入当前项目 denominator，也不支持
generalization claim。
