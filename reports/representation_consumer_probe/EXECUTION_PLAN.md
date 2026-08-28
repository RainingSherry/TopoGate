# Representation-Consumer Probe — Execution Plan

本项目独立于 V 系列。它只定位 frozen candidate-relation family 的 opportunity 上界，不搜索
新 selector 或新 backbone。执行过程遵循 `auto-review-loop`：每个阶段先做审查，修复合同问题，
再进入下一阶段；审查不能替代原始实验工件。

## 已冻结边界

- stress panel：`cnae9`、`Mouse_retina`、`sms_spam_collection`、`Baron Human`、`Campbell`、
  `hate_speech`；只作机制压力面板。
- 公共输入：一次性 `TruncatedSVD(H0, d0=128, random_state=0)`。
- candidate pool：`k=20`，严格 positive cosine，统一 H0 cosine weights，统一对称化。
- budget：`budget_cap=8`，每行实际使用 `b_i=min(8, positive_count_i)`；R、O_pool、O_full
  共享同一个 budget vector 和 budget hash。
- seeds：S1/S2 pilot `[42,123,7]`；seed 是 paired repeat，不是独立统计单位。
- readout：benchmark-known `K` 的 clean embedding + `KMeans(n_init=20, random_state=seed)`。
  Spectral 的 K 进入 representation；SimpleCut/F 的 K 只进入 readout。
- 标签隔离：F/U/R/Spectral/SimpleCut 的 fit 接口不接收标签；标签只构造诊断 oracle 和外层
  metrics。当前不存在可估计的 `T_adapter`。
- GPU：S1 不训练深度模型，以 CPU 稀疏图计算为主；若 S2 需要 GPU，只允许显式池
  `[1,2,3,4,5,6]`，物理 GPU 0/7 禁止使用。资源充足不改变冻结合同。

## Step 0 — S0 freeze and adapter audit

1. 核对六个输入快照的 path、SHA、shape、H0 和 K-source；写入 formal `S0_freeze` artifacts。
2. 运行 `audit_adapter_semantics`。当前预期为 `adapter_not_estimable`，这会永久关闭 T/S3/S4。
3. 对每个 dataset 保存一次 `H0.npy`、candidate pool、effective-budget profile、budget hash、
   graph/loss numerical contract 和 source provenance。
4. synthetic apparatus 分成 `graph_numerical_sanity` 与 `spectral_recovery_sanity`：clean block
   必须 `ARI>=0.95`，且 clean 优于 contaminated；isolate rows 必须为零。
5. S0 不产生聚类性能结果，不训练模型，不启动 GPU 队列。

## Step 1 — S1 opportunity-only

在同一个 H0/candidate family 上构造：

- `F`：H0 + known-K KMeans；
- `U`：全部 candidate edges；
- `R`：每行从 positive candidate 中 uniform without replacement 选择 `b_i`；
- `O_pool`：candidate pool 内 same-class 优先，余量按 H0 cosine 的异类 positive edges 补齐；
- `O_full`：全样本空间 same-class 优先，余量按相同 H0 cosine 规则补齐，但仍只选 `b_i`。

所有 arm 统一 weights、symmetrization、spectral solver 和 readout。记录 candidate recall、edge
purity、components、isolates、degree profile、ground-truth NCut、graph hash 和 budget hash。
S1 只形成 `H_pool/H_full/C`，不形成 `S_graph`。

## Step 2 — S2 opportunity confirmation

仅当 S1 的 Spectral 结果可能受 relaxation 影响时，运行小型 SimpleCut probe；它使用固定
`128→64→32` encoder 和 normalized cut/orthogonality/variance loss，不引入 GNN、Transformer、
OT、DEC、GAN 或新 selector。S2 只确认 opportunity，不解锁新的项目阶段。

当前 S2 已完成：Baron Human 与 Mouse_retina 的 `R/O_pool/O_full × [42,123,7]` 共 `18/18`
completed-valid。Baron Human 的 `H_pool=+0.033242` 仅支持 Spectral relaxation miss 的
限定解释；Mouse_retina 的 `H_pool=+0.008880` 为 observed-small。两者均没有 material
candidate gap，S2 不估计 `S_graph`。

## Step 3 — Decision

按 frozen materiality margin `delta=0.03` 对 `H_pool` 与 `H_full` 分类：

- 两者都小：`opportunity_absent_under_frozen_relation_family`；
- `H_full >> H_pool`：`candidate_family_requires_review`；
- `H_pool` 明显为正：只说明冻结 relation family 存在 diagnostic opportunity，不能说明当前
  TopoGate selector 或新 representation consumer 已经可识别。

本轮终局为 `heterogeneous_with_spectral_relaxation_caveat`；selector 为 `not_estimable`，
representation-consumer promotion 未授权。审计 WARN 仅涉及 history pre-step/post-step
loss 的 metadata timing，不改变上述 terminal decision。

## 永久锁定阶段

本项目不执行 S3 objective isolation、S4 strong backbone、S5 holdout、S6 paper-scale expansion，
也不创建 TopoCut。holdout manifest 保留用于记录既有冻结输入，但状态为
`dormant_due_to_adapter_not_estimable`。任何 sample-edge selector 研究必须新建
`relation_selection_probe`。

## 结果工件

正式结果只能写入 `result/representation_consumer_probe/`。S0 至少保存：

```text
resolved_config.json
input_provenance.json
dataset_manifest.json
selection_to_relation_adapter.json
synthetic_apparatus.json
s0_manifest.json
s0_decision.json
artifact_hashes.json
```

临时 smoke、外部 review 失败、OOM 或只读挂载不得写入正式性能表；失败保留为
`incomplete_compute` 或执行边界记录。
