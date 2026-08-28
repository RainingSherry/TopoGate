# Pre-registration: Representation-Consumer Probe

状态：`draft_for_review`。本文件在 S0 commit 前不得视为实验授权。

## Primary estimands

对 dataset `d`、consumer `c` 和 paired seed `s`：

```text
S_graph(d,c) = ARI(T_adapter,c) - ARI(R,c)
H_pool(d,c)  = ARI(O_pool,c)   - ARI(R,c)
H_full(d,c)  = ARI(O_full,c)   - ARI(R,c)
C(d,c)       = H_full(d,c)    - H_pool(d,c)
```

`T_adapter` 只有在 S0 的 relation adapter 合同通过后才存在；否则相关 estimand 为 `not_estimable`，不能用旧 V25 `S_d` 代替。

机会捕获率只在 `H_pool >= 0.03` 且分母严格为正时报告：

```text
rho(d,c) = S_graph(d,c) / H_pool(d,c)
```

不把 `rho` 当作独立 primary claim；分母接近零时报告 `undefined`。

## Materiality and seed stability

`delta=0.03` 是预注册的 descriptive materiality margin，继承 V25 的审计口径，不是统计显著性阈值。dataset-level 判定要求完整三 seed：

- `material_positive`：dataset mean ≥ `delta`，且至少 2/3 seed effects > 0；
- `material_negative`：dataset mean ≤ `-delta`，且至少 2/3 seed effects < 0；
- `observed_small`：完整三 seed 且 dataset mean 严格落在 `(-delta, +delta)`；
- `inconclusive`：少于三条 `completed_valid` seed、存在 `invalid_design`/`incomplete_compute`，或 primary endpoint 不可计算。

由于六个面板的 ARI 基线异质，绝对 `delta=0.03` 只作为预先固定的 descriptive margin，不被解释为跨数据集等价尺度或显著性。最终表格必须同时给出 `delta` 的 absolute effect、相对于 `max(1-ARI(R), 1e-6)` 的 normalized sensitivity，以及每个 seed 的原始值；normalized sensitivity 不能改写 gate，也不能在结果后选择。

Pilot 不计算跨 dataset p-value，不把 18/108 runs 当独立样本。

## Frozen gates

### Gate 0 — adapter validity

必须通过 source/config/branchpoint/label-isolation audit，并证明 `T_adapter` 输出的是 sample-edge membership；否则 `T_adapter` 路线永久标记 `not_estimable`，S1 只做 opportunity diagnostic。

### Gate 1 — opportunity

在 Spectral 与 SimpleCut 两个 consumer 中取 `H*_d=max(H_pool,d,c)`。至少 2/6 压力数据集必须达到 `material_positive`，且至少一个 consumer 在这些数据集上满足；否则停止 topology-consumer 路线。

若 `H_full` 有 ≥2 个 material-positive 数据集，而 `H_pool` 没有对应 material-positive 数据集，并且至少 2 个数据集的 `C>=delta`，结论冻结为 candidate bottleneck，禁止更换 backbone。

### Gate 2 — selection gap

在 `H_pool` material-positive 的数据集中，至少 2 个必须有 `S_graph` 为 `material_negative` 或 `observed_small`，才能称为 missed/harmful selection。只有一个数据集不解锁 S3。

### Gate 3 — objective interaction

S3 报告：

```text
I_d = [ARI(T_adapter)-ARI(R)]_Cut - [ARI(T_adapter)-ARI(R)]_Rec
```

只有至少 2/6 数据集 `I_d >= delta` 且不与 Gate 1/2 矛盾时，才解锁 S4。`Cut_R > Rec_R` 但 `Cut_T ≈ Cut_R` 只能支持“cut consumer 可能更强”，不能支持 TopoGate 贡献。

### Gate 4 — topology specificity

S4 的 `TopoCut_T` 必须同时高于 `TopoCut_R` 和 degree-matched endpoint-shuffle；否则 topology-specific claim 为 No-Go。

### Gate 5 — holdout

在查看任何 holdout outcome 前冻结 12 个 dataset。只跑 frozen config；至少报告 dataset-level median/mean interaction、win/tie/loss 和 direction-stable count。holdout 失败时停止 paper-scale 扩展，并按结果是 `inconclusive_not_completed` 还是性能 No-Go 分开记录。

## Oracle non-tuning firewall

`O_pool/O_full` 的 label-derived graph 和 ARI 只能用于 Gate 1/2 的外层 diagnostic decision。它们不得选择或修改 `T_adapter` 的 relation features、threshold、budget、alpha、loss weights、optimizer、epoch、readout、dataset membership、seed 或 holdout manifest。所有这些字段必须先出现在 `S0_FREEZE.md` 和 resolved config 中；若 oracle 结果与冻结配置冲突，记录 No-Go/diagnostic mismatch，不进行 rescue。

## Falsifiers and prohibited rescue

任何一项以下情况都不能通过 gate：labels 进入 fit；edge budget/weight rule 在 arm 间不匹配；T/R 使用不同 donor/noise/schedule；graph consumer 没有实际接收 W；oracle 修改 feature、K、weight 或 readout；只剩单一 consumer 的阴性结果；用 outcome 选择 holdout、超参、epoch、seed 或 dataset；或通过增加 GNN/Transformer/OT/DEC/GAN 等新自由度救场。
