# V25：Systematic Failure Atlas and Mechanism Localization

更新时间：2026-08-15

V25 是对 V1--V24 的系统机制研究，不是新的 TopoGate architecture。项目目录固定为
`V25_systematic_mechanism_study/`；本研究结束后不进入 V26，也不追加新的 Gate、loss、selector、
DCBoost、V18/V22/V24 rescue 或开放式 utility sweep。

核心问题是：

> 为什么看起来有用的结构信息，无法稳定转化为聚类收益？

统一定位链为：

```text
Opportunity -> Selection -> Intervention -> Representation -> Readout
```

本报告是当前仓库中 V25 代码、结果、审计和论文材料的发布索引。所有数字都来自已保存的
CSV/JSON/Markdown/figure/paper 工件；没有在本报告中重新估计指标。

## 1. 研究范围与证据边界

V25 保留两个层次的论文证据：

1. V1--V22 的 Failure Atlas 是观察性证据。历史重复行不是独立数据集，不能把 1,637 条 paired
   rows 当成 1,637 个独立样本，也不把不同 protocol、preprocessing、readout 或 budget 混合为一个
   因果总体。
2. 只有 A2 保留 E1 后，V21 的 matched N/R/T 才作为狭窄的 prospective mechanism case study。
   该 case study 使用真实标签做外层 benchmark 和 known-K readout，但 labels 不进入 preprocessing、
   graph、Gate、loss 或 model update，因此应称为 `real-GT, known-K benchmark`，而不是 fully
   label-free fitting。

V23/V24 只登记为 boundary evidence：V23 的 Cycle dependency-specific utility 为 No-Go，V24 的
calibration 为 No-Go。它们没有与 V1--V22 的 structural-intervention rows 合并统计。

## 2. Phase A：历史证据

### A0 Evidence Registry

输入为已经审计的 V1--V22 unified long table，以及隔离的 V23/V24 boundary records。统计单位为
dataset/protocol/readout；seed 和 variant 是重复测量。

| 量 | 结果 |
|---|---:|
| V1--V22 registry rows | 2,209 |
| completed rows | 2,175 |
| reported rows | 32 |
| incomplete rows | 2 |
| paired Delta ARI rows | 1,637 |
| dataset/protocol/readout units | 431 |
| unique datasets | 342 |
| V23/V24 boundary records | 2 |
| replay-eligible rows | 0 |

A0 还登记 source/preprocess/readout/K hashes、结构来源、selection、intervention location、training
target、measurement timing、causal status、artifact status、labels/K isolation、`reused_from` 和
alternative explanation。A0 没有重新加载标签来构建 registry。

输出：

- `result/V25_systematic_mechanism_study/A0/mechanism_evidence_registry.csv`
- `result/V25_systematic_mechanism_study/A0/dataset_protocol_units.csv`
- `result/V25_systematic_mechanism_study/A0/artifact_availability.csv`
- `result/V25_systematic_mechanism_study/A0/V23_V24_boundary_evidence.csv`
- `result/V25_systematic_mechanism_study/A0/registry_summary.json`
- `result/V25_systematic_mechanism_study/A0/A0_REGISTRY.md`

### A1 Failure Atlas 与离线 E3 gate

A1 只使用 artifact-complete 结果。metadata-only snapshot 不进入 replay；raw Delta ARI 是主描述量，
baseline/headroom、representation displacement、effective corruption 和 final-neighborhood 是
敏感性或 post-treatment descriptor。

| 量 | 结果 |
|---|---:|
| paired rows | 1,637 |
| material positive (`Delta ARI > +0.03`) | 194 |
| material negative (`Delta ARI < -0.03`) | 680 |
| observed-small | 763 |
| dataset/protocol/readout units | 239 |
| unique datasets | 207 |
| local/global boundary rows | 6 |
| artifact-complete E3 replay candidates | 0 |

因此 A1 的主要观察是：

```text
structural quality != intervention utility
```

这不是跨历史 protocol 的普遍因果结论。由于没有 artifact-complete replay candidate，正式 E3
没有重训或离线 replay；V23 的 local/global boundary rows 保留为独立边界证据。embedding 变化使用
可比的 geometry descriptors，而不使用跨版本 raw L2；label-free geometry 与 post-hoc supervised
geometry 分开。

输出：

- `result/V25_systematic_mechanism_study/A1/failure_atlas.csv`
- `result/V25_systematic_mechanism_study/A1/version_family_summary.csv`
- `result/V25_systematic_mechanism_study/A1/structural_opportunity_summary.csv`
- `result/V25_systematic_mechanism_study/A1/baseline_headroom_summary.csv`
- `result/V25_systematic_mechanism_study/A1/magnitude_gain_summary.csv`
- `result/V25_systematic_mechanism_study/A1/local_global_boundary.csv`
- `result/V25_systematic_mechanism_study/A1/failure_localization_taxonomy.csv`
- `result/V25_systematic_mechanism_study/A1/e3_replay_summary.json`
- `result/V25_systematic_mechanism_study/A1/a1_summary.json`
- `result/V25_systematic_mechanism_study/A1/FAILURE_ATLAS.md`

Failure localization taxonomy 显式保留 `causal_status`、`confidence`、`evidence_source` 和
alternative explanation，不能把 observational pattern 写成已证实原因。

### A2 Mechanism Triage

A2 具有真实否决权；E1 只是 provisionally reserved experiment，不是预先强制执行的主线。

最终决策：

```text
decision = retain_e1
no_new_e4 = true
```

保留理由是：V21 历史结果具有可审计的异质性符号，且 matched random/none counterfactual 缺失但
可识别。A2 没有授权任何临时 E4 或 V26。

A2 同时冻结了 holdout candidate/adapter manifest、measurement schema、`delta=0.03` 理由、
primary endpoint 和判定规则。

输出：

- `result/V25_systematic_mechanism_study/A2/A2_decision.json`
- `result/V25_systematic_mechanism_study/A2/CLAIM_EVIDENCE_MATRIX.csv`
- `result/V25_systematic_mechanism_study/A2/holdout_candidate_manifest.json`
- `result/V25_systematic_mechanism_study/A2/measurement_schema.json`
- `result/V25_systematic_mechanism_study/A2/A2_DECISION.md`

## 3. Phase B：prospective mechanism test

### E1：V21 matched three-arm selection policy

E1 仅因 A2=`retain_e1` 执行。三臂为：

```text
N = matched none
R = matched random assignment
T = topology-dependent selection
```

主分解为：

```text
I_d = Q(R) - Q(N)
S_d = Q(T) - Q(R)
Q(T) - Q(N) = I_d + S_d
```

primary readout 是 clean embedding + benchmark-known-K KMeans；Student-t head 只作 secondary
diagnostic。三臂共享 warmup branchpoint、base corruption、donor、eligible set、effective budget、
batch order、topology statistics、selection noise 和 readout。正式 random 不复用旧
`random_assignment_control`，而是使用相同冻结 Gumbel tensor：

```text
s_T = f_theta(phi_ij) + epsilon_ij
s_R = 0 + epsilon_ij
```

None 保留 Student-t head、InfoMax、warmup 初始化和 optimizer，但不执行 assignment forward，也不
执行 JS；donor/eligible/budget 对 N 只作为 shadow/audit schedule。

判定阈值为 `delta=0.03`：`Positive`、`Negative`、`Observed-Small`、`Inconclusive` 分别按
dataset mean、三 seed 同号规则判定；不使用 3-seed bootstrap 宣称 equivalence，也不使用 3-cluster
robust CI。Pilot continuation 允许稳定的异号效果。

#### Pilot：cnae9、Mouse_retina、sms_spam_collection，seeds `[42, 123, 7]`

共 27 arm jobs，9 个 dataset-seed panels，正式审计 `9/9 audit_ok`，coverage complete。

| Dataset | `I_d` | 状态 | `S_d` | 状态 |
|---|---:|---|---:|---|
| cnae9 | +0.002057 | Observed-Small | +0.006010 | Observed-Small |
| Mouse_retina | +0.027889 | Inconclusive | -0.067033 | Negative |
| sms_spam_collection | +0.069251 | Positive | +0.000553 | Inconclusive |

Pilot gate 通过：3 个数据集中有 2 个产生 seed-stable material effect；不要求跨数据集同号。

#### Confirmation：Baron Human、Campbell、hate_speech，seeds `[42, 123, 7]`

共 27 arm jobs，9 个 dataset-seed panels，正式审计 `9/9 audit_ok`，coverage complete。

| Dataset | `I_d` | 状态 | `S_d` | 状态 |
|---|---:|---|---:|---|
| Baron Human | -0.011463 | Inconclusive | +0.044617 | Positive |
| Campbell | +0.132841 | Positive | -0.065332 | Negative |
| hate_speech | -0.002671 | Observed-Small | -0.033410 | Negative |

E1 的确认结果是 sign-heterogeneous：Baron Human 的 topology selection 增量为正，Campbell 和
hate_speech 为负。因此冻结的机制措辞是：

> topology-dependent selection has conditional incremental utility in the audited V21 case study;
> this is not a universal population claim.

E1 同时通过了：shared branchpoint、T/R donor/eligible/budget/Gumbel hash equality、None 无
assignment/JS forward、actual Adam one-step、labels-after-fit-only 和 explicit-K audit。独立 contract
审计从保存的 predictions 重算 primary ARI、`I_full_ARI` 和 `S_full_ARI`，结果匹配。

E1 输出根：

- `result/V25_systematic_mechanism_study/E1/pilot/`
- `result/V25_systematic_mechanism_study/E1/confirmation/`
- `result/V25_systematic_mechanism_study/E1/e1_manifest.json`
- 每个 panel 的 `resolved_config.json`、`runner_profile.json`、`summary.json`、`audit.json`、
  `predictions.npy`、`labels_true.npy`、`embedding_final.npy`、schedule/hash artifacts 和
  relationship artifacts
- `E1/*/Audit/phase_summary.json`
- `E1/*/Audit/panel_audit.csv`
- `E1/*/Audit/pair_effects.csv`
- `E1/*/Audit/gradient_probe.csv`
- `E1/*/Audit/one_step_metrics.csv`

### E2：feature 与 objective localization

E2 不新增模型。E2-A 的 inferential unit 是 dataset x seed；coordinate-level distributions 只作
图形描述，不能把数十亿 sample-feature coordinates 当作独立样本，也不能据此报告虚假 p-value。

确认阶段 E2-A：

- 9 个 audited panels；
- 3 datasets x 3 seeds；
- 90 个 dataset-seed semantic rows；
- 30 个 dataset summaries；
- 9 个 panel audits；
- selected/eligible-not-selected coordinate 统计只作描述；post-hoc Fisher、MI、support enrichment
  没有进入 fitting。

E2-B/C：

- 27 个 gradient geometry rows，覆盖 T0/T1/T2；
- 保存 `cos(g_b,g_a)`、`cos(g_b,g_i)`、`cos(g_b,g_a+g_i)` 和 norm；
- 从相同 model/head/Adam state 执行 N/R/T actual Adam one-step；
- one-step 使用 N/R/T 分解，纯 scMAE step 只可作为 secondary diagnostic；
- E2 没有被单独升级成 objective-conflict 主张。

输出：

- `result/V25_systematic_mechanism_study/E1/confirmation/Audit/e2_feature_audit.json`
- `result/V25_systematic_mechanism_study/E1/confirmation/Audit/gradient_probe.csv`
- `result/V25_systematic_mechanism_study/E1/confirmation/Audit/one_step_metrics.csv`
- `result/V25_systematic_mechanism_study/E1/confirmation/Audit/pair_effects.csv`
- `result/V25_systematic_mechanism_study/E1/confirmation/Audit/panel_audit.csv`

## 4. Phase C：claim freeze

冻结的唯一 primary claim family 是 `selection`，primary endpoint 是：

```text
S_full_ARI = ARI_T - ARI_R
```

Phase C 没有因为结果不漂亮而更换中心句，也没有把 E2 feature/gradient diagnostics 提升为独立主张。

输出：

- `result/V25_systematic_mechanism_study/PhaseC/FROZEN_PAPER_CLAIM.json`
- `result/V25_systematic_mechanism_study/PhaseC/FROZEN_PAPER_CLAIM.md`
- `result/V25_systematic_mechanism_study/PhaseD/holdout_activation_manifest_claim_dependent.json`

## 5. Phase D：independent holdout

A2 前冻结的 candidate/adapter manifest 通过 source、input adapter、preprocessing、K 和 hash
preflight；实际可执行候选为：

- `news20__libsvm_sparse_highdim`
- `rcv1_train__libsvm_sparse_highdim`

预期 6 个 panels（2 datasets x 3 seeds），但冻结 dense V21 decoder/Adam state 在 news20 的三个
seed 于 Adam state 初始化阶段触发 CUDA OOM；其余 panels 在同一资源边界下停止或未启动。

最终状态：

```text
expected panels       = 6
completed panels      = 0
audit_ok panels       = 0
primary endpoint      = not evaluable
status                 = inconclusive_not_completed
performance result     = false
```

这不是模型负结果、不是 frozen claim 被证伪，也不是 independent replication。后续对 runner 做了
host-backed batch/statistics、`foreach=false, fused=true` 和 CPU-backed snapshot 的资源等价修复审计；
bounded engineering smoke 仍未在时间窗内完成，未生成新的 holdout performance artifact，也没有
改变 `0/6` 事实。

输出：

- `result/V25_systematic_mechanism_study/PhaseD/holdout_activation_manifest.json`
- `result/V25_systematic_mechanism_study/PhaseD/holdout_activation_manifest_claim_dependent.json`
- `result/V25_systematic_mechanism_study/PhaseD/holdout_e1_manifest.json`
- `result/V25_systematic_mechanism_study/PhaseD/E1/manifest_snapshot.json`
- `result/V25_systematic_mechanism_study/PhaseD/E1/queue_state.json`
- `result/V25_systematic_mechanism_study/PhaseD/Audit/phase_summary.json`
- `result/V25_systematic_mechanism_study/PhaseD/E1/logs/` 和 per-panel `launch_record.json`/
  `manifest_record.json`

## 6. Phase E：closure

最终决策：

```text
closure_decision = close_without_v26
```

允许保留的论文结论：

1. V1--V22 提供 structural-intervention failure atlas，显示 structural quality 与 intervention
   utility 不等价；这是 observational evidence。
2. 审计 V21 case study 显示 topology-dependent selection 的 conditional、sign-heterogeneous
   incremental utility；不支持 universal topology superiority。
3. 独立 holdout replication 尚未建立。

明确关闭：V26、新 Gate、新 loss、新 selector、DCBoost、V18/V22/V24 rescue 和开放式 utility
sweep。

输出：

- `result/V25_systematic_mechanism_study/PhaseE/closure.json`
- `result/V25_systematic_mechanism_study/PhaseE/CLOSURE.md`
- `result/V25_systematic_mechanism_study/PhaseE/closure_audit.json`

## 7. 论文与证据包

正式论文位于 `papers/V25_systematic_mechanism_study/paper/`：

- `main.tex`、`main.pdf`、`main.bbl`、`references.bib`；
- `sections/`：introduction、study design、results、discussion、limitations；
- `tables/`：atlas、E1 effects、evidence layers 和 LaTeX asset manifest；
- `figures/`：五张图，每张 PNG/PDF/SVG；
- `PAPER_ACCEPTANCE_CONTRACT.md`、`PROVENANCE.md`；
- `FINAL_PAPER_AUDIT.json`、`FINAL_PAPER_AUDIT.md`；
- `CITATION_AUDIT.json`、`CITATION_AUDIT.md`。

证据包位于 `result/V25_systematic_mechanism_study/PaperEvidence/`，包含：

- A0/A1/A2 exports；
- E1 dataset/seed/pair effects；
- E2 semantic/gradient summaries；
- E3 boundary evidence；
- 五张 publication figures；
- source SHA256 manifest；
- `paper_evidence_summary.json`；
- `claim_scope_audit.json`；
- `ClaimAudit/V25_PAPER_CLAIM_AUDIT.json`。

证据包的关键审计结果：

```text
claim_scope_audit = audit_ok
final_paper_audit = audit_ok
all 9 confirmation panels = audit_ok
five figures x three formats = hash matched
forbidden universal/holdout claims = absent
```

## 8. 代码与验证

核心代码：

- `methods/TopoGate/V25_systematic_mechanism_study/e1_protocol.py`
- `methods/TopoGate/V25_systematic_mechanism_study/e2_metrics.py`
- `methods/TopoGate/V25_systematic_mechanism_study/README.md`
- `methods/TopoGate/V25_systematic_mechanism_study/PROTOCOL.md`

编排、生成、审计和导出脚本：

- `scripts/V25/build_a0_registry.py`
- `scripts/V25/build_a1_failure_atlas.py`
- `scripts/V25/build_a2_triage.py`
- `scripts/V25/build_e1_manifest.py`
- `scripts/V25/launch_e1_pilot.py`
- `scripts/V25/run_e1_matched_protocol.py`
- `scripts/V25/summarize_e1.py`
- `scripts/V25/audit_e1_phase.py`
- `scripts/V25/build_e2_feature_audit.py`
- `scripts/V25/freeze_claim.py`
- `scripts/V25/build_holdout_manifest.py`
- `scripts/V25/build_holdout_e1_manifest.py`
- `scripts/V25/preflight_holdout.py`
- `scripts/V25/build_paper_evidence.py`
- `scripts/V25/build_paper_figures.py`
- `scripts/V25/build_closure_artifacts.py`
- `scripts/V25/audit_v25_contract.py`
- `scripts/V25/audit_paper_claims.py`
- `scripts/V25/audit_final_paper.py`

已完成验证：

- `pytest -q scripts/V25/tests`：44 passed；
- `python -m compileall -q methods/TopoGate/V25_systematic_mechanism_study scripts/V25`：通过；
- `V25_CONTRACT_AUDIT.json`：`status=audit_ok`；
- paper claim audit：`audit_ok`；
- final paper audit：`audit_ok`；
- branchpoint serialize/restore、T/R schedule/noise hash equality、None no-assignment/JS、actual
  Adam one-step、coordinate aggregation、label isolation、explicit-K、holdout preflight 和三臂 CPU
  engineering smoke 均已覆盖。

## 10. 闭环差距图与后续决策产物

由 `scripts/V25/build_closure_artifacts.py` 从上述冻结 CSV/JSON 自动生成的非权重闭环产物为：

- `result/V25_systematic_mechanism_study/V25_GAP_MAP.md`
- `result/V25_systematic_mechanism_study/V25_GAP_MAP.csv`
- `result/V25_systematic_mechanism_study/failure_localization_taxonomy.csv`
- `result/V25_systematic_mechanism_study/E1_MECHANISM_SUMMARY.csv`
- `result/V25_systematic_mechanism_study/V25_NEXT_SERIES_DECISION.md`
- `result/V25_systematic_mechanism_study/V25_CLOSURE_ARTIFACTS.json`

其中 Gap Map 按 `Opportunity -> Selection -> Intervention -> Representation -> Readout` 逐项区分
已解决、部分解决、未解决、No-Go 和未完成验证；taxonomy 覆盖 V1--V24，但没有为缺失审计行的
V1--V8、V15、V16、V17 强行指定机制。`E1_MECHANISM_SUMMARY.csv` 恰好包含六个 E1 数据集，
pilot 的 E2-A 缺失明确写为 `deferred`，confirmation 的 E2 只以 dataset x seed 为统计单位。
Holdout `0/6` 明确写为 `inconclusive_not_completed`，绝不写成负结果。

这些文件不包含 checkpoint、branchpoint、原始数据或缓存；可提交的发布副本位于
`papers/V25_systematic_mechanism_study/results/`，并保留源工件 SHA256 和 closure audit。

## 11. GitHub 发布范围

将以下相对路径原样发布到 GitHub `origin/main`：

```text
methods/TopoGate/V25_systematic_mechanism_study/
scripts/V25/
papers/V25_systematic_mechanism_study/
result/V25_systematic_mechanism_study/
reports/V25_systematic_mechanism_study_summary.md
```

发布时保留代码、论文、JSON/CSV/Markdown 审计表、figures、predictions、embeddings、histories、
manifests、logs 和 non-weight result arrays 的相同相对路径。不会上传原始 datasets、dataset symlink
和系统缓存。

本地完整结果盘约 160 GB，其中约 159.63 GB 为 checkpoint/branchpoint 权重。以下文件保留在本地
`result/V25_systematic_mechanism_study/`，不进入 GitHub：

```text
checkpoint.pt
branchpoint.pt
```

原因是单文件约 0.5--8.8 GB，超过 GitHub 单文件限制。非权重可审计结果约 167 MB，发布前检查每个
纳入文件均小于 100 MB。GitHub 发布摘要必须明确：缺少权重不代表结果缺失；可审计的配置、指标、
预测、embedding、history、hash、audit 和论文证据均按原路径发布，权重仍由本地结果盘保存。

## 12. 最终状态

V25 已完成从历史证据登记、Failure Atlas、A2 否决式 triage、matched N/R/T case study、E2
localization、claim freeze、independent holdout boundary 到 closure 的完整闭环。最强且可防守的
结论不是“topology selection 总是有效”，而是：

> useful structure is not sufficient; utility depends on how and where structure intervenes.

在当前证据下，这句话应被解释为 V1--V22 的观察性总括，以及 V21 case study 的 conditional
mechanistic evidence；不扩展为 universal population claim 或 independent replication claim。
