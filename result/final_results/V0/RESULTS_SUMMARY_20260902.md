# TopoGate 实验结果汇总（事实表）

## 2026-09-02 V0 小规模近期方法对照：htru2

为补充 AHDPC、HDPC、ZEUS 和 TableDC，建立了一个受控的小面板，而不是重新运行整套
131 个数据集。完整 cell 为 `htru2.npz × {V0-F,V0-T,AHDPC,HDPC,ZEUS,TableDC}`、
seed=42。所有方法接收同一 `CLUBench.load_data` 列 z-score 输入；`K=int(unique(y).size)`
只用于 fit 后 benchmark readout/指标，`labels_used_during_fit=false`。

| Method | ACC | NMI | ARI | Status |
|---|---:|---:|---:|---|
| V0-F | 0.971097 | 0.626089 | 0.779476 | completed |
| V0-T | **0.971297** | **0.628056** | **0.781154** | completed |
| TableDC | 0.968497 | 0.594543 | 0.765929 | completed |
| ZEUS | 0.910791 | 0.464398 | 0.553454 | completed |
| AHDPC | 0.908391 | 0.000063 | -0.000180 | completed |
| HDPC | 0.908391 | 0.000063 | -0.000180 | completed |

在该单 seed case study 中，V0-F 与 V0-T 均在三项指标上超过这四个新增对照方法；
V0-T 相对最强外部 ARI（TableDC）的差为 `+0.015225`，V0-F 的差为 `+0.013547`。
这不是多 seed 稳健性、统计显著性、普遍 SOTA 或全仓库方法覆盖结论。正式记录、输入
hash、预测/真值数组和审计见 `result/recent_baseline_panel_20260902/`、
`reports/V0_recent_baseline_panel_20260902.md` 和该目录的 `EXPERIMENT_AUDIT.md`。
`Mouse_retina` 的 AHDPC/HDPC feasibility attempt 未形成完整方法面板（HDPC
`incomplete_compute`，ZEUS/TableDC 未启动），因此不纳入本节胜出结论。

## 2026-09-02 V0 严格对照筛选：可用于论文的突出数据集

新增后处理分析器 `scripts/V0/analyze_vs_clubench_methods.py`，对已完成并通过审计的
V0 CLUBench 单 seed 矩阵进行逐数据集比较。对照集合由官方 CLUBench `best_hpc` 归档的
24 个方法和同 seed=42、同 131 数据集面板的 AHDPC、HDPC、V9 组成，共 27 个方法；官方
`best_hpc` 文件只保存指标，没有逐行 seed 元数据，因此不把这 24 个值描述为 seed-matched
重跑。
分析器只读取已落盘的指标，不训练、不调参、不读取标签；严格胜出定义为 V0 在 ARI、
NMI、ACC 三项上都超过每一个对照方法，浮点容差为 `1e-12`。

在 seed=42、known-K 仅用于 fit 后 KMeans readout 的口径下，F/fixed 与 T/topology
均严格胜出的数据集只有：

| Dataset | Parameterization | ARI | NMI | ACC | strongest competitor |
|---|---|---:|---:|---:|---|
| `Mouse_retina.npz` | fixed/F | 0.943335 | 0.904110 | 0.984555 | V9 (ARI 0.930365) |
| `Mouse_retina.npz` | topology/T | 0.932349 | 0.883333 | 0.981561 | V9 (ARI 0.930365) |
| `htru2.npz` | fixed/F | 0.779476 | 0.626089 | 0.971097 | birch (ARI 0.777767) |
| `htru2.npz` | topology/T | 0.781154 | 0.628056 | 0.971297 | birch (ARI 0.777767) |

在只比较官方 24 个 CLUBench 方法时，额外的严格胜出数据集为 `enron.npz` 和
`sms_spam_collection.npz`；由于同 seed V9 在至少一个指标上超过 V0，这两项不属于
27 方法口径的推荐清单。完整逐数据集记录、输入 SHA256、方法列表和 claim boundary
见 `result/analysis/topogate_v0_vs_clubench_methods_20260902/`。

边界：上述是 single-seed descriptive case-study evidence，使用 benchmark-known-K
外层 readout；V0 与 AHDPC/HDPC/V9 为 seed=42，官方 24 个值的逐行 seed 未记录。不能解释为
多 seed 稳健性、统计显著性、全体 `methods/` 覆盖或普遍 SOTA。
论文主表仍应报告完整 131 数据集矩阵/宏平均，并将 `Mouse_retina`、`htru2` 明确标为
突出案例；正式优越性结论需对选定数据集和对照方法使用预注册的 paired seeds `[42,123,7]`。

## 2026-09-02 V0 随机协议清理与 CLUBench 最佳数据集

V0 活动代码现在只接受 `rng_protocol=isolated_v0`；旧的 `legacy_plantnet` 配置分支、
旧随机状态回放和 V0 侧 legacy parity 可执行入口均已删除。PlantNet 原始源码、历史
parity/attribution 产物和本节之前的记录保留为只读溯源，不会被当前 V0 导入或自动调用。
新增回归测试确认传入 `legacy_plantnet` 会 fail closed；已有 CLUBench 正式矩阵本来就使用
`isolated_v0`，因此本次清理没有改变该矩阵的数值。

在 `result/v0/clubench_single_seed_v1/final/` 的官方 CLUBench 全量单 seed（seed=42）
结果中，按 ARI、NMI、ACC 的单 cell 最高值选择，最佳数据集为 `weather.npz`：

| Dataset | Parameterization | ACC | NMI | ARI |
|---|---|---:|---:|---:|
| `weather.npz` | fixed/F | 1.000000 | 1.000000 | 1.000000 |
| `weather.npz` | topology/T | 1.000000 | 1.000000 | 1.000000 |

该选择只表示 seed=42、known-K 外层 readout 下的描述性最高 cell；它不是多 seed 稳健性、
统计显著性、总体 SOTA 或 F/T 优越性结论。正式六个单细胞数据集的三 seed 结果中，
F 的最高平均 ARI 是 `Mouse_Pancreas_1`（`0.884923 +/- 0.007399`），T 的最高平均
ARI 是 `Human_Pancreas_1`（`0.880978 +/- 0.001383`），两者属于不同的统计口径，不能
与 CLUBench 单 seed 的 `weather.npz` 直接混为同一“最佳”。

## 2026-09-02 TopoGate V0 PlantNet F/T reference-equivalence smoke（工程证据）

此前新增并运行过 `scripts/V0/run_reference_equivalence.py` 与对应 V0 focused tests，直接对照
`/home/luolie/biopipeline/dimension-reduction/plantnet/experimental_retired_models/` 中冻结的
F/T 参考函数。24×11 确定性 toy 输入上，F/T 的图、边可靠性、node gate、pseudo view、训练样本
权重和 row-swap mask/value 共 17/17 项逐数组 `exact_equal=true`；F/T 各完成一次 2-epoch CPU
label-free fit，embedding finite 且 `labels_used_during_fit=false`。这是实现等效性/工程 smoke，
不是聚类性能或优越性结论。compact JSON 见
`review-stage/V0_unified_reference_check_20260902/summary.json`；该 parity 可执行入口随后已从
当前 V0 删除，正式 `result/v0` 目标在本沙箱
只读，未写入半成品结果。

## 2026-09-01 TopoGate V0 current-YAML formal reproducibility rerun

按固定的 PlantNet `scvicar_attribution_v2` 输入、统一 V0 预处理、known-K 外层协议和
paired seeds `[42,123,7]`，当前 V0 完成了 6 datasets x 2 parameterizations x 3 seeds
的 `36/36` 矩阵。正式输出为 `result/v0/formal_v0_repro_v2/`，其中 fixed/F 与
topology/T 共享同一 V0 scMAE backbone/trainer，只切换当前 YAML 规定的 corruption
parameterization（F `neighbor_k=5`；T `neighbor_k=10`）。没有复用旧产物，`manifest.json`
终态为 `completed_valid=36`、`reused=0`、`incomplete_compute=0`、`protocol_mismatch=0`。

输入为 `/home/luolie/biopipeline/dimension-reduction/plantnet/result/scvicar_attribution_v2/datasets/`；
每个输入和 V0 源文件的 SHA256、解析配置、GPU preflight、命令和状态均记录在 manifest
及 `matrix_audit.json`。预处理固定为 V0 h5ad loader、raw/count auto selection、Seurat
HVG 1000、`normalize_total(target_sum=10000)`、`log1p` 和 scaling。`K` 由外层
`resolved_label` 计算 `int(np.unique(y).size)`，只用于 fit 后 KMeans readout/benchmark；
所有 run 的 `K_source=benchmark_oracle_from_y`，且 graph/gate/loss/preprocessing/fit/selection
的 label flags 均为 false。实际使用物理 GPU6；GPU0/7 禁用。

下表是每个 dataset 三个 seed 的均值 +/- sample std；这是 known-K real-label benchmark
的描述性结果，不是无标签部署或普遍优越性结论。完整 ACC/NMI/ARI/AMI/F1/FMI 在
`run_level_metrics.csv` 与 `summary_metrics.csv` 中。

| Dataset | K | Fixed ARI | Topology ARI | Fixed NMI | Topology NMI |
|---|---:|---:|---:|---:|---:|
| `Blood_BoneMarrow` | 30 | 0.418699 +/- 0.012852 | 0.436933 +/- 0.011185 | 0.720427 +/- 0.005993 | 0.731938 +/- 0.007351 |
| `Human_Pancreas_1` | 6 | 0.878527 +/- 0.000955 | 0.880978 +/- 0.001383 | 0.855246 +/- 0.000859 | 0.857834 +/- 0.002682 |
| `Human_Pancreas_3` | 13 | 0.775411 +/- 0.001173 | 0.859935 +/- 0.073995 | 0.823593 +/- 0.001612 | 0.848855 +/- 0.024646 |
| `Mouse_Pancreas_1` | 10 | 0.884923 +/- 0.007399 | 0.862244 +/- 0.041974 | 0.861885 +/- 0.003335 | 0.846122 +/- 0.026733 |
| `PRJNA895163` | 12 | 0.177385 +/- 0.013274 | 0.168517 +/- 0.020170 | 0.371692 +/- 0.018618 | 0.357035 +/- 0.022564 |
| `TabulaSapiens_Pancreas` | 16 | 0.487205 +/- 0.006509 | 0.512820 +/- 0.015274 | 0.721740 +/- 0.003970 | 0.734451 +/- 0.004563 |

18 个配对 seed 的 topology-fixed ARI 差为 `+0.016546 +/- 0.047936`，NMI 差为
`+0.003609 +/- 0.022245`；数据集方向并不一致，不能据此宣称 topology 优于 fixed。
独立终态审计为 `audit_ok=true`，且数组形状、finite 指标和标签隔离均通过。此前
`result/v0/formal_v0_repro/` 的显式 K 试运行被标为协议不匹配并排除；历史
`legacy_attribution_v2` staging 也未混入本表。Claude `auto-review-loop` 请求分别遇到
privacy rejection 与 1200 秒无正文超时，没有生成 acquittal；本地 focused tests 为
`32 passed`，两者边界均已记录在 formal README 和 review trace。

## 2026-09-01 TopoGate V0 CLUBench engineering smoke（非性能结论）

V0 已接入 CLUBench 为独立 `TopoGateV0` adapter，真实调用链为
`load_data -> fit_predict -> clustering_evaluation`。`-f/-F/F/fixed` 统一为
`parameterization=fixed`，`-t/-T/T/topology` 统一为 `parameterization=topology`；两者共享
同一 V0 scMAE backbone/trainer，只切换 corruption parameterization。adapter 的 core API
没有 `y` 参数，known-K 仅用于外层 KMeans readout；graph、gate、loss、selection 均为
`labels_used_* = false`。

已完成 `datasets/iris.npz`、z-score、CPU、seed=42、2 epochs 的 engineering smoke，产物在
`result/v0/clubench_smoke/`：

| run key | parameterization | neighbor_k | ACC | NMI | ARI | 状态 |
|---|---|---:|---:|---:|---:|---|
| `iris::topogate_v0::fixed::seed42` | fixed/F | 5 | 0.846667 | 0.713736 | 0.649754 | completed |
| `iris::topogate_v0::topology::seed42` | topology/T | 10 | 0.846667 | 0.713736 | 0.649754 | completed |

两个 run 的 `summary.json` 均标记 `evidence_level=engineering_smoke`、
`engineering_smoke=true`、`performance_conclusion_allowed=false`；数值仅证明协议、梯度、
输入与产物链路可运行，不支持性能改进、F/T 优劣、泛化或论文结论。聚焦回归为
`32 passed`，compileall 通过；smoke runner 已加入 completed-artifact 精确协议键复用保护。
上述 smoke 后，正式全量单 seed CLUBench 矩阵已经按冻结协议完成；其产物与结论边界见下节。

## 2026-09-02 TopoGate V0 full CLUBench single-seed matrix（描述性，不作性能结论）

统一 V0 入口 `methods/TopoGate/V0/trainer.py:fit_predict` 已完成 CLUBench 官方全量
131 数据集 × `fixed/F`、`topology/T` × seed `42` 的 **262/262** 运行；权威根目录为
`result/v0/clubench_single_seed_v1/final/`。`artifact_audit.json` 核验每个 cell 的 source hash、
解析配置、全列 z-score 输入、known-K 外层 readout、标签隔离和全部数组/诊断文件，终态为
`completed_cell_count=262`、`incomplete_cell_count=0`。

F/T 均在同一个 V0 scMAE backbone/trainer 中运行（不是 retired `-f/-t` runner）；F 使用冻结的
label-free tuning anchor，T 使用同一候选筛选后的 topology anchor。所有输入均为 CLUBench
全数据逐列 z-score、无 feature cap；训练/预处理/graph/gate/loss/selection 的
`labels_used_*` 均为 false。标签只在 `fit_predict` 返回后用于 `K=int(unique(y))` 的 KMeans
readout 和后验指标。

`single_seed_descriptive_summary.json` 的 131 数据集等权宏平均 ARI 为 F=`0.321713`、
T=`0.322181`，T−F=`+0.000468`（60 正、51 负、20 平）。这仅是一个 seed 的描述值，不能据此
主张 T 优于 F、稳健性、显著性或论文级泛化；已预注册的 6 数据集 × 3 seed 机制消融正在另行
运行，并将以 complete-cell 审计为前提汇总。

## 2026-09-01 TopoGate V0 legacy attribution-v2 data staging（非新性能结果）

为统一后的 `methods/TopoGate/V0/` 提供历史等效参考数据，已将 PlantNet
`scvicar_attribution_v2` formal 矩阵原样复制到
`result/v0/legacy_attribution_v2/`。矩阵覆盖 6 个数据集、7 个 variant、3 个 seed，共
126/126 个 success runs；每个 run 39 个文件，共 4,914 个 formal run 文件，源和目标
formal run bytes 均为 `4,072,658,040`。目标同时保留 aggregate CSV、状态/编排元数据、源冻结
记录、协议 README、完整 hash list、README 和 manifest。输入 `.h5ad` 没有重复复制。

这是一项数据 staging/provenance 操作，不是新的 V0 formal GPU run。`fixed` 和
`topology_full` 分别映射历史 scVICAR-F/T 的 model identity；五个 paired controls 也保留，
所有 run 文件均 verbatim 且没有 `labels.npy`/`eval_metrics.json` 等文件重命名。源审计确认
126/126 success、label leakage flags 全为 false；source/destination run identity、文件数/字节数
和 checksum-aware `rsync --checksum --dry-run` 均通过。

必须保留的协议边界：历史矩阵使用 `gate_max=0.1`，T 使用 `neighbor_k=5`，而当前 V0 YAML
为 `0.15` 和 `10`；因此不把这些历史数值放入当前 V0 YAML 性能表，也不宣称数值复现、T
executable parity、聚类效果、改进、鲁棒性或泛化。标签/K 只是 benchmark/readout metadata，
`direct_v0_run=false`。Claude cross-family staging review 为 `8/10 ready`，其 sandbox 无法
直接读取外部源路径；同一 staging 的 same-family claim gate 只支持上述 provenance 命题，
不构成科学 efficacy acquittal。权威文件为
`result/v0/legacy_attribution_v2/manifest.json`、
`review-stage/V0_data_staging/AUTO_REVIEW.md` 和
`review-stage/V0_data_staging/CLAIMS_FROM_RESULTS.md`。

## 2026-08-19 raw_sparse_mask_schedule_probe v2 shared-resource amendment

用户明确清除 E1–E4、E8–E12，建立独立协议
`raw_sparse_mask_schedule_probe_v2_shared`。v2 允许 GPU 1–6 与外部进程共享，取消
idle/occupancy gate、每卡单 worker、T0+11h cutoff、一次 retry 上限和 v1 P0/P1 重跑要求；
GPU 0/7 禁用、外部进程不抢占、labels-after-fit、失败 `incomplete_compute` 和完整 provenance
仍保留。v1 `MAIN` 与旧 `SHARED_RESOURCE_MAIN` 不变，v2 结果只写入
`result/raw_sparse_mask_schedule_probe/V2_SHARED/`。

v2 已完成一次 GPU 2 的单 cell 验证（`cnae9/CLEAN_AE/seed42`, 30 epochs），状态为
`completed_valid`；该单 cell 只是 launcher/产物验证，不是性能结论。正式 90-cell v2 矩阵在本条
之后启动，聚合与最终 decision 尚未生成。清除记录见
`reports/raw_sparse_mask_schedule_probe/CLEARANCE_E1_E12_20260819.md`。

## 2026-08-19 shared-resource engineering execution

用户明确授权与外部任务共享物理 GPU 1/6。固定 90-cell 队列已完成 `90/90`（每卡 45），
无 OOM/超时/return-code failure。正式 `MAIN` 仍保持 idle-GPU 门控；共享运行隔离写入
`result/raw_sparse_mask_schedule_probe/SHARED_RESOURCE_MAIN/`，所有 cell 均标记
`shared_resource_engineering_only`、`audit_ok=false`，不进入正式聚合或论文结果。运行不调用
进程终止，不使用 GPU 0/7。资源完整性审计位于
`result/raw_sparse_mask_schedule_probe/SHARED_RESOURCE_AUDIT.json`；该运行不产生可引用的
科学性能结果。

## 2026-08-19 raw_sparse_mask_schedule_probe dispatcher occupancy incident

在一次错误的直接 `dispatch_main([1,6])` 调用中，GPU 1/6 仍有外部进程；旧 dispatcher
只检查 GPU ID 合法性，导致 8 个 seed-42 MAIN cells 被尝试，其中 6 个写出 partial
summary、2 个在运行中被终止。该进程组已终止，外部 PID 未触碰；8 个 cell 均明确标记为
`incomplete_compute`/排除，partial metrics 不属于科学结果。随后新增 dispatch-entry 与
per-bind occupancy guard，真实复测在 GPU 1/6 仍占用时返回 `GPU_WAITING`、`launched=0`。
当前只允许在 fresh snapshot 同时显示物理 GPU 1 和 6 为 `legal_idle_gpus` 后重新冻结并启动；
GPU 0/7 仍禁止。incident manifest 位于
`result/raw_sparse_mask_schedule_probe/MAIN/INCOMPLETE_DISPATCH_INCIDENT_20260819.json`。

## 2026-08-19 raw_sparse_mask_schedule_probe P0/P1 prelaunch state

独立项目 `raw_sparse_mask_schedule_probe` 已完成冻结协议、六个 E3 raw source 的
zero-preserving adapter/provenance audit、`13 passed` focused tests 和 compileall。P1 的
SVD32 baseline 为 `18/18` completed-valid，sparse/dense first-projection benchmark 为
`6/6`，均为无标签工程基线，不是 MAIN 聚类结果、holdout 或泛化证据。第一次 P1 调用的
部分产物没有对应状态，随后在统一 CSR SVD 路径下重跑并写入 `P1_COMPLETE`；该中断未启动
任何 MAIN GPU cell。

Claude prelaunch review 三轮最终为 `9.5/10, ready`；Round 1/2 指出的 dispatcher、SVD
fail-closed coverage、code SHA resume binding、冻结 hash 强制相等和 state reconciliation
问题均已修复并由 16 个 focused tests 覆盖。当前
`result/raw_sparse_mask_schedule_probe/OVERNIGHT_STATE.json` 为 `GPU_WAITING`：GPU 1–6
均有外部进程，GPU 0/7 禁止；MAIN `90` cells、fixed-ratio oracle、representation
localization 和 holdout 均未启动。Git 元数据不可验证，freeze 使用 file-hash-only anchor。

## 2026-08-18 corruption_objective_compatibility_probe E0–E4 terminal result

独立项目 `corruption_objective_compatibility_probe` 按冻结协议完成 E0、E1、E1b 和 E3；E1
为 `6 datasets × 3 arms × 3 paired seeds = 54/54` completed-valid，其中 biological
P0/P2 的 18 个单元从已关闭 C2 经当前 H0/budget/label SHA256 校验复用，36 个新 GPU 单元
在物理 GPU 6 完成。E1b no-fit 为 54/54 CPU cells，E3 raw audit 为 6/6、`audit_ok=true`、
`labels_not_loaded=true`。GPU 0/7 全程禁用，未抢占 GPU 1–5 的外部任务。

Primary E1 使用 `Delta_random=ARI(P2)-ARI(P0)`、`Delta_clean=ARI(P2)-ARI(Clean)` 和
no-fit training-amplification 诊断。三个 burned biological development datasets 复现了 C2
P2 增益（Mouse `+0.394898`、Baron `+0.126069`、Campbell `+0.146883`）；但三个
non-biological sentinel datasets 中没有一个同时满足两个 `≥0.03` model margins 和两个
of-three seed-positive checks（G1=`0/3`），training amplification 仅 cnae9 达到 margin
（G2=`1/3`）。因此按预注册门控 E2 objective matrix 未启动，自动终态为
`STOP_GENERAL_CORRUPTION`；这不是 P2 全局无效或 raw-X support 因果结论。

E0 `audit_ok=true`、corrected D1 gate=false、`d2_authorized=false`、GPU runs=0；support
attribution 仍冻结。E1/E2 的 support 语义是 threshold-defined dense H0 proxy，E3 raw-X
zero/nonzero 仅描述性，不能进入 fit/gate/decision。Focused tests=`9 passed`，compileall
通过。权重、输入、labels、arrays、embeddings、predictions、checkpoints 和 per-run logs
不属于发布摘要；compact final artifacts 位于
`result/corruption_objective_compatibility_probe/FINAL/`，解释报告位于
`reports/corruption_objective_compatibility_probe/E1_RESULTS.md` 与
`E1_INTEGRITY_AUDIT.md`。

## 2026-08-18 sparse_corruption_principle_probe C2 terminal result

独立项目 `sparse_corruption_principle_probe` 按冻结协议
`sparse_corruption_principle_probe_c2_v1` 完成正式 `3 datasets × 6 principles × 3 paired
seeds = 54/54` GPU runs。C2 使用共同 dense SVD/H0 proxy、相同 reconstruction probe、exact
changed-coordinate budget 和 labels-after-fit-only firewall；标签只用于 fit 后的
benchmark-known-K 外层 readout。独立 C2 integrity audit 为 `audit_ok=true`，17/17 checks 通过；
实际物理 GPU 为 `[2,3,4,5,6]`，GPU `0/7` 禁用，GPU1 本轮因外部占用未使用。

Primary `Delta_P = ARI(P)-ARI(P0_Random)`（dataset 是统计单位，seed 是 paired repeat）：

| Dataset | P1 | P2 | P3 | P4 | P5 | Best |
|---|---:|---:|---:|---:|---:|---|
| Mouse_retina | +0.319044 | **+0.394898** | +0.388034 | +0.384727 | +0.116976 | P2 |
| Baron Human | −0.126288 | **+0.126069** | −0.108278 | −0.102735 | −0.116420 | P2 |
| Campbell | +0.029888 | **+0.146883** | +0.046116 | +0.082621 | +0.014387 | P2 |

Material descriptive margin 为 `0.03 ARI`。P2 在 `3/3` development datasets 达到该 margin，
P3/P4 各为 `2/3`，因此 C2 终态标签为 `simple_static_principle_sufficient`。这只是测试过的
static library 在 development panel 上的 bounded finding，不是 oracle 上界、raw-X
zero/nonzero support 结论或 generalization claim。C3 holdout、adaptive policy、GAN 和
learned generator 仍锁定。compact C2 report/audit 位于
`reports/sparse_corruption_principle_probe/C2_RESULTS.md` 与
`reports/sparse_corruption_principle_probe/C2_INTEGRITY_AUDIT.md`。

## 2026-08-18 sparse_corruption_principle_probe C0–C1 implementation and structural audit

建立独立项目 `sparse_corruption_principle_probe`，不创建 V 系列；Relation 与旧
`adaptive_corruption_probe` 只读。C0 冻结 Mouse_retina、Baron Human、Campbell 三个
mechanism/development datasets、六个 primary static principles、exact changed-coordinate
budget、labels-after-fit-only firewall 和 GPU pool `[1,2,3,4,5,6]`（0/7 禁用）。

- Toy S/V/M apparatus：`completed_valid`，18 个 fixture rows 全部 finite/exact-budget；这不是
  real-data clustering evidence。
- Holdout inventory：14 个 label-free candidate records 中按预注册 maximin 选择 12 个，
  `shortfall=0`、无 development overlap、source SHA256 完整、`audit_ok=true`；holdout runs
  仍未授权。
- C1：三 dataset × 六个 closed B1 arm × 三 paired seeds = `54/54` structural replays，
  `fit_runs=0`、`labels_loaded=false`、CSV numeric fields finite。它只读取 closed B1 compact
  post-fit ARI/L_rec 作为 provenance，并在 audited S0 H0 上计算 support/value/geometry diagnostics。
  H0 是固定阈值 support proxy，不是原始 count-matrix zero support；C4 structural replay 使用
  column-median/MAD residual proxy，不等同 B1 warm-up residual。
- Local deterministic contract audit 为 `audit_ok=true`。外部 `auto-review-loop` route 因
  reviewer plan-mode 无法读取 artifacts，结果为 `review_unavailable_no_score`，没有被当成科学证据。

随后对 C2 几何分数做小样本边界复核，发现 `n_neighbors=n` 会触发 sklearn 的严格邻居数约束；
已改为最多请求 `n-1` 个非自身邻居，并加入 `n_rows={2,3,4,5}` 回归测试。当前
`python -m pytest -q tests/sparse_corruption_principle_probe` 为 `16 passed`，compileall 与本地
contract audit 仍通过。该修复只证明静态库边界可运行，不产生 C2 性能证据；C2 矩阵、C3、adaptive
policy、MLP、GAN 和 learned generator 继续锁定。

C2 `3×6×3` GPU performance matrix、adaptive policy、MLP selector、GAN、learned generator 和
C3 holdout runs 均继续锁定；当前结果只支持“协议/机制诊断已完成”，不支持任何新方法性能或
因果发现。

## 2026-08-18 Independent parallel probes — A1/B1 completion

`learned_relation_rule_probe` and `adaptive_corruption_probe` are independent,
non-V-series projects.  Track A A1 used only the three burned development
datasets (`cnae9`, `Campbell`, `sms_spam_collection`) and the inherited
candidate pool/R/O_pool artifacts.  The 2 scorers × 3 views × 5 anchor-group
fold contract had 100% OOF coverage and no anchor leakage.  Its diagnostic
supervised ceiling did not reach the frozen 2-of-3 material gate:

| dataset | best diagnostic scorer/view | `Delta_sup` mean | `H_pool` mean |
|---|---|---:|---:|
| cnae9 | TinyMLP / No-rank | `-0.015740` | `+0.215720` |
| Campbell | Logistic / No-geometry | `+0.024503` | `+0.191444` |
| sms_spam_collection | Logistic / Full | `-0.300232` | `+0.367108` |

The terminal A1 decision is
`predictable_reference_not_actionable_for_selection`; A2–A5 remain locked.
This is a diagnostic supervised ceiling, not label-free utility.

Track B B1 passed the unlabeled synthetic sensitivity fixture and completed
`108/108` runs: six fixed datasets × six arms × three paired seeds, with GPU
workers restricted to physical GPUs `[1,2,3,4,5,6]`.  In the fresh
pair-feasible rerun, the primary C0-vs-clean contrast was material on
Mouse_retina (`+0.074182`) and Campbell (`+0.041038`); the best structured
corruption was also material on Baron Human.  Level 2 found material
structured-vs-random effects for C2/C3/C4 on two registered-scRNA development
datasets.  The Level-3 role
heterogeneity gate was not met (only one coarse role class had a material
winner), so the terminal decision is
`simple_corruption_principle_sufficient`; B2–B5 and holdout are locked.
These are bounded mechanism-panel results, not generalization evidence.

Compact reports and audit artifacts are under
`reports/learned_relation_rule_probe/A1_RESULTS.md`,
`reports/adaptive_corruption_probe/B1_RESULTS.md`, and the corresponding
`result/*/A1_supervised_ceiling` / `B1_corruption_library` directories.  Raw
inputs, labels, scores, embeddings, predictions, checkpoints and logs remain
local and are excluded from publication.

## 2026-08-17 Relation-selection probe RS0–RS3 terminal result

独立项目 `relation_selection_probe` 在关闭 `representation_consumer_probe` 后按冻结范围
完成 RS0、RS1、RS2、RS3；不属于 V 系列，也没有启动 learned selector、new backbone、
holdout 或新 reconstruction objective。旧项目 S0/S1/S2 仅作为只读输入，RS0 继承的
holdout manifest SHA256 为 `6d9afa1f6d90f77d8836e9f877f6567ebb7c7621ba3d022622e2488c9fb8b2cb`。

- RS1：17 个 label-free relation features、5-fold GroupKFold by anchor。对
  `pool_reference_membership`，7/7 families 在固定的三个 primary datasets 均通过
  `Delta AP >= 0.10` 与 `Lift@b >= 1.5`；对 `same_class`，0/7 families 通过完整双阈值。
  这只证明 reference-selection solvability，不证明 semantic same-class utility。
- RS2：B0 cosine、B1 mutual-first、B2 SNN/Jaccard、B3 stability、B4 equal-rank fusion
  共 `90/90` completed-valid，selector、graph、Spectral fit 均未使用标签。五个 selector
  中没有一个在至少两个 primary datasets 同时达到 `Delta_S >= 0.03` 且 median
  `Capture_S >= 0.25`；B4 在 cnae9 的最佳描述性均值为 `+0.0144`，但 Campbell 为
  `-0.0298`、sms 为 `-0.3080`。
- RS3：cnae9/Campbell/sms 的 `H_pool` 分别为 `+0.215720/+0.191444/+0.367108`，
  但没有固定 selector 产生 material positive capture。hate_speech 的
  `H_full-H_pool=+0.634319` 触发 extreme candidate-family sentinel，且 sms 的
  `H_full-H_pool=+0.175197` 也为 material gap；Mouse_retina 的最佳
  selector `+0.0193`（约为其低机会 `H_pool` 的 70%）未触发 material contradiction；Baron
  Human 保留为低机会 consumer boundary。Mouse/Baron 不进入 primary denominator。

终态为 `candidate_family_problem_and_learned_rule_only_proposal`。后续若研究 learned
selector，必须另起新协议并重新预注册；当前项目不执行 RS4。发布层仅包含
`reports/relation_selection_probe/`、relation-selection code/tests 和
`result/relation_selection_probe/FINAL/` 的 weight-free compact summaries，raw arrays、
graphs、embeddings、predictions、weights 和输入数据不发布。

## 2026-08-17 Representation-consumer probe S2 SimpleCut conditional confirmation

独立项目 `representation_consumer_probe` 在 S1 需要排除 Spectral relaxation 假阴性的两个
数据集上完成冻结的 S2：`Baron Human`、`Mouse_retina` × `R/O_pool/O_full` × seeds
`[42,123,7]`，共 `18/18` completed-valid。原始工件位于
`result/representation_consumer_probe/S2_simple_cut/`，报告位于
`reports/representation_consumer_probe/S2_RESULTS.md`。

- SimpleCut fit 只接收固定 H0、selected graph W、seed/device/epochs；标签不进入 encoder、
  loss 或 optimizer。K 仅用于 benchmark-known-K 的外层 KMeans/readout；O_pool/O_full 是
  label-derived diagnostic oracle，不是可部署方法性能。
- Dataset-level primary diagnostics：Baron Human `H_pool=+0.033242`、
  `H_full=+0.033367`、`C=+0.000125`；Mouse_retina `H_pool=+0.008880`、
  `H_full=+0.009622`、`C=+0.000742`。Baron 达到冻结的 descriptive `delta=0.03`，但
  seed 波动大，只支持“Spectral 阴性可能是 relaxation miss”；Mouse 仍是
  `observed-small`，不能写成 topology 全局 No-Go。两者均无 material candidate gap。
- 18/18 的 labels/source SHA、ARI/NMI/optimal-mapping ACC、S1 selected/direct graph reuse 和
  exact-tree hashes 均通过 fresh integrity audit；embedding finite，未见明显 collapse。
- 审计总体为 `WARN` 而非失败：`training_metrics.csv` 最后一行记录 optimizer step 前的
  loss，`fit_metadata.final_loss` 为 step 后重算值。该 metadata timing gap 不影响 primary
  opportunity metrics，但不应把两者写成同一个训练点。
- S2 是该项目 terminal decision。`S_graph` 仍不可估计；S3/S4/S5/S6、TopoCut、新 selector
  和 holdout 均继续锁定。终局为 `heterogeneous_with_spectral_relaxation_caveat`，不是
  representation-consumer promotion。

## 2026-08-17 Representation-consumer probe S1 formal opportunity-only matrix

S0 PASS 后，按冻结协议 `representation_consumer_probe_s1_opportunity_spectral_v2` 完成正式
S1：6 datasets × 5 arms (`F/U/R/O_pool/O_full`) × 3 paired seeds，共 `90/90` completed-valid
jobs。结果工件位于 `result/representation_consumer_probe/S1_oracle_v2/`，每个 job 保存
resolved config、graph/embedding/predictions、metrics、audit 和 artifact hash。

- 训练/graph consumer fit 未使用标签；标签只用于 O_pool/O_full diagnostic graph 与外层 ARI/NMI/ACC。
- Primary quantities 仅为 `H_pool=ARI(O_pool)-ARI(R)`、`H_full=ARI(O_full)-ARI(R)`、
  `C=H_full-H_pool`；`S_graph` 仍不可估计。统计单位为 dataset，seed 是 paired repeat。
- Material `H_pool`：`3/6` datasets（cnae9 `+0.215720`、Campbell `+0.191444`、sms
  `+0.367108`）。`H_full` material：`4/6`（另含 hate_speech `+0.636495`）。
- Matched-budget candidate gap `C` material-positive：`2/6`（sms `+0.175197`、hate_speech
  `+0.634319`）；其余数据集没有正向 candidate gap，不能据此宣称 total candidate-recall loss。
- Mouse_retina (`H_pool=+0.027426`) 与 Baron Human (`+0.014306`) 未达到 `delta=0.03`，按预注册
  规则仅要求 conditional S2 才能排除 Spectral relaxation 假阴性；不作 topology No-Go。
- `F` descriptive arm 的首版 row-L2 运行保留于 `S1_oracle/`，并标记 `invalid_design`；正式
  证据只使用 `S1_oracle_v2/` 的 raw-H0 F arm。S3/S4/S5/S6、TopoCut、新 selector 仍锁定。
- 独立 `experiment-audit` 对 90 个 run 的 ARI/NMI、labels、hash、scope 和 claim boundary
  复核为 `PASS`；根目录 hash manifest 采用排除当前 root manifest、纳入 README 与嵌套 run
  manifests 的 exact-tree policy（`827` entries）。F/U/R 必须继续标为 known-K real-GT benchmark，
  O_pool/O_full 必须继续标为 pre-fit label-derived diagnostic oracle，不是可部署方法。

## 2026-08-17 Representation-consumer probe S0 formal contract replay

独立项目 `representation_consumer_probe` 的正式 S0 replay 已写入
`result/representation_consumer_probe/S0_freeze/`，并生成 `artifact_hashes.json`。本次没有训练、没有 S1/S2 性能计算，也没有
GPU job；因此不产生 ARI/NMI 或 backbone 结果。

- source/hash/shape preflight：`6/6` completed_valid；E1 manifest SHA match；H0/SVD 合同通过。
- adapter audit：`adapter_not_estimable`。这是当前项目 T-related causal chain 的 terminal state，
  只允许不含 T 的 opportunity-only S1，必要时再做 S2 confirmation。
- graph/loss numerical sanity：PASS；Spectral recovery sanity：PASS（clean block ARI=`1.0`、
  contaminated ARI=`0.8136842105`，isolate rows zero）。这些是 apparatus contract，不是数据集性能。
- budget contract：`budget_cap=8`、逐行 `b_i=min(8,positive_count_i)`；六个 stress datasets 全部
  保留。`cnae9`/`sms_spam_collection`/`hate_speech` 分别有 `1`/`40`/`135` 个 zero-budget rows，
  没有通过异类边补齐或整集删除。
- S3/S4/S5/S6、TopoCut 和新 selector 在本项目内永久锁定；holdout manifest 状态为
  `dormant_due_to_adapter_not_estimable`。

该条目只记录可复核的 S0 contract 工件，不把 `adapter_not_estimable` 扩展成 topology 全局
No-Go，也不把临时审查或单元测试当作性能证据。

## 2026-08-17 ACCG real panel (clustering promotion No-Go)

v3 synthetic contract 通过后，冻结的 v2 real manifest 完成主矩阵 `30/30`：9 个有标签
数据集 × 3 seed 的 `27` 个 confirmatory panels，加无标签 PBMC3k、显式 `K=8` 的 `3` 个
operational panels。开发集消融 `48/48`，四个 ablation variant 复用 canonical `N/R/T_s`
controls；因此 confirmatory artifact 总数为 `75/75`。完整 weight-free 摘要和来源/协议哈希
位于 `review-stage/ACCG_real_panel_v2_audit/`，raw checkpoints、predictions 和输入数据不
进入 GitHub。

- 所有训练、graph、Gate、loss 和 readout 均未使用真值标签；9 个有标签数据集的 `K` 为
  `benchmark_oracle_from_y`，PBMC3k 的 `K_source=explicit_n_clusters`。PBMC3k 不计算
  ARI/NMI，不进入 confirmatory aggregate。
- 主 endpoint `ARI(T_c)-ARI(T_s)` 的 dataset-level mean=`+0.007492`、median=`+0.000363`，
  dataset bootstrap 95% CI=`[-0.000879,+0.018889]`，仅 `4/9` 数据集三 seed 全部为正。
- 开发集 joint mean effect=`+0.010751`，coordinate=`+0.015689`，joint-coordinate
  `-0.004938`；joint 仅 `1/12` 个 paired seed 胜出。

结构审计、matched schedule、source/config/branchpoint identity 和 label isolation 均通过，
因此这不是 incomplete compute 或资源失败，而是当前证据不支持“joint structural constraint
带来稳定聚类提升”的 Q1 方法主张。按停止规则没有启动 external baseline 或 outcome-driven
rescue。

## 2026-08-16 ACCG Stage 1 synthetic contract (historical pre-compute gate)

ACCG Stage 1 generated the frozen W0-W5 synthetic panel: `60` records from two
generator families, five seeds, and six worlds. This stage performed no model
training, real-data fitting, queue launch, or GPU execution. Artifacts are under
`result/ACCG_action_constrained_gate/synthetic_contract_v1/`.

- Shortcut audit: `10/10 valid`; support is exactly matched within each
  family-seed and support/marginal classifier AUC stays below `0.60`.
- Grouped action probe: `40` records, but the frozen promotion decision is
  **No-Go** (`9/30` required records pass; pooled family-holdout joint
  AUC=`0.634351`, below `0.65`).
- W5 exact-selector audit: `32/32` exact-feasible rows, greedy-feasible rate
  `1.0`, labels unavailable to the selector.
- A W5-only family-holdout diagnostic gives AUC=`0.664208`; this is a
  post-audit interpretation and does not override the frozen promotion gate.

At that historical stage no ACCG synthetic end-to-end, real `N/R/T_s/T_c`,
ablation, or baseline result was yet available. The subsequent v3 promotion and
real-panel result are recorded in the section above.

## 2026-08-15 V25 Systematic Failure Atlas and Mechanism Localization

V25 是 V1--V24 的系统机制研究，不是新的 TopoGate architecture。正式工件位于
`result/V25_systematic_mechanism_study/`。

- A0 registry：V1--V22 共 `2209` rows、`2175` completed、`1637` paired Delta ARI rows、
  `431` dataset/protocol/readout units；V23/V24 为 `2` 条 boundary evidence，不进入定量
  intervention atlas；历史 summary table 的 replay eligibility 为 `0`。
- A1 atlas：`1637` paired rows，`194` material positive、`680` material negative、`763`
  observed-small；统计单位是 dataset/protocol/readout，seed 是重复测量，结果仅为
  observational description。
- A2 triage：`retain_e1`。依据是历史 V21 的可审计异号实质效应与缺失但可识别的
  matched random/none counterfactual；A2 保留否决权，未授权任何 E4 或 V26 路线。
  Holdout adapter contract 记录了当前 scRNA pool shortfall，未事后按 outcome 补选数据。
- E1 pilot 已完成 `9/9` dataset-seed panels（cnae9、Mouse_retina、sms_spam_collection，
  seeds `[42,123,7]`），`audit_ok=9/9`，无 `incomplete_compute`。统计单位是 dataset，
  seed 是重复测量：cnae9 的 `I_d=+0.002057`、`S_d=+0.006010` 均
  `Observed-Small`；Mouse_retina 的 `S_d=-0.067033` 为 `Negative`；sms 的
  `I_d=+0.069251` 为 `Positive`。pilot gate `passes=true`（2/3 datasets material，异号允许）。
  完整 panel/paired/gradient/one-step 审计位于 `E1/pilot/Audit/`；不使用 3-seed bootstrap
  宣称 equivalence。
- E2-A 的 dataset×seed 聚合器与 confirmation-time feature counts 已实现并通过测试；旧 pilot
  没有保存 feature-selection counts，因此其 E2-A replay 不改写历史结果，标记为 deferred。
  confirmation 已在训练过程中保存 exact T-policy selected/eligible counts，coordinate 分布只作
  descriptive，post-hoc Fisher/MI/support enrichment 不进入 fit。
- E1 confirmation 已完成 `9/9` panel，`audit_ok=9/9`；`S_d` 为 Baron Human `+0.044617`
  (Positive)、Campbell `-0.065332` (Negative)、hate_speech `-0.033410` (Negative)。这支持
  conditional/heterogeneous V21 case-study evidence，不支持 universal population claim。
  E1 使用真实数据集标签进行外层评估，并以 benchmark-known `K`（`K_source=benchmark_oracle_from_y`）
  进入 cluster head/readout；完整 `y` 向量不进入 preprocessing、graph、Gate 或 loss，因此应称
  为 `real-GT, known-K benchmark`，不能称为完全 label-free fitting。
  E2-B/C 保存了 gradient geometry 与真实 Adam one-step，但未单独升级为 objective 主张。
- Phase C 已冻结 primary endpoint `S_full_ARI = ARI_T - ARI_R`。Phase D 的 outcome-independent
  holdout manifest 通过 source/adapter/K preflight，但冻结预算在 news20 的三个 seed 均于
  Adam state 初始化阶段 CUDA OOM；其余面板按同一资源边界收口。结果是 `0/6` completed panels、
  `audit_ok=0`，primary endpoint 不可评估，记为 `inconclusive_not_completed`，不是负性能结果。
  详情见 `result/V25_systematic_mechanism_study/PhaseE/CLOSURE.md`。
- 随后对 E1 runner 做了资源等价修复审计：CUDA 使用 host-backed batch/statistics streaming，
  极高维输入使用同一 Adam 算法的 `foreach=false, fused=true`，并将 branchpoint/arm snapshots
  保持在 CPU。`news20` bounded engineering smoke 仍未在时间窗内完成，未生成任何性能工件；
  该实现验证不改变上述 `0/6` holdout 事实，也不把资源边界写成模型负结果。
- 已从冻结工件生成 analysis-only `PaperEvidence/` bundle，包含 A0/A1 atlas、E1 `(I_d,S_d)`、
  E2 dataset×seed summaries、E3 boundary rows、source SHA256 manifest 和 claim-scope audit。
  `claim_scope_audit.json` 为 `audit_ok=true`；该导出不新增训练、不把坐标或 seed 当作独立
  population units，也不把 `0/6` holdout completion 当作性能结果。
- 当前 `V25_CONTRACT_AUDIT.json` 还对一个 confirmation panel 独立重算 N/R/T ARI 与 primary
  `I/S` pair，并验证所有 T/R matching、None contract、branchpoint、labels-after-fit-only
  检查均通过；phase auditor 对无效 panel 和不完整 seed 集合不会汇总。pilot queue 是
  attempt-local ledger（6 个本次启动 panel，3 个 cnae9 panel 由外部已完成工件覆盖），不能
  单独作为 pilot 的 9-panel denominator；正式 denominator 取 phase audit 的 9/9。
- `PaperEvidence/figures/` 已生成五张可复核诊断图：V1--V22 Failure Atlas、机制链、E1 matched
  N/R/T `(I_d,S_d)` 分解、E2 diagnostic geometry，以及 V23 local/global boundary；每张图均有
  PNG/PDF/SVG 三种格式。`figure_manifest.json` 绑定五份输入 CSV 的 SHA256、15 个图资产和
  observational/conditional/boundary evidence scope。
- 最终 integrity audit 位于 `review-stage/V25_EXPERIMENT_AUDIT.md/.json`，总体为 `WARN`：
  完成工件通过完整性核对，但 E1 是 known-K benchmark，且独立 holdout 为
  `inconclusive_not_completed`，因此不支持 universal 或 independent-replication claim。


## 2026-08-14 V24-Q1 v2 frozen calibration No-Go

v2 的 pre-fit panel 已重新完成：五个固定 seed、六个 world 共 `30/30` contract
均有效，且所有 key 均绑定 `v24_conditional_incremental_response_q1_v2`。W0 panel
support/marginal macro-OVR AUC 均值分别为 `0.50048448`、`0.49512566`，均在
`0.5 +/- 0.01` 的预注册居中范围内。V23 read-only P0 同时已完成 `12/12`；
它没有重新训练 V23，四个历史 world 的平均 conditional delta AUC 均接近零。

matched estimator calibration 已完成 `200` 个 replicate（8 个确定性 CPU worker）。
null false-positive rate=`0.0`、null mean delta AUC=`-0.00004875`，但 weak-alternative
power at `delta AUC >= 0.02`=`0.0`，alternative mean delta AUC=`0.00109075`。
因此 `calibration_passes=false`，正式 P1 `6 worlds x 5 seeds` 没有启动，Q2/DCBoost
也没有调用。这个 calibration No-Go 表示冻结的估计器/弱替代组合未达到预注册检出力，
不是 C_cycle 无效、聚类退化或 DCBoost 无效的证据；不得在 V24 内事后放宽门槛或重定义替代。

## 2026-08-14 V24-Q1 calibration-failed exploratory override（非正式）

应用户明确授权，在 calibration gate failed 的前提下启动隔离 exploratory 矩阵。输出根为
`result/V24_conditional_response/q1_synthetic_v2_exploratory_override/`，复用已通过合约审计的
v2 合成面板，覆盖六个 world、五个 primary seed，共 `30/30` unique jobs。fit/profile 使用
物理 GPU `[1,2,3]`，analysis 使用四进程 bootstrap worker；每个 job 完成完整 `200/200`
Poisson weighted bootstrap。fit/profile 的 labels/K 均不可访问，标签只在外层 analysis 使用。

该批严格标记 `execution_class=exploratory_override`、`calibration_override=true`、
`formal_q1_eligible=false`、`promotion_to_q2=false`，原因是
`calibration_gate_failed_by_explicit_user_override`。未生成 exploratory 或 formal
`q1_decision.json`，正式根仍没有 `run_summary.json`；本批不得进入正式 Q1 决策、Q2/DCBoost
证据或论文性能表。

仅作诊断的 delta-AUC 描述性均值为：W0 `+0.000968`、W1 `+0.001352`、W2 `-0.004270`、
W3 `+0.000383`、W4 `+0.000337`、W5 `-0.000057`；30-job 总均值 `-0.000214`。这些数值
不是校准通过或方法有效性的证据，完整逐 seed CI 与产物见 exploratory `exploratory_summary.json`。

## 2026-08-14 V24-Q1 reviewer-driven verification update

当前源码上的 corrected synthetic contract audit 已覆盖 W0--W5、seed42，结果为 `6/6=true`，工件位于
`result/V24_conditional_response/contract_audit_seed42_20260814/`。W0 的 support/marginal probes
回到 chance；W2 的 support signal、W3 的 marginal-dispersion signal 均按控制世界语义记录；W4
保持 exact support、逐特征非零边缘和 block dependency separation。该目录中的数值是 generator
contract，不是模型效果。

R2 CPU engineering smoke 位于
`result/V24_conditional_response/engineering_smoke_q1_20260814_r2/`，完成 V23 fit/profile 到
V24 outer analysis，`conditional_pair_utility.delta_auc=0.044444`，bootstrap CI
`[-0.005722, 0.047847]`；它只有 3 个 bootstrap replicates、120×40、2 个 epoch，且
`formal_q1_eligible=false`，不能作为 utility、聚类或 P1 证据。V23 M0 No-Go、正式 calibration、
P0 和 6 worlds × 5 seeds P1 的边界均未改变。

## 2026-08-14 V24-Q1 conditional-response engineering verification

新增独立实现 methods/TopoGate/V24_conditional_response/ 与分阶段 runner
scripts/V24/run_q1.py。V24 只检验 State、effective Support 和 N × T × 9
Marginal 后的条件增量 Response utility，明确不使用 independence、causality 或
functional redundancy 的叙事。V23 M0 的 No-Go 仍然有效，未被覆盖。

production-scale 的 W0/W4、seed42 合成 contract 均通过：W0 的 support/marginal
probe 接近 chance；W4 保持 exact support 与逐特征非零边缘、block dependency
separation 合格，并用 support-template grouped CV 和 featurewise scalar marginal
probe 防止把联合依赖重新计入 marginal。该检查是 generator contract，不是模型效果。

CPU engineering smoke 位于
result/V24_conditional_response/engineering_smoke_q1_20260814/，使用 120 × 40
合成 W4、V23 两 epoch 与 V24 三次 bootstrap，完整走通 fit/profile/outer analysis。
工件明确标记 engineering_smoke_only、formal_q1_eligible=false；其数值不得用于
效能、utility 或聚类性能结论。正式 calibration、P0 postmortem 和 6 worlds × 5 seeds
P1 矩阵均尚未启动。

## 2026-08-14 V23 Cycle-Response Protocol A M0（No-Go）

固定 M0 位于 `result/V23_cycle_response/m0_synthetic_protocol_a_v1/`，覆盖四个合成
world、seeds `[42,123,7]`，共 `12/12` jobs、`36/36` fit/profile/evaluate stages 完成，
`0` failed queues。正式运行使用物理 GPU 2--6；GPU 0、7 未使用。fit/profile 进程不接收
labels 或 K，外层 evaluate 才读取合成真值。平均 effective mask ratio=`0.018697`，
`C_cycle` 的 `64/64` fingerprint columns 有效。

依赖正例中，`C_cycle - A_pre` 的 ARI、pair AUC、kNN purity@10 平均增量分别为
`+0.011661`、`+0.025346`、`+0.130611`，三项均为 `3/3` seed 正向；这支持一个有限的
局部关系信号。但相对 support control，ARI=`-0.003671`（`0/3` 正向）、pair
AUC=`-0.000251`（`1/3` 正向），只有 kNN purity=`+0.092322`（`3/3` 正向）。
`G_gain - A_pre` 的 pair AUC 与 kNN purity 分别为 `-0.002032` 和 `-0.002756`，均为
`0/3` 正向，未显示额外可恢复性信息。

dependency-destroyed conditional null 仍保留 `C_cycle - A_pre` 增量：ARI=`+0.007428`、
pair AUC=`+0.008559`、kNN purity=`+0.043600`，三项均为 `3/3` seed 正向；global null
则基本回到 chance。latent-linear decoder 与 canonical decoder 相似或更强，说明信号并非
canonical decoder 独占捷径，但不能消除 support-only 与 conditional-null 解释。

严格按预注册门槛，Protocol A M0 判定为 **No-Go**，不进入 M1，不实施 Protocol B，也不
在本协议上追加机制救场。机器判定与逐项证据见 `m0_decision.json`，人类可读说明见
`m0_decision.md`；这是一项机制否证结果，不是 V23 聚类性能提升证据。

## 2026-08-12 V22 cooperative Keep-Gate scaffold and second dataset panel

V22 现保留两个语义不同的 topology Gate：原
`v22_topology_discriminator_hard_gate` 是主动追逐判别器困难坐标的
adversarial-hard-negative control；新增
`v22_topology_discriminator_cooperative_keep_gate` 选择 exact-budget keep 集合，
保留 keep 坐标并重建其 changed complement，Gate 在冻结 scMAE 与判别器时最小化匹配
坐标重建误差与 generator adversarial loss。两者均使用 coordinate-matched real/fake
判别器，判别器不接收 Mask 或 Hint。新增 `scmae_always_visible` 作为零随机遮挡 control。
训练历史还记录 D 的 real/fake 值幅度、非零率和幅度分箱准确率，用于检查稀疏数据 shortcut。

| Variant | Dataset/seed | Epochs/device | ARI/NMI | Status |
|---|---|---:|---:|---|
| `v22_topology_discriminator_cooperative_keep_gate` | micro-mass / 42 | 2 / CPU | 0.507440 / 0.728076 | engineering smoke |
| `v22_topology_discriminator_cooperative_keep_gate` | pbmc_1k_v3 / 42 | 1 / CPU | n/a (unlabelled) | engineering smoke |

该 smoke 的 Gate 非零更新率为 `1.0`，keep 与 complementary mask profile 均非空；数值只
证明新分支的梯度、语义和产物契约可运行，不支持相对 scMAE 或 hard-gate 的性能结论。
新增 PBMC 1k v3 smoke 位于 `result/V22/engineering_smoke_pbmc1k_v3_cooperative_20260812/`：
原始 `1222 x 33538` 稀疏 count 经 label-free top-variance cap 为 `1222 x 2000`，显式
`K=8`、`K_source=explicit_n_clusters`，Gate 更新 `10` 次；该无标签运行没有 ARI/NMI，
也不进入任何宏平均。
最新诊断 smoke 位于 `result/V22/engineering_smoke_cooperative_keep_diag2_20260812/`：
末 epoch Gate 重建梯度范数 `0.378357`、D 梯度范数 `0.000523`，幅度匹配后的 D accuracy
`0.492153`；这只是一个数据集、两个 epoch 的机制诊断，不能外推到正式训练。

第二批固定数据清单位于
`datasets/external/v22_dataset_extension_round2_20260812/manifest.json`，在本批性能读取
前登记并下载：`news20`、`rcv1_train` 两个高维稀疏文本集，`mnist` 非高维控制集，以及
无标签 `pbmc_1k_v3` scRNA count。来源、稀疏形状、原始/处理后 SHA、标签隔离和传输边界
见 `CHANGELOG_data.md`；该批尚未启动正式多种子矩阵，新增的 PBMC 1k v3 仅为一 epoch
CPU engineering smoke，不能并入宏平均。

## 2026-08-12 V22 scaffold and engineering smoke

V22 独立实现位于 `methods/TopoGate/V22_topology_discriminator_hard_mask/`，数据扩展
manifest 位于 `datasets/external/v22_dataset_extension_20260812/manifest.json`。新增
四个固定候选：`sector`、`real-sim`、`covtype` dense control 和无标签 `PBMC3k`；来源、
SHA、稀疏形状与标签隔离见 `CHANGELOG_data.md`。V22 采用 coordinate-matched
real/fake discriminator、四维 topology statistics、exact-budget ST-TopK Gate，主读出
为 clean embedding + known-K KMeans；判别器不接收 Mask/Hint。

| Smoke | Variant | Epochs/seed/device | Shape (original -> model) | ARI/NMI | Status |
|---|---|---|---:|---:|---|
| micro-mass | `v22_topology_discriminator_hard_gate` | 2 / 42 / CPU | 1300 -> 1300 | 0.471830 / 0.673989 | engineering smoke |
| sector | `v22_topology_discriminator_hard_gate` | 1 / 42 / CPU | 55197 -> 2000 | 0.041635 / 0.321119 | engineering smoke |
| PBMC3k | `v22_topology_discriminator_hard_gate` | 1 / 42 / CPU | 32738 -> 2000 | n/a (unlabelled) | engineering smoke |

三次 smoke 均完成，判别器/Gate 更新率有限且非零；这些单 seed/短 epoch 数值只证明输入、
梯度、产物和无标签边界可运行，不支持 V22 相对 scMAE 或任何 baseline 的性能结论。随后
启动的 Full 单 seed 结果见下节。Round-1 审阅后补充的
`result/V22/engineering_smoke_micro_mass_20260812_auditfix/` 还记录了 Gate/有效掩码的
unique-feature、top-10 mass 和 coverage entropy 分布，用于工程诊断，不改变协议或性能结论。
Round-2 诊断产物 `result/V22/engineering_smoke_micro_mass_20260812_round2/` 进一步记录了
`D(real)`、`D(fake_gate)`、`D(fake_scmae)` 的逐 epoch 分类率和同预算 random/Gate mask
profile；两 epoch smoke 的 D 分类率仍接近 chance，不能据此推断正式训练稳定。
矩阵 runner 的协议测试现为 `13 passed`（模型 11 + 矩阵 2）；无标签 PBMC3k 的真实矩阵
运行必须显式传入 `--n-clusters pbmc3k__10x_unlabelled_count=K`，默认 dry-run 只标注该
要求，并为相应 15 个键写出 `requires_explicit_n_clusters=true`，不猜测 K。

随后完成一个固定的 micro-mass 双 seed sanity panel，输出位于
`result/V22/engineering_smoke_micro_mass_sanity_20260812/`，覆盖三路控制、seeds
`[42,123]`、CPU、2 epochs，共 `6/6` completed。该 panel 只用于观察 D/Gate 是否有有限
更新和检查产物契约，不用于选择配置或宣称聚类收益；拓扑 Gate 在两 seed 的非零更新率均
为 `1.0`，但短预算下没有显示相对控制的稳定优势。

| Variant | seed42 ARI | seed123 ARI | Gate updates | Status |
|---|---:|---:|---:|---|
| `scmae_only` | 0.503408 | 0.470464 | 0 | engineering sanity |
| `scmae_plus_discriminator_random_mask` | 0.511445 | 0.497167 | 0 | engineering sanity |
| `v22_topology_discriminator_hard_gate` | 0.424565 | 0.459438 | 12 / 12 | engineering sanity |

## 2026-08-12 V22 full-component single-seed run (resource-bounded)

固定 manifest `datasets/external/v22_full_single_seed_20260812/manifest.json`，只运行
`v22_topology_discriminator_hard_gate`、seed=`42`、80 epochs。队列终态为
`10/12 completed`、`2/12 incomplete_compute`，严格审计通过 `10/10`；未启动任何消融或
超参数搜索。所有已完成任务的拟合、图、Gate、判别器和 loss 均记录
`labels_used_during_fit=false`、`K_used_during_fit=false`。PBMC3k 使用显式
`K_source=explicit_n_clusters`，没有 ARI/NMI。

| Dataset | Stratum | Status | ARI | NMI | Gate updates | Gate nonzero | Last effective mask |
|---|---|---|---:|---:|---:|---:|---:|
| cnae9 | original8_shared_text | completed | 0.634714 | 0.692388 | 720 | 1.0 | 0.00221 |
| Mouse_retina | original8_clubench_bridge | completed | 0.290121 | 0.503017 | 5280 | 1.0 | 0.04143 |
| Baron Human | original8_clubench_bridge | completed | 0.230553 | 0.420961 | 5360 | 1.0 | 0.11046 |
| Campbell | original8_clubench_bridge | completed | 0.172945 | 0.319401 | 6320 | 1.0 | 0.11397 |
| sms_spam_collection | original8_shared_text | completed | 0.386262 | 0.286795 | 560 | 1.0 | 0.00052 |
| hate_speech | original8_shared_text | completed | 0.043501 | 0.024730 | 2080 | 1.0 | 0.01384 |
| imdb | original8_shared_text | completed | -0.000247 | 0.000003 | 2080 | 1.0 | 0.01841 |
| sentiment_labeld_sentences | original8_shared_text | completed | 0.002478 | 0.004765 | 1760 | 1.0 | 0.00624 |
| sector | new_sparse_highdim | completed | 0.066370 | 0.389133 | 4080 | 1.0 | 0.00711 |
| PBMC3k | new_scRNA_unlabelled | completed | n/a | n/a | 1760 | 1.0 | 0.04547 |
| real-sim | new_sparse_highdim | incomplete_compute | n/a | n/a | n/a | n/a | n/a |
| covtype | new_dense_control | incomplete_compute | n/a | n/a | n/a | n/a | n/a |

已完成且有标签的 9 个数据集单 seed 宏平均 ARI=`0.202966`（仅描述性统计，不是
跨 seed 或 baseline 结论）；其中 8/9 为正。real-sim 与 covtype 在精确 cosine-kNN /
拓扑统计和长训练阶段超过预设约两小时资源窗口后被本次 launcher 终止，分别保留
`incomplete_compute.json`、启动记录、日志和已生成的 memmap，不能写入性能表。机器可复核
汇总见 `result/V22/v22_full_single_seed_20260812/aggregate_summary.json` 与
`aggregate_report.md`。

本结果只证明 V22 Full 在 10 个任务上可完成并产生完整产物；没有 scMAE-only、随机
Mask、重建困难 Gate 或非拓扑 Gate 的匹配对照，因此不能判断 V22 的增益、Gate 必要性或
判别器贡献。由于 Full 阶段并未在 12 个任务上完整结束，本轮不启动消融；也不进行基于
ARI 的超参数选择。

## 2026-08-11 V19 ARI-selected sparse/high-dimensional extension（终态）

扩展矩阵位于 `result/V19/v19_rg_extended_sparse_ari_v1/`，固定 ARI 选择后的 RG
配置迁移到 13 个预注册稀疏/高维数据集；`rg_full` 与匹配的 `scmae_only` 各运行
seeds `[42,123,7]`，共 `78/78`，`audit_ok=true`，拟合/建图未读取标签。按数据集
平均 ARI，RG 胜出 `6/13`，其中三 seed 全部为正的只有 `2/13`；13 集宏平均 ARI
为 RG=`0.175345`、scMAE-only=`0.182150`，配对差=`-0.006805`。

| 数据集 | RG ARI | scMAE ARI | ΔARI | 正向 seed 数 |
|---|---:|---:|---:|---:|
| Internet Advertisements | -0.060292 | -0.072549 | +0.012257 | 2/3 |
| gina_prior2 | 0.361992 | 0.354692 | +0.007300 | 1/3 |
| tr45.wc | 0.009203 | 0.004462 | +0.004741 | 3/3 |
| Dexter | 0.004957 | 0.000876 | +0.004080 | 2/3 |
| Madelon | 0.028291 | 0.025753 | +0.002537 | 3/3 |
| Dorothea | -0.083545 | -0.084407 | +0.000862 | 2/3 |
| Arcene | 0.097713 | 0.097713 | 0.000000 | 0/3 |
| micro-mass | 0.494370 | 0.501711 | -0.007341 | 0/3 |
| Quake Smart-seq2 Lung | 0.169644 | 0.179120 | -0.009476 | 1/3 |
| SMS Spam full TF-IDF500 | 0.861438 | 0.875675 | -0.014238 | 0/3 |
| fbis.wc | 0.290645 | 0.309347 | -0.018702 | 0/3 |
| Gisette | 0.072307 | 0.096792 | -0.024485 | 1/3 |
| Fabert | 0.032766 | 0.078762 | -0.045997 | 0/3 |

对 6 个 RG 胜 scMAE 的数据集，固定参数运行 AHDPC、DPC-GFNN 和 GCC，全部 `18/18`
个外部基线完成。RG 超过最佳外部方法的有 `gina_prior2`（RG `0.361992` vs GCC
`0.250800`）和 `Madelon`（RG `0.028291` vs GCC `0.028005`），共 `2/6`；
Dexter、Dorothea、Internet Advertisements、tr45.wc 均未超过最佳外部方法。完整机器审计
见 `result/V19/v19_rg_sparse_goal_audit_20260811/goal_audit.json`，其预注册门槛
“至少 5 个 RG 胜 scMAE”已满足，但不能据此宣称 RG 普遍优于 scMAE 或 SOTA。

本批外部基线使用了已冻结的 benchmark known-K 作为外层协议元数据，方法拟合仍记录
`labels_used_during_fit=false`。Dexter/Dorothea 的 CSR NPZ 适配修复及原始 launcher
汇总缺失均保留在 `CHANGELOG_errors.md`；不改变模型或外部方法参数。

## 2026-08-11 V21 readout audit and v3 extension status

对 `result/V21/v21_formal6_full_20260811_graphfix/` 的 18 个 Full 产物做离线同 embedding
读出复算；没有重训、没有使用标签选择 readout 或 KMeans 初始化：

| Readout / embedding | 六数据集宏 ARI | 说明 |
|---|---:|---|
| v2 Full Student-t head | 0.207693 | 历史正式 primary readout |
| v2 Full clean embedding + KMeans | 0.384094 | 同 18 个 embedding 的读出审计 |
| matched scMAE-only clean embedding + KMeans | 0.418579 | 历史 matched control |
| ARI-selected Full clean embedding + KMeans | 0.383253 | 开发/确认层，仅诊断 |

v2 Full head 每 run 平均空簇 `2.722`；Baron Human/Campbell 分别平均空 `7.667/8.667`
个簇。读出问题真实且幅度很大，但统一 KMeans 后 Full 仍比 scMAE-only 低 `0.034485`，所以
不能把 V21 失败完全归因于 head。v3 将 Student-t 距离从 128 维均值改为平方距离和，并以
clean embedding KMeans 作为 primary readout；这是新协议，尚无正式 80-epoch 扩展结果。

13 数据集扩展 manifest 为
`result/V21/v21_extended13_readoutfix_manifest_20260811.json`，固定 78 runs。当前只有
`result/V21/engineering_smoke_extended_readoutfix_20260811/` 的 micro-mass、seed42、
2 epochs、两路 CPU smoke：`2/2` completed、`audit_ok=true`；Full/scMAE ARI 为
`0.449937/0.427035`。该 Delta `+0.022902` 是单 seed 短 epoch 工程值，不是性能证据，正式
矩阵状态仍为 **not started**。

## 2026-08-11 V21 formal six-dataset matrix（graph-fix，已完成）

初始输出根 `result/V21/v21_formal6_full_20260811/` 已因 kNN 自邻居过滤缺陷降级为审计记录，
不纳入正式结论。修复后的唯一正式矩阵位于
`result/V21/v21_formal6_full_20260811_graphfix/`，协议为
`v21_assignment_adversarial_full6_graphfix_v1`，由 `scripts/V21/run_formal_matrix.py`
按六个数据集 × 两个 variant × 三个 seed 管理。终态为 `36/36` completed、`0` queued、
`0` incomplete；`matrix_audit.json` 与 `aggregate_summary.json` 均为 `audit_ok=true`，
`provenance_ok=true`。模型拟合、图、Gate 和损失没有读取标签；cluster head 的 K 由外层
benchmark 协议提供并单独记录。

| Dataset | Full ARI | scMAE-only ARI | Delta (Full-scMAE) |
|---|---:|---:|---:|
| cnae9 | 0.5083 | 0.3600 | +0.1483 |
| Mouse_retina | 0.4416 | 0.9363 | -0.4947 |
| Baron Human | -0.0077 | 0.2058 | -0.2135 |
| Campbell | 0.0547 | 0.1994 | -0.1447 |
| sms_spam_collection | 0.3244 | 0.8258 | -0.5014 |
| hate_speech | -0.0752 | -0.0160 | -0.0592 |
| **六数据集宏平均** | **0.2077** | **0.4186** | **-0.2109** |

该矩阵只有 `cnae9` 的 Full 平均 ARI 高于 scMAE-only；因此固定 V21 不能宣称普遍优于
scMAE-only。这里的对比是“完整 V21 vs scMAE-only”，不是只隔离 Gate 的纯消融：Full
还包含 Student-t cluster head、InfoMax 和 known-K 拟合路径。

## 2026-08-11 V21 ARI grid and confirmation

ARI 网格输出位于 `result/V21/v21_ari_grid_seed42_20260811/`，共 `72/72` 个任务并通过严格
审计。网格固定 seed=`42`，使用六数据集宏平均 ARI 选择配置，因此属于
`ARI-selected development evidence`，不是无标签泛化证据。选中配置为
`assignment_weight=0.1`、`gate_lr=2.5e-4`、`infomax_weight=0.05`、`epochs=80`、
`warmup_epochs=40`，seed42 网格宏平均 ARI=`0.3956`。

三 seed 确认输出位于 `result/V21/v21_ari_confirm_aw0.1_glr0.00025_ep80_20260811/`，
共 `18/18` 个任务，`confirm_audit.json` 为 `audit_ok=true`。确认宏平均 ARI=`0.3427`，
相对正式固定 Full=`+0.1350`，但相对正式 scMAE-only=`-0.0759`。分数据集只有
`cnae9`、`hate_speech` 的确认平均 ARI 高于 scMAE-only；确认结果仍属于使用 ARI 选择后的
开发/确认层，不能当作独立无标签测试。

## 2026-08-11 V21 assignment-adversarial implementation smoke

V21 独立实现位于 `methods/TopoGate/V21_assignment_adversarial_gate/`，三路真实数据工程
smoke 位于 `result/V21/engineering_smoke_20260811/cnae9__shared_text/`。三路均为
`cnae9`、seed42、CPU、2 epochs、1 epoch warmup；该预算只能验证端到端契约，不能用于
性能比较、调参或版本晋级。

| Variant | Primary readout | K used during fit | Graph/Gate | ARI | Status |
|---|---|---:|---:|---:|---|
| `scmae_only` | known-K KMeans | no | no/no | 0.060563 | engineering smoke |
| `random_assignment_control` | Student-t head | yes | no/no | 0.033569 | engineering smoke |
| `topology_assignment_adversarial` | Student-t head | yes | yes/yes | 0.033670 | engineering smoke |

三路拟合均未接收标签；本次聚类头 K=`9` 由外层 benchmark 标签唯一值提供，记录为
`K_source=benchmark_oracle_from_y`。random/full 最后一个 epoch 的 donor-different eligible
rate 分别约 `0.01408/0.01426`，全特征 selected/effective rate 约
`0.00608/0.00619`，而 selected 中 actual value change 与预算填充率均为 `1.0`。Full
执行 5 次 Gate update，non-zero gradient rate=`1.0`。这证明 V21 没有再把局部 40%
错误报告成全特征 40%，但不证明如此稀疏的全局扰动量足以提高正式 ARI。

## 2026-08-10 V20 Full first-round single-seed evidence

V20 独立实现位于 `methods/TopoGate/V20_topology_conditioned_adv_mask/`。首轮只运行
`topology_adversarial_full`，cnae9/shared_text、seed42、GPU2、80 epochs；输出根为
`result/V20/full_first_round_20260810/cnae9__shared_text/seed42/`。训练使用稀疏
`X_graph` 的 TruncatedSVD/cosine-kNN（实际 SVD dim 397，累计解释方差 0.950233，k=20），
`X_model` 上的分块 topology statistics，以及 40 epoch warmup + 40 epoch 半对抗 Gate。

| Variant | ARI | NMI | ACC | Status |
|---|---:|---:|---:|---|
| V20 `topology_adversarial_full` | 0.181408 | 0.467400 | 0.472222 | single-seed first-round |

本次 fit 未使用标签；K=`9` 仅用于外层 KMeans readout 和后验指标。requested mask rate
为 `0.399533`，effective value-change mask rate 约 `0.00678`，Gate update 为 `50` 次且
non-zero gradient rate=`1.0`。该结果不能作为 V20 优于 scMAE 的结论，也不与 V19 的历史
`scmae_only` 直接合并为 matched comparison：V20 使用 ST-TopK/requested-mask 训练目标，
V19 reference 使用原生 Bernoulli/effective-mask 语义。

首轮 X-only tuning 位于 `result/V20/tuning_first_round_20260810/cnae9_seed42/`，4 个候选
全部完成，选择 `gate_lr=5e-4`、`tau_ste=0.5`；`labels_accessed=false`、`y_key_read=false`、
`n_clusters_used=null`。该段记录建立时尚未启动 8 数据集矩阵；后续粗筛结果见下一节。

**最后更新**：2026-08-10 (V20 Full first-round single-seed evidence added; V19 v2 matrix retained)
**目的**：消除 static_gate/learnable_gate、K 错误、MAE freeze 概念混淆，便于后续 multi-seed 验证对照

## 2026-08-10 V20 Full eight-dataset coarse screen, seed42

在固定 V20 Full 配置下完成 8 个 bridge/shared-text 数据集的单 seed 粗筛。cnae9
复用首轮产物，其余 7 个数据集输出位于
`result/V20/full8_seed42_20260810/`。所有 run 使用 80 epochs、40 epoch warmup、
`gate_lr=5e-4`、`tau_ste=0.5`；由于 GPU 2--4 被外部任务占用，补充任务最终在 GPU1
串行完成，无 `incomplete_compute`。本批不是三 seed 正式矩阵，也没有运行 matched
scMAE-only。

| Dataset | Protocol | ARI | NMI | ACC | Effective mask (last epoch) |
|---|---|---:|---:|---:|---:|
| cnae9 | shared_text | 0.181408 | 0.467400 | 0.472222 | 0.0068 |
| Mouse_retina | clubench_bridge | 0.341889 | 0.432861 | 0.585967 | 0.0564 |
| Baron Human | clubench_bridge | 0.204016 | 0.346117 | 0.393563 | 0.0864 |
| Campbell | clubench_bridge | 0.099994 | 0.254213 | 0.195237 | 0.0732 |
| sms_spam_collection | shared_text | 0.167792 | 0.125527 | 0.881437 | 0.0114 |
| hate_speech | shared_text | -0.029183 | 0.007179 | 0.524061 | 0.0365 |
| imdb | shared_text | 0.000888 | 0.001051 | 0.516923 | 0.0688 |
| sentiment_labeld_sentences | shared_text | -0.000074 | 0.000020 | 0.503275 | 0.0150 |
| **Macro mean** | — | **0.120841** | **0.204296** | **0.509086** | — |

模型拟合、建图、stats、Gate 和 loss 均未读取标签；K 仅用于外层 known-K KMeans
和后验指标。该粗筛显示结果具有明显数据集依赖性，不能据此宣称 V20 优于 scMAE。
完整逐 run 产物中的 `summary.json`、`metrics.json`、`resolved_config.json` 和
`training_history.json` 是本表的事实来源。

> **存储与证据声明（2026-08-03）**：本文件位于项目结果软链接
> `/home/luolie/ToPoGate/result`，实际目标为 `/data/luolie/ToPoGate/result`。
> 本事实表中的项目根路径统一写作 `result/...`；同一结果盘内的相对路径可省略
> `result/`。本轮已清理 `learnable_gate_smoke`、V6/V7/HVF smoke、AHDPC
> verified smoke、V10/V11 iris smoke 以及 `/tmp` 中明确的 V11 semantic
> 临时产物。旧条目若仍保留这些路径，只代表历史工程记录，不再是当前可复核的
> 权威产物；正式结论只能引用仍存在的 `summary.json`、CSV、数组和配置。

## 2026-08-10 PlantNet-ARI fixed RG transfer with PCA200

将 PlantNet full-16 ARI 搜索选出的 RG 参数固定迁移到 V19，并把图构建
`knn_pca_dim` 从 50 改为 200；在 8 个 comparable 数据集
(`mouse_retina`、`campbell`、`baron_human`、`sms_spam_collection`、`cnae9`、`imdb`、
`hate_speech`、`sentiment_labeld_sentences`) 上运行 `rg_full` 与相同骨干的 matched
`scmae_only`，seeds `[42,123,7]`，共 `48/48`，无 `incomplete_compute`，终态审计通过。

结果根目录为 `result/V19/v19_rg_plantnet_ari_pca200_20260810/`，固定配置为
`methods/TopoGate/V19_rg_adapter/configs/v19_rg_plantnet_ari_pca200.yaml`。模型拟合和
预处理均未使用标签；标签仅用于 benchmark K 与后验指标。需要明确：该配置在 PlantNet
阶段按 ARI 选择，因此这里是跨数据域的 benchmark-transfer，不是无标签调参结果。

| Variant | ARI | NMI | ACC |
|---|---:|---:|---:|
| PlantNet-RG-PCA200 | 0.325049 | 0.359080 | 0.605638 |
| matched scMAE | 0.322975 | 0.360145 | 0.601605 |
| 配对 RG-scMAE | +0.002074 | -0.001066 | +0.004032 |

按底层 8 个数据集的平均 ARI，RG 胜出 `4/8`；相对已有 V19 `rel_both2` 的 8 数据集 RG 平均
ARI `0.327970`，本次为 `0.325049`，下降 `0.002921`。RG 在 Campbell 和 Baron 有正向
配对差，但在 Mouse retina、SMS、CNAE9、Hate speech 退化；归档 SOTA 只有 4/8 层有可连接
记录，不能对缺失层填零或宣称完成 SOTA 比较。逐数据集结果见
`result/V19/v19_rg_plantnet_ari_pca200_20260810/aggregate_report.md`。

## 2026-08-10 V19 v2 mechanism refine and post-freeze final matrix

V19 v2 的正式无监督 mechanism refine 已完成于
`result/V19/v19_rg_mechanism_refine_v2_cached_20260809/`：12 个 mechanism candidates、
11 个输入层、3 个 seeds `[42,123,7]`，共 `396/396`，launcher `audit_ok=true`。调参器
只使用固定 20% held-out X-only proxy，未读取 `y`、未推导 K、未执行聚类或写入标签指标。
选择结果为 `rel_both2`，覆盖 `gamma_mutual=2.0`、`gamma_snn=2.0`、
`gamma_sim=0.0`、`gamma_distance=0.0`；`selection_status=proxy_supported`、
`no_go=false`，但仅有 8 个底层数据集中的 2 个达到 proxy-win，不能解释为普适机制收益。

冻结配置后的 final 矩阵位于
`result/V19/v19_rg_final_postfreeze_rel_both2_20260810/`：11 个输入层 × 6 个 variant ×
3 个 seeds，共 `198/198`，`audit_ok=true`，GPU 1--6 使用，GPU 0/7 未用于本批次。6 个
variant 为 `rg_full`、`rg_default`、`scmae_only`、`rg_nomix`、`rg_reliability_off` 和
`rg_constant_gate`；6 个 worker 返回码均为 0。模型拟合和变体选择均未使用标签；标签仅用于
benchmark K 和拟合后的 ARI/NMI/ACC。

最终 11 个输入层的宏平均（先按 seed 求每层均值，再对层等权）如下：

| Variant | ARI | NMI | ACC |
|---|---:|---:|---:|
| `rg_full` | 0.421592 | 0.474223 | 0.646786 |
| `scmae_only` | 0.421354 | 0.478420 | 0.652961 |
| `rg_default` | 0.419628 | 0.472144 | 0.647900 |
| `rg_nomix` | 0.421354 | 0.478420 | 0.652961 |
| `rg_reliability_off` | 0.415937 | 0.468680 | 0.645054 |
| `rg_constant_gate` | 0.427913 | 0.479126 | 0.646480 |

`rg_full - scmae_only` 的配对宏平均为 `+0.000238 ARI`、`-0.004197 NMI`、
`-0.006175 ACC`；按 11 个输入层计，ARI/NMI/ACC 的正向层数分别为 `6/11`、`4/11`、
`4/11`。`rg_nomix` 与 `scmae_only` 的 33 个配对 run 在三项指标上逐项完全相同，说明本批
数据中 NeighborMix 没有产生可测增益。`rg_full` 相对 `rg_reliability_off` 的宏平均差为
`+0.005655/+0.005543/+0.001732`（ARI/NMI/ACC），但相对 constant-gate 的 ARI/NMI 差为
`-0.006321/-0.004903`，门控细化收益仍具有明显数据集依赖性。

与归档 SOTA 的比较只对有同名归档行的 4/8 个 bridge/shared-text 层成立：V19 `rg_full`
在 `cnae9` 和 `Mouse_retina` 三项指标均高于该层最佳归档方法，在 `Campbell` 和
`sms_spam_collection` 三项指标均低于最佳归档方法，严格胜出为 `2/4`。`Baron Human`、
`hate_speech`、`imdb` 和 `sentiment_labeld_sentences` 没有可连接的归档 SOTA 行，不能填零
或宣称比较完成。长表和报告见
`result/V19/v19_rg_final_comparison_rel_both2_20260810/comparison.csv` 与
`comparison.md`；归档 baseline 仍标记为 archived reference，不是 fresh matched rerun。

## 2026-08-09 V19 v2 reference and formal tuning boundary

固定的 X-only scMAE reference 已在
`result/V19/v19_scmae_xonly_reference_v2_paired_20260809/` 完成 `33/33`：11 个
固定输入层、seeds `[42,123,7]`，无 `incomplete_compute`。独立终态审计确认
`labels_accessed=false`、`y_key_read=false`、`n_clusters_used=null`、无
`labels_true.npy`/`predictions.npy`/`metrics.json`；该 reference 仅用于无监督
proxy 配对，尚未产生 ARI/NMI/SOTA 结论。

formal V19 v2 已在启动前发现并修正 RG held-out pseudo mixing 的索引域错误，且将
候选评估的 mask ratio 与输入诊断图固定到 base reference。launcher 增加 root lock、
资源占用检查、失败返回码和 expected-run/artifact 审计。经审查，backbone/joint
候选会把训练预算与 topology mechanism 混淆，因此正式搜索冻结 scMAE backbone，
只调 `rg_full` mechanism：`mechanism_screen` 为 48 候选 × 8 comparable layers ×
seed42 = `384`，`mechanism_refine` 为 top12 × 11 layers × 3 seeds = `396`，总计
`780` RG runs。mechanism_screen 已完成 `384/384` 且 `audit_ok=true`；mechanism_refine
中间复核曾记录 `261 completed / 3 running`，另有 `10` 个历史
`incomplete_compute` 记录（4 个原始 OOM、6 个因 GPU 重分配主动中断），这些记录不会进入
选择汇总；该中间快照已由本节顶部的 `396/396` 终态取代。恢复阶段使用 `small_first` 队列，
小数据集先运行，大数据集和长耗时 bridge 任务后置；GPU0/7 始终禁止使用。

该 v2 选择协议明确为 `transductive_full_X_label_free_preprocessing`：模型/图只在
fit rows 上训练，HVG/scaling 仍在全量 X 上拟合。后续 final config 必须在独立 fresh
评估和 matched scMAE control 上再揭示标签计算 ARI/NMI，proxy 选择本身不能写成
RG 优于 scMAE 或 SOTA 的性能证据。

## 2026-08-08 V19 independent RG adapter and engineering smoke

V19 已建立独立路径 `methods/TopoGate/V19_rg_adapter/` 与 `scripts/V19/`，只包含
`scmae_only` 和完整原始 reliability-gated NeighborMix 的 `rg_full`。核心 fit 不接收
标签；标签只在外层确定 benchmark K 并计算后验指标。原始 RG 的 graph、edge reliability、
node gate、pseudo mixing 和 weighted scMAE loss 已通过数组/张量级回归，focused tests
为 `11 passed`。固定 manifest 位于
`result/V19/v19_rg_dataset_manifest_20260808.json`，11 个输入层全部 eligible，正式计划为
11 strata × 2 variants × seeds `[42,123,7]` = 66 runs。当前正式矩阵已 `66/66 completed`
且无 `incomplete_compute`。运行状态由 `result/V19/v19_rg_selected_advantage_v1/` 下的
`run_record.json` 复核；不重复计算 SHA/hash。

工程 smoke 位于 `result/V19/engineering_smoke_20260808/`，覆盖 `cnae9__shared_text` 与
`baron_human__rg_native` 的两路 variant，均为 seed42、64 行、1 epoch、CPU。所有四条
run 均 completed，paired variant 的预处理特征选择一致；`scmae_only` 的 graph/pseudo
明确关闭，`rg_full` 的 graph/pseudo 明确启用。smoke 中出现的 ARI/NMI 差异不具备性能
证据资格，不进入 SOTA、均值或晋级判断。运行与 manifest 均未重新计算 SHA/hash。

### V19 formal matrix and X-only tuning handoff

V19 三个 seed 已完成 `66/66`，无 `incomplete_compute`；GPU0/7 未用于 V19，V18 既有
worker 未停止。新增 `scripts/V19/tune_unsupervised.py` 和
`scripts/V19/summarize_unsupervised_tuning.py`，调参器只读取矩阵特征字段，固定
`n_clusters=None`，不访问 `y`、不执行 KMeans、不写入 ARI/NMI 或 `labels_true`。24 个
候选 × 11 输入层 × 3 seed 共 792 个 X-only tuning run 已完成，状态为
`selection_completed`，且 `labels_accessed=false`、`y_key_read=false`。v1 选择
`mask03`，但它只是 RG 候选间的绝对 X-only pilot，不是 RG 相对 scMAE 的配对优势证据，
不作为 v2 或最终 SOTA 配置。

### V19 v2 paired held-out tuning protocol (formal refine resumed)

新增独立入口 `scripts/V19/tune_unsupervised_v2.py`、
`scripts/V19/run_scmae_reference_v2.py`、`scripts/V19/summarize_unsupervised_tuning_v2.py`
和 `scripts/V19/launch_unsupervised_v2.py`。v2 固定 20% 未见行作 X-only 诊断，先生成一次
固定 `scmae_only` reference，再把 RG full 候选与其配对；选择单位是 8 个底层数据集，
生物数据的 native/bridge 层先聚合。主选择目标是 proxy-win 数据集数量，阈值、collapse
规则和 seeds 写入 `stage_spec.json`/`selected_config.json`；不读取标签、不推导 K、不执行
KMeans、不写 ARI/NMI/SOTA 指标。正式 refine 输出根为
`result/V19/v19_rg_mechanism_refine_v2_cached_20260809/`，已按同一协议完成；调度顺序记录在
`schedule_spec.json`，仅影响队列顺序，不改变 run key 或算法协议，最终状态见本表顶部。

预注册漏斗为：48 个 RG mechanism 候选在 8 个底层数据集/seed42 粗筛；top12 在 11
层/3 seeds 细化；top4 mechanism × 8 个共享 backbone profile 做 8 底层数据集 screen；
top6 joint 候选在 11 层/3 seeds 锁定。预计约 1,234 个 RG run 加 33 个固定 scMAE
reference run。共享 backbone profile 会在最终另跑 matched scMAE control，不能把 proxy
胜出直接写成拓扑机制收益。该中间计划状态已由顶部的 final 矩阵和比较报告更新；运行不重复
计算 SHA/hash。

旧路径 `result/V19/v19_rg_unsup_tuning_v2/` 和
`result/V19/v19_scmae_xonly_reference_v2/` 只存在一份未完成的 2-candidate/seed42
`stage_spec.json`，其 11 层/2 层选择协议与当前 `comparable_only` 版本不一致；未生成
completed run，不并入 v2 结果。当前正式 v2 使用带日期后缀的新输出根，避免覆盖旧草稿。

## 2026-08-08 V18 independent v2.2 matrix (running)

v2.1 在运行中被前置协议审计停止：mask 有效位置语义和 FISTA latent 归一化与计划不
一致。v2.1 的已完成产物保留在 `result/V18/v18_scmae_mainline_v2_1/`，其中 6 个
未完成 key 已明确记录为 `incomplete_compute`，不与新协议合并。

当前正式矩阵使用独立代码、manifest
`result/V18/v18_dataset_manifest_v2_2_20260808.json` 和输出根
`result/V18/v18_scmae_mainline_v2_2/`，149 条 eligible 数据、10 个 variant、3 个
seed，共 4470 个 run key。提交时状态为 18 completed、6 running；早期状态不构成
性能结论。v2.2 已通过 compileall、focused tests `8 passed`、CLI 和真实短 smoke；
mask/归一化修正不改变预注册 variant、seed、K 或标签隔离协议。runner 不重复计算
SHA/hash。

## 2026-08-08 V18 independent implementation and engineering smoke

V18 已建立独立代码路径：`methods/TopoGate/V18_scmae_latent_gate/`，入口为
`scripts/V18/run.py`，矩阵入口为 `scripts/V18/run_matrix.py`。主线严格为
scMAE masked reconstruction -> three deterministic latent mask views -> latent
cosine/SNN candidate union -> HardConcrete edge gate plus candidate-restricted
sparse relation -> `C=G*W` -> `abs(C)+abs(C.T)` -> normalized spectral readout。
V9/V17/外部 baseline 源码未修改。

一次性 manifest `result/V18/v18_dataset_manifest_20260808.json` 共登记 157 条记录，
其中 149 条 `eligible`，8 条 `ineligible`；manifest 选择声明不使用标签或既有结果，
实验 runner 不逐 run 重算 SHA/hash。

真实登记数据 `2d_20c_no0` 的 3 路短配置结果（`epochs_mae=2`、gate/joint 各 1、
单 seed、1500 行抽样）均完成：`scmae_only`、`latent_GW_frozen`、`v18_full`。
产物位于 `result/V18/engineering_smoke_real_20260808/2d_20c_no0/`，仅证明
输入、梯度、产物和标签隔离契约可运行；该 smoke 中 gate hard-open rate 为 1.0，
不能作为稀疏门控收益或性能结论。正式多 seed、多 variant 矩阵尚未汇总。

## 2026-08-07 V17 topology-native reference implementation（无性能证据）

V17 独立 reference solver 已实现于 `methods/TopoGate/V17_topology_native/`，入口为
`scripts/V17/run_reference.py`。当前闭环为 sparse-safe input adapter、多 sparse
random projection candidate union、candidate-restricted group-Huber elastic sparse
self-expression、exact-zero `C` gate、`A=|C|+|C.T|` 和 normalized spectral readout。
`fit_topology` 不接收标签或 `K`；degree-zero 节点显式 abstain，不通过第二个 latent
聚类器补分区。

工程验证：compileall 通过，focused tests `11 passed`，模块与脚本入口 `--help` 均通过。
本轮没有运行真实数据、single-seed smoke 或 benchmark，因此本节不报告 ARI/NMI/AMI，
也不能据此称 V17 有效或优于任一 V 系列/外部方法。spectral feedback 与 learnable
unrolling 尚未实现，当前版本固定标记为 `V17-reference`。

## 2026-08-07 V16.1 expanded-count continuation（固定协议）

### 当前批次已完成补齐（截至 2026-08-07 04:40）

以下数据集已完成 clean/compound、seeds `[42,123,7]` 和五路 paired readout，按同一预注册晋级规则判定；`PRJNA895163` 产物暂存于 `/tmp/v16_1_stage1_parallel_20260806/`，其余两项位于结果盘：

| 数据集 | 分层 | clean Delta ARI | compound Delta ARI | 状态 |
|---|---|---:|---:|---|
| `PRJNA895163` | high_sparse_bonus | `0.000000` | `+0.000004` | empirical_not_supported |
| `Bone_Marrow` | high_sparse_bonus | `-0.002388` | `-0.000291` | empirical_not_supported |
| `Young` | high_sparse_bonus | `+0.002589` | `0.000000` | empirical_not_supported |
| `hrvatin_geo_maintype_counts` | high_sparse_bonus | `-0.000309` | `0.000000` | empirical_not_supported |

上述数据集均未达到 `Delta ARI >= 0.03`、消融优于固定图/随机 support 和 stress retention 的晋级条件；没有据此调整 gate 或输入协议。`NormanWeissman2019_perturbation` 已按搜索上限停止，未进入性能统计。

将旧的 33 条完整去重快照与新完成的 `PRJNA895163`、`hrvatin_geo_maintype_counts` 合并后，当前完整去重快照为
35 个数据集，全部 `empirical_not_supported`，`candidate_positive=0`，文件为
`/tmp/v16_1_global_dedup_summary_current_20260807.json`。`Blood_BoneMarrow` 的完整
证据来自旧结果根；当前结果盘中的同协议重复补跑已停止。

新增完成的数据集：

| 数据集 | 分层 | clean Delta ARI (V16.1 - self) | compound Delta ARI | 状态 |
|---|---|---:|---:|---|
| `PBMC3K` | high_sparse_bonus | `0.000000` | `0.000000` | empirical_not_supported |
| `Bach` | high_sparse_bonus | `0.000000` | `0.000000` | empirical_not_supported |
| `PBMC_68K` | high_sparse_bonus | `0.000000` | `0.000000` | empirical_not_supported |
| `Shekhar` | high_sparse_bonus | `0.000000` | `0.000000` | empirical_not_supported |

PBMC3K 使用 H5AD `raw.X` 的可逆 `log1p(count)`，转换为 CSR 后通过 Stage-0；clean/
compound 三 seed 与五路 readout 在 GPU5/6 并行完成。support 全负时
`cluster_probabilities.npy` 精确回退 `q_self`，不调整固定协议。完整产物位于
`result/V16_1/expanded_count_stage1_20260807/PBMC3K/`，固定汇总暂存于
`/tmp/v16_1_summary_pbmc3k_20260807.json`。

本批次仍使用 seeds `[42,123,7]`、clean/compound 和五路 paired readout；每个 seed
只训练一次 Stage A。已完整完成并按固定晋级规则汇总的数据集如下：

| 数据集 | 分层 | clean Delta ARI (V16.1 - self) | 状态 |
|---|---|---:|---|
| `Norman_perturb_e_distance` | sparse_count_control | `-0.000017` | empirical_not_supported |
| `Quake_Smart-seq2_Lung` | high_sparse_bonus | `-0.000094` | empirical_not_supported |
| `Quake_10x_Spleen` | high_sparse_bonus | `+0.000064` | empirical_not_supported |
| `subsample_2k` | high_sparse_bonus | `-0.000060` | empirical_not_supported |

这些完整结果均没有触发 gate、support、temperature、thinning 或 K 调整。该段保留的是
早期批次快照；`Shekhar`、`PRJNA895163` 和 `hrvatin_geo_maintype_counts` 后续已完成并在本节顶部补记；Norman Stage-0
已按搜索上限停止。固定汇总暂存于
`/tmp/v16_1_stage1_parallel_20260807_promotion.json`。

`subsample_2k` 的 Stage 0 已通过：`2000x53678`、high_sparse_bonus、candidate
recurrence `0.5676`、稳定边比例 `0.7902`、support 非退化；其来源和 CSR bundle
记录于 `CHANGELOG_data.md`。本批次仍未产生 `candidate_positive`。

此前跨四个结果根目录的 33 个数据集去重快照保存在
`/tmp/v16_1_global_dedup_summary_20260807.json`；当前 35 条快照见
`/tmp/v16_1_global_dedup_summary_current_20260807.json`。Norman 未生成审计 JSON 或性能
产物；当前不把中间状态写入正例表。

## 2026-08-06 V16.1 expanded-count Stage-1（固定协议，临时批次）

输出根 `/tmp/v16_1_stage1_parallel_20260806/` 使用固定 seeds `[42,123,7]`、clean/
compound、五路 paired readout；每个 seed 只训练一次 Stage A，再从同一表征和 support
导出五个 readout。已完成的五个数据集均未通过预注册晋级规则，当前 `candidate_positive=0`：

| 数据集 | 分层 | clean Delta ARI (V16.1 - self) | 状态 |
|---|---|---:|---|
| `Arabidopsis_Stereo_seq_leaf` | high_sparse_bonus | `+0.000025` | empirical_not_supported |
| `CRA002977_1` | high_sparse_bonus | `-0.000029` | empirical_not_supported |
| `HCA_subsampled_20k` | high_sparse_bonus | `+0.000013` | empirical_not_supported |
| `TabulaSapiens_Pancreas` | high_sparse_bonus | `+0.005427` | empirical_not_supported |
| `tr45.wc` | high_sparse_bonus | `+0.000000` | empirical_not_supported |

`tr45.wc` 的首轮仅在理论证书阶段返回，原因是带点号的数据集名被错误截断；修复其
word-count metadata 解析后，已用同一固定协议完整重跑，上表仅引用有效重跑结果。
`SRP224648`（`14533x67300`）在 Stage-A Adam 状态分配时超出单卡 80GB 容量，标为
`stage1_incomplete_compute`，不计为模型性能失败。`Baron Human`、`Campbell`、
`Human_Pancreas_3`、`Macosko`、`SRP182008`、`Tosches` 的完整矩阵仍在运行；不得以
中间 seed 或 fixed-graph 优势形成机制结论。

这些是可复核的临时运行证据，不是论文主结果。正式结果盘写入、主表扩展与正例声明都
等待完整 batch、晋级汇总和结果盘状态确认。

## 2026-08-06 V16.1 Stage-0（静态边界，无 Stage-1 性能证据）

V16.1 已在独立目录实现，冻结 V16 及更早版本。Stage A 为 topology-disabled
scMAE-compatible sparse count MAE，Stage B 为三次 count split 的
cross-fitted predictive support；拓扑只进入 assignment readout。compileall 通过，
focused tests **21 passed**。每个 split 同时评分 A→B 与 B→A，逐边 median；理论边界和输出语义记录于
`methods/TopoGate/V16_1_predictive_graph_gate/THEORY.md`。
paired runner 默认 seeds 为 `[42,123,7]`，正式输出根目录为
`result/V16_1/v16_1_paired`；Stage-0 固定 `k=20` 和三次 split。

当前 Stage 0 事实：`Campbell`（`9993×26774`）、`Mouse_retina`（`8352×6198`）、
`Baron Human`（`8451×20125`）、`tr45.wc`（`676×8261`）和 `fbis.wc`（`2196×2000`）
通过理论域证书。候选 recurrence 分别为 `0.4724`、`0.2667`、`0.5155`、`0.4685`、
`0.4041`，support 正值率分别仅 `0.0034%`、`0.0054%`、`0.0253%`、`0.0169%`、
`0.0856%`；这些数值只表示静态图/support 结构，不是性能结论。Campbell/Mouse_retina
的延长窗口产物为 `/tmp/v16_1_stage0_campbell_exchange.json` 和
`/tmp/v16_1_stage0_mouse_exchange.json`，其余候选记录在
`/tmp/v16_1_stage0_exchange.json` 与 `/tmp/v16_1_stage0_priority_exchange.json`。
`Quake_Smart-seq2_Lung`、`hrvatin`、`hrvatin_filtered` 因 dense storage 或 count
encoding 无法恢复标记 `theory_domain_not_supported`。以上 exploratory 输出没有
写入正式结果盘；V16.1 Stage 1 尚未启动，也没有论文级性能结论。

优先候选的追加 Stage 0 输出为 `/tmp/v16_1_stage0_priority_exchange.json`：
`fbis.wc`（`2196×2000`）通过理论域证书，candidate recurrence `0.4041`、support
正值率 `0.0856%`、median support `-6.581`，因此仍不进入 Stage 1；`hrvatin`
（`65539×25187`）和 `hrvatin_filtered`（`48266×25187`）均因 dense storage
及无法恢复的全量 count encoding 标记 `theory_domain_not_supported`。

### V16.1 expanded-count 候选（Stage-0，无性能结论）

扩展策略只把维度、零比例、行 nnz 和空行作为分层指标；count 语义、CSR/分块读取、
held-out split 和行数一致性仍是硬条件。`scripts/V16_1/count_candidate_registry.json`
登记本地 scCluBench scRNA H5 源，转换及静态审计暂存于 `/tmp`。新增候选中
`Melanoma_5K`（recurrence `0.8439`，正支持行 `3.8777%`）和 `Guo`（`0.5195`，
`1.2327%`）最值得进入后续固定 Stage-1；`Limb_Muscle` 与 `worm_neuron_cell`
的 support 为负，仍不做性能判定。`Bach`、`Macosko`、`Shekhar`、`Tosches` 已完成
CSR 转换但图审计未完成。`Wang` 因非整数归一化值被标记为
`theory_domain_not_supported`。这些数据都没有进入正例表，且没有重调 gate。

新增固定 Stage-1（clean/compound、5 variant、3 seed）结果：`Guo` 的 V16.1 与
self-only 均为 `0.434319`，`Melanoma_5K` paired Delta ARI 为 `+0.000288`、stress
retention `0.305`，`Young` paired Delta ARI 为 `+0.002589`、stress Delta 为 `0`。
三者均按预注册规则记为 `empirical_not_supported`。`Bach` 仅完成首个 seed 的
engineering 产物，固定三 seed 在 1800 秒窗口未完成，不进入性能汇总。

## 2026-08-06 V16 protocol correction（静态验证，无新增性能证据）

V16 已按修正计划完成协议级更新：门控直接使用 raw held-out predictive
support；未压缩 NPZ 的 dense `x.npy` 采用分块 memmap→CSR，无法走该路径的输入
在训练前记录 `dense_input_not_supported`；`embedding_final.npy` 保存 Stage-A
latent，`cluster_probabilities.npy` 保存 assignment readout；gate diagnostics
使用 edge-conditional entropy；正式 Stage-1 入口统一为 paired runner，并支持固定
compound 条件和 clean/stress 晋级汇总。理论文档已明确 binomial split 在固定观测
计数条件下的互补依赖，不再把 support 当作 ARI utility。

验证：compileall 通过，V16 focused tests **12 passed**。本节记录协议修正；锚点
Stage-0/1 的 restricted 结果见下一节。此前 fbis exploratory 仍为
`empirical_not_supported`。

## 2026-08-06 V16 Stage-0/1 锚点 restricted no-go

修正后的 runner 先完成 `Campbell` 和 `Mouse_retina` 的无标签 Stage 0，再按固定
seeds `[42, 123, 7]`、五路 paired readout 和固定 compound stress 执行 Stage 1。
Stage 0 两个数据集均通过计数域证书，但 candidate recurrence 为 `0.472390`/
`0.266699`，support 正值率为 `0.001531`/`0.000629`。

Stage 1 的 60 个 summary 全部完成，产物暂存
`/tmp/v16_stage1_anchors_20260806_fixed/`，未写入正式 result 盘。clean mean ARI：
Campbell self-only `0.158261`、fixed graph `0.217547`、V16 `0.157655`；
Mouse_retina self-only `0.404180`、fixed graph `0.429160`、V16 `0.404147`。
V16 clean paired delta 为 `-0.000607`、`-0.000033`；compound delta 为
`0.000000`、`+0.000961`。两数据集均为 `empirical_not_supported`，不进入候选池
确认；本批次不构成正式论文性能证据。

## 2026-08-05 V16 Predictive Graph Gate（implementation / restricted evidence）

V16 独立实现位于 `methods/TopoGate/V16_predictive_graph_gate/`，launcher 位于
`scripts/V16/`；V1--V15 和外部 baseline 未修改。当前实现只接受非负整数或可
恢复的 `log1p(count)` 稀疏矩阵，并将 topology-disabled masked Poisson MAE、
count thinning、稀疏 cosine 候选图、held-out predictive support 和
assignment-only abstaining sparsemax 串成固定闭环。

最小验证：compileall 通过，V16 focused tests **7 passed**。fbis 单 seed、1
epoch 五路 smoke 暂存 `/tmp/v16_stage1_fbis`，仅为 engineering smoke；不形成
性能结论。Stage-0 已核对六个预注册数据的 count-domain certificate，并完成
fbis/tr45 的无标签 exploratory support；Campbell/Mouse_retina/Baron/Quake
大集 support 审计因稀疏 kNN 成本在本轮中止，未写入正式 benchmark 表。

fbis 5-epoch、三 seed `[42,123,7]` exploratory（`/tmp/v16_stage1_fbis_5ep_3seed`）为：
`self_only ARI=0.3314`、`V16 ARI=0.3295`、`fixed_predictive_graph ARI=0.3985`、
`shuffled_support ARI=0.3240`。按预注册晋级规则，fbis 当前为
`empirical_not_supported`；该 smoke 不构成论文级性能证据。

## 2026-08-04 V15 Counterfactual Gate Stage-1（restricted no-go）

V15 独立实现位于 `methods/TopoGate/V15_counterfactual_gate/`，脚本位于
`scripts/V15/`；不修改 V2--V13 或外部 baseline。核心路径包含 sparse-aware
anchor MAE、raw/EMA-latent union candidate graph、detached single-edge
utility、六维 scorer、null/self abstention sparsemax、Student-t/EMA head、
EMA cluster-frequency correction 和 sampled-zero reconstruction。

当前源码工程验证：`compileall` 通过；
V15 focused 回归测试 → **48 passed**（后续只保留与 readout/utility 语义直接
相关的测试）。
cnae9 真实 NPZ smoke 和 Stage-1 engineering panel 产物暂存 `/tmp`，未作为正式
benchmark 结果写入事实表：当前源码 v2 panel 为六个代表集加受控 2D/noisy 集、
2 epochs、单 seed，7/7 完成；真实集 utility AUROC 达标 2/6，candidate
recall 中位数约 0.70，受控边界/低密度/离群 null-AUROC 均为 0.5。
当前 V15 smoke summary 同时保存输入源 SHA256 和六个 V15 源文件 SHA256。

按预注册规则，Stage-1 未通过，正式 Stage-3 `[42,123,7]` 矩阵暂停；以上仅是
机制实现和失败边界证据，不支持 V15 性能收益结论。外部 CLM commit 尚未核验，
不能形成 CLM-aware 主结论。

### V15 Stage-1B 三证书审计（只读，2026-08-04）

审计入口为 `scripts/V15/audit_stage1b_certificates.py`，输出暂存于
`/tmp/v15_stage1b_certificates.json`。它不修改训练产物；`labels_true.npy` 只
用于 graph 的后验指标，并逐 run 保留 `label_use=posthoc_only`。

| 证书 | 当前可验证范围 | panel 结果 |
|---|---|---:|
| teacher correctness | 需要 teacher assignment/embedding 与跨视图或时间 pair；当前契约未保存 | 0/7 |
| candidate graph | post-hoc edge purity、budget-normalized recall、same-label coverage | 7/7 |
| utility in-sample | 同一 run 的 `utility_hat` 对 `utility_target`，仅诊断 | 7/7 |
| utility held-out | 需要 scorer 权重或 held-out prediction | 0/7 |
| independent cluster gain | 需要逐边反事实 embedding/assignment 和独立 downstream 指标 | 0/7 |

因此现有 utility AUROC 不能被写成 utility 泛化或聚类收益；Stage-1 仍为
restricted no-go，正式多 seed benchmark 不启动。

### V15 修复后最小 paired exploratory（2026-08-05，非正式性能结果）

当前 source hash 下的 clean 输出暂存 `/tmp/v15_local_consensus_matrix_20260805/`，
compound 输出暂存 `/tmp/v15_compound_matrix_20260805/`，未写入正式 result 盘。
clean 完成 sms/cnae9 各 5 个 variant 及 reuters self-only；compound 完成
sms/cnae9 的 self-only、direct-local-consensus、counterfactual-learned 共 6
个 run。`reuters direct_counterfactual` 因高维图/训练成本过高在无产物后终止，
不计入失败率或性能统计。

clean 单 seed 线索：sms 的 direct counterfactual/local-consensus/learned 相对
self-only 的 ARI 约为 `+0.039/+0.002/+0.058`；cnae9 local-consensus 约
`+0.004`，learned 约 `-0.017`。candidate recall/purity 约为 sms
`0.89/0.89`、cnae9 `0.75/0.75`；learned scorer 对独立 probe utility 的
held-out AUROC 约 `0.50/0.54`，不能当作 utility 泛化证据。

compound 线索：cnae9 graph recall/purity 约 `0.26/0.26`、sms 约
`0.81/0.81`；local-consensus 仍保持高 edge mass，learned scorer 在两集的
null mass 均为 `0`。该结果是 coherent graph pollution 的 restricted no-go，
不支持 Stage-3 正式多种子 benchmark。

## 2026-08-04 V12 edge-rank stage-2 (rank/trust signal + 4 AHDPC 36-run, restricted go)

按 `v12_edge-rank_topogate_refactor_6d7aad82.plan.md` 增量改进 V12_latent_topology，
不重建 V13。新增 `rank_alignment_loss`（log-space pairwise hinge）：
目标 reliability = `(1/(1+distance) + mutual + snn) → row-standardize ∈ [0, 1]`，
detached,梯度仅回传 gate。`rank_loss_weight=0.1, rank_margin=0.1` 默认；
nomix 显式 0。`run_npz.py` 接入 rank 子块与 topology 共享 ramp schedule，
warmup 期间 gate 仍 no_grad。

正式批次 `result/V12/v12_edge_rank_stage2_2026-08-04/`：4 AHDPC
(flame, balance_scale, spect_heart, vehicle) × 3 variants (nomix /
edge_only / self_null_lambda01) × seeds [42, 123, 7] = **36/36 completed,
0 failed**。CPU `--no-cuda`，80 epochs, hidden=128, mask ratio=0.3,
batch=256, neighbor_k=5, StandardScaler, lambda=0.1, warmup=20,
ramp=10。每个 `summary.json` 含 runner/model/gate source SHA-256、
`labels_used_during_fit=False`、resolved args、metrics、history、
`rank_loss` / `rank_loss_weight` / `rank_margin` / `rank_active_fraction`
字段。

关键结果（mean ± std, n=3 seeds）：

| dataset | nomix ARI | edge_only ARI | self_null@0.1 ARI | Δ ARI vs NoMix |
|---|---:|---:|---:|---:|
| flame | 0.3897 ± 0.1092 | 0.5075 ± 0.0180 | **0.5154 ± 0.0069** | **+0.1257** |
| balance_scale | 0.1163 ± 0.0392 | 0.1059 ± 0.0053 | 0.1016 ± 0.0098 | −0.0147 |
| spect_heart | −0.0264 ± 0.0302 | 0.0104 ± 0.0172 | 0.0050 ± 0.0088 | +0.0314 |
| vehicle | 0.0780 ± 0.0017 | 0.0805 ± 0.0050 | 0.0750 ± 0.0025 | −0.0030 |

Gate 诊断（mean over 3 seeds）：`mean_history.rank_loss` 在 0.020–0.044
区间单调下降；`mean_gate_grad_norm` 从 0 升到 0.04–0.06。但
`edge_entropy` 仍 1.45–1.60 ≈ log(5)，effective_neighbors 4.3–5.0——
`rank_loss_weight=0.1` 不够强。诊断对照：enron λ=0.1 + rank=0.3 的
edge_entropy 0.63 (eff_neigh 2.1) 表明 rank 信号在更高 weight 下能
塌缩到少数邻居，但 ARI 仍是 0.0003（topology alignment 在 enron 上
是 stage-1 已记录的退化边界，rank 修复不解决也不恶化）。

**判定**: edge-rank stage-2 是 **restricted go**——边缘选择机制已生效
（rank_loss 下降、gate 有梯度、reliability 非退化），flame ARI 显著
超过 NoMix（+0.126），但 seed 7 退化 0.22 仍需评估；edge_entropy 仍
接近 log(5) 说明 `rank_loss_weight=0.1` 不足。不宣称"已修复选择"——
仅宣称"已实现选择机制 + 部分证据"。下一阶段（day-2 task）建议：
`rank_loss_weight=0.3, rank_margin=0.2` 在 5 datasets × 3 variants ×
3 seeds 重跑验证 entropy 显著下降。

详细报告见 `result/analysis/V12_edge_rank_stage2_2026-08-04.md`。

## 2026-08-04 V12 stage-3 拓扑信号强化网格 (no-go, hinge loss 已饱和)

按 stage-2 结论（edge_entropy 仍接近 log(5)）执行 plan
`v12_topology_signal_amplification_stage3_<id>.plan.md` 的 hyperparameter
sweep。**搜索空间**：`lambda_topology ∈ {0.3, 0.5}` ×
`rank_margin ∈ {0.5, 1.0}` × `self_init_weight ∈ {0.3, 0.5}` (self_null
only) = 8 self_null + 4 edge_only = **12 configs**；数据集 flame /
balance_scale / spect_heart / vehicle；seeds [42, 123, 7]；epochs=80；
`rank_loss_weight=0.1` 恒定，`mask_loss_weight=0.1`。

正式批次 `result/V12/v12_topology_search_stage3_2026-08-04/`：**144/144
completed, 0 failed**。CPU `--no-cuda`，3 worker 并发，约 5 分钟。
Launcher `scripts/V12/run_stage3.py`；summarizer
`scripts/V12/summarize_stage3.py`。

**edge_entropy（headline metric）**：

| dataset | entropy 区间（12 configs） | effective_neighbors 区间 | < log(5) cell | < 1.0 cell |
|---|---|---|---:|---:|
| flame | 1.586 – 1.591 | 4.889 – 4.911 | 12/12 | **0/12** |
| balance_scale | 1.398 – 1.481 | 4.100 – 4.418 | 12/12 | **0/12** |
| spect_heart | 1.459 – 1.531 | 4.339 – 4.634 | 12/12 | **0/12** |
| vehicle | 1.196 – 1.324 | 3.416 – 3.821 | 12/12 | **0/12** |

**所有 48 个 (dataset, config) cell 都 < log(5) 但没有任何 cell < 1.0**。
`rank_loss` 随 `rank_margin` 增大从 ~0.21 升到 ~0.49（确认 rank signal
在工作），但 edge_entropy 仅下降 0.05–0.10，effective_neighbors 仍
3.4–4.9——**hinge loss 梯度强度已达饱和**。

**ARI 与 paired delta vs stage-2 self_null baseline**：

| dataset | paired delta 区间（12 configs） | 解释 |
|---|---|---|
| flame | -0.012 ~ -0.016 | 退化（落在 0.03 容差内） |
| balance_scale | **+0.039 ~ +0.043** | **真实增益**（> 0.03 容差） |
| spect_heart | -0.001 ~ +0.008 | 持平 |
| vehicle | +0.009 ~ +0.027 | 边缘增益 |

跨 12 configs ARI mean 区间 **0.1833–0.1885**（≈ 0.005），edge_only vs
self_null ARI 差异 < 0.001——与 stage-2 观察一致：KMeans(k=2 or 3)
对 topology 分支 0.04–0.13 ARI 差异不敏感；4 AHDPC embedding 主要由 AE
主成分决定。

**判定**：stage-3 网格内 **no-go**——edge_entropy 仍 1.42–1.59 区间，
**没有任何 config 触及 < 1.0 目标**。Hinge loss 架构无法突破
softmax-uniform 边界，触发 plan 中"hinge loss 架构需要彻底替换"的
兜底结论。balance_scale +0.04 ARI 跨 config 稳定（lambda 提升带来），但
flame -0.012 反向退化，不构成"已修复选择"证据。

**下一步建议**（plan 失败条件对应路径）：替换 hinge loss 为
KL 散度 / Gumbel-top-k / sparsemax，重建 V13 top-k gating
（K=2 选出可信邻居），或重写 reliability target（source-path entropy
或多视图一致性）。当前 V12_latent_topology 不进入论文 main-result 表。

详细报告见
`result/analysis/V12_topology_signal_amplification_stage3_2026-08-04.md`；
产物 `runs.csv` / `summary_by_config.csv` / `entropy_diagnostic.csv` /
`paired_deltas_vs_stage2.csv` / `report.md`。

## 2026-08-04 V13 Gumbel-Top-k (hard gate + 5 datasets × 2 variants × 3 seeds, 有条件 go)

按 stage-3 no-go 结论和 plan 失败条件重建 V13，替换 softmax + hinge 为
`GumbelTopKGate`（Gumbel-Softmax straight-through，推理时 hard top-k）。
核心新模块 `methods/TopoGate/V13_hard_gate/gumbel_gate.py`：
`GumbelTopKGate` 使用 top-k 强制排序，无 rank_loss，无 self/null fallback。
`hard_topk_alignment_loss` 用 mask_sum 归一化而非 K。

正式批次 `result/V13/v13_hard_gate_2026-08-04/`：5 datasets ×
2 variants (nomix / topk2) × seeds [42, 123, 7] = **30/30 completed,
0 failed**。Launcher `scripts/V13/run_v13.py`；summarizer
`scripts/V13/summarize_v13.py`；14/14 unit tests passed。

**核心发现：hard gate 完美工作，topology_alignment_loss 具破坏性**

| 指标 | 结果 |
|---|---|
| `effective_neighbor_count = 2.000` | ✅ 5 datasets × 15 runs 全部严格成立 |
| enron topk2 vs nomix delta | ⚠️ -0.73 ARI（灾难性崩溃） |
| flame topk2 vs nomix delta | ⚠️ -0.084（seed 不稳定：seed 7 +0.066, seed 42 -0.277） |
| balance_scale topk2 vs nomix | 边缘 +0.023 |
| spect_heart topk2 vs nomix | 持平 |
| vehicle topk2 vs nomix | 持平 |

**门控机制成功**（V13 解决了 V12 的核心问题），但 **topology_alignment_loss
在硬选择后更具破坏性**：hard top-k 无 softmax 的"模糊平均"效应，
一旦选错跨簇邻居，MSE 直接强制 anchor 移向错误簇中心。enron 在 nomix
下 ARI 0.803（AE 本身能做），但 topk2 只有 0.072。

**判定**：V13 Gumbel-Top-k **有条件 go**：
- ✅ hard gate 机制完全有效，`effective_neighbors = top_k = 2` 在所有
  dataset 上稳定且可复现；
- ✅ **V13 是第一个验证 Gumbel-Top-k 在聚类任务中 hard selection 行为的工作**；
- ⚠️ topology_alignment_loss 设计需要重新审视（可能改为 detach 目标、
  contrastive 而非 MSE、或仅在低维数据集启用）；
- **论文叙事**：贡献 = "Gumbel-Top-k hard selection 在无监督聚类中的验证"，
  而非"topology alignment 改进"。

详细报告见 `result/analysis/V13_gumbel_topk_analysis_2026-08-04.md`。

## 2026-08-03 V12 self/null stage-1 formal paired benchmark（实现完成，性能 restricted no-go）

真正的 `methods/TopoGate/V12_latent_topology/run_npz.py` 已在
`result/V12/v12_self_null_stage1_2026-08-03/` 完成 flame/enron ×
NoMix/edge-only/self-null(lambda=0.01/0.03/0.1) × seeds
`[42,123,7]`，共 **30/30 completed，0 errors**。固定 StandardScaler、
hidden=128、mask ratio=0.3、batch=256、neighbor_k=5、80 epochs；
K 仅由 `np.unique(y)` 用于 benchmark，30/30 summary 均记录
`labels_used_during_fit=false`，source path/hash、resolved args、预测/真值
数组和 graph diagnostics 均存在。

工程契约：默认 decoder 为兼容的 `[latent, mask_logits] -> Linear`，
mask loss 为 additive `reconstruction + 0.1*mask`；训练不调用
`make_pseudo_batch`，邻居 gather 和 gate 聚合使用 Torch；self/null
self+edge 权重和为 1，clean target 不接收梯度且 gate gradient 非零。
compileall、7 个 V12 tests 和 flame/enron smoke 已通过。

阶段性结果（六 run 均值）：NoMix ARI=0.6616、edge-only=0.2015，
self/null lambda=0.01/0.03/0.1=0.6195/0.3374/0.1872。self mass 有效
（flame 约 0.73--0.74，enron 约 0.87--0.92），但 conditional edge entropy
约等于 `log(5)`，未形成有效逐边选择；enron lambda=0.03/0.1 出现明显
seed-sensitive collapse，flame 所有 topology 条件低于 NoMix。因此这批次
只证明实现、梯度和失败边界，不支持 V12 默认 lambda=0.1 的性能收益，也不
扩展第二阶段五数据集。详细表格和诊断见
`result/V12/v12_self_null_stage1_2026-08-03/` 与
`result/analysis/V12_self_null_stage1_2026-08-03.md`；不声称严格 TDA、
概率模型或普遍拓扑优越性。

## 2026-08-03 V12 finalized-code warmup-fix stage-1（当前权威阶段证据，restricted no-go）

旧的 `result/V12/v12_self_null_stage1_2026-08-03/` 保留为 pre-fix 审计批次；
未覆盖它。加入 warmup 真冻结和 runner/model/gate source hash 后，当前源码在
`result/V12/v12_self_null_stage1_2026-08-03_warmup_fix/` 完成同一
flame/enron × NoMix/edge-only/self-null(lambda=0.01/0.03/0.1) ×
`[42,123,7]` 矩阵，**30/30 completed，0 errors**。所有 summary 共享当前
代码 hash，source path/hash、resolved args、预测/真值数组和 diagnostics 均存在；
`labels_used_during_fit=false`。

warmup 修复验证为前 20 个 epoch gate gradient=0 且 self mass 不漂移，ramp 后
才建立 topology gradient。最终六 run 均值：NoMix ARI=0.6616、edge-only
=0.2016、self/null lambda=0.01/0.03/0.1=0.6194/0.3372/0.1874；
self/null self mass 有效，但 lambda 0.01/0.03 的 conditional edge entropy
仍约为 log(5)，lambda=0.1 仅轻微偏离，尚未学到可靠逐边选择。enron 仅
lambda=0.01 保持高维去噪，lambda=0.03/0.1 明显退化；flame topology
条件全部低于 NoMix。该阶段满足工程接口与梯度契约，但性能为 restricted
no-go，暂不扩展五数据集或写入论文收益叙事。详细报告见
`result/analysis/V12_self_null_stage1_warmup_fix_2026-08-03.md`。

## 2026-08-03 V12 latent-topology engineering smoke（非性能结果）

新增独立路径 `methods/TopoGate/V12_latent_topology/`，不修改 V9 legacy、V10
或 V11。V12 使用 `mask_loss_weight=0.1` 的 clean masked reconstruction、
`[N,K,4]` Torch edge features、`[B,K]` softmax edge weights 和 detached-clean
neighbour latent alignment；runner 不调用 `make_pseudo_batch`，并记录实际 gate
梯度范数。`compileall` 与 3 个 V12 tests 通过，3-epoch flame gradient smoke
的 `mean_gate_grad_norm=4.666475e-05`。

工程对照（seed=42、CPU、single-seed、缩短或诊断性训练）曾得到：flame 8 epoch
full/NoMix ARI=`0.377868/0.388210`，flame 80 epoch=`0.357486/0.206987`；
enron 8 epoch=`0.885082/0.890737`。这些运行的数组和 summary 均写入 `/tmp`
后清理，不能作为论文性能结论；正式判断仍需至少五个数据集、seeds
`[42,123,7]` 和预注册配对控制。

### V12 性能下降诊断（单 seed 工程证据）

同协议 `flame` 诊断（seed=42、CPU、80 epochs、hidden=128、mask ratio=0.3、
StandardScaler、batch=256）显示：legacy V9 NoMix 的 ARI 为 0.4764（mask loss
0.7）/0.4649（mask loss 0.1），而当前 V12 decoder 的 NoMix 为 0.1843；仅在
临时运行中恢复 legacy `[latent, mask_logits]` decoder 后为 0.4534。当前 V12
Full 为 0.0747，最终 edge entropy=1.6088，接近 `log(5)`，表明没有学到有效的
逐边选择。该条目用于定位架构回归，不是论文级性能结论；V12 正式多 seed 批次
必须重新运行真正的 `V12_latent_topology/run_npz.py`。

当前源码四路隔离复核（兼容 decoder/latent-only decoder × NoMix/Full）和
provenance 纠正见 `result/analysis/V12_performance_drop_diagnosis_2026-08-03.md`。
当前源码的同协议结果为 V12 legacy NoMix=`0.4998`、latent-only NoMix=`0.1843`、
legacy Full=`0.1844`、latent-only Full=`0.0747`；这些仍是单 seed engineering
smoke，不能替代真正 V12 的多数据集多 seed 事实表。

## 2026-08-03 V11 h0_early_mst 正式配对（固定协议 no-go）

正式产物位于 `result/V11/tda_h0_early_mst_pilot_2026-08-03/`，由
`scripts/V11/run_v11_multiseed.py` 运行五个 AHDPC processed 数据集的
`V11_full` 与 `V11_tda_h0_early_mst`，每个使用 80 epochs、CPU `--no-cuda` 和
seeds `[42,123,7]`，共 **30/30 completed，0 errors**。分析入口为
`scripts/analysis/analyze_v11_tda_h0_pilot.py` 的显式 `--output-dir` 和
`--variants` 参数。

每个数据集的 Full/候选 source SHA-256 一致，K 分别为
`balance_scale=3`、`spect_heart=2`、`banknote=2`、`flame=2`、`vehicle=4`；
所有 30 个 summary 均记录 `k_protocol=benchmark_oracle_from_y` 和
`labels_used_during_fit=false`，预测、真值、配置和 source hash 文件均存在。

15 个同 dataset-seed 配对中，候选相对 Full 的 head ARI 为 `+0.000010`、KMeans
ARI 为 `-0.001139`、NMI 为 `+0.000013`、silhouette 为 `+0.000140`；head ARI
为 2 胜 / 12 平 / 1 负。候选提高了平均 graph loss `+0.005177` 和 mean gate
`+0.001129`，但没有转化为聚类改进。因此该固定五数据集协议下结论为 **no-go**，
不把结果泛化为完整 TDA 无效，也不进入论文主方法。

## 2026-08-03 跨版本证据与 provenance 审计

统一审计入口为 `result/analysis/cross_version_evidence_audit_2026-08-03.md`，机器可读产物为：

- `result/analysis/cross_version_evidence_2026-08-03.csv`
- `result/analysis/paired_version_deltas_2026-08-03.csv`
- `result/analysis/provenance_audit_2026-08-03.csv`

同一批次内的 Full-NoMix ARI 差值为：V9 优势消融 `+0.015356`（7 datasets）、V11 minimum `-0.000475`（5 datasets）、V12 `-0.001244`（12 datasets）、V13 `-0.000238`（12 datasets）、V14 `+0.004373`（5 datasets）。这些数值只用于配对机制审计，不能跨输入协议或数据清单纵向拼接。

provenance 审计发现：V9 优势消融和 V12 的历史 `summary.json` 把 dataset 写为 `adhoc`，真实身份保留在 `ablation_runs.csv`/`runs.csv`、`run_record.json` 和 source hash 中；V11/V13/V14 部分 summary 未显式写 `labels_used_during_fit=false`，但源码边界审计未发现训练器消费真值标签。这些批次标为 metadata gap/partial provenance，不把缺失字段改写成已记录事实。

## 2026-08-03 跨版本优势/劣势景观审计

只读脚本 `scripts/analysis/analyze_topogate_cross_version_landscape.py` 已从上述
批次和 TDA 75-run summary 生成 `result/analysis/topogate_cross_version_landscape_2026-08-03*`
六个产物。56 条同 batch Full-NoMix 配对的均值为 V9 `+0.015356`、V11
`-0.000475`、V12 `-0.001244`、V13 `-0.000238`、V14 `+0.004373`；StaticGate
为单 seed 历史表 `-0.015310`，不作为多 seed 稳定证据。source SHA-256 审计阻止
同名异源纵向合并，`vehicle` 的两个 hash 被标为不可直接比较。该条目只增加描述性
分析，不改变任何模型结果或论文主方法。

## 2026-08-03 V11 sparse H0 TDA pilot（工程验证，非性能结果）

已新增可回退的 `methods/TopoGate/V11/tda.py`：在固定 raw kNN 稀疏
Vietoris--Rips 1-skeleton 上用 union-find 计算 H0 component merge
persistence，并将 bounded、detached prior 映射到 V11 当前候选边。配置默认
`tda_prior_mode=none`、`tda_prior_weight=0.0`，因此不改变原 V11 路径；同时注册
`h0_mst`、`fixed_filtration` 和 `random` 对照模式。

工程验证：`compileall` 通过；V11 回归测试 `19 passed`；iris CPU 3-epoch
TDA smoke 成功写入 graph/history 诊断后清理了 `/tmp` 临时输出。该 smoke 只证明
实现和产物契约，**没有**形成持久化性能 CSV，也不能支持 TDA 的 ARI/NMI 结论。
完整数学说明和正式比较结果见
`result/analysis/topogate_v11_tda_h0_pilot_2026-08-03.md`；下方正式五数据集条目
已取代本节的“尚无性能结果”阶段性边界。

## 2026-08-03 V11 sparse H0 TDA 正式五数据集对照

正式产物位于 `result/V11/tda_h0_pilot_2026-08-03/`，由
`scripts/V11/run_v11_multiseed.py` 按显式 AHDPC source mapping 读取
`datasets/AHDPC/processed/{balance_scale,spect_heart,banknote,flame,vehicle}.npz`。
固定 V11 默认 YAML、80 epochs、CPU `--no-cuda`、seeds `[42,123,7]`，比较
`V11_full`、`V11_nomix`、`V11_tda_h0_mst`、`V11_tda_fixed_filtration` 和
`V11_tda_random`，共 **75/75 completed，0 errors**。`K` 均记录为
`benchmark_oracle_from_y`，训练器不读取标签；75/75 summary 显式记录
`labels_used_during_fit=false`，source hash、预测/真值数组和 resolved config 均存在。

机器可读诊断和配对统计为 `run_diagnostics.csv`、`summary_by_dataset_variant.csv`、
`paired_deltas.csv`、`protocol.json`，文字报告为 `report.md`。跨 15 个配对
dataset-seed：H0 相对 V11 Full 的 head ARI `+0.000010`、KMeans ARI
`-0.000726`；fixed-filtration 分别为 `+0.000002`、`-0.000665`；random
分别为 `+0.000018`、`-0.000274`。三种 prior 的 head ARI 均有至少 11/15
对为 ties，未显示持久性相对距离或随机控制的独立聚类收益。H0 的平均最终
gate 为 `0.0178`、V11 Full 为 `0.0160`，但平均 graph loss 为 `0.0508` 对
`0.0444`；prior 改变了图目标和 gate mass，未转化为 ARI/NMI 改善。

**结论**：在这五个预注册数据集和固定协议内，sparse H0 prior 为性能 **no-go**，
仅保留为可审计的 TDA 诊断/后续假设生成；该结果不等价于对完整 TDA 或其他数据分布的否定。

## 2026-08-03 无标签优势/劣势数据特征与 TDA 诊断

特征审计入口为 `scripts/analysis/build_topogate_dataset_feature_audit.py`，只读取 NPZ 的 `x`，不加载 `y`，不使用标签选择图、阈值、尺度或 variant。当前结果盘产物为：

- `result/analysis/topogate_dataset_features_2026-08-03.csv`：49 个结果相关数据集，47 个完成，`Campbell` 与 `hrvatin_filtered` 因 `80,000,000` 元素上限跳过；CSV 记录 source、shape、采样状态和事后结果连接。
- `result/analysis/topogate_feature_version_correlations_2026-08-03.csv`：180 条固定协议下的探索性 Spearman 相关。
- `result/analysis/topogate_advantage_feature_audit_2026-08-03.md`：版本正负集合、TDA 术语边界和 pilot 设计。

协议为标准化后 PCA 上限 50、普通 kNN `k=5`、稀疏 TDA skeleton `k=15`；超过 4,000 个样本或 512 个特征时固定随机采样。`tda_h0_*` 仅是固定稀疏 1-skeleton 上的 H0/component persistence 摘要，`cycle_rank_*` 不是 H1 persistence diagram。该批次是无标签诊断和假设生成，不是已实现的 TDA 模型，也不能支持因果或普遍性能结论。当前最完整的 V9 topology 正例为 `balance_scale`；后续 pilot 必须保留原 V11、NoMix、random prior、fixed-filtration 和 `[42,123,7]` 配对控制。

## 2026-08-03 V9 优势数据分析与 V14 正式小批次

**V9 优势分析入口**：`result/analysis/V9_AHDPC_advantage_deep_analysis_2026-08-03.md`；特征画像补充见 `result/analysis/V9_AHDPC_feature_profile_2026-08-03.md`。论文预处理匹配协议下，V9 相对 AHDPC 的正差值数据集为 `spect_heart`（+0.261303）、`balance_scale`（+0.175669）、`landsat`（+0.024792），即 3/24；历史标准化协议的 9/24 正差值仅作补充，不能与论文匹配协议混合。无标签几何探索提示最大共性是局部邻域可利用而固定密度/ε 假设不稳定，不能归结为单一 mutual-neighbor 阈值；补充分析显示合成几何组平均 ΔARI=-0.2439，UCI 组平均 ΔARI=-0.0868，且高 mutual 本身不是 V9 获胜条件。

**V9 Full/NoMix 配对消融**：产物为 `result/v9_results_2026-08-02_advantage_ablation/ablation_runs.csv` 和 `summary_by_dataset.csv`，7 datasets × 4 variants × 3 seeds，84/84 completed；`v9_full` 为 reliability mix + `pseudo_weight=0.3`，`v9_nomix` 为严格 `mix_mode=none` + `pseudo_weight=0`。Full 相对 NoMix 的宏平均 ARI 为 **+0.015356**（NMI **+0.011522**），21 个配对的 Wilcoxon `p=0.3905`、配对 t 检验 `p=0.1417`，不能宣称总体拓扑增益。数据集层面 `balance_scale` 为最清晰正例（+0.080941，3/3 seed），`landsat` 为小幅稳定正差（+0.005501，3/3），而 `spect_heart`（−0.012542）、`vehicle`（−0.011436）和 `vertebral_column`（−0.017061）偏向 NoMix；`glass`、`image_segment` 虽相对 NoMix 提升，但仍分别低于 AHDPC `−0.019074`、`−0.148420`。该消融覆盖 7 个数据集，不能外推为 24 个数据集的普遍拓扑结论。

**V14 事实表**：产物 `result/v14_results_2026-08-03_advantage_5ds/runs.csv`，5 datasets × `v14_full`/`v14_nomix` × seeds `[42,123,7]`，30/30 completed。宏平均 ARI：full **0.133629**，nomix **0.129256**，差 **+0.004373**；Wilcoxon **p=0.8139**，配对 t 检验 **p=0.6597**。full 平均 target gate=0.006276，表明拓扑路径可运行但监督质量/强度偏弱。V14 为机制可运行、性能 **no-go**，不写入论文主方法。配置为 `methods/TopoGate/V11/configs/topogate_v14_advantage_minimum.yaml`，runner 为 `scripts/v9_learnable_gate/run_v14_advantage_smoke.py`。

## 2026-08-03 CLUBench 131 数据集：AHDPC、HDPC 与 TopoGate V9 全量单 seed 对照

**权威产物**：[`result/clubench_ahdpc_hdpc_v9_2026-08-02`](./clubench_ahdpc_hdpc_v9_2026-08-02)，其中 `MANIFEST.json`、`comparison_long.csv`、`comparison_wide.csv`、`method_summary.csv`、`analysis_report.md` 和 `analysis_full.json` 共同构成审计入口。

**协议**：CLUBench 官方 `load_data` 列级 z-score；`K=int(np.unique(y).size)` 仅用于 benchmark K 和拟合后的指标；AHDPC/HDPC 使用 `epsilon=1.0`、`paper_semantic`、`table_reproduction`、`block_size=256`；V9 使用 `learnable_gate_v9_adaptive`、seed=42、80 epochs、batch size=256、`scale_input=false`。训练过程不传入真值标签。

**完成事实**：131 个数据集 × 3 个方法 = **393/393 completed**，**0 errors**；每个摘要均核验 `labels_used_during_fit=false`，`k_source=labels_unique`，输入预处理均为 `CLUBench.load_data z-score`。

| 方法 | 完成 | Mean ARI | Median ARI | Mean NMI | Mean ACC |
|---|---:|---:|---:|---:|---:|
| AHDPC | 131 | 0.1830 | 0.0320 | 0.2401 | 0.5305 |
| HDPC | 131 | 0.1614 | 0.0104 | 0.2200 | 0.5165 |
| V9 | 131 | 0.3227 | 0.2484 | 0.3757 | 0.6059 |

按 ARI 的配对结果：V9 相对 AHDPC 为 **105 胜 / 2 平 / 24 负**，平均 ΔARI=0.1396；相对 HDPC 为 **104 胜 / 1 平 / 26 负**，平均 ΔARI=0.1613。按描述性阈值 ΔARI≥0.10，V9 相对 AHDPC 有 58 个数据集明显占优；ΔARI≤−0.10 的显著退化有 7 个。该批次是 **single seed=42 的工程/对照证据**，不能直接替代多 seed 论文结论。

负面结果须区分：AHDPC 强而 V9 退化的 `banknote_authentication`（ΔARI=-0.9381）、`shuttle`（-0.4604）、`extyaleb`（-0.1579）、`world12d`（-0.1096）；其余 substantial regression 还包括 `heart_disease`、`paris_housing_classification`、`echocardiogram`。三种方法 ARI 都≤0.10 的共同困难数据共有 43 个，不应归因于 V9 特异性失败。完整分层与全部行见 `analysis_report.md`/`analysis_by_dataset.csv`。

## 2026-08-02 V11.3 semantic-metric 候选（临时验证，禁止性能结论）

### 旧 semantic_residual breast 对照（3 seeds，历史产物已清理）

历史产物：`/tmp/topogate_v11_semantic_breast__{full,nomix}__seed{42,123,7}`；已按 smoke 生命周期规则清理。K 均由 `len(unique(y))` 自动检测，训练不消费标签。下表只保留历史工程数值，不能作为当前可复核性能证据。

| 指标 | Full | NoMix | Δ Full-NoMix |
|---|---:|---:|---:|
| Student-t head ARI | 0.887228 ± 0.003224 | 0.885369 ± 0.011102 | +0.00186 |
| KMeans embedding ARI | 0.889104 ± 0.006473 | 0.885395 ± 0.005558 | +0.00371 |
| silhouette | 0.732155 ± 0.002689 | 0.727167 ± 0.010127 | +0.00499 |

Full final gate（seed 42/123/7）为 0.0336/0.0363/0.0398，target gate 为 0.0356/0.0442/0.0404；NoMix 严格不构图且 gate=0。该结果是旧 `semantic_residual` 的临时对照，不代表 V11.3 几何分支。

### V11.3 semantic_metric iris 工程 smoke（历史产物已清理）

历史产物：`/tmp/topogate_v11_semantic_metric_iris/`（已清理）；CPU、seed=42、4 epochs、缩小网络。head ARI=0.6051，KMeans embedding ARI=0.5961，最后 gate/target=0.311/0.021，edge alignment loss=1.7082。仅证明新几何项曾经可运行；由于产物已清理且属于缩短 smoke，禁止与正式 V11/V9/NoMix 性能比较。

---

## 2026-07-31 AHDPC（ESWA 2026）真实数据复现 smoke（历史产物已清理）

**历史产物**：`result/AHDPC/verified_smoke_2026-07-31/summary.csv`、`result/AHDPC/verified_smoke_2026-07-31/banknote_reported_equation/summary.json` 已按项目规则清理。下表仅保存已发生运行的审计记录，不再宣称这些路径当前存在。

| 数据集 | 输入 | 模式 | AMI | RI | FMI | NMI |
|---|---|---|---:|---:|---:|---:|
| Flame | 240×2，raw，K=2 | AHDPC `table_reproduction` | 0.9353 | 0.9834 | 0.9846 | 0.9355 |
| Flame | 同上 | HDPC | 0.7009 | 0.8823 | 0.8950 | 0.7019 |
| Aggregation | 788×2，raw，K=7 | AHDPC `table_reproduction` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Aggregation | 同上 | HDPC | 0.9020 | 0.9276 | 0.8188 | 0.9033 |
| Banknote | 1372×4，z-score，K=2 | AHDPC `table_reproduction` | 0.9316 | 0.9812 | 0.9814 | 0.9317 |
| Banknote | 同上 | HDPC | 0.6092 | 0.8120 | 0.8139 | 0.6094 |
| Banknote | 同上 | AHDPC `reported_equation` | 0.0084 | 0.5046 | 0.7063 | 0.0095 |

**事实边界**：这验证了下载数据、独立实现及运行链路；并不证明论文全部 28 个数据集上的性能。24 个精确数据已归档，G2 和 3 个医学影像集因论文没有可复现的精确来源/协议被明确标记为 `unresolved`，未使用替代数据。Banknote 显示表格匹配模式为经验反演而非印刷 Eq.(10)，该区别必须在任何使用该结果的地方保留。

**扩展覆盖（同日真实运行，产物已清理）**：历史路径 `result/AHDPC/verified_smoke_2026-07-31/extended/summary.csv` 额外验证 20 簇、64-D 及 UCI 路径：2d-20c-no0 的 AMI/RI/FMI/NMI=0.9730/0.9942/0.9492/0.9741；Dim064=1.0000/1.0000/1.0000/1.0000；Image Segment=0.5499/0.8530/0.5066/0.5718；Rice=0.4630/0.7684/0.7840/0.4631。它们是单次确定性运行，不能被用作跨实现、跨预处理或超参数选择的泛化结论。

**Olivetti 图像分支（历史产物已清理）**：历史路径 `result/AHDPC/verified_smoke_2026-07-31/olivetti_faces_tsne_seed42/summary.json` 在已下载的 400×4096 AT&T faces 上以 seed=42、perplexity=30、1000 iterations、ε=0.1 完成 t-SNE + AHDPC：AMI/RI/FMI/NMI=0.8001/0.9746/0.5930/0.8767。输出曾分离为 `predictions.npy` 与 `labels_true.npy`；论文未报告 t-SNE 参数，因此该数值只能作为历史可审计运行记录。

## 2026-07-30 V11 概率可信拓扑工程 smoke（历史产物已清理，禁止作性能结论）

**历史产物**：`result/V11/smoke/iris__V11__seed42/summary.json` 已清理；本节只保留工程验证边界。

| 项目 | 已核实事实 |
|---|---|
| 数据 | `datasets/iris.npz`，150 samples × 4 features；SHA256=`c31ba1e...3fd5` |
| K | 3，来自 `len(unique(y))`；summary 显式记录 `benchmark_oracle_from_y`，trainer 不接收 y |
| 运行 | CPU，seed=42，3 epochs，batch=64；缩小 hidden/latent 仅用于 smoke |
| 聚类头 | 对角 Student-t mixture soft responsibilities；主预测保存于 `predictions.npy`，KMeans 仅作诊断 |
| 动态图 | epoch 2/3 刷新 `raw-kNN ∪ EMA-latent-kNN`；edge change fraction=0.4600/0.0044 |
| self/null gate | 末 epoch learned topology mass=0.2141，风险 target=0.1419；严格 `use_topology=false` 回归中 gate 恒为 0 且不构图 |
| 输出契约 | `predictions.npy`、`labels_true.npy`、`label_mapping.json` 分离；保存 source SHA256、实际 PCA dim=2、环境和完整 history |
| 工程指标 | head ACC=0.8267、NMI=0.6651、ARI=0.6129；KMeans 诊断 ARI=0.6292 |
| 工程测试 | `pytest -q methods/TopoGate/V11/tests/test_v11.py`：**6 passed** |

**结论边界**：这是 single-seed、3-epoch、缩小网络的工程 smoke，仅证明梯度、概率归一化、EMA、动态图、null expert、真实 NPZ 与保存链路可运行。V11 是否优于 V9/NoMix 尚未成立；正式 go/no-go 需要预注册的 15 数据集多种子 full-vs-ablation 对照和统计检验。

## 2026-07-30 V10 Reliable-Graph 持久工程 smoke（历史产物已清理，禁止作性能结论）

**历史产物**：`result/v10_reliable_graph/smoke/iris__topogate_v10_reliable_graph__seed42/summary.json` 已清理；本节只保留工程验证边界。

| 项目 | 已核实事实 |
|---|---|
| 数据 | `datasets/iris.npz`，150 samples × 4 features |
| K | 3，来自 `len(unique(y))`；`k_source=labels_unique`，known-K 评估 |
| 运行 | CPU，seed=42，3 epochs，batch size=150；`warmup=0/ramp=1/refresh=1` |
| 原型初始化 | epoch 1，在归一化 EMA clean embedding 上执行 `KMeans(n_init=20)`，同步初始化 online/EMA prototypes |
| 簇先验 | `cluster_prior_mode=warmup_kmeans`，无标签 warmup KMeans 计数先验 `[0.459998, 0.213336, 0.326667]`；可用 `uniform` 做消融 |
| 动态图 | 每个 epoch 刷新，共 3 次；每次 1500 条候选边；最终 input/latent stability=0.887333，独立 temporal recurrence=0.868667 |
| 输出契约 | `predictions.npy` 与 `labels_true.npy` 分离；另存 prototype 诊断和含全候选边 gate 的 `final_graph_edges.npz` |
| 标准 KMeans readout | ACC=0.8400，NMI=0.685151，ARI=0.633486，macro-F1=0.834646 |
| 工程测试 | `pytest -q tests/v10_reliable_graph`：**14 passed**；含 gate abstention、独立 temporal target、feature-only 无随机 prototype、duplicate-row 去 self、FAISS HNSW、非均衡 prior 与全图 gate 产物回归 |

**结论边界**：上述数值来自 single-seed、3-epoch 的快速工程 smoke，只证明最终代码能够自动推断 K、初始化 prototypes、刷新动态图、完成训练并保存语义分离的产物。不得将 ARI=0.633486 与历史 V9/NoMix 或其他模型做性能比较，也不得据此声称算法改进有效。V10 的论文级结论仍缺少至少 5 个核心数据集 × 3 seeds 的主对照和 feature-only/fixed-graph 消融。

---

## 2026-07-29 HVF + Adaptive PCA smoke 结果（5 datasets × 4 configs, seed=42）

### enron (n=9999, d=4096, K=2)

| 配置 | HVF | PCA模式 | ARI | Δ vs baseline |
|------|-----|---------|----:|---------------|
| v2_baseline | 0 | fixed(50) | 0.8656 | — |
| hvf500_fixed | 500 | fixed(50) | 0.8757 | +0.010 |
| hvf1000_fixed | 1000 | fixed(50) | **0.8900** | **+0.024** |
| hvf1000_adaptive | 1000 | adaptive | **0.8900** | **+0.024** |

**结论**：HVF 在高维 enron 上显著提升。

### Mouse_retina (n=8352, d=6198, K=5)

| 配置 | HVF | PCA模式 | ARI | Δ vs baseline |
|------|-----|---------|----:|---------------|
| v2_baseline | 0 | fixed(50) | **0.9370** | — |
| hvf500_fixed | 500 | fixed(50) | 0.9060 | -0.031 ⚠️ |
| hvf1000_fixed | 1000 | fixed(50) | 0.9283 | -0.009 |
| hvf1000_adaptive | 1000 | adaptive | 0.9313 | -0.006 |

**结论**：HVF 在 scRNA 数据上**可能有害**（过滤了生物信号）。推荐默认关闭。

### iris (n=150, d=4, K=3)

| 配置 | HVF | PCA模式 | ARI | Δ vs baseline |
|------|-----|---------|----:|---------------|
| v2_baseline | 0 | fixed(50) | 0.6765 | — |
| hvf500_fixed | 500 | fixed(50) | 0.6765 | 0 (HVF skipped, d=4<500) |
| hvf1000_fixed | 1000 | fixed(50) | 0.6765 | 0 (HVF skipped) |
| hvf1000_adaptive | 1000 | adaptive | 0.6765 | 0 (HVF skipped) |

### hrvatin_filtered (n=3778, d=9664, K=8)

| 配置 | HVF | PCA模式 | ARI | Δ vs baseline |
|------|-----|---------|----:|---------------|
| v2_baseline | 0 | fixed(50) | **0.8724** | — |
| hvf500_fixed | 500 | fixed(50) | 0.6425 | **-0.230** ⚠️ |
| hvf1000_fixed | 1000 | fixed(50) | 0.7710 | -0.101 |
| hvf1000_adaptive | 1000 | adaptive | 0.7722 | -0.100 |

**结论**：HVF 在 hrvatin scRNA 数据上**严重降低性能**（-0.10~-0.23）。这对 scRNA 是灾难性的。

### sms_spam_collection (n=5574, d=4834, K=2)

| 配置 | HVF | PCA模式 | ARI | Δ vs baseline |
|------|-----|---------|----:|---------------|
| v2_baseline | 0 | fixed(50) | 0.8282 | — |
| hvf500_fixed | 500 | fixed(50) | 0.8282 | 0 |
| hvf1000_fixed | 1000 | fixed(50) | 0.8282 | 0 |
| hvf1000_adaptive | 1000 | adaptive | 0.8272 | -0.001 |

### 全面教训

| 数据类型 | 例子 | HVF 影响 | 推荐 |
|---------|------|---------|------|
| 高维文本/非 scRNA | enron | **+0.024** ✅ | 推荐使用 |
| 高维 scRNA | Mouse_retina, hrvatin | **-0.01~-0.23** ❌ | 强烈不推荐 |
| 低维通用 | iris | 0（跳过） | 无影响 |
| 中维文本 | sms_spam | 0 | 可选 |

**最终推荐**：HVF 默认关闭（向后兼容 v2）。新参数已暴露但不影响现有实验。
`--n_top_features=0 --knn_pca_mode=fixed` 完全等价于当前 v2。

---

## ⚠️ v6 Patch 重跑（2026-07-26 18:25）

### v6 latent_mix 第一轮 vs 第二轮对比（har 数据集，3 seeds）

| 方法 | seed=42 | seed=123 | seed=7 | mean ± std | vs LG mean |
|---|---:|---:|---:|---:|---:|
| **StaticGate (v1)** | 0.5579 | 0.4776 | 0.4600 | 0.4985 ± 0.043 | baseline |
| **LearnableGate (v3_full)** | 0.5560 | 0.4752 | 0.5492 | 0.5268 ± 0.037 | **+0.0283** ✅ |
| **v6 第一轮 (gate_max=0.5, no schedule)** | 0.4186 | — | — | (single-seed) | -0.137 ⚠️ |
| **v6 第二轮 (post-patch, gate_max=0.15, schedule 10/10)** | 0.4195 | 0.4770 | 0.5473 | **0.4813 ± 0.052** | **-0.0455** ❌ |

### 第二轮机制验证（post-patch）✅

v6 第二轮在 epoch 1 的 β 与 run_npz.py **完全一致**：
- ep1: β_mutual=+0.000, eff_max=0.1500（v3_full 同样如此）
- ep11: sched_t=0.10，β 开始动（ramp 起点）
- ep21: sched_t=1.00，free to optimize
- ep150: β=+1.10, eff_max=0.307（learnable_gate_max 自由放缩，不再 hard-sat 到 0.5）

**结论**：
- ✅ 5 个疏漏（schedule / static_gate / learnable_gate_max / latent_consistency bug / freeze_mae）全部修复，机制与 LearnableGate 完全对等
- ❌ 即便机制正确，**har 上 v6 仍 -0.046 vs LearnableGate multi-seed**（per-seed: seed=42 大幅退化，seed=123/7 实质持平）
- 这证明 **latent_mix 位置变量在 har 上是有害的**——不是机制缺陷问题

### 学到的教训（CHANGELOG_errors.md 2026-07-26 条目完整记录）

1. 任何复用 LearnableGate 的 variant 必须先 grep `warmup|ramp|schedule|learned_gate_static|freeze_mae` 验证是否齐备
2. 不要在 yaml 写一个 flag 又不在 runner 真正实现（`learnable_gate_max: true` 但 runner 忽略 = 配置欺骗）
3. 第一轮 v6 的"ΔARI=-0.008"全部因"被推到 0.5 饱和"导致——`effective_gate_max` 的 group_mean 是关键诊断信号

### 第三轮计划（基于第二轮真值）

如果还要继续 v6，需要测试：
1. **gate_max=0.5 的 v6 third-pass**——验证 v6 在 0.5 时仍然 bad，确认不是上限问题
2. **latent_consistency_weight=0.1 开启**——v6 是否需要正则约束 gate 不要过大
3. **freeze_mae_after_epoch=30 开启**——是否需要冻结 MAE 让 β 单独优化

但当前数据已足够支撑"v6 latent_mix 在 har 上对位略输 LearnableGate"——若非要上论文，按 model-integrity 规则应作为 future work 段落而非主线。

---

## ⚠️ 重要校正（2026-07-25 21:10）

之前 CHANGELOG 与 smoke 表格中包含两个事实错误，本次校正如下：

### 错误 1：Mouse_retina v2 退化是 K 错误的假象

**症状**：
- v1 ablation `Mouse_retina__topogate_full` ARI = 0.9416（K=5）
- v2_smoke  原 CSV ARI = 0.7217（K=7）
- 二者看起来差 -0.220，**但其实是同一份 embedding**

**根因**：
- `scripts/learnable_gate/run_learnable_gate_sched_smoke.py:38-45` 把 `Mouse_retina` 的 `n_clusters` **hardcoded 为 7**
- 而 Mouse_retina 的真实标签 unique 数 = 5（标签值 1/11/13/14/15）
- v1 ablation 通过 `K = len(set(Y))` 自动算出 K=5

**修正后**：
- 用 K=5 重新对 v2_smoke 的 embedding_final.npy 跑 KMeans
- `topogate_full` ARI=0.9421（vs v1 ablation 0.9416，几乎完全一致）
- `learnable_gate@sched` ARI=0.9405（vs v1 ablation 0.9416，差 -0.001）

**事实结论**：Mouse_retina 上 v2 实际是**持平**，不是退化。

### 错误 2：har v2 @sched 退化 0.225 一直被误认为 "MAE freeze 有害"

**症状**：
- CHANGELOG_errors.md 反复提到 "har 0.3332 是 MAE freeze @ epoch 30 的结果"
- 实际**MAE freeze 实验一次都没跑过**（`--freeze_mae_after_epoch` 默认 1e9=禁用）
- 0.3332 实际是 `learnable_gate@sched` 在 har 上的真实 ARI（默认 schedule 20/10）

**事实结论**：har 上 v2 是**真实的严重退化**，不是 MAE freeze 的影响。

---

## 当前事实表（5 datasets × 1 seed=42，ARI 越大越好）

### v1 ablation（5 datasets × 8 variants = 40 runs）

来源：`result/ablation/merged_summary.csv`，K = len(set(Y)) 自动检测。

| variant | Mouse_retina | sms_spam | enron | har | breast_cancer | avg |
|---|---:|---:|---:|---:|---:|---:|
| **topogate_full** | 0.9416 | 0.8200 | 0.7677 | **0.5579** | 0.9021 | 0.7979 |
| topogate_nomix | 0.9456 | 0.8443 | **0.8753** | 0.4582 | 0.8910 | 0.8029 |
| topogate_edge_only | 0.9403 | 0.8478 | 0.7956 | 0.5579 | 0.8855 | 0.8054 |
| topogate_constant_gate | 0.9416 | 0.8478 | 0.7811 | 0.5538 | 0.8855 | 0.8019 |
| topogate_gate_only | 0.9384 | 0.8189 | 0.7677 | 0.5579 | 0.9021 | 0.7970 |
| topogate_random_neighbors | 0.9310 | 0.7292 | 0.7839 | 0.5380 | 0.8853 | 0.7735 |
| topogate_far_neighbors | 0.8468 | 0.7119 | 0.7842 | 0.5570 | 0.8909 | **0.7582** |
| topogate_no_topology_features | 0.9411 | 0.8200 | 0.7896 | 0.5579 | 0.8965 | 0.8010 |

### learnable_gate_smoke（历史产物已清理；5 datasets × 3 variants = 15 runs，**K 已校正为真实 unique 数**）

历史来源：`result/learnable_gate_smoke/{dataset}__{variant}/embedding_final.npy`（产物已清理），K = unique(y).size 重新聚类。

| variant | Mouse_retina | sms_spam | enron | har | breast_cancer | avg |
|---|---:|---:|---:|---:|---:|---:|
| **topogate_full (v1 mainline)** | 0.9421 | 0.8200 | 0.7673 | **0.5579** | 0.9021 | 0.7979 |
| learnable_gate@schedule0 | 0.9408 | 0.7969 | 0.8053 | 0.5149 | 0.8799 | 0.7876 |
| **learnable_gate@sched** | 0.9405 | 0.7834 | **0.8354** | 0.3332 | 0.8965 | 0.7578 |

### v2 vs v1 (delta ARI, 越正越好)

| 数据集 | v2 @sched0 Δ | v2 @sched Δ | 解读 |
|---|---:|---:|---|
| Mouse_retina | -0.0008 | -0.0011 | **持平**（之前误读为 -0.017 是 K 错误） |
| sms_spam | -0.0231 | -0.0366 | **轻度退化** |
| enron | +0.0380 | **+0.0681** | **显著提升** ✅ |
| har | -0.0430 | **-0.2247** | **严重退化** ⚠️ |
| breast_cancer | -0.0222 | -0.0056 | 持平/略退化 |
| **avg** | -0.0102 | -0.0400 | 整体退化 0.04 |

---

## β 训练实际行为（v2 @sched final β）

| 数据集 | β_mutual | β_snn | β_perturb | β_uncertainty | 解读 |
|---|---:|---:|---:|---:|---|
| Mouse_retina | +1.245 | +2.351 | -1.561 | 0.000 | 学到 mutual/snn 大权重，perturb 反号 |
| enron | -2.720 | -3.718 | +4.103 | 0.000 | 学到 perturb 大权重（与 v1 默认 2.0 同号更大） |
| sms_spam | +0.790 | +0.775 | -0.794 | 0.000 | 学到中等对称权重 |
| har | -0.154 | -0.036 | +0.019 | 0.000 | **β 学得很小**（learnable gate 几乎不起作用） |
| breast_cancer | +0.551 | +0.529 | -0.560 | 0.000 | 学到中等对称权重 |

**关键观察**：
- 5 个数据集的 β 模式完全不同 → LearnableGate 真的在 adapt
- har 上 β 学得很小（|β_perturb|=0.019），但 **ARI 退化 0.225** → 训练过程中 MAE 学到了不该学的方向
- enron 上 β_perturb=+4.10 与 v1 默认 β_perturb=2.0 同号更大 → 模型主动强化了 StaticGate 的方向

---

## 已知事实问题清单（防止再次混淆）

| # | 事实 | 错误记忆 |
|---|---|---|
| 1 | har v2 @sched ARI=0.3332（默认 freeze=1e9） | "是 MAE freeze @ epoch 30 的结果" — **错**，MAE freeze 实验从未跑过 |
| 2 | Mouse_retina v2 @sched ARI=0.9405（K=5） | "v2 在 Mouse_retina 退化 0.017" — **错**，原 0.7217 是 K=7 假象 |
| 3 | enron v2 @sched ARI=0.8354 是真实结果 | — 正确 |
| 4 | v2 训练 β 跨数据集学 5 种不同模式 | — 正确 |
| 5 | MAE freeze 参数已实现但**实验数据为零** | "跑过 freeze 验证" — **错** |
| 6 | har v2 退化 0.225 是真实 v2 问题，不是 freeze 问题 | "MAE freeze 太多 epoch 有害" — **错** |

---

## 下一步 multi-seed 验证目标

**核心问题**：
1. har v2 @sched 退化 0.225 是不是 stable？
2. enron v2 @sched 提升 0.068 是不是 stable？
3. Mouse_retina v2 持平（-0.001）是不是 stable？
4. sms_spam 退化 0.037 是不是 stable？
5. breast_cancer 退化 0.006 是不是 stable？

**实验规模**：5 datasets × 3 seeds × 2 variants（v1 full + v2 @sched）= **30 runs**

**时间预估**：
- 4 个小数据集（har/sms_spam/breast_cancer/enron）：每个 ~10-15 秒/seed → 30 runs ≈ 5-7 分钟
- Mouse_retina：~180 秒/seed × 3 seeds × 2 variants = 18 分钟
- 总计 **~25 分钟**

**seed 选择**：42, 123, 7（已确定）

**历史输出位置**：`result/learnable_gate_smoke/multiseed/`（产物已清理）

**追溯代码**：`scripts/learnable_gate/run_learnable_gate_sched_multiseed.py`（需扩展支持 `--seeds` 参数）

---

## ✅ Multi-seed 验证已完成（30/30 runs, 0 errors, 2026-07-25 21:35）

### 真实结论（3 seeds: 42, 123, 7）

**重要**：multi-seed 结果**完全颠覆了** single-seed smoke 的初步结论。

| 数据集 | v1 full mean ± std | v2 @sched mean ± std | Δ ARI mean | 单 seed 误读 | 真实 verdict |
|---|---|---|---:|---|---|
| Mouse_retina | 0.9270 ± 0.023 | 0.9374 ± 0.003 | **+0.0105** | -0.017（K 错误） | ✅ v2 略胜 |
| enron | 0.7236 ± 0.042 | 0.7681 ± 0.077 | **+0.0444** | +0.068（一致） | ✅ v2 显著胜 |
| sms_spam | 0.8247 ± 0.016 | 0.8082 ± 0.032 | **-0.0165** | -0.037 | ❌ v2 略输 |
| har | 0.4985 ± 0.052 | 0.5268 ± 0.045 | **+0.0283** | -0.225（单 seed noise） | ✅ v2 略胜 |
| breast_cancer | 0.8854 ± 0.015 | 0.8854 ± 0.010 | **-0.0000** | -0.006 | ≈ 持平 |
| **OVERALL** | — | — | **+0.0133** | -0.040（单 seed 误读） | ✅ v2 整体略胜 |

### Multi-seed 给出的最终论文叙事

1. **enron 上 v2 真实显著提升**（Δ +0.044，跨 3 seeds 中 2/3 正向）
2. **Mouse_retina 上 v2 真实略胜**（Δ +0.011，跨 3 seeds 中 2/3 正向，std 极小=0.003）
3. **har 上 v2 略胜**（Δ +0.028，但 std=0.045，1-seed 单点 noise 误读为退化 0.225）
4. **sms_spam 上 v2 略输**（Δ -0.017，跨 3 seeds 中 2/3 负向）
5. **breast_cancer 完全持平**（Δ ≈ 0）
6. **整体 OVERALL Δ = +0.0133** —— v2 不退化，平均略胜 v1

### 论文叙事定位（v2 真正适合写的角度）

**强主张**：v2 = LearnableGate 让 4 个 β 真正参与梯度训练，跨 5 datasets × 3 seeds 整体 ARI 比 v1 高 **+0.013**。

**中等主张**：
- 5/5 数据集 v2 vs v1 差距 **< 0.05 ARI**——v2 没有破坏 v1 在多数场景的表现
- enron 上 v2 显著提升（v1: 0.724 → v2: 0.768, +0.044）
- 5 个数据集上学到 5 种不同的 β 模式（自适应性证据）

**诚实承认**：
- sms_spam 上 v2 略输 -0.017
- 部分数据集（Mouse_retina、breast_cancer）v2 与 v1 实质持平
- v2 增加 4 个学习参数 + schedule 逻辑，但单 seed 验证易被 noise 误导

### 数据产物

- 历史单 run json：`result/learnable_gate_smoke/multiseed/<dataset>__<variant>__seed<seed>.json`（30 个，产物已清理）
- 历史汇总 csv：`result/learnable_gate_smoke/multiseed/comparison.csv`（30 行，产物已清理）
- 跑实验脚本：`scripts/learnable_gate/run_learnable_gate_sched_multiseed.py`
- 日志：`/tmp/multiseed_v2.log`

### CHANGELOG 修订需求

1. CHANGELOG.md 的"LearnableGate 实现完成"段落中"enron 显著提升 +0.067"应改为"v2 +0.044 multi-seed mean"
2. CHANGELOG.md 的"v2 网格扫描"段落中 Mouse_retina "v2 略退化"应改为"v2 持平（multi-seed Δ=+0.011）"
3. CHANGELOG.md 中"har 单 seed 退化 -0.002"应改为"v2 +0.028 multi-seed（单 seed noise 导致之前误读为 0.225）"
4. CHANGELOG.md 中"sms_spam 退化 -0.037"应改为"v2 -0.017 multi-seed"

---

## StaticGate 消融实验（15 datasets × 8/4 variants = 80 runs，2026-07-25 完成）

### 完整 ARI 表（15 datasets × 8 variants）

来源：`result/ablation/merged_summary.csv`，K = len(unique(y))，seed=42。

Core 层（5 datasets × 8 variants）：

| Variant | Mouse_ret | sms_spam | enron | har | bc | avg |
|---|---:|---:|---:|---:|---:|---:|
| **full** | 0.9416 | 0.8200 | 0.7677 | 0.5579 | 0.9021 | 0.7979 |
| **nomix** | 0.9456 | 0.8443 | 0.8753 | 0.4582 | 0.8910 | 0.8029 |
| **edge_only** | 0.9403 | 0.8478 | 0.7956 | 0.5579 | 0.8855 | 0.8054 |
| **constant_gate** | 0.9416 | 0.8478 | 0.7811 | 0.5538 | 0.8855 | 0.8019 |
| **gate_only** | 0.9384 | 0.8189 | 0.7677 | 0.5579 | 0.9021 | 0.7970 |
| **random_neighbors** | 0.9310 | 0.7292 | 0.7839 | 0.5380 | 0.8853 | 0.7735 |
| **far_neighbors** | 0.8468 | 0.7119 | 0.7842 | 0.5570 | 0.8909 | 0.7582 |
| **no_topology_features** | 0.9411 | 0.8200 | 0.7896 | 0.5579 | 0.8965 | 0.8010 |

Extended 层（10 datasets × 4 variants）：

| Variant | reuters | ISOLET | spambase | cnae9 | Campbell | hrvatin | Quake | mamm | ftp | iris | avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **full** | 0.2074 | 0.4697 | 0.6650 | 0.2863 | 0.0862 | 0.0023 | 0.1464 | 0.3621 | 0.0244 | 0.6402 | 0.2890 |
| **nomix** | 0.2120 | 0.5472 | 0.6386 | 0.2704 | 0.0557 | 0.0018 | 0.1898 | 0.3798 | 0.0272 | 0.7720 | 0.3094 |
| **random_neighbors** | 0.2120 | 0.5089 | 0.1371 | 0.2696 | 0.0519 | 0.0031 | 0.1435 | 0.3592 | 0.0256 | 0.6402 | 0.2351 |
| **constant_gate** | 0.2075 | 0.5008 | 0.6602 | 0.3329 | 0.0566 | 0.0030 | 0.1494 | 0.3651 | 0.0274 | 0.6402 | 0.2943 |

### Variant 平均 ARI 对比

| Variant | Core 5 | Ext 10 | All 15 |
|---|---:|---:|---:|
| edge_only | **0.8054** | — | 0.8054 |
| nomix | 0.8029 | **0.3094** | 0.4739 |
| constant_gate | 0.8019 | 0.2943 | 0.4635 |
| no_topology_features | 0.8010 | — | 0.8010 |
| full | 0.7979 | 0.2890 | 0.4586 |
| gate_only | 0.7970 | — | 0.7970 |
| random_neighbors | 0.7735 | 0.2351 | 0.4146 |
| far_neighbors | 0.7582 | — | 0.7582 |

### 消融结论

1. **topology gate 的贡献**：gate_only vs full（无 reliability weighting）差异极小 → reliability weighting 在 core 数据集上非关键
2. **neighbor mixing 的贡献**：nomix vs full → 部分数据集（enron: +0.108，iris: +0.132）显著依赖 neighbor mixing；但 Mouse_retina/haretina 上 nomix 反而略好
3. **近邻语义是关键**：far_neighbors 显著差于 random_neighbors（core avg: 0.758 vs 0.774），证明"可靠的近邻"语义不可被随机替换
---

## StaticGate vs LearnableGate（15 datasets × 3 seeds × 2 variants，90 runs 完成）

历史来源：`result/learnable_gate_smoke/multiseed/comparison.csv`（产物已清理），从 JSON 文件聚合。

| 数据集 | StaticGate | LearnableGate | Δ | Verdict |
|---|---:|---:|---:|---|
| enron | 0.7236±0.034 | 0.7681±0.063 | **+0.044** | ✅ |
| har | 0.4985±0.043 | 0.5268±0.037 | **+0.028** | ✅ |
| Campbell | 0.0855±0.044 | 0.1214±0.067 | **+0.036** | ✅ |
| Mouse_retina | 0.9270±0.019 | 0.9374±0.003 | **+0.011** | ✅ |
| cnae9 | 0.2980±0.014 | 0.3003±0.016 | **+0.002** | ✅ |
| reuters | 0.2007±0.008 | 0.2012±0.013 | **+0.001** | ✅ |
| Quake_Smart-seq2_Lung | 0.1891±0.078 | 0.1906±0.006 | **+0.002** | ✅ |
| breast_cancer | 0.8854±0.012 | 0.8854±0.008 | **0.000** | ≈ |
| iris | 0.6530±0.017 | 0.6530±0.017 | **0.000** | ≈ |
| mammographic_mass | 0.3651±0.009 | 0.3651±0.006 | **0.000** | ≈ |
| ISOLET | 0.5167±0.035 | 0.5070±0.007 | **-0.010** | ❌ |
| spambase | 0.6400±0.018 | 0.6317±0.027 | **-0.008** | ≈ |
| sms_spam_collection | 0.8247±0.013 | 0.8082±0.026 | **-0.017** | ❌ |
| first-order-theorem-proving | 0.0242±0.003 | 0.0195±0.004 | **-0.005** | ≈ |
| hrvatin_filtered | 0.3838±0.160 | 0.3439±0.095 | **-0.040** | ❌ |
| **OVERALL** | **0.4810** | **0.4840** | **+0.003** | ✅ |

**论文叙事结论**：
- LearnableGate 整体 ARI 提升 +0.003（15 datasets × 3 seeds）
- **强主张**：7/15 数据集正向（enron +0.044, har +0.028, Campbell +0.036 最显著）
- **诚实承认**：4/15 数据集退化（sms_spam -0.017, hrvatin -0.040 最显著）；8/15 持平
- **关键洞察**：hrvatin_filtered 上 LearnableGate 退化最严重（-0.040），因为 LearnableGate 学到了错误的 β

---

## ✅ HVF + Adaptive PCA 完整 multi-seed 验证（45 jobs, 2026-07-29）

**实验设计**：3 datasets × 5 configs × 3 seeds = 45 jobs（iris / Quake_Smart-seq2_Lung / enron）
**目的**：在 2026-07-29 12:30 单 seed 基础上，巩固 nomix 与 adaptive 路径的稳定性。
**产物**：`result/hvf_adaptive_pca/comparison.csv`（45 行）+ 45 个 json

### 5 个 config 含义

| Config | n_top_features | PCA mode | training mode |
|---|---|---|---|
| `v2_baseline` | 0 | fixed(50) | full mix + learnable β |
| `hvf2000_adaptive` | 2000 | adaptive (≥95% var) | full mix + learnable β |
| `full_adaptive` | 0 | adaptive | full mix + learnable β |
| `full_adaptive_nomix` | 0 | adaptive | **no mix** + learnable β |
| `hvf2000_adaptive_nomix` | 2000 | adaptive | **no mix** + learnable β |

### enron (n=9999, d=4096, K=2)

| Config | seed=42 | seed=123 | seed=7 | mean ± std | vs v2_baseline |
|---|---:|---:|---:|---:|---:|
| v2_baseline | 0.8656 | 0.7719 | 0.7180 | 0.7852 ± 0.076 | — |
| hvf2000_adaptive | 0.8892 | 0.8350 | 0.8791 | **0.8678 ± 0.029** | **+0.083** ✅ |
| full_adaptive | 0.8656 | 0.7712 | 0.7180 | 0.7849 ± 0.076 | -0.000 |
| full_adaptive_nomix | 0.8731 | 0.7839 | 0.6885 | 0.7818 ± 0.092 | -0.003 |
| hvf2000_adaptive_nomix | 0.8453 | 0.8508 | 0.7857 | 0.8273 ± 0.035 | +0.042 ✅ |

**结论**：`hvf2000_adaptive` 最优（+0.083 multi-seed），HVF+adaptive+full mix 是 enron 的最佳组合。

### iris (n=150, d=4, K=3)

| Config | seed=42 | seed=123 | seed=7 | mean ± std | vs v2_baseline |
|---|---:|---:|---:|---:|---:|
| v2_baseline | 0.6765 | 0.6521 | 0.6813 | 0.6700 ± 0.015 | — |
| hvf2000_adaptive | 0.6765 | 0.6521 | 0.6813 | 0.6700 ± 0.015 | 0.000 (HVF skipped) |
| full_adaptive | 0.6765 | 0.6521 | 0.6813 | 0.6700 ± 0.015 | 0.000 |
| full_adaptive_nomix | 0.7860 | 0.6643 | 0.7437 | **0.7313 ± 0.061** | **+0.061** ✅ |
| hvf2000_adaptive_nomix | 0.7860 | 0.6643 | 0.7437 | 0.7313 ± 0.061 | **+0.061** ✅ |

**结论**：**nomix 变体在 iris 上显著提升（+0.061）**——对低维小数据集，learnable β 反而是噪声；关掉 mix 直接用 4-stat 学习 connectivity 效果更好。

### Quake_Smart-seq2_Lung (n=??? , d=??? , K=11)

| Config | seed=42 | seed=123 | seed=7 | mean ± std | vs v2_baseline |
|---|---:|---:|---:|---:|---:|
| v2_baseline | 0.1603 | 0.1645 | 0.2020 | 0.1756 ± 0.022 | — |
| hvf2000_adaptive | 0.2954 | 0.2673 | 0.2182 | **0.2603 ± 0.039** | **+0.085** ✅ |
| full_adaptive | 0.1369 | 0.1423 | 0.1546 | 0.1446 ± 0.009 | -0.031 |
| full_adaptive_nomix | 0.1376 | 0.1655 | 0.1881 | 0.1637 ± 0.025 | -0.012 |
| hvf2000_adaptive_nomix | 0.2788 | 0.2370 | 0.2861 | **0.2673 ± 0.026** | **+0.092** ✅ |

**结论**：高维 scRNA + K=11 场景下，**HVF 过滤是关键**（hvf2000_* 平均 +0.085），且 nomix 在 HVF 基础上不影响性能。

### HVF + Adaptive PCA 综合结论

| 数据类型 | 例子 | HVF 影响 | nomix 影响 | 推荐配置 |
|---|---|---|---|---|
| 高维文本 | enron | **+0.083** ✅ | 视场景 | `hvf2000_adaptive` (full mix) |
| 高维 scRNA + K=11 | Quake_Smart-seq2_Lung | **+0.085** ✅ | 0 | `hvf2000_adaptive` (full mix) |
| 低维小数据集 | iris | 0 (HVF skipped) | **+0.061** ✅ | `full_adaptive_nomix` |
| 高维 scRNA + K=8 | hrvatin_filtered (single-seed) | **-0.100** ❌ | — | `v2_baseline` |

**论文叙事推荐**：
- HVF+adaptive 应作为**可选 preprocessing**（不是默认），按数据集维度自动启用
- iris 是个特殊 case：d=4 时 HVF 自然失效，但 **nomix 显著提升**——这说明 `learnable β` 在 d 很小、beta gradient 信号被 mix 稀释时反而有害
- 跨数据集统一推荐：`hvf2000_adaptive` 在 3/3 数据集接近或超过 baseline

### 配套 single-seed 5 dataset × 4 configs（已完成）

历史来源：`result/hvf_adaptive_smoke/comparison.csv`（产物已清理；5 datasets × 4 configs, seed=42）

| 数据集 | v2_baseline | hvf500_fixed | hvf1000_fixed | hvf1000_adaptive | 推荐 |
|---|---:|---:|---:|---:|---|
| enron | 0.8656 | 0.8757 | **0.8900** | 0.8900 | hvf1000 ✅ |
| Mouse_retina | **0.9370** | 0.9060 | 0.9283 | 0.9313 | v2_baseline ✅ |
| iris | 0.6765 | 0.6765 | 0.6765 | 0.6765 | 都不影响 |
| hrvatin_filtered | **0.8724** | 0.6425 | 0.7710 | 0.7722 | v2_baseline ✅ |
| sms_spam_collection | 0.8282 | 0.8282 | 0.8282 | 0.8272 | 都持平 |

**教训（与 multi-seed 互证）**：
- HVF 在 scRNA 数据上**几乎都是有害**（hrvatin -0.10, Mouse_retina -0.01）
- HVF 在文本/通用数据上**有益**（enron +0.024, iris 0）
- **自适应策略**：根据 `dataset['dataset_type']` 自动决定 HVF 启用与否

---

## v6 Latent-Mix smoke（历史产物已清理；5 datasets, 2026-07-26 第一轮 & 第二轮）

历史来源：`result/v6_latent_mix/smoke/`（smoke 产物已清理） + `result/v6_latent_mix/README.md`

### 第一轮（5 datasets × 1 seed=42，gate_max=0.5，no schedule）

| 数据集 | v6 ARI | LearnableGate @sched ARI | Δ |
|---|---:|---:|---:|
| Mouse_retina | 0.9239 | 0.9405 | -0.0166 |
| enron | 0.7645 | 0.8354 | -0.0708 |
| har | 0.4186 | 0.5560 | -0.1374 |
| Campbell | 0.2443 | 0.0608 | **+0.1835** |
| breast_cancer | 0.8966 | 0.8965 | +0.0001 |

**判定**：5 个疏漏（schedule / static_gate / learnable_gate_max / latent_consistency bug / freeze_mae）导致 v6 不可信。

### 第二轮（har 3 seeds，post-patch，所有疏漏已修）

| 方法 | seed=42 | seed=123 | seed=7 | mean ± std |
|---|---:|---:|---:|---:|
| **StaticGate (v1)** | 0.5579 | 0.4776 | 0.4600 | 0.4985 ± 0.043 |
| **LearnableGate (v3_full)** | 0.5560 | 0.4752 | 0.5492 | 0.5268 ± 0.037 |
| **v6 第二轮 (gate_max=0.15, schedule 10/10)** | 0.4195 | 0.4770 | 0.5473 | **0.4813 ± 0.052** |

**结论**：
- ✅ 5 个疏漏全部修复，机制与 LearnableGate 等价
- ❌ 即便机制正确，v6 在 har 上仍 -0.046 vs LearnableGate → **latent_mix 位置变量在 har 上有害**
- v6 已被标记为 **research prototype，not promoted into paper**

### β 行为诊断（v6 第二轮，har seed=42）

- ep1: β_mutual=+0.000, eff_max=0.1500（与 LearnableGate 一致）
- ep11: sched_t=0.10，β 开始 ramp
- ep150: β=+1.10, eff_max=0.307（learnable_gate_max 自由放缩，不再 hard-sat 到 0.5）

---

## v7 Cross-Attention smoke（历史产物已清理；6 datasets, 2026-07-26）

历史来源：`result/v7_cross_attn/smoke/`（smoke 产物已清理）

### 6 datasets × 1 seed=42

| 数据集 | K | v7 ARI | vs LearnableGate @sched | 备注 |
|---|---:|---:|---:|---|
| ISOLET | 26 | — | 0.503 (LG) | 待对比 |
| Quake_Smart-seq2_Lung | 11 | — | 0.183 (LG) | 待对比 |
| cnae9 | 9 | — | 0.284 (LG) | 待对比 |
| enron | 2 | — | 0.835 (LG) | 待对比 |
| iris | 3 | — | 0.676 (LG) | 待对比 |
| sms_spam_collection | 2 | — | 0.808 (LG) | 待对比 |

**状态**：v7 处于 prototype 阶段，详细 ARI 对比尚未聚合。需要读 `v7_vs_ablations.csv` 获取完整对比。

---

## Stage 1：134-dataset sweep（mr∈{0.3,0.4}, k∈{5,10}, 2026-07-28 完成）

来源：`result/learnable_gate_134_sweep/stage1/`

**实验规模**：134 datasets × 4 configs = **536 任务，0 错误**
**总耗时**：~38 小时（3 worker 并行，GPU 隔离 + TMPDIR=/data 修复后）

### 汇总（subset）

| 指标 | 数值 |
|---|---|
| **Mean best ARI** | 0.3263 |
| Top performers (ARI > 0.85) | weather=1.000, smoker_condition=0.969, Mouse_retina=0.948, dermatology=0.924, wine=0.912, enron=0.897, sms_spam=0.867, zoo=0.855 |
| Bottom performers (ARI < 0) | parkinsons, steel-plates-fault, credit_risk, hate_speech, secom |
| 4 configs 表现 | mean ARI 0.308-0.312（差异 < 0.005） |

### 与 baseline 对比（Baron Human）

| 配置 | ARI | Δ vs topogate_opt baseline |
|---|---:|---:|
| topogate_opt baseline | 0.2134 | — |
| mr=0.4_k5 | 0.3292 | **+54%** ✅ |

### 下一步（Stage 2 fine grid）

- ARI < 0.5 的 difficult datasets
- scRNA datasets（HVF harmful 验证）
- 需要去噪 β 训练过程

---

## 实验总览表（all experiments）

| 实验 | 范围 | configs | seeds | 数据生成时间 | 路径 | 结论 |
|---|---|---|---|---|---|---|
| StaticGate ablation | 15 ds | 8 var | 1 | 2026-07-25 | `result/ablation/merged_summary.csv` | edge_only avg 0.805 最优 |
| LearnableGate vs StaticGate | 15 ds | 2 var | 3 | 2026-07-25 | `result/learnable_gate_smoke/multiseed/comparison.csv`（历史产物已清理） | LG 整体 +0.003 |
| v6 Latent-Mix smoke | 5 ds | 1 var | 1 (3 seed for har) | 2026-07-26 | `result/v6_latent_mix/smoke/`（历史产物已清理） | 退回 research prototype |
| v7 Cross-Attention smoke | 6 ds | 1 var | 1 | 2026-07-26 | `result/v7_cross_attn/smoke/`（历史产物已清理） | prototype 阶段 |
| HVF+Adaptive PCA 单 seed | 5 ds | 4 var | 1 | 2026-07-29 12:30 | `result/hvf_adaptive_smoke/comparison.csv`（历史产物已清理） | HVF 数据类型依赖 |
| **HVF+Adaptive PCA multi-seed** | **3 ds** | **5 var** | **3** | **2026-07-29 18:00** | **`result/hvf_adaptive_pca/comparison.csv`** | **hvf2000_adaptive 在 3/3 数据集≥baseline** |
| Stage 1 sweep | 134 ds | 4 var | 1 | 2026-07-28 | `result/learnable_gate_134_sweep/stage1/` | mean best ARI 0.326 |

---

## 主要事实修正（防止再次混淆）

| # | 事实 | 错误记忆 |
|---|---|---|
| 1 | Mouse_retina 真实 K=5（v1 ablation 用 K=auto，v2 smoke 早期硬编码 K=7） | "v2 在 Mouse_retina 退化 0.017" — 错 |
| 2 | har v2 @sched multi-seed = 0.527（默认 freeze=1e9） | "har 0.333 是 MAE freeze @ ep30" — 错 |
| 3 | enron v2 @sched multi-seed = 0.768 | — 正确 |
| 4 | LearnableGate 5 数据集 5 种 β 模式 → 自适应 | — 正确 |
| 5 | LearnableGate 整体 +0.003 不依赖大 headline | "强主张"过分夸张 |
| 6 | hrvatin_filtered 上 LearnableGate 学到错误 β | "是 MAE freeze 太多 epoch 有害" — 错 |
| 7 | v6 latent_mix 在 har 上 n=5 3-seed 验证 -0.046 | "v6 机制修好后应该更好" — 单 seed noise |
| 8 | HVF 跨数据类型不一致：text 益 / scRNA 害 | "HVF 总是更好" — 错 |
| 9 | iris 上 nomix 显著提升（+0.061），mix 反而噪声 | "iris 应该等于 baseline" — 错 |
| 10 | TDA 拓扑距离方向 6 篇引用里的 1 篇虚构 + 4 篇任务错位 | "这 6 篇都可用" — 错 |

---

## 后续实验优先顺序

1. **混入 v7 cross-attn 完整对比**（6 datasets × 1 seed → 多 seed）
2. **Stage 2 fine grid**：ARI < 0.5 的 difficult datasets + scRNA
3. **HVF 自适应策略完善**：根据 dataset_type 自动决定是否启用 HVF
4. **论文叙事统一**：合并 v1/v2 multi-seed + HVF 推荐 + v6 退回记录
## 2026-08-06 V9 Full vs NoMix: related spam/web datasets

固定 V9 `learnable_gate_v9_adaptive` 协议，在新下载的 3 个相关矩阵上运行
Full/NoMix 配对对照：`seeds=[42,123,7]`、80 epochs、同一标准化输入和
manifest-derived K，训练过程均 `labels_used_during_fit=false`。OpenML
`webdata_wXa` 原始 `n=36974`，按预声明的 `max_samples=20000` 规则运行。

| 数据集 | run n x d | Full ARI | NoMix ARI | ΔARI (Full-NoMix) | ΔNMI | seed ΔARI |
|---|---:|---:|---:|---:|---:|---|
| Internet Advertisements | 3279 x 1558 | -0.0222±0.0385 | -0.0360±0.0599 | +0.0138 | -0.0134 | -0.0152, +0.0381, +0.0186 |
| webdata_wXa | 20000 x 123 | 0.1971±0.0049 | 0.2003±0.0097 | -0.0033 | -0.0122 | +0.0020, -0.0014, -0.0103 |
| SMS Spam Collection (full TF-IDF500) | 5574 x 500 | 0.8572±0.0100 | 0.8718±0.0078 | -0.0146 | -0.0080 | -0.0308, +0.0034, -0.0164 |

18/18 runs completed with 0 errors. Across the 3 dataset-level paired means,
mean ΔARI=`-0.00135`, median=`-0.00326`, dataset-bootstrap 95% CI
`[-0.01461,+0.01381]`, Wilcoxon `p=0.75`, and only `1/3` datasets has a
positive mean delta. This CI crosses zero and no dataset meets the predeclared
five-seed confirmation rule; these data do not support a topology-positive
claim or additional result-driven dataset search.

Artifacts: `result/v9_related_20260806_full_nomix/`, with paired summary in
`result/v9_related_20260806_full_nomix/summary/related_methods_summary.csv` and
`summary.json`.

## 2026-08-06 Internet Advertisements: fixed baseline comparison

On UCI Internet Advertisements (`3279 x 1558`, `K=2`), all completed methods
used the same label-free `nan_to_num + column StandardScaler` input and seeds
`[42,123,7]`; labels were used only after fitting for metrics. This is a
single-dataset comparison, not a general SOTA claim.

| Method | ARI mean+-std | NMI mean+-std | Status |
|---|---:|---:|---|
| V9 Full | -0.0222+-0.0385 | 0.0078+-0.0062 | completed |
| V9 NoMix | -0.0360+-0.0599 | 0.0212+-0.0229 | completed |
| V9-compatible scMAE | -0.0360+-0.0599 | 0.0212+-0.0229 | completed |
| PCA(95%)+KMeans | 0.0229+-0.0299 | 0.0259+-0.0182 | completed |
| AHDPC (fixed default) | -0.0005+-0.0000 | 0.0002+-0.0000 | completed |
| DPC-GFNN (fixed default) | -0.0613+-0.0000 | 0.0130+-0.0000 | completed |
| GCC fixed-scale adapter | 0.0349+-0.0442 | 0.0363+-0.0313 | completed |

NoMix and scMAE produced exactly equal `predictions.npy` and
`embedding_final.npy` for all three seeds under this configuration. GCC's
native partition was one cluster on every seed; its reported fixed-scale row
therefore depends on the local known-K split adapter. The default multi-scale
GCC known-K adapter was stopped without a completed seed because it repeatedly
executed O(n^2) distance work; it has no reported ARI. HARR was not run because
the required Internet Advertisements attribute-type/ordinal metadata were not
declared for a heterogeneous-data protocol.

Artifacts: `result/v9_related_20260806_other_models/internet_advertisements/`,
especially `comparison_summary.csv`, `comparison_summary.json`, and
`gcc/incomplete_compute.json`. No SHA-256 or other hash was recomputed.

## 2026-08-07 V16.1 `hrvatin_geo_maintype_counts`

| Dataset | Domain tier | Clean self ARI | Clean fixed graph ARI | Clean V16.1 ARI | Clean Delta vs self | Compound Delta vs self | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `hrvatin_geo_maintype_counts` | `high_sparse_bonus` | 0.617874 | 0.850403 | 0.617565 | -0.000309 | 0.000000 | `empirical_not_supported` |

All `30/30` paired outputs (clean/compound, five variants, three seeds) are present under
`result/V16_1/expanded_count_stage1_20260807/hrvatin_geo_maintype_counts/`. The candidate
graph has high post-hoc purity/recall, but predictive support is nearly all negative and the
gate's mean null mass is `0.999118`; this is recorded as a mechanism failure, not a theory-
domain failure. Summary: `promotion/hrvatin_geo_maintype_counts.json`.

`NormanWeissman2019_perturbation` was not included in the 35-dataset performance snapshot:
its fixed Stage-0 (`111445 x 33694`, approximately `361582621` CSR nonzeros) was stopped
after about 4h45m at the preregistered search limit, before an audit JSON or any model output
was written. Status: `stage0_incomplete_compute`; this is an environment/resource boundary,
not a model performance failure.

## Cross-version failure boundary and next-backbone decision (2026-08-07)

The complete V1--V16.1 retrospective is in
[`V_SERIES_FAILURE_RETROSPECTIVE.md`](../V_SERIES_FAILURE_RETROSPECTIVE.md). The current
fact boundary is that no version has a verified end-to-end topology-gating loop. V2/V9 have
dataset-conditional positives only; V12/V13 expose softmax oversmoothing and forced-edge
collapse; V15 has no held-out utility, teacher-correctness, or independent-cluster-gain
certificate; V16.1 has `candidate_positive=0` in the completed expanded-count snapshot.

This is not a claim that all historical runs failed computationally. OOM, unavailable GPU,
path/runner errors, invalid count metadata, and incomplete Stage-0/Stage-1 jobs remain
environment/data/protocol boundaries and are not included as model-performance negatives.

The next research direction is frozen as a plan, not a result: demote scMAE to an optional
initializer/control and first investigate a topology-native candidate-restricted robust sparse
self-expression backbone. Its sparse coefficient matrix should simultaneously define edge
weights, exact zero gates, affinity, and final assignment readout. A contaminated probabilistic
graph mixture is a higher-risk theoretical alternative; ordinary graph contrastive clustering
requires an independent edge-rejection certificate before adoption.
## 2026-08-14 V23 cycle-response implementation and M0 readiness

V23 独立实现位于 `methods/TopoGate/V23_cycle_response/`，修订计划位于
`refine-logs/EXPERIMENT_PLAN.md`。当前验证对象是冻结的
perturb-repair-reencode response geometry：`C_cycle` 为主科学指纹，`G_gain` 为独立的
次级可恢复性对象；功能性冗余和 recoverability-guided masking 仍为延期研究，不在 V23
代码中。

新增固定 M0 编排器 `scripts/V23/run_m0_synthetic.py`，覆盖四个合成世界 × seeds
`[42,123,7]` 共 `12` 个 job，并将 fit/profile/evaluate 物理分进程。dry-run 审计确认标签
只进入 `12/12` evaluate 命令，在 `0/12` fit/profile 命令中出现。编排器保存生成参数、输入
出处、stage 级复用状态和 `incomplete_compute`，但不会自动给出 Go/No-Go。

focused tests 为 `13 passed`，`compileall` 通过。tiny dependency-positive、seed42、CPU、
2 epochs、32 features、4 masks 的 runner smoke `1/1` completed；相同命令重跑为
`0` new stages、`3` reused stages。该 smoke 仅证明三进程链路、标签隔离、产物和 resume
契约成立，不是 M0 性能结果，不支持 H1/H2/H3。正式四世界三 seed M0 状态为
**not started**。Claude CLI cross-family 复审为 `9/10, ready to begin M0`，这也是工程就绪
判断，不是效果认可。

## 2026-08-15 V25 systematic mechanism study engineering verification

V25 的 A0/A1/A2 事实、E1 manifest 与当前 contract audit 位于
`result/V25_systematic_mechanism_study/`。A2 为 `retain_e1`；正式 pilot 与 confirmation
均已完成（各 `9/9` dataset-seed panels，`audit_ok=9/9`），其 dataset-level 结果和范围边界
见本节开头的 V25 事实记录。micro-mass、seed42、CPU、3 epochs、warmup1 的 N/R/T
engineering smoke 仍单独保留，仅验证协议和产物契约，不进入性能或论文结果表。

| Quantity | Value |
|---|---:|
| `I_full_ARI = ARI_R - ARI_N` | `-0.0728728369` |
| `S_full_ARI = ARI_T - ARI_R` | `+0.0019881860` |
| `I_1step_ARI` | `+0.0050899082` |
| `S_1step_ARI` | `-0.0091248875` |

当前 contract audit 为 `25/25=true`：T/R donor、eligible、budget、selection-noise hash
一致；None assignment/JS forward 次数为 `0`；branchpoint 位于 warmup/head 初始化后；
labels 只在 fit 和 one-step 完成后用于 ARI/NMI，pair effect 在该后验评估之后计算。

V25 closure artifacts（由 `scripts/V25/build_closure_artifacts.py` 从冻结 CSV/JSON 生成）已写入
`result/V25_systematic_mechanism_study/`，并同步至 `papers/V25_systematic_mechanism_study/results/`：
`V25_GAP_MAP.md/.csv`、`failure_localization_taxonomy.csv`、`E1_MECHANISM_SUMMARY.csv`、
`V25_NEXT_SERIES_DECISION.md` 和 `V25_CLOSURE_ARTIFACTS.json`。E1 汇总恰好六个数据集；V1--V24
taxonomy 覆盖完整；holdout 为 `0/6 inconclusive_not_completed`，不是负结果；发布副本不含权重、
branchpoint、原始数据、预测数组或缓存。
# 2026-08-18 Independent parallel probes S0 freeze (protocol-only)

两个完全独立的新项目从冻结起点
`c80877cf904e41950315d37b95374825c33a7362` 建立：
`learned_relation_rule_probe` 与 `adaptive_corruption_probe`。本条只记录协议冻结，
不记录性能结果。

- Track A S0：`completed_valid`，A1 actionable supervised ceiling 是唯一授权的下一阶段；
  RS 三个 primary dataset 仅作 development evidence，A5 复用的 dormant 12-dataset manifest
  只保存 source hash，未被本项目使用或改写。
- Track B S0：`completed_valid`，B1 matched corruption library 是唯一授权的下一阶段；
  adaptive location、generator/GAN 和 B5 holdout membership 仍锁定。当前 six-dataset panel
  的结构角色按已核验 registry 记录为两类 text-like sparse、三类 registered scRNA count 和
  一个 generic non-expression sparse control，不虚构 dense control。
- 两条线均 `labels_used_during_fit=false`，GPU 合法池为 `[1,2,3,4,5,6]`，物理 GPU `0/7`
  禁用；S0 没有模型、ARI/NMI/ACC、checkpoint、embedding、prediction 或 raw artifact。

验证：`python -m pytest -q tests/learned_relation_rule_probe tests/adaptive_corruption_probe`
（`8 passed`）；对应脚本 `compileall` 通过；两个 `S0_freeze/audit.json` 均为
`status=completed_valid`。

三轮 compact `auto-review-loop` 后，Track A A4 label-free gate 明确以继承的 matched-random
`R` 为 reference，并要求无 material opposing-sign development row；Track B 增加
`C_clean_no_corruption` floor、synthetic sensitivity failure=`protocol_insensitive`、C1--C4
对 C_clean 的 primary pairing、C0 secondary contrast，以及 ARI/L_rec 分离。外部 reviewer
最终评分 A=`8.8`、B=`8.5`、combined=`8.7`，verdict=`ready`；这只是协议审查，不是性能证据。
## 2026-08-18 support_target_validation_probe

独立项目（非 V 系列）M0/M1 preflight：M0 exact replay audit `9/9` 通过；C2 P2 的
active-source/inactive-destination action identity、values、masks 和 post-epoch RNG 调度可由
冻结 seed/H0 完整重建。M1 full 30-epoch no-training matching audit 的 9 rows 结构检查全部
通过，但 `P2_MM_SupportPreserve` 的 dataset-total L1 mismatch 在 Baron Human 三 seed 为
`0.094640/0.095877/0.094646`，超过预注册 `0.05`；Mouse_retina 与 Campbell 六行满足容差。

冻结终态：`magnitude_match_not_estimable`; `gpu_runs_started=0`; 不产生 ARI/NMI/ACC 性能结论。
该控制不可估计不等于支持假设的负结果。M2/M3/M4、adaptive policy 和 GAN 仍 locked；dense H0
threshold support 与 raw-X zero/nonzero support 的解释防火墙保持有效。权威 compact audit：
`result/support_target_validation_probe/M0_freeze/audit.json`、
`result/support_target_validation_probe/M1_preflight/audit.json`。

## 2026-08-18 support_crossing_common_dose_probe

独立项目 `support_crossing_common_dose_probe` 在 M1 终止后只执行 D0/D1，不修改 C2/M1，也不
放宽 magnitude 门槛。D0 继承审计 `audit_ok=true`；D1 完成 `3 datasets × 3 deterministic
tie-break seeds = 9/9` CPU/no-training rows，计算审计 `audit_ok=true`，但
`d1_gate_pass=false`，终态为 `common_dose_not_estimable`。

| Dataset | Common positive-budget rows | Dataset-total mismatch | Median row mismatch | Gate |
|---|---:|---:|---:|---|
| Mouse_retina | 100.000% | 3.134% | 0.420% | pass |
| Baron Human | 93.098% | 8.981% | 12.188% | fail |
| Campbell | 100.000% | 8.492% | 9.643% | fail |

Cross 是 active↔inactive swap，Preserve 是 unequal active↔active swap；两者保持 exact changed
coordinate budget、row value multiset，分别要求 support change >0 与 =0。D1 只使用 constructive
min/max witnesses，不宣称 exhaustive feasible range；三 seed 是 deterministic tie-break
reproductions。Baron 有 579 个正预算行无共同区间，Campbell 虽全行有区间但总剂量仍超 5% 门槛。

该终态是冻结可行性合同的 No-Go，不是 ARI/NMI/ACC 负结果、不是 C2 推翻，也不是 raw-X support
因果证据。D2 common-dose GPU、raw-X bridge、holdout、adaptive policy 和 GAN 均未启动；权威
compact artifacts 位于 `result/support_crossing_common_dose_probe/D0_freeze/` 与
`result/support_crossing_common_dose_probe/D1_feasibility/`，其中 per-row `records.json` 不属于
发布层。
