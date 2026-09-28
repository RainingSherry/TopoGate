# 运行监测最终审计

日期：2026-09-27  
远端实验目录：`/data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922/advantage_story_20260927`  
本地目录：`S:\Creation\写论文\武汉理工\论文们\TopoGate\papers\advantage_story_20260927`

本记录覆盖基础队列执行、只读 E5 response 队列及轻量产物同步。它是执行与标签隔离审计，不是对实验结论的科学审阅；最终结果由主编辑、Sol 与 Claude 继续复核。

## 基础队列

冻结队列包含 80 项：21 项 GPU 训练任务和 59 项 CPU cohort 事后分析。最终控制器状态为 `PIPELINE_COMPLETED`，80 项完成、0 项失败；21 项训练均使用 GPU 2 或 GPU 4。所有 80 项均一次完成，状态记录中无待运行或仍在运行的任务。

全量检查了每项已完成任务对应的 `record.json`，记录均存在且状态为 `completed`。21 项训练记录全部满足：

- `test_labels_loaded=false`
- `labels_used_for_model_fit=false`
- `core_source_unchanged=true`
- `core_hashes_before` 与 `core_hashes_after` 相同

59 项 cohort 事后分析记录全部满足 `posthoc_full_x_labels_loaded=true`、`labels_used_for_model_fit=false`。基础状态文件为 `outputs/base_queue_audit_20260927/PIPELINE_STATUS.json`；控制器状态、队列定义、审批材料、任务配置、事件记录和日志保存在同一审计目录下。

## E5 response 队列

只读 response 队列的 9 个 case-seed 任务全部完成，`new_training=false`，未进行 encoder 训练。每项都生成了 `probe_rows.csv`、`probe_summary.json` 和 `record.json`；全量记录满足 `test_labels_loaded=false`、`labels_used_for_model_fit=false`、`core_source_unchanged=true`。

原始 `QUEUE_RESULT.json` 中的 `reused=true` 是 runner 在任务完成后递归检查预期产物时写入的状态，不能解读为 9 项任务在启动前都已存在或被跳过。根据启动前状态，只有 `student_grade_seed7` 已完成并跳过；其余 8 项在本轮执行。该字段属于队列行政记录，主编辑负责保留原始文件并修正其说明。本地同步不包含 `QUEUE_RESULT.json`，也没有复制 `adapter_*` 目录。

## E2 cohort depth 修正版

同步的是 H1 等价规则修正后的结果：59 个匹配数据集、4560 个排除项。记录声明 `labels_used_for_model_fit=false`、`new_training=false`、`posthoc_full_x_readouts_loaded=true`。H1 比较忽略不生效的 `high_hop_probability_multiplier`，H2 及以上仍要求有效参数严格一致。

本地 8 个文件的 SHA256 与远端一致：`record.json`、`input_hashes.json`、`per_seed.csv`、`summary.csv`、`exclusions.csv`，以及热图 PNG、PDF、SVG。

## 本地同步范围

产物位于 `papers/advantage_story_20260927/outputs/`：

- `train/`：21 个基础训练 `record.json`。
- `cohort/`：59 个 cohort `record.json` 和 177 个 E3 CSV。
- `base_queue_audit_20260927/`：控制器状态、`PIPELINE_STATUS.json`、冻结队列与任务配置、审批与来源清单、事件记录及日志。
- `E5_response/`：9 组 `probe_rows.csv`、`probe_summary.json`、`record.json`，共 27 个文件；未同步 adapter 目录或权重。
- `E2_cohort_depth/`：修正版的 8 个表格、记录和图文件。

已有的 `outputs/Baron/`、`outputs/Quake_Smart-seq2_Lung/` 与 `outputs/student_grade/` 中 E4 文件未被覆盖。

## 保留的同步 bundle

两份解包后的归档仍保留在仓库根目录，按主编辑要求不再尝试删除：

- `S:\Creation\写论文\武汉理工\论文们\TopoGate\topogate_base_queue_light_20260927.tar.gz`（225,009 bytes）
- `S:\Creation\写论文\武汉理工\论文们\TopoGate\topogate_E5_response_light_20260927.tar.gz`（7,452,385 bytes）

