# TopoGate V0_RG 传导式消融（0911 winners / 第一数据集 26 集 + 11 生物数据集）

本目录归档生成以下三份数据的**全部相关代码**与溯源说明：

| 本地交付物 | 对应代码（本目录） |
|---|---|
| `TopoGate消融_第一数据集_20260917/csv_final/*`（26 数据集 × 7 分支 × 3 种子） | `scripts/V0_RG/run|dispatch|summarize_transductive_tabular_fixed.py` 等 |
| `TopoGate消融_11生物数据集_ablation+macro_20260917.xlsx` | `scripts/V0_RG/summarize_transductive_0911winners_fixed.py`（数据）+ `build_ablation_xlsx.py`（装配） |
| `合并汇总_3源_56模型×147数据集_均值±std_含iris.xlsx` | `local_merge_assembly/`（本地 Windows 装配：finalize.py / apply_format.py / spec_canonical.py 等） |

## 目录
- `scripts/V0_RG/` —— 服务器端消融脚本（15 个，见下）
- `methods/TopoGate/V0_RG/` —— V0_RG 模型库（config/graph/mixing/model/run/trainer/tuning/input_adapter），依赖 `methods/NeighborMix_scMAE`
- `csv_final/` —— 第一数据集消融汇总 CSV（31 数据集 × 7 分支 × 3 种子，含生物套件全集）
- `local_merge_assembly/` —— 本地 Windows 装配脚本（3 源合并、格式、校验）
- `build_ablation_xlsx.py` —— 由 `csv_final/ablation_summary.csv` 生成两张表 xlsx（每指标第一名加粗、第二名下划线）

## scripts/V0_RG 代码 ↔ 操作对照

| 脚本 | 操作 |
|---|---|
| `prepare/run/dispatch/summarize_abl_0911_11ds.py` | 0911 归档根 `result/topogate_transductive_clubench131_scmae16_64x3_20260911` 的 11 数据集调参+final 流程（winner 配置来源） |
| `run_transductive_old_winner_ablation.py` 等 3 件 | 0915 首轮生物消融（读旧 winner 配置，7 分支×3 种子；224 任务）→ `..._old_winner_20260915` 根 |
| `prepare_transductive_old_winner_ablation.py` | 0915 任务清单/manifest 生成 |
| `run/dispatch/summarize_transductive_0911winners_fixed.py` | 0916 修正版生物消融（**全样本拟合预处理**修复指纹漂移；147 任务）→ `..._0911winners_20260916_fixed` 根 |
| `run/dispatch/summarize_transductive_tabular_fixed.py` | **2026-09-17 第一数据集消融**：20 新增数据集（npz，input_kind=general）7 分支×3 种子 = 420 任务 → `v0_rg_ablation_0911winners_first26_20260917_tabular_fixed` 根 |
| `audit_full_reproducibility.py` | 全模型复现审计（归档 vs 消融 full 分支） |

## 关键协议（与归档一致，勿改）
- 预处理 `SplitPreprocessor`：input_kind ∈ {general, raw_count, log1p_expression}；`general` 用 train_variance 选特征（count=min(width,n_top_features)）；指纹 `sha256(json.dumps(..., sort_keys=True, allow_nan=False))`；**必须全样本拟合**（feature_scope=all_samples_unlabeled），否则指纹漂移。
- 拆分 `split_rows(n, 9102026)` 60/20/20，只评 test 20%；传导式训练（fit_X=Xall）；`n_clusters=len(unique(y))`（benchmark oracle）。
- winner 配置来源 `selected.json['winner']['config']`（归档根 datasets/<ds>/）；每任务记录 config_hash / 预处理指纹 / 复现增量。
- 7 分支：full / base_scmae / anchor_equal_mask / neighbor_equal / no_edge_modulation / fixed_mix_original_aux / mixed_observation_reconstruction；estimator=current。
- GPU 0、7 禁用；调度池 [1,2,3,5]；`CUDA_VISIBLE_DEVICES` 由调度器按 GPU 设置；worker 幂等（存在 final_summary.json 即跳过）。
- 外部依赖：`methods/NeighborMix_scMAE`（scMAE encoder 契约）、scanpy/sklearn/torch（服务器 conda 环境 scssl_bench_py310）。

## 重跑步骤（服务器）
1. `python scripts/V0_RG/run_transductive_tabular_fixed.py --dataset <ds> --variant <v> --estimator current --seed <s> --gpu 1`
2. 或用 `dispatch_transductive_tabular_fixed.py`（读取结果根 `scheduler/tasks.tsv`，自动跳过已完成，失败可重复启动）
3. `python scripts/V0_RG/summarize_transductive_tabular_fixed.py` → 产出 `csv_final/` 6 张表
4. 本地 `python build_ablation_xlsx.py --csv-dir csv_final --out out.xlsx --filter first26` → 两张表 xlsx

## 数据/结果位置（服务器，不入库）
- 归档根：`/data/luolie/ToPoGate/result/topogate_transductive_clubench131_scmae16_64x3_20260911/datasets/`
- 本次消融根：`/data/luolie/ToPoGate/result/v0_rg_ablation_0911winners_first26_20260917_tabular_fixed/`
- 生物消融根：`.../v0_rg_biology_ablation_transductive_old_winner_20260915/`、`..._0911winners_20260916_fixed/`

## 与仓库内 methods/TopoGate/V0_RG 的关系（重要）
仓库 `methods/TopoGate/V0_RG/` 已在 2026-09-10 由提交 c852f68/7f27809 推送（统一门控/邻域控制 + held-out 调参协议版本，含 configs/docs/tests）。
本消融运行于**该提交之前的服务器工作树版本**；为保证可复现性，将实际运行所用库文件冻结于
`v0_rg_library_snapshot_20260917/`。scripts/V0_RG 的 worker 若直接运行，请用本快照库替换/对照 `methods/TopoGate/V0_RG`。
