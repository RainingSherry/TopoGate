# -*- coding: utf-8 -*-
"""
Canonical constants for the 3-source merge task.
All agents MUST import this module and use these EXACT strings,
so that the metadata sheet and file structure are byte-identical.
"""
import os

# ---- sheet names ----
SHEET1_NAME = "合并汇总"
SHEET2_NAME = "说明与溯源"

# ---- fixed notes rows for sheet 2: [字段, 内容] ----
NOTES_FIXED = [
    ["说明", "本文件合并自三个来源文档；每个单元格为 模型×数据集×指标 的 均值±标准差，均值与标准差均保留 6 位小数。"],
    ["单元格格式", "均值与标准差均存在：如 0.070877±0.005747；仅有均值无标准差：只填均值（如 0.774280）；两者均缺失：填 NA。"],
    ["来源A", "TopoGate_CLUBench自带_Benchmark.xlsx — 147 数据集 × 25 模型（仅 ToPoGate_strict8 含标准差，其余 24 模型仅有均值）"],
    ["来源B", "PlantNet_23模型×16数据集_ARI_NMI_ACC.xlsx — 16 数据集 × 23 模型（完整均值±标准差）"],
    ["来源C", "TopoGateBenchmark.xlsx — 147 数据集 × 11 模型（完整均值±标准差）"],
    ["合并规则", "模型并集（56 个）按 来源A→B→C 的首次出现顺序排列；数据集并集（147 个）按 来源A 的行顺序排列；每个指标 ARI/NMI/ACC 各占一列。"],
    ["同名模型处理", "同一字面模型名出现在多个来源时，按 来源A、B、C 优先级取首个非空值：ToPoGate_strict8（A∪C，数值完全一致）、dec（A 的 131 个 CLUBench 数据集仅均值 + B 的 16 个生物数据集完整值，互补合并）。"],
    ["scMAE处理", "scMAE 在 来源B 与 来源C 中数值不同，经用户确认仅保留 来源C（TopoGateBenchmark）的数据，来源B 的 scMAE 数据不纳入。"],
    ["模型名处理", "不同来源中仅大小写/空格不同的模型名（如 kmeans/KMeans、PCA+KMeans/PCA + KMeans、scdeepcluster/scDeepCluster、scname/scNAME、idec/IDEC、edesc/EDESC、scdcc/scCDCG）按字面名称视为不同模型，不进行归一化合并。"],
]

# ---- input file names (display only) ----
SRC_A_NAME = "TopoGate_CLUBench自带_Benchmark.xlsx"
SRC_B_NAME = "PlantNet_23模型×16数据集_ARI_NMI_ACC.xlsx"
SRC_C_NAME = "TopoGateBenchmark.xlsx"

# ---- merged filename inside the agent's own output directory ----
OUTPUT_FILENAME = "merged_result.xlsx"
