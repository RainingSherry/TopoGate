# TopoGate 跨版本优势/劣势景观审计

生成时间：2026-08-03。本报告是对已完成结果的只读、事后描述性分析；不重新训练、不用标签选择配置、不修改既有模型或外部 baseline。

## 结果与范围

- 输出根目录：`source-repository/result`，实际目标：`external-result-storage/result`。
- Full/NoMix 只在同一 batch、同一数据集、同一 seed 配对；`delta_ari = Full - NoMix`。
- seed 方向阈值为 `|delta| > 0.001`；小于等于该阈值只标为 `near_neutral`。
- StaticGate 的 `merged_summary.csv` 是单 seed/单行表格，不能被解释为多 seed 稳定性证据。
- 同名数据集仅当 source SHA-256 一致时才允许纵向解释；多个 hash 或缺 hash 的行保留为审计警告。

## Full-NoMix 总览

| Version | Batch | Datasets | Mean ΔARI | Median ΔARI | Positive | Negative | Near-neutral | Stable + | Stable - | Mixed seed | Single/table |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V9 | v9_advantage_ablation | 7 | +0.0154 | +0.0055 | 4 | 3 | 0 | 2 | 0 | 5 | 0 |
| V11 | v11_minimum_5x3 | 5 | -0.0005 | +0.0001 | 1 | 1 | 3 | 0 | 0 | 4 | 0 |
| V12 | v12_advantage | 12 | -0.0012 | -0.0038 | 4 | 8 | 0 | 2 | 2 | 8 | 0 |
| V13 | v13_advantage | 12 | -0.0002 | +0.0000 | 3 | 5 | 4 | 0 | 0 | 9 | 0 |
| V14 | v14_advantage_5ds | 5 | +0.0044 | +0.0012 | 3 | 1 | 1 | 1 | 0 | 3 | 0 |
| StaticGate | static_gate_legacy_table | 15 | -0.0153 | -0.0040 | 5 | 9 | 1 | 0 | 0 | 0 | 15 |

## 稳定正向、稳定负向与混合数据集

### V9

- stable positive: `balance_scale, landsat`
- stable negative: `none`
- mixed across seeds: `glass, image_segment, spect_heart, vehicle, vertebral_column`
- near neutral: `none`

### V11

- stable positive: `none`
- stable negative: `none`
- mixed across seeds: `Mouse_retina, breast_cancer_wisconsin_original, enron, spambase`
- near neutral: `har`

### V12

- stable positive: `banknote_authentication, statlog_image_segmentation`
- stable negative: `rice_dataset_cammeo_and_osmancik, seeds`
- mixed across seeds: `glass_identification, image_segmentation, ionosphere, satellite_image, spectf_heart, vehicle, vertebral_column, wine`
- near neutral: `none`

### V13

- stable positive: `none`
- stable negative: `none`
- mixed across seeds: `banknote_authentication, image_segmentation, ionosphere, satellite_image, seeds, statlog_image_segmentation, vehicle, vertebral_column, wine`
- near neutral: `glass_identification, rice_dataset_cammeo_and_osmancik, spectf_heart`

### V14

- stable positive: `balance_scale`
- stable negative: `none`
- mixed across seeds: `landsat, vehicle, vertebral_column`
- near neutral: `spectf_heart`

### StaticGate

- stable positive: `none`
- stable negative: `none`
- mixed across seeds: `none`
- near neutral: `hrvatin_filtered`

## 逐数据集 Full-NoMix 表

| Version | Dataset | n/d/K | ΔARI | Seed pattern | Seed range | Source identity |
|---|---|---|---:|---|---|---|
| StaticGate | Campbell | 9993/26774/14 | +0.0305 | single_seed_positive | +0.0305..+0.0305 | missing |
| StaticGate | ISOLET | 7797/617/26 | -0.0776 | single_seed_negative | -0.0776..-0.0776 | missing |
| StaticGate | Mouse_retina | 8352/6198/5 | -0.0040 | single_seed_negative | -0.0040..-0.0040 | missing |
| StaticGate | Quake_Smart-seq2_Lung | 1676/23341/11 | -0.0434 | single_seed_negative | -0.0434..-0.0434 | missing |
| StaticGate | breast_cancer_wisconsin_original | 683/9/2 | +0.0111 | single_seed_positive | +0.0111..+0.0111 | missing |
| StaticGate | cnae9 | 1080/856/9 | +0.0158 | single_seed_positive | +0.0158..+0.0158 | missing |
| StaticGate | enron | 9999/4096/2 | -0.1076 | single_seed_negative | -0.1076..-0.1076 | missing |
| StaticGate | first-order-theorem-proving | 6118/51/6 | -0.0027 | single_seed_negative | -0.0027..-0.0027 | missing |
| StaticGate | har | 735/561/6 | +0.0997 | single_seed_positive | +0.0997..+0.0997 | missing |
| StaticGate | hrvatin_filtered | 48266/500/8 | +0.0005 | near_neutral | +0.0005..+0.0005 | missing |
| StaticGate | iris | 150/4/3 | -0.1318 | single_seed_negative | -0.1318..-0.1318 | missing |
| StaticGate | mammographic_mass | 830/5/2 | -0.0176 | single_seed_negative | -0.0176..-0.0176 | missing |
| StaticGate | reuters | 6576/4096/3 | -0.0046 | single_seed_negative | -0.0046..-0.0046 | missing |
| StaticGate | sms_spam_collection | 835/500/2 | -0.0243 | single_seed_negative | -0.0243..-0.0243 | missing |
| StaticGate | spambase | 4601/57/2 | +0.0265 | single_seed_positive | +0.0265..+0.0265 | missing |
| V11 | Mouse_retina | 8352/6198/5 | +0.0006 | mixed_seed | -0.0216..+0.0197 | d3bc2eb08d95 |
| V11 | breast_cancer_wisconsin_original | 683/9/2 | -0.0000 | mixed_seed | -0.0112..+0.0056 | 3ba1713b62cf |
| V11 | enron | 9999/4096/2 | -0.0078 | mixed_seed | -0.0226..+0.0178 | b948a1340304 |
| V11 | har | 735/561/6 | +0.0001 | near_neutral | -0.0005..+0.0007 | d721df683494 |
| V11 | spambase | 4601/57/2 | +0.0047 | mixed_seed | +0.0008..+0.0103 | 28ecfe66283f |
| V12 | banknote_authentication | 1372/4/2 | +0.0192 | stable_positive | +0.0105..+0.0304 | cc9c529e27bf |
| V12 | glass_identification | 214/9/6 | +0.0051 | mixed_seed | +0.0000..+0.0100 | bd1e797ec7e1 |
| V12 | image_segmentation | 210/19/7 | -0.0046 | mixed_seed | -0.0400..+0.0474 | fa7e9febb8b3 |
| V12 | ionosphere | 351/34/2 | -0.0079 | mixed_seed | -0.0190..+0.0098 | 8a1cce5b11f2 |
| V12 | rice_dataset_cammeo_and_osmancik | 3810/7/2 | -0.0397 | stable_negative | -0.0623..-0.0061 | 67bc38e177b9 |
| V12 | satellite_image | 6435/36/6 | -0.0032 | mixed_seed | -0.0124..+0.0071 | 173f3f03a97f |
| V12 | seeds | 210/7/3 | -0.0098 | stable_negative | -0.0218..-0.0025 | 9c1375865055 |
| V12 | spectf_heart | 80/44/2 | +0.0333 | mixed_seed | -0.0140..+0.0804 | 8fa734ef4644 |
| V12 | statlog_image_segmentation | 2310/19/7 | +0.0135 | stable_positive | +0.0115..+0.0159 | 91e660327f9b |
| V12 | vehicle | 846/18/4 | -0.0038 | mixed_seed | -0.0096..-0.0003 | 2e8709f54a32 |
| V12 | vertebral_column | 310/6/2 | -0.0038 | mixed_seed | -0.0372..+0.0194 | a02522583395 |
| V12 | wine | 178/13/3 | -0.0133 | mixed_seed | -0.0553..+0.0152 | a594c68a514b |
| V13 | banknote_authentication | 1372/4/2 | +0.0014 | mixed_seed | -0.0025..+0.0043 | cc9c529e27bf |
| V13 | glass_identification | 214/9/6 | +0.0000 | near_neutral | +0.0000..+0.0000 | bd1e797ec7e1 |
| V13 | image_segmentation | 210/19/7 | -0.0013 | mixed_seed | -0.0065..+0.0018 | fa7e9febb8b3 |
| V13 | ionosphere | 351/34/2 | -0.0029 | mixed_seed | -0.0136..+0.0049 | 8a1cce5b11f2 |
| V13 | rice_dataset_cammeo_and_osmancik | 3810/7/2 | +0.0000 | near_neutral | +0.0000..+0.0000 | 67bc38e177b9 |
| V13 | satellite_image | 6435/36/6 | -0.0019 | mixed_seed | -0.0065..+0.0066 | 173f3f03a97f |
| V13 | seeds | 210/7/3 | -0.0116 | mixed_seed | -0.0230..+0.0000 | 9c1375865055 |
| V13 | spectf_heart | 80/44/2 | +0.0000 | near_neutral | +0.0000..+0.0000 | 8fa734ef4644 |
| V13 | statlog_image_segmentation | 2310/19/7 | +0.0087 | mixed_seed | -0.0027..+0.0252 | 91e660327f9b |
| V13 | vehicle | 846/18/4 | +0.0058 | mixed_seed | -0.0020..+0.0187 | 2e8709f54a32 |
| V13 | vertebral_column | 310/6/2 | -0.0019 | mixed_seed | -0.0118..+0.0061 | a02522583395 |
| V13 | wine | 178/13/3 | +0.0008 | mixed_seed | +0.0000..+0.0024 | a594c68a514b |
| V14 | balance_scale | 625/4/3 | +0.0028 | stable_positive | +0.0023..+0.0036 | aebeaa2574dc |
| V14 | landsat | 4435/36/6 | +0.0316 | mixed_seed | -0.0365..+0.1316 | 22ab32bfc065 |
| V14 | spectf_heart | 80/44/2 | +0.0000 | near_neutral | +0.0000..+0.0000 | 8fa734ef4644 |
| V14 | vehicle | 846/18/4 | +0.0012 | mixed_seed | -0.0034..+0.0042 | 2e8709f54a32 |
| V14 | vertebral_column | 310/6/2 | -0.0138 | mixed_seed | -0.0357..+0.0050 | a02522583395 |
| V9 | balance_scale | 625/4/3 | +0.0809 | stable_positive | +0.0633..+0.1071 | aebeaa2574dc |
| V9 | glass | 214/10/6 | +0.0407 | mixed_seed | -0.0105..+0.1300 | 82ba9fce4cb9 |
| V9 | image_segment | 210/19/7 | +0.0214 | mixed_seed | -0.0213..+0.0648 | ef764bd1d712 |
| V9 | landsat | 4435/36/6 | +0.0055 | stable_positive | +0.0019..+0.0119 | 22ab32bfc065 |
| V9 | spect_heart | 267/22/2 | -0.0125 | mixed_seed | -0.0277..+0.0160 | 07ce03b3056b |
| V9 | vehicle | 846/18/4 | -0.0114 | mixed_seed | -0.0360..+0.0021 | 8b080d62deb4 |
| V9 | vertebral_column | 310/6/2 | -0.0171 | mixed_seed | -0.0258..+0.0000 | a02522583395 |

## 同名数据集的 source hash 审计

- `Mouse_retina`：`hash_missing_or_partial`；版本轨迹 `StaticGate:negative|V11:near_neutral`；hash `d3bc2eb08d95acd12d324f668c537ae4208de57c355862f8e5800d4ba1e727c1`。不能把这些版本强行合并为一个纵向结论。
- `breast_cancer_wisconsin_original`：`hash_missing_or_partial`；版本轨迹 `StaticGate:positive|V11:near_neutral`；hash `3ba1713b62cf59ab7944eb4c3d13d579c92f6d9ec6632f8e6b4977045ecb15bf`。不能把这些版本强行合并为一个纵向结论。
- `enron`：`hash_missing_or_partial`；版本轨迹 `StaticGate:negative|V11:negative`；hash `b948a134030457668937d969f8848d81aaff669a2a730475b2ea83addbab9347`。不能把这些版本强行合并为一个纵向结论。
- `har`：`hash_missing_or_partial`；版本轨迹 `StaticGate:positive|V11:near_neutral`；hash `d721df683494b0b820863c166038634f0ca7922b87d51820b15c6bcc9b6aa9f6`。不能把这些版本强行合并为一个纵向结论。
- `spambase`：`hash_missing_or_partial`；版本轨迹 `StaticGate:positive|V11:positive`；hash `28ecfe66283f4220b62aa7a5afeb3c5e17d04eb9b5f17c99c2fb3cfcbe371bbc`。不能把这些版本强行合并为一个纵向结论。
- `vehicle`：`multiple_sha256_do_not_merge`；版本轨迹 `V12:negative|V13:positive|V14:positive|V9:negative`；hash `2e8709f54a322f2d93b142a820062d2be1f6b154682a12954cfe804224fe65ef|8b080d62deb44713e36ec78a67a7b90c7e5c9e1bfea09f59dcad669b3e6da33e`。不能把这些版本强行合并为一个纵向结论。

同一 SHA-256 的可比纵向条目：
- `balance_scale`：`V14:positive|V9:positive`。
- `banknote_authentication`：`V12:positive|V13:positive`。
- `glass_identification`：`V12:positive|V13:near_neutral`。
- `image_segmentation`：`V12:negative|V13:negative`。
- `ionosphere`：`V12:negative|V13:negative`。
- `landsat`：`V14:positive|V9:positive`。
- `rice_dataset_cammeo_and_osmancik`：`V12:negative|V13:near_neutral`。
- `satellite_image`：`V12:negative|V13:negative`。
- `seeds`：`V12:negative|V13:negative`。
- `spectf_heart`：`V12:positive|V13:near_neutral|V14:near_neutral`。
- `statlog_image_segmentation`：`V12:positive|V13:positive`。
- `vertebral_column`：`V12:negative|V13:negative|V14:negative|V9:negative`。
- `wine`：`V12:negative|V13:near_neutral`。

## TDA H0 pilot：相对 V11_full

TDA 结果单独与同一 pilot batch 的 `V11_full` 配对，不与 V11 minimum 5x3 混合。

| Variant | Dataset | Head ΔARI | KMeans ΔARI | Head seed pattern | KMeans seed pattern | Effect |
|---|---|---:|---:|---|---|---|
| V11_nomix | balance_scale | +0.0025 | +0.0129 | mixed_seed | mixed_seed | both_positive |
| V11_nomix | banknote | -0.0005 | +0.0018 | mixed_seed | mixed_seed | readout_split |
| V11_nomix | flame | +0.0000 | -0.0074 | near_neutral | mixed_seed | readout_split |
| V11_nomix | spect_heart | +0.0106 | -0.0002 | mixed_seed | mixed_seed | readout_split |
| V11_nomix | vehicle | +0.0002 | -0.0001 | near_neutral | mixed_seed | both_near_neutral |
| V11_tda_fixed_filtration | balance_scale | +0.0000 | -0.0013 | near_neutral | mixed_seed | readout_split |
| V11_tda_fixed_filtration | banknote | -0.0001 | -0.0018 | near_neutral | mixed_seed | readout_split |
| V11_tda_fixed_filtration | flame | +0.0000 | +0.0000 | near_neutral | near_neutral | both_near_neutral |
| V11_tda_fixed_filtration | spect_heart | +0.0000 | +0.0000 | near_neutral | near_neutral | both_near_neutral |
| V11_tda_fixed_filtration | vehicle | +0.0001 | -0.0002 | near_neutral | near_neutral | both_near_neutral |
| V11_tda_h0_mst | balance_scale | +0.0000 | -0.0022 | near_neutral | mixed_seed | readout_split |
| V11_tda_h0_mst | banknote | +0.0001 | -0.0014 | near_neutral | mixed_seed | readout_split |
| V11_tda_h0_mst | flame | +0.0000 | +0.0000 | near_neutral | near_neutral | both_near_neutral |
| V11_tda_h0_mst | spect_heart | +0.0000 | +0.0000 | near_neutral | near_neutral | both_near_neutral |
| V11_tda_h0_mst | vehicle | -0.0000 | -0.0000 | near_neutral | near_neutral | both_near_neutral |
| V11_tda_random | balance_scale | +0.0000 | -0.0013 | near_neutral | mixed_seed | readout_split |
| V11_tda_random | banknote | +0.0001 | +0.0002 | near_neutral | near_neutral | both_near_neutral |
| V11_tda_random | flame | +0.0000 | +0.0000 | near_neutral | near_neutral | both_near_neutral |
| V11_tda_random | spect_heart | +0.0000 | +0.0000 | near_neutral | near_neutral | both_near_neutral |
| V11_tda_random | vehicle | +0.0000 | -0.0003 | near_neutral | near_neutral | both_near_neutral |

- `V11_nomix`：head mean ΔARI `+0.002555`，KMeans mean ΔARI `+0.001397`，head positive/negative/neutral `2/0/3`。

- `V11_tda_h0_mst`：head mean ΔARI `+0.000010`，KMeans mean ΔARI `-0.000726`，head positive/negative/neutral `0/0/5`。

- `V11_tda_fixed_filtration`：head mean ΔARI `+0.000002`，KMeans mean ΔARI `-0.000665`，head positive/negative/neutral `0/0/5`。

- `V11_tda_random`：head mean ΔARI `+0.000018`，KMeans mean ΔARI `-0.000274`，head positive/negative/neutral `0/0/5`。

## 无标签特征的描述性关系

特征来自已有 `topogate_dataset_features` 审计，结果差值来自事后运行。下表仅报告 Spearman 描述，不是因果分析、显著性证明或配置选择依据；样本量很小且跨版本协议不同。

| Analysis | Outcome | Feature | n | Spearman rho |
|---|---|---|---:|---:|
| Full_minus_NoMix | StaticGate_delta_ari | d | 15 | +0.166 |
| Full_minus_NoMix | StaticGate_delta_ari | effective_neighbor_proxy | 13 | +0.291 |
| Full_minus_NoMix | StaticGate_delta_ari | log_nd | 15 | +0.068 |
| Full_minus_NoMix | StaticGate_delta_ari | mean_mutual_ratio | 13 | -0.341 |
| Full_minus_NoMix | StaticGate_delta_ari | mean_snn | 13 | +0.280 |
| Full_minus_NoMix | StaticGate_delta_ari | n | 15 | +0.004 |
| Full_minus_NoMix | StaticGate_delta_ari | sparse_graph_components | 13 | +0.291 |
| Full_minus_NoMix | StaticGate_delta_ari | sparse_graph_cycle_rank | 13 | +0.022 |
| Full_minus_NoMix | StaticGate_delta_ari | sparse_graph_largest_component_fraction | 13 | -0.245 |
| Full_minus_NoMix | StaticGate_delta_ari | tda_h0_q90_death_norm | 13 | +0.297 |
| Full_minus_NoMix | StaticGate_delta_ari | tda_h0_tail10_share | 13 | +0.242 |
| Full_minus_NoMix | StaticGate_delta_ari | tda_h0_total_persistence_norm | 13 | +0.110 |
| Full_minus_NoMix | V11_delta_ari | d | 5 | +0.000 |
| Full_minus_NoMix | V11_delta_ari | effective_neighbor_proxy | 5 | +0.200 |
| Full_minus_NoMix | V11_delta_ari | log_nd | 5 | +0.000 |
| Full_minus_NoMix | V11_delta_ari | mean_mutual_ratio | 5 | -0.300 |
| Full_minus_NoMix | V11_delta_ari | mean_snn | 5 | -0.200 |
| Full_minus_NoMix | V11_delta_ari | n | 5 | -0.100 |
| Full_minus_NoMix | V11_delta_ari | sparse_graph_components | 5 | +0.707 |
| Full_minus_NoMix | V11_delta_ari | sparse_graph_cycle_rank | 5 | +0.200 |
| Full_minus_NoMix | V11_delta_ari | sparse_graph_largest_component_fraction | 5 | -0.707 |
| Full_minus_NoMix | V11_delta_ari | tda_h0_q90_death_norm | 5 | -0.200 |
| Full_minus_NoMix | V11_delta_ari | tda_h0_tail10_share | 5 | +0.400 |
| Full_minus_NoMix | V11_delta_ari | tda_h0_total_persistence_norm | 5 | +0.300 |
| Full_minus_NoMix | V12_delta_ari | d | 12 | +0.214 |
| Full_minus_NoMix | V12_delta_ari | effective_neighbor_proxy | 12 | +0.182 |
| Full_minus_NoMix | V12_delta_ari | log_nd | 12 | +0.098 |
| Full_minus_NoMix | V12_delta_ari | mean_mutual_ratio | 12 | +0.182 |
| Full_minus_NoMix | V12_delta_ari | mean_snn | 12 | +0.070 |
| Full_minus_NoMix | V12_delta_ari | metadata_k | 12 | +0.091 |
| Full_minus_NoMix | V12_delta_ari | n | 12 | +0.042 |
| Full_minus_NoMix | V12_delta_ari | sparse_graph_cycle_rank | 12 | +0.049 |
| Full_minus_NoMix | V12_delta_ari | tda_h0_q90_death_norm | 12 | +0.084 |
| Full_minus_NoMix | V12_delta_ari | tda_h0_tail10_share | 12 | +0.049 |
| Full_minus_NoMix | V12_delta_ari | tda_h0_total_persistence_norm | 12 | -0.028 |
| Full_minus_NoMix | V13_delta_ari | d | 12 | -0.095 |
| Full_minus_NoMix | V13_delta_ari | effective_neighbor_proxy | 12 | +0.077 |
| Full_minus_NoMix | V13_delta_ari | log_nd | 12 | +0.331 |
| Full_minus_NoMix | V13_delta_ari | mean_mutual_ratio | 12 | +0.169 |
| Full_minus_NoMix | V13_delta_ari | mean_snn | 12 | +0.127 |
| Full_minus_NoMix | V13_delta_ari | metadata_k | 12 | +0.195 |
| Full_minus_NoMix | V13_delta_ari | n | 12 | +0.162 |
| Full_minus_NoMix | V13_delta_ari | sparse_graph_cycle_rank | 12 | +0.176 |
| Full_minus_NoMix | V13_delta_ari | tda_h0_q90_death_norm | 12 | +0.021 |
| Full_minus_NoMix | V13_delta_ari | tda_h0_tail10_share | 12 | +0.092 |
| Full_minus_NoMix | V13_delta_ari | tda_h0_total_persistence_norm | 12 | +0.254 |
| Full_minus_NoMix | V14_delta_ari | d | 5 | +0.000 |
| Full_minus_NoMix | V14_delta_ari | effective_neighbor_proxy | 5 | -0.700 |
| Full_minus_NoMix | V14_delta_ari | log_nd | 5 | +0.700 |
| Full_minus_NoMix | V14_delta_ari | mean_mutual_ratio | 5 | -0.400 |
| Full_minus_NoMix | V14_delta_ari | mean_snn | 5 | -0.700 |
| Full_minus_NoMix | V14_delta_ari | metadata_k | 5 | +0.872 |
| Full_minus_NoMix | V14_delta_ari | n | 5 | +0.800 |
| Full_minus_NoMix | V14_delta_ari | sparse_graph_cycle_rank | 5 | +0.800 |
| Full_minus_NoMix | V14_delta_ari | tda_h0_q90_death_norm | 5 | +0.300 |
| Full_minus_NoMix | V14_delta_ari | tda_h0_tail10_share | 5 | +0.300 |
| Full_minus_NoMix | V14_delta_ari | tda_h0_total_persistence_norm | 5 | +0.800 |
| Full_minus_NoMix | V9_delta_ari | d | 7 | -0.286 |
| Full_minus_NoMix | V9_delta_ari | effective_neighbor_proxy | 7 | -0.071 |
| Full_minus_NoMix | V9_delta_ari | log_nd | 7 | -0.071 |
| Full_minus_NoMix | V9_delta_ari | mean_mutual_ratio | 7 | +0.214 |
| Full_minus_NoMix | V9_delta_ari | mean_snn | 7 | -0.143 |
| Full_minus_NoMix | V9_delta_ari | metadata_k | 7 | +0.564 |
| Full_minus_NoMix | V9_delta_ari | n | 7 | -0.143 |
| Full_minus_NoMix | V9_delta_ari | sparse_graph_cycle_rank | 7 | -0.179 |
| Full_minus_NoMix | V9_delta_ari | tda_h0_q90_death_norm | 7 | +0.107 |
| Full_minus_NoMix | V9_delta_ari | tda_h0_tail10_share | 7 | +0.214 |
| Full_minus_NoMix | V9_delta_ari | tda_h0_total_persistence_norm | 7 | +0.143 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | d | 5 | -0.344 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | effective_neighbor_proxy | 5 | +0.447 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | log_nd | 5 | -0.447 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | mean_mutual_ratio | 5 | +0.447 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | mean_snn | 5 | +0.447 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | metadata_k | 5 | -0.750 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | n | 5 | +0.224 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | sparse_graph_cycle_rank | 5 | +0.224 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | tda_h0_q90_death_norm | 5 | +0.224 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | tda_h0_tail10_share | 5 | +0.224 |
| TDA_H0_minus_V11_full | V11_tda_h0_mst_head_delta | tda_h0_total_persistence_norm | 5 | +0.447 |

## 解释边界与下一步

1. V9 的拓扑相关收益最清晰地出现在 `balance_scale`，但同一批次也有 `spect_heart`、`vehicle`、`vertebral_column` 的负向差值；这支持数据集依赖，而不是普遍优势。
2. V11--V14 的 Full-NoMix 平均差值接近零，且正负数据集并存；不能把较新的版本写成拓扑增益已经稳定解决。
3. H0 pilot 的正向增强项没有在 head 与 KMeans 两个 readout 上形成一致、稳定的跨数据集收益；当前应保留为诊断 no-go。源码中的正向 persistence 分数会强调晚合并边，可能包含跨组件 bridge，下一候选只能作为默认关闭、可回退的 detached prior 假设，需先做 toy graph 单元测试再决定是否训练。
4. 当前 `kNN`、mutual/SNN、动态图和 edge reliability 是有限度量图结构；它们不能被扩写为完整 persistent homology。拓扑学、数学分析、概率混合与深度聚类的本地教材审计见 `TopoGate_whole_project_math_TDA_audit_2026-08-03.md`。
5. 本报告不改变 V9、V10、V11/V12/V13/V14 或外部 baseline 的实现，也不把单 seed StaticGate 表格升级为正式多种子证据。

## 可复核文件

- `scripts/analysis/analyze_topogate_cross_version_landscape.py`
- `result/analysis/cross_version_evidence_2026-08-03.csv`
- `result/analysis/paired_version_deltas_2026-08-03.csv`
- `result/analysis/topogate_dataset_features_2026-08-03.csv`
- `result/V11/tda_h0_pilot_2026-08-03/**/summary.json`（75 runs）
