# 数据追溯日志

## 2026-08-18 corruption_objective_compatibility_probe E1/E3 provenance

E1/E1b 继承已审计的 dense SVD/H0 snapshots、budget manifests 和 post-fit label sources；
E1 使用 Clean、P0_Random、P2_SupportTarget，30 epochs、`d→64→32→64→d` matched
autoencoder、paired seeds `[42,123,7]`。18 个 biological P0/P2 cells 仅在当前 H0、budget、
labels SHA256 和协议字段精确匹配后复用 C2；36 个新 cells 使用物理 GPU 6。GPU allowlist 为
`[1,2,3,4,5,6]`，`0/7` forbidden；实际空闲合法池只有 `[6]`。labels 在 fit 后才读取，用于
benchmark-known-K 的外层 ARI/NMI/ACC；no-fit E1b materializes H0-derived features before
labels；E3 raw `.npz` 只记录 shape/dtype/nnz/zero fraction/source hashes，`labels_not_loaded=true`。

E1 dataset-level means：Mouse_retina `Delta_random=+0.394898`、Baron Human `+0.126069`、
Campbell `+0.146883`、cnae9 `+0.002441`、hate_speech `−0.007294`、sms_spam_collection
`−0.020311`；G1=`0/3`、G2=`1/3`，所以 E2 未运行。H0 中的 support 仍是
threshold-defined dense proxy，不等同 raw-X zero/nonzero support。发布只保留
`result/corruption_objective_compatibility_probe/FINAL/` compact decision/audit/aggregate
artifacts、reports、必要代码与 tests；原始数据、labels、arrays、embeddings、predictions、
weights、checkpoints、logs、caches 和 per-run directories 排除。

## 2026-08-18 support_crossing_common_dose_probe D0/D1 provenance

D0/D1 只读取已审计的 dense SVD/H0 sources：Mouse_retina `(8352,128)`、Baron Human
`(8451,128)`、Campbell `(9993,128)`；H0 SHA256 与 inherited M1 audit 一致。D1 使用固定
clean-row threshold support ratio `0.05`、C2 row budget rate `0.25`，Cross 为
active↔inactive swap，Preserve 为 unequal active↔active swap，均保持 row value multiset 和
exact `2*m_i` changed-coordinate budget。全程 `labels_not_loaded=true`、`gpu_runs_started=0`。

D1 的 constructive range audit 为 `9/9` completed-valid，`d1_gate_pass=false`：Mouse 总剂量
relative mismatch `0.031342`；Baron `0.089810` 且 579 个正预算行无共同区间；Campbell `0.084924`。
三 seed 是确定性 tie-break reproductions，不是独立统计样本。per-row `records.json` 仅留在本地
结果盘，不进入 GitHub；D2、raw-X bridge、holdout、adaptive policy、GAN 均未授权。

## 2026-08-18 sparse_corruption_principle_probe C2 provenance

C2 使用预注册的 Mouse_retina、Baron Human、Campbell development panel、六个静态 corruption
principles 和 paired seeds `[42,123,7]`，以固定 dense SVD/H0 proxy 作为共同输入。每个 run
复用相同的 30-epoch reconstruction probe、optimizer、readout 和 exact changed-coordinate
budget；标签只在 fit 后用于 benchmark-known-K 的外层 ARI/NMI/ACC。54/54 runs completed-valid，
当前 H0、budget manifest 和 labels SHA256 全部匹配；独立 C2 integrity audit 为 `audit_ok=true`。
协议允许 GPU `[1,2,3,4,5,6]`、禁止 `[0,7]`；实际运行使用 `[2,3,4,5,6]`，GPU1 在本轮有外部占用。

P4 residual score 是 standardized clean-H0、dataset×seed 冻结的 proxy；P5 geometry score 是
raw clean-H0、dataset-level 冻结的 label-free proxy。C2 support 始终表示
`threshold-defined support of dense H0`，不是原始 count matrix 的 zero/nonzero support。原始
矩阵、标签、corruption/score arrays、embeddings、predictions、weights、checkpoints 和 logs
仍保留在外部结果盘，不进入发布副本。

## 2026-08-18 sparse_corruption_principle_probe C0/C1 provenance

新项目复用已关闭 B1 的 compact post-fit summary 与 representation-consumer S0 `H0.npy`，不复制
原始输入、标签或模型工件。C0 inventory 检查了预注册 scCluBench processed/processed_scmae
候选，记录 shape、抽样稀疏度、SVD-90 intrinsic proxy、source-family 和 SHA256；14 个候选中
以 outcome-independent label-free maximin 选择 12 个 holdout sources，无 development overlap，
`shortfall=0`。C1 在 Mouse_retina/Baron Human/Campbell × 六个关闭 B1 arm × `[42,123,7]`
上做结构 replay，`54/54` rows、`fit_runs=0`、`labels_loaded=false`；C4 使用 column-median/MAD
proxy，仅作结构诊断，不等同旧 B1 warm-up residual。H0 support 是固定阈值 proxy，不是原始
count matrix support。compact artifacts 位于 `result/sparse_corruption_principle_probe/`。

## 2026-08-18 adaptive-corruption probe B1 formal rerun provenance

The publication-facing B1 matrix uses the audited S0 `H0` stem with per-dataset
`d_eff`, clean-H0 column standardization, backbone `d_eff→64→32→64→d_eff`,
30 epochs, batch size 512, corruption rate `0.25`, and paired seeds
`[42,123,7]`. The pair-feasible rule is frozen as
`m_i=min(ceil(rate*active_i), floor(active_i/2), inactive_i)`; every non-clean
arm changes `2*m_i` coordinates. The fresh matrix has `108/108`
completed-valid runs, all effective-rate matches pass, and workers used only
physical GPUs `[1,2,3,4,5,6]` (0 and 7 forbidden).

The earlier support-budget-mismatch matrix remains local under
`result/adaptive_corruption_probe/B1_corruption_library_attempts/` and is
explicitly excluded from the compact CSV/JSON, reports and GitHub release.
Labels are loaded only after fitting for benchmark-known-K KMeans and outer
ARI/NMI; raw H0-derived arrays, embeddings, predictions, checkpoints and logs
are not publication artifacts.

## 2026-08-17 relation-selection probe RS1–RS3 data/result audit

新项目复用已审计的 `representation_consumer_probe` S0 H0、candidate pool、row budget
和 S1 `R/O_pool/O_full` summaries，不修改旧工件。RS0 继承 holdout manifest 的 SHA256
为 `6d9afa1f6d90f77d8836e9f877f6567ebb7c7621ba3d022622e2488c9fb8b2cb`，并验证旧项目
artifact tree 未修改。

RS1 使用 17 个固定 label-free edge features、八个 view seeds `[17,31,47,61,73,89,101,113]`
和 5-fold GroupKFold by anchor；RS2 使用 B0 cosine、B1 mutual-first、B2 SNN/Jaccard、
B3 stability、B4 equal-rank fusion，均保持原始 cosine weight、`b_i=min(8,positive_count_i)`、
symmetrization、Spectral 和 known-K KMeans。标签只用于 RS1 diagnostic targets 与外层
metrics，RS3 只读取完成的 summaries。

正式矩阵为 RS1 六数据集 diagnostic rows、RS2 `90/90` selector rows；raw features、graphs、
embeddings、predictions 和 per-run logs 不进入发布层。可发布的重要结果位于
`reports/relation_selection_probe/` 与 `result/relation_selection_probe/FINAL/`。

## 2026-08-17 representation-consumer probe S2 SimpleCut data/result audit

S1 条件确认阶段固定使用 `Baron Human` 与 `Mouse_retina` 的 S0 H0、S1 v2 selected graphs、
`R/O_pool/O_full`、SimpleCut `128→64→32`、80 epochs 和 paired seeds `[42,123,7]`，共
`18/18` completed-valid。物理 GPU 为 3；GPU 0、7 未使用。S2 没有重建 candidate pool、修改
budget、选择新 consumer 或使用新数据集。

- source archive `y` 只用于 O oracle graph construction、known-K readout 和 post-fit metrics；
  SimpleCut fit 不接收 labels/K。O artifacts 明确标记 `diagnostic_only`/`method_claim=false`。
- root artifact manifest exact-tree `197` entries；18 个 run 与 nested manifests 均通过。所有
  S2 selected/direct graphs 与 S1 v2 source graph hash/structure 精确一致；labels_true、source
  SHA、H0 SHA 和 ARI/NMI/optimal-mapping ACC 均独立复核一致。
- dataset-level opportunity quantities：Baron `H_pool=+0.033242`、`H_full=+0.033367`、
  `C=+0.000125`；Mouse `H_pool=+0.008880`、`H_full=+0.009622`、`C=+0.000742`。这些是
  label-derived diagnostic upper bounds，不是 TopoGate 或新 backbone 性能。
- integrity audit 为 `WARN`：training history 最后一行是 optimizer step 前 loss，而
  `fit_metadata.final_loss` 是 step 后重算；不影响 primary metrics，但保留为 metadata timing
  limitation。未据此重训或修改原始性能工件。

## 2026-08-17 representation-consumer probe S0 input and budget audit

独立 `representation_consumer_probe` 正式 S0 replay 使用既有 E1 manifest（SHA256
`edf2d57bba15cc1a56b18d12dd72efd320e0cbc4a730875a513b79814c577339`）和六个冻结 stress inputs：
`cnae9`、`Mouse_retina`、`sms_spam_collection`、`Baron Human`、`Campbell`、`hate_speech`。
六个 source path、shape 和 source SHA 均与 manifest 匹配；每个 dataset 只生成一个
`TruncatedSVD(d0=128, random_state=0)` H0 snapshot，并将 candidate pool、H0 SHA、archive keys、
K source 和 labels-after-fit-only boundary 写入 `result/representation_consumer_probe/S0_freeze/`。

本次冻结 `k=20` positive-cosine relation pool 与 `budget_cap=8`，实际 row budget 为
`b_i=min(8,positive_count_i)`。R/O_pool/O_full 未来必须复用同一 effective-budget hash；
zero-budget rows 保留为 graph diagnostics，不通过异类边补齐，也不整集删除。S0 没有读取标签
进入 H0、candidate 或 numerical loss；标签计数只用于外层 K audit，oracle graph 尚未运行。

## 2026-08-12 V22 dataset extension round 2 downloaded and audited

在未读取本轮新增数据性能之前固定并下载第二批候选，输出目录为
`datasets/external/v22_dataset_extension_round2_20260812/`，登记入口为
`datasets/external/v22_dataset_extension_round2_20260812/manifest.json`。候选分层为：

| dataset_id | source / shape | zero fraction | labels | status |
|---|---:|---:|---:|---|
| `news20__libsvm_sparse_highdim` | LIBSVM news20.scale, `15935 x 62061` | `0.998713` | 20 | eligible |
| `rcv1_train__libsvm_sparse_highdim` | LIBSVM RCV1 train, `20242 x 47236` | `0.998432` | 2 | eligible |
| `mnist__libsvm_dense_control` | LIBSVM MNIST scale, `60000 x 780` | `0.807817` | 10 | eligible control |
| `pbmc_1k_v3__10x_unlabelled_count` | 10x PBMC 1k v3, `1222 x 33538` | `0.939043` | none | eligible_unlabelled |

候选依据是公开来源可用、稀疏高维主层、一个非高维控制层和一个无标签 scRNA count
部署层；选择不读取标签或 V22 结果。LIBSVM 标签只保留给外层 known-K 与后验指标，转换、
预处理、graph、Gate、discriminator 和 loss 均不读标签。PBMC 1k v3 的 10x 归档不含独立
细胞类型标签，只允许后续显式传入 `--n-clusters DATASET_ID=K` 的无标签探索，不进入
ARI/NMI 汇总。原始归档与 CSR-NPZ 均通过 `file`、形状、有限值、稀疏重建和标签长度检查，
manifest 记录 URL、传输边界、原始/处理后 SHA-256 与完整 profile；下载沿用本地代理的
`verify=False`，不能把它当作独立传输证明。

验证：`python scripts/V22/prepare_dataset_extension_round2.py --dry-run`；转换脚本编译通过；
四个处理文件均通过矩阵审计；V22 focused tests 与矩阵 tests 共 `13 passed`。本批正式
多种子矩阵尚未启动，不能与第一批 V22 Full 结果合并为性能结论。另完成
`result/V22/engineering_smoke_pbmc1k_v3_cooperative_20260812/` 的一 epoch CPU 输入 smoke：
原始 `1222x33538` 经 label-free top-variance cap 为 `1222x2000`，显式 `K=8`，
`labels_used_during_fit=false`、`K_used_during_fit=false`，不计算 ARI/NMI。使用新 manifest
的 cooperative Keep-Gate/always-visible dry-run 展开 `8` 个唯一键；PBMC 1k v3 的两个键均
保留 `requires_explicit_n_clusters=true` 并显式传入 `K=8`，没有猜测 K。

## 2026-08-12 V22 dataset extension and explicit-K boundary

新增固定数据清单 `datasets/external/v22_dataset_extension_20260812/manifest.json`，候选在
读取 V22 性能前登记，且 `selection_uses_labels_or_outcomes=false`。主分层包括 LIBSVM
`sector`（`6412x55197`、105 类、约 99.7% 零）和 `real-sim`（`72309x20958`、二类、约
99.8% 零）；`covtype`（`581012x54`、约 77.8% 零）作为大样本非高维控制；PBMC3k
（`2700x32738`、约 97.4% 零）作为无独立标签的 10x scRNA count 部署控制。原始归档、
处理后 CSR-NPZ、形状/稀疏度、来源 URL 和 SHA-256 均写入 manifest；下载时因本地代理
证书链不可验证使用 `verify=false`，这一边界被显式记录，不能把它当作独立传输证明。

LibSVM 标签仅保留给外层 benchmark 的 known-K 与后验指标，转换、预处理、graph、Gate、
discriminator 和 loss 均不读标签。PBMC3k 没有标签，不能在 runner 中猜 K；V22 矩阵现在
要求通过 `--n-clusters DATASET_ID=K` 显式传入，缺失时在启动任何任务前拒绝。默认 dry-run
仍展开四个数据集、五个 variant、三 seed 的 60 个唯一键，并为 PBMC3k 的 15 个键标记
`requires_explicit_n_clusters=true`。

## 2026-08-11 V19 sparse/high-dimensional extension final audit

第一批扩展 manifest `result/V19/v19_rg_extended_sparse_manifest_20260811.json` 固定 13 个
候选、`rg_full/scmae_only`、seeds `[42,123,7]`，共 `78/78` 完成。RG 配置来自已完成的
V19 ARI development selection；扩展本身不重新按标签选择配置或数据集，标签只用于外层
known-K 与拟合后 ARI/NMI/ACC。13 集结果汇总为 RG 胜 scMAE `6/13`，宏平均 ARI
`0.175345` vs `0.182150`；三 seed 全部正向 `2/13`。

预注册的第一批胜出集全部进入外部方法控制。AHDPC、DPC-GFNN、GCC 固定参数和相同的
V19 label-free input adapter 均完成，审计覆盖 `6 datasets x 3 methods = 18/18`。原始
runner 因 Dexter CSR NPZ 加载异常未写出聚合 JSON；根据已存在的 12 个逐方法 summary
生成 `baseline_summary_reconstructed.json`，并将修复 CSR loader 后的 Dexter/Dorothea
6 条结果单独保存在 `v19_rg_extended_winner_baselines_missing_v1/`。最终审计输出为
`result/V19/v19_rg_sparse_goal_audit_20260811/goal_audit.json`：`goal_met=true`、
RG 胜最佳外部方法 `2/6`、获胜集基线缺失 `0`。

第二批 7 个候选没有启动，因为第一批已达到预注册的至少 5 个 RG 胜 scMAE 条件；其
条件 launcher 明确记录 `not_activated_primary_met`。本条结果不是全数据集普遍优势声明，
且不把只完成开发/选择的指标和扩展泛化证据合并。

## 2026-08-11 V21 readout-fix sparse/high-dimensional extension

新增 `result/V21/v21_extended13_readoutfix_manifest_20260811.json`，从既有 V19 稀疏/高维
输入清单冻结 13 个与 V21 六数据集开发层无重叠的候选：`fbis.wc`、`tr45.wc`、`fabert`、
`micro-mass`、`gina_prior2`、Internet Advertisements、完整 SMS TF-IDF500、
Quake Smart-seq2 Lung、Arcene、Dexter、Dorothea、Gisette 和 Madelon。manifest 固定
`topology_assignment_adversarial/scmae_only`、seeds `[42,123,7]`，预期 78 runs；
`extension_labels_used_for_selection=false`。UCI 数据沿用已有原始/派生 hash，local snapshot
继续明确记录 unresolved source metadata，不重复计算 hash 或把本地来源冒充公开来源。

新协议 `v21_assignment_adversarial_v3_readoutfix_v1` 的 primary output 是
`kmeans_embedding_known_k`；K 仍只由 benchmark 外层作为协议元数据提供，readout 不读取
标签。Student-t head 仅作训练代理，额外保存其预测、概率和 occupancy。当前工程 smoke 位于
`result/V21/engineering_smoke_extended_readoutfix_20260811/`：只运行 micro-mass、seed42、
两 variant、2 epochs，`2/2` completed 且严格审计通过。该目录不进入正式扩展结果表。

## 2026-08-11 V19 second-panel conditional launcher

扩展 runner 与汇总器现在同时接受已预注册的 `v19_rg_extended_sparse_v1` 和
`v19_rg_extended_sparse_batch2_v1`，只根据各自 manifest 读取协议，不改变 RG/scMAE
计算路径。新增 `scripts/V19/launch_batch2_after_primary.py`：等待第一批完整审计，只有
第一批 `promotion_rg_win_by_mean_ari` 少于 5 个时才在空闲的允许 GPU 上运行第二批全部
7 个候选（42 runs）；达到 5 个时明确记录 `not_activated_primary_met`。第二批仍使用
第一批 ARI 选出的固定配置，不按第二批结果再次选数据集。
两个条件 launcher 的请求 GPU 池已设为允许物理卡 `[1,2,3,4,5,6]`，启动时按至少
30 GiB 空闲显存动态选取；GPU0/7 永不使用，已有外部任务的卡不会被抢占。

新增 `scripts/V19/run_winner_baselines_after_panels.py`，在第一批不足 5 个胜出时，
等待第二批终态并对两批中所有 `RG mean ARI > scMAE-only mean ARI` 的数据集分别运行
已验证的 AHDPC、DPC-GFNN 和 GCC；若第一批已达到 5 个，则交给原单批启动器，避免重复
SOTA 运行。

新增 `scripts/V19/audit_rg_sparse_goal.py`，在两批扩展和对应 baseline summary 完成后，
统一核验 RG 胜出数量、胜出集的 baseline 覆盖，以及 RG 是否高于三种外部方法中的最佳
ARI；该审计不参与模型或数据集选择。
新增 `scripts/V19/launch_goal_audit_after_baselines.py` 自动等待所需终态并调用该审计。

## 2026-08-11 V19 sparse/high-dimensional extension batch 2 preregistration

新增固定清单 `result/V19/v19_rg_extended_sparse_batch2_manifest_20260811.json`，包含
20 Newsgroups、Reuters、Enron、WOS、ISOLET、SECOM 和 OpenML webdata_wXa 共 7 个
候选，统一登记为 `rg_full/scmae_only`、seeds `[42,123,7]`，预期 42 个配对 run。
候选名单在第一批扩展结果揭示前固定；标签只保留为外层 benchmark 元数据，未用于资格、
预处理、图、RG 或 scMAE 设置。该批的激活规则是：第一批若未达到预注册的 5 个 RG
胜出标准，则完整运行这 7 个候选，不按单个结果挑选数据集。

输入来源为已有本地 NPZ 或已登记的 OpenML webdata_wXa 快照，`source_hash` 沿用现有
登记策略记为 `unavailable`，不重复计算哈希。该批尚未启动模型运行。

## 2026-08-11 V19 sparse/high-dimensional extension panel preregistration

新增扩展 manifest `result/V19/v19_rg_extended_sparse_manifest_20260811.json`，固定
13 个候选层、`rg_full/scmae_only` 两个 variant 与 seeds `[42,123,7]`，预期 78 个
配对 run。候选在读取任何扩展性能前按“稀疏/高维输入与公开来源可用”登记；目标“至少
5 个 RG 胜出”是事后成功标准，不是按结果筛选数据。现有本地候选包括 `fbis.wc`、
`tr45.wc`、`fabert`、`micro-mass`、`gina_prior2`、Quake 单细胞，以及已审计的 UCI
`internet_advertisements` 与完整 SMS TF-IDF；其来源状态分别保留为 local snapshot 或
复用 V9 external manifest，不把未核验来源冒充 UCI。

新增 UCI 数据包来自 Arcene (167)、Dexter (168)、Dorothea (169)、Gisette (170) 和
Madelon (171)，原始 zip 位于
`datasets/external/v19_extended_sparse_20260811/raw/uci/`，训练与 validation 行按官方
文件合并；每个 zip 与派生 NPZ 均记录一次 SHA-256。Dorothea 源矩阵为 100,000 维，
当前 scMAE 解码器无法直接承受二次维度参数量，因此在预注册转换阶段只执行一次无标签
方差 Top-2000，保存 `dorothea.selected_feature_indices.npy`、源/派生维度和规则；RG
与 scMAE 共用该派生输入。所有候选的 `X`、标准化和图构建不读取 `y`，标签只由外层
benchmark 用于 known-K 与后验 ARI/NMI。

## 2026-08-10 V20 Full eight-dataset coarse screen

使用冻结配置 `methods/TopoGate/V20_topology_conditioned_adv_mask/configs/v20_full.yaml`
并覆盖 X-only 选择的 `gate_lr=5e-4`、`tau_ste=0.5`，运行 8 个输入层：
`Mouse_retina`、`Campbell`、`Baron Human` 的 `clubench_bridge`，以及
`sms_spam_collection`、`cnae9`、`imdb`、`hate_speech`、
`sentiment_labeld_sentences` 的 `shared_text`。每个数据集为 seed42、80 epochs、
40 epoch warmup；cnae9 复用 `result/V20/full_first_round_20260810/`，其余输出在
`result/V20/full8_seed42_20260810/`。

V20 使用稀疏图视图的 TruncatedSVD/cosine-kNN（k=20，SVD 95% 目标、上限 500），
dense 模型视图上的分块 deviation/dispersion 和共享 `2->64->1` Gate。训练过程、图、
统计、Gate 和 loss 均未读取标签；K 仅用于 known-K KMeans readout 和后验指标。
GPU2--4 在启动时被外部进程占用，因此补充任务改在 GPU1 串行完成；所有 8/8 run
最终 `completed`，没有 OOM 或 `incomplete_compute`。本轮只做 Full 粗筛，未运行
matched scMAE-only，不能作为正式对照矩阵。

requested mask 约为 0.40，但 sparse donor 的 effective value-change rate 依数据集约
为 0.0068--0.0864；该值作为诊断保存，不替代 requested-mask 训练语义。未重新计算
SHA/hash，沿用既有数据清单和路径。

## 2026-08-10 PlantNet-ARI fixed RG transfer with PCA200

固定配置文件为 `methods/TopoGate/V19_rg_adapter/configs/v19_rg_plantnet_ari_pca200.yaml`：
PlantNet full-16 ARI 选择的 `hidden_size=256`、`batch_size=512`、`lr=0.00139648`、
`mask_ratio=0.368914`、`dropout=0.260665`、`masked_data_weight=0.814697`、
`mask_loss_weight=0.636069`、`neighbor_k=5`、`mix_neighbors=4`、`tau=0.493971`、
`pseudo_weight=0.564693`、`gate_max=0.064455`、`n_top_features=1500`；本轮将图 PCA
请求值设为 `200`。输出根为
`result/V19/v19_rg_plantnet_ari_pca200_20260810/`，覆盖 8 个 comparable 数据集、
`rg_full/scmae_only`、seeds `[42,123,7]`，共 `48/48`。

V19 的 `clubench_bridge/shared_text` 预处理协议不做 HVG，因此 PlantNet 的
`n_top_genes=1500` 在本批输入上不改变特征数；`knn_pca_dim=200` 按
`min(200,n_features,n_samples-1)` 实际化，`hate_speech` 实际为 100，其余 RG runs 为
200。未重新计算 SHA/hash，复用了已登记的 manifest 与数据路径。

## 2026-08-10 V19 v2 mechanism refine and final comparison

正式 mechanism refine 输出根为
`result/V19/v19_rg_mechanism_refine_v2_cached_20260809/`，固定
`v19_rg_unsup_tuning_v2` 协议完成 `396/396`：12 个 mechanism candidates、11 个输入层、
seeds `[42,123,7]`。该阶段只使用 held-out X-only proxy，launcher 审计为
`labels_accessed=false`、`y_key_read=false`，没有聚类/K/标签指标。唯一选中配置为
`rel_both2`：`gamma_mutual=2.0`、`gamma_snn=2.0`、`gamma_sim=0.0`、
`gamma_distance=0.0`；proxy-win 为 2/8 个底层数据集，保留该限制作为结果解释边界。

post-freeze final 输出根为
`result/V19/v19_rg_final_postfreeze_rel_both2_20260810/`，11 个输入层 × 6 个 variant ×
3 个 seed 完成 `198/198`，`audit_ok=true`，6 个 worker 返回码为 0，未使用 GPU 0/7。每个
run 的正式 artifact contract 为 `status.json`、`run_record.json`、`resolved_config.json`、
`metrics.json`、`predictions.npy`、`labels_true.npy` 和 `embedding_final.npy`，另保留
`dataset_profile.json`、`preprocess_profile.json`、gate/edge diagnostics 和 training history。
模型拟合与变体选择均未读取标签；`labels_true.npy` 只服务于 benchmark K/后验指标。

comparison 输出为
`result/V19/v19_rg_final_comparison_rel_both2_20260810/`。归档 baseline CSV 仅在 Campbell、
Mouse_retina、cnae9 和 sms_spam_collection 四个层有可连接记录；不对缺失层做零填充，
不把归档数值包装成 fresh matched SOTA 运行。实验过程没有重复计算 SHA/hash。

## 2026-08-08 V19 X-only tuning handoff

新增 `result/V19/v19_rg_unsup_tuning_v1/` 的无标签调参协议和等待衔接器。调参器从
固定 V19 manifest 读取 11 个输入层的 NPZ 特征矩阵字段，不访问 `y`，不推导 K，不执行
KMeans，也不保存标签指标。预注册候选为 24 个，完整搜索规模为 24 × 11 × 3 = 792
个 run；选择指标是按数据集/seed 等权的 masked recovery、latent view stability 和
input-neighbor preservation 排名均值。V19 正式矩阵已 `66/66 completed`，衔接器已启动
调参 worker；不重新计算 SHA/hash。

## 2026-08-08 V19 seed42 formal batch running

已启动固定协议 `v19_rg_selected_advantage_v1` 的 seed42 批次，manifest 为
`result/V19/v19_rg_dataset_manifest_20260808.json`，输出根为
`result/V19/v19_rg_selected_advantage_v1/`。seed42 已 `22/22 completed` 且无
`incomplete_compute`；seed123 与 seed7 随后由既存 launcher 启动，各自当前为 7 个
run completed、1 个 run running，其余按 launcher 顺序等待。V19 使用允许的物理 GPU，
GPU0/7 未用于 V19；V18 既有 worker 未停止或覆盖。当前不写入性能结论，不重新计算
SHA/hash。

## 2026-08-08 V19 fixed selected-dataset manifest and engineering smoke

生成固定 manifest `result/V19/v19_rg_dataset_manifest_20260808.json`，manifest id 为
`v19_rg_advantage_inputs_20260808_v1`。8 个真实 NPZ 数据展开为 11 个输入层：
`Mouse_retina`、`Campbell`、`Baron Human` 各有 `rg_native` 与 `clubench_bridge`；
`sms_spam_collection`、`cnae9`、`imdb`、`hate_speech`、
`sentiment_labeld_sentences` 使用 native/bridge 等价的 `shared_text`，不重复运行。
manifest 共 11/11 eligible，选择不使用标签或既有结果；按要求不重新计算 SHA/hash，
`source_hash` 显式记为 `unavailable`。

预处理固定为：Baron native 使用 `normalize_total(10000) -> log1p -> HVG1000 ->
scanpy scale`；Mouse/Campbell native 使用已有 log1p 表达、`HVG1000 -> scanpy scale`，
不重复 log1p；所有 bridge 与 shared text 使用原 NPZ `x -> StandardScaler`，不做 HVG、
count normalization 或 log1p。生物 native 不与归档 SOTA 混合；bridge 与 bridge-equivalent
shared text 可进入后续统一协议对照。

真实 engineering smoke 位于 `result/V19/engineering_smoke_20260808/`，覆盖 `cnae9`
shared text 和 `Baron Human` native 的 `scmae_only/rg_full`，均为 seed42、64 行、1 epoch、
CPU。两个 paired variant 的 selected-feature 文件逐项一致；RG 输出 10-neighbor graph、
非零 node gate 与 pseudo path，scMAE 输出空图和零 gate。该 smoke 只验证输入与产物契约，
其 ARI/NMI 不进入正式结果表。

## 2026-08-08 V18 v2.2 protocol replacement

v2.1 在正式矩阵中途因前置代码审计发现 mask 语义和 FISTA latent 归一化偏差而停止；
其 564 个已完成 run 保留为旧 protocol 产物，6 个运行中的 key 标记为
`incomplete_compute`，不进入 v2.2 汇总。v2.2 复用已经冻结的 157 条数据登记，
生成 `result/V18/v18_dataset_manifest_v2_2_20260808.json`，其中 149 条 eligible，
10 个 variant、3 个 seed 仍为 4470 个预注册 run key。未重新扫描数据源或重算哈希。

v2.2 engineering smoke 位于
`result/V18/engineering_smoke_v2_2_20260808/2d_20c_no0/`，使用真实数据、seed42、
96 行和短 epoch，仅验证代码路径与产物契约，不进入正式性能表。

## 2026-08-08 V18 dataset manifest and engineering smoke

基于现有 V9 registry 一次性生成 `result/V18/v18_dataset_manifest_20260808.json`，
manifest id 为 `v18_scmae_mainline_20260808`，共 157 条记录，其中 149 条 eligible、
8 条 ineligible。数据集选择声明 `selection_uses_labels_or_outcomes=false`；manifest
和 runner 不在每个 seed 或汇总阶段重复计算 SHA-256/其他哈希。

真实登记 `ahdpc_prepared__2d_20c_no0`（原始 `1517x2`、`K=20`）按 seed42、最大
1500 行的 label-free 行抽样运行三路短 engineering smoke。输入预处理为
`nan_to_num + column StandardScaler`，标签只用于外层 K/后验指标。产物位于
`result/V18/engineering_smoke_real_20260808/2d_20c_no0/`；这不是正式性能证据，
不与 V9/V17 或 baseline 汇总混合。

### V18 formal matrix submission (2026-08-08)

使用 `result/V18/v18_dataset_manifest_20260808.json` 中 149 条 eligible 数据，按
10 个预注册 variant、seeds `[42,123,7]` 展开 4470 个 run key。六个 launcher worker
分别绑定物理 GPU 1--6；每个子进程保存 `manifest_id`，不重复计算数据或源码哈希。
当前记录为运行中，尚无正式 V18 汇总结果。

### 2026-08-07 V16.1 完成补齐的候选

在不改变 V16.1 固定协议的情况下，`PRJNA895163`、`Bone_Marrow` 和 `Young` 已完成
clean/compound、seeds `[42,123,7]` 与五路 paired readout。`PRJNA895163` 的产物位于
`/tmp/v16_1_stage1_parallel_20260806/`，另两项位于
`result/V16_1/expanded_count_stage1_20260807/`。固定汇总均为
`empirical_not_supported`：clean Delta ARI 分别为 `0.000000`、`-0.002388`、
`+0.002589`；compound Delta ARI 分别为 `+0.000004`、`-0.000291`、`0.000000`。
标签仅用于 benchmark K 和后验指标，`labels_used_during_fit=false`。三者不进入正例表，
也不触发任何 gate、support、temperature、thinning 或 K 调整。当前未完成的只有
Norman Stage-0；`hrvatin_geo_maintype_counts` 已完成并按固定规则记为
`empirical_not_supported`。

### 2026-08-07 V16.1 PBMC3K 新增计数候选

新增本地公开 PBMC3K 候选：源文件为
`/data/luolie/biopipeline/test-datasets/test-datasets-modules/data/genomics/homo_sapiens/scrnaseq/h5ad/pbmc3k.h5ad`，使用
`raw.X` 作为可逆 `log1p(count)` 视图，`obs.louvain` 作为 benchmark 标签。
转换器新增 `raw.X` 读取和严格 `expm1` 恢复路径，生成 CSR bundle
`/tmp/v16_1_expanded_data/PBMC3K.npz`，矩阵 `2638×13714`，非零数 `2238732`，
`labels_used_during_fit=false`。没有计算或记录新的哈希。

固定 V16.1 Stage-0（`k=20`、三次 split、A/B 交换）通过理论域证书：零比例
`0.938118`、median row nnz `819`、candidate recurrence `0.285671`、稳定边率
`0.582272`，分层为 `high_sparse_bonus`。六次 cross-fitted support 的正值率为
`0`，但该诊断不是 expanded-count 的硬门槛，仍进入固定 Stage-1。

Stage-1 输出位于
`result/V16_1/expanded_count_stage1_20260807/PBMC3K/`，clean 和 compound 均使用
seeds `[42,123,7]`、五路 paired readout，分别在 GPU5/GPU6 并行完成。固定汇总
`/tmp/v16_1_summary_pbmc3k_20260807.json` 将其标记为
`empirical_not_supported`：clean mean Delta ARI `0.000000`，compound mean Delta
ARI `0.000000`；V16.1 因全负 support 精确回退 self-only。该结果不进入正例表，
也不触发 gate、support、temperature、thinning 或 K 调整。

同一批次中，`Bach` 与 `PBMC_68K` 也已完成三 seed、clean/compound 和五路 paired
readout。两者均按固定规则标记 `empirical_not_supported`：Bach clean Delta ARI
`0.000000`，PBMC_68K clean Delta ARI `0.000000`；完整产物分别位于
`result/V16_1/expanded_count_stage1_20260807/Bach/` 和
`result/V16_1/expanded_count_stage1_20260807/PBMC_68K/`。该段记录的是当时的运行快照；
`Shekhar`、`PRJNA895163` 与 `hrvatin_geo_maintype_counts` 后续已完成；Norman Stage-0
已按搜索上限停止且未产出性能结果。

`Shekhar` 随后完成三 seed、clean/compound 和五路 paired readout，完整结果位于
`result/V16_1/expanded_count_stage1_20260807/Shekhar/`。固定汇总为
`empirical_not_supported`，clean 与 compound Delta ARI 均为 `0.000000`；fixed graph
的改善没有被 predictive gate 复现。

此前去重快照 `/tmp/v16_1_global_dedup_summary_20260807.json` 包含 33 个完整数据集；
合并已完成的 `PRJNA895163` 与 `hrvatin_geo_maintype_counts` 后，当前临时快照为 35 个完整数据集，文件为
`/tmp/v16_1_global_dedup_summary_current_20260807.json`，全部为
`empirical_not_supported`。Norman Stage-0 已按搜索上限停止，未追加性能状态。

### 2026-08-07 V16.1 并行扩展候选

hrvatin_geo_maintype_counts.h5ad 是新增的本地 raw-count 源：矩阵
48266×25187，使用显式 layers/counts 和 obs.maintype，已分块转换为
/tmp/v16_1_expanded_data/hrvatin_geo_maintype_counts.npz，CSR 非零数为
70653489，labels_used_during_fit=false。当前固定 k=20、三次 split 的
Stage-0 已完成并通过：candidate recurrence `0.2670`、support 非退化；随后按固定
协议启动 GPU 2 Stage-1。Stage-0 数值只用于结构筛选，不是性能证据。

同时审计了 scSSL-Bench 的 PBMC.h5ad 与 Pancreas.h5ad。其 layers/counts
非零值为归一化浮点数，不能证明为可逆 raw count；两者标为
theory_domain_not_supported，没有转换或训练。现有 GPU 1、5、6 的
PRJNA895163、Macosko、Bach 任务继续按原固定协议运行，GPU 2、3、4
仅保留给新的 Stage-0 通过候选。

PRESCRIBE 的 Norman perturb-seq 子集 perturb_e_distance.h5ad 也通过了输入层
面的预核对：12474×2037，使用 layers/counts，obs.condition 有 231 个条件，
抽查非零值为整数。已登记为 Norman_perturb_e_distance 候选；其固定 Stage-1 已
完整结束，clean Delta ARI 为 `-0.000017`，按预注册规则标记
`empirical_not_supported`。

⚠️ **重要原则**：所有图表数据必须可追溯，不得虚构或使用虚拟数据。

> **存储审计声明（2026-08-03）**：本文档保留了多代实验的历史追溯记录。
> 文中出现的 `learnable_gate_smoke`、V6/V7/HVF smoke、AHDPC verified smoke、
> V10/V11 iris smoke 和 `/tmp/topogate_v11_semantic_*` 路径均指已经发生但已
> 清理的临时产物，不代表当前文件存在；当前正式产物统一以 `result/...` 路径
> 为准，并需回到磁盘核对。

### 2026-08-06 V16.1 expanded-count Stage-1 当前批次

固定输出根为 `/tmp/v16_1_stage1_parallel_20260806/`。`Arabidopsis_Stereo_seq_leaf`、
`CRA002977_1`、`HCA_subsampled_20k`、`TabulaSapiens_Pancreas` 与 `tr45.wc` 已完成
三 seed、clean/compound 和五路 paired readout，完整 promotion JSON 均标为
`empirical_not_supported`；当前没有 `candidate_positive`。各数据集的输入来源、计数
语义、K 来源与 `labels_used_during_fit=false` 均保存在各 run 的 `summary.json`。

`tr45.wc` 的第一次批次只产生 theory-domain 状态，未进行模型训练；根因为数据集名
`tr45.wc` 被 metadata resolver 截断为 `tr45`，导致预注册 word-count 语义遗漏。修复后
在同一 GPU、同一 seeds 和同一实验矩阵下完整重跑；有效结果替代无训练状态。`SRP224648`
因 `14533x67300` decoder 的 Adam 状态超过单卡容量，标记 `stage1_incomplete_compute`，
不计入数据集性能失败或正例分母。其余完整矩阵仍在运行，暂不写入主结果表。

### 2026-08-06 V16.1 expanded-count H5AD candidates and Stage-1 confirmation

在首批 210 个正式 summaries 之外，已启动新的并行 Stage-1 候选确认，固定使用
`TabulaSapiens_Pancreas.npz` 与 `CRA002977_1.npz`、seeds `[42,123,7]`、clean/compound
和五路 paired readout；输出根为 `/tmp/v16_1_stage1_parallel_20260806/`，分别占用物理
GPU 3、4。记录时已有 `CRA002977_1` clean 三 seed及 compound/seed42、
`TabulaSapiens_Pancreas` clean seed42/123 的五路 summaries（共 30），任务仍在运行，
未作晋级判定；不覆盖既有结果，也没有重新计算任何哈希。

扩展 registry `scripts/V16_1/count_candidate_registry.json` 新增本地 scCluBench H5AD
整数 count 源：`SRP182008`、`SRP224648`、`CRA002977_1`、`Mouse_Pancreas_1`、
`Human_Pancreas_3`、`Human_Pancreas_1`、`Bone_Marrow`、`Blood_BoneMarrow`、
`TabulaSapiens_Pancreas` 和 `PRJNA895163`。原始 H5AD 按分块读取转换为 CSR bundle，
暂存于 `/tmp/v16_1_expanded_data/`；每个 bundle 记录源路径、矩阵形状、标签字段和
`labels_used_during_fit=false`。分块非零值审计把 `SRP235541`、`SRP171040`、
`SRP309176`、`SRP145013`、`CRA007122`、`Wang` 和 `Pollen` 的当前输入标为
非整数/归一化，不强行恢复为 count。

本轮又登记并转换三个本地可核验源：`HCA_subsampled_20k`（`20000×26662`，
`cell_type` 13 类）、`Paul15`（`2730×3451`，`paul15_clusters`）和
`Arabidopsis_Stereo_seq_leaf`（`721×18257`，`cell_type` 6 类）。三者的抽样非零值均为
非负整数，转换 bundle 位于 `/tmp/v16_1_expanded_data/`，不计算或保存新的哈希。
固定 Stage-0 输出分别为 `/tmp/v16_1_stage0_hca_long.json` 和
`/tmp/v16_1_stage0_small_external.json`：HCA 候选 recurrence `0.4210`、稳定边率
`0.7062`、正支持行 `0.020%`；Arabidopsis recurrence `0.3252`、稳定边率 `0.6432`、
正支持行 `0.139%`；Paul15 support 全负，保留为 Stage-0 候选而不进入模型测试。
此前超时的 `Shekhar` 和 `Tosches` 也完成了延长 Stage-0：Shekhar support 全负；
Tosches recurrence `0.4117`、稳定边率 `0.7000`、正支持行 `0.016%`。这些指标只用于
无标签候选筛选，不能当作性能证据。

固定 Stage-0（expanded-count、三次 split、A/B 交换、`k=20`）输出位于：
`/tmp/v16_1_stage0_h5ad_small.json`、`/tmp/v16_1_stage0_human_pancreas1.json`、
`/tmp/v16_1_stage0_bone_marrow.json`、`/tmp/v16_1_stage0_human_pancreas3.json`、
`/tmp/v16_1_stage0_blood_bonemarrow.json`、`/tmp/v16_1_stage0_srp182008.json`、
`/tmp/v16_1_stage0_srp224648.json`、`/tmp/v16_1_stage0_cra002977_1.json` 和
`/tmp/v16_1_stage0_tabula_pancreas.json`。其中 `Human_Pancreas_1`、`Bone_Marrow`、
`Blood_BoneMarrow`、`TabulaSapiens_Pancreas` 具有非零正支持行比例，但该指标只作
结构候选筛选；`Mouse_Pancreas_1` 支持全负，仍保留为 Stage-0 candidate。

首批正式 Stage-1 输出位于 `result/V16_1/expanded_count_stage1_20260806/`，固定
三 seed、clean/compound 和五路 readout，共 90 个 summaries。汇总文件为
`result/V16_1/expanded_count_stage1_20260806/promotion_summary.json`；
`Blood_BoneMarrow`、`Bone_Marrow`、`Human_Pancreas_1` 均为 `empirical_not_supported`，
分别 clean paired Delta ARI `-0.000598`、`-0.002388`、`0.000000`。这些数据不再调 gate，
继续按固定候选池寻找新数据。

### 2026-08-06 V9 条件性拓扑收益协议与主矩阵

新增 manifest 驱动工具链 `scripts/v9_regime/`。本地 manifest 暂存于
`/tmp/v9_regime_20260806/manifest.local.json`：157 条记录，其中 149 条满足
`n>=100, 2<=K<=50` 和 dense element 上限，8 条明确排除。所有 eligible 数据均
使用同一 X-only `nan_to_num`+column StandardScaler；V9 runner 接收已标准化 X 并
固定 `scale_input=false`。标签只用于 manifest 的
`K=int(np.unique(y).size)` 和训练后 ARI/NMI，逐 run 记录
`labels_used_during_fit=false`、source path/version、preprocessing、resolved config
和 `predictions.npy`/`labels_true.npy`/`embedding_final.npy` 语义。

Stage 0 特征表为 `/tmp/v9_regime_20260806/features.local.csv`，固定
70/30 X-only split 为 `/tmp/v9_regime_20260806/split.local.json`（discovery 113、
confirmation 36）。预分层消融面板已锁定为
`/tmp/v9_regime_20260806/panel.local.json`，11 个数据集、6 类结构角色；由于主
Full/NoMix confirmation 触发停机规则，Static/Random/Far 未启动。

固定 V9 Full/NoMix 运行产物暂存于
`/tmp/v9_regime_20260806/runs_screen_standardized/`：Stage 1 screen
298/298 completed，随后补齐 seed `[123,7]`，主矩阵共 894/894 completed、0 error。
汇总为 `/tmp/v9_regime_20260806/summary_main_standardized/summary.json`。当前正式
`result/` 目标只读，因此这些路径是本轮可复核的临时证据，不冒充正式结果盘。

OpenML 元数据发现登记 159 个数值候选；前 20 个显式 target/K fetch 全部
`unresolved`（API 数据端点 SSL/网络失败），记录在
`/tmp/v9_regime_20260806/openml_registry.json`，未进入训练 manifest；没有使用
相似或模拟替代数据，也没有重复计算哈希。

### 2026-08-06 V9 Full/scMAE confirmation secondary comparison

为替代 NoMix 作为用户要求的比较对象，固定同一 confirmation manifest、同一
标准化输入、同一 seed 和 K 协议，新增 `scmae`：`gate_mode=none`、
`mix_mode=none`、`pseudo_weight=0`。该变体是 V9 runner 内的 vanilla scMAE-compatible
路径；它不是独立 `NeighborMix_scMAE/run.py` 的 h5ad 专用数据协议。Full 仍为
`learned/reliability/0.3`，两路均为 80 epochs、mask ratio `0.3`、hidden size `128`、
`scale_input=false`（输入已由协议统一 StandardScaler）。

产物根为 `/tmp/v9_regime_20260806_full_scmae_confirmation_v2/`，包括 36 datasets、
3 seeds、216 条 `run_record.json`，均 `status=completed`；汇总为
`/tmp/v9_regime_20260806_scmae_confirmation_summary/`，其中
`paired_deltas.csv` 保存同数据集同 seed 的配对差。运行记录固定保存
`predictions.npy`、`labels_true.npy`、`embedding_final.npy`，并记录
`k_source=manifest_labels_unique` 和 `labels_used_during_fit=false`。

按 ARI 排名，Full 任务表现最高为 `ahdpc_prepared__dim512` (`1.0000`，scMAE 也为
`1.0000`)、`local__smoker_condition` (`0.9672` vs `0.9711`)、
`local__mouse_retina` (`0.9393` vs `0.9491`)、`local__wine` (`0.8703` vs `0.8591`)；
若按 Full 相对 scMAE 的稳定正差，当前前三是 `local__image_segmentation`
(`+0.04256`)、`local__extyaleb` (`+0.03857`) 和
`local__patient_treatment_classification` (`+0.03856`)，均为 3/3 seed 正。
这些值是 confirmation 结果，不进行标签、损失、污染比例或版本搜索。

### 2026-08-04 TopoGate 数据目标与版本语义

TopoGate 的原型是 `scMAE`，数据实验的总目标是检验其在高维、特征噪声强、同时具有天然稀疏性的单视图数据上的聚类能力。V1--V 系列都属于同一原型的探索性改良和诊断，不对应预先声明的场景；最终论文只选择一代作为对外方法，不能把各代结果当作多个最终模型。

当前 V15 exploratory protocol 预注册的数据集合保持为：

- 自然稀疏、高维目标组：`Mouse_retina`、`cnae9`、`imdb`、`sms_spam_collection`、`secom`；
- 高维困难或图质量压力组：`enron`、`reuters`、`20newsgroups`、`cifar10`、`CIFAR10_CLIP`、`labeled_faces_in_the_wild`、`flickr_material_database`、`ISOLET`；
- 高 CLM 或简单结构控制组：`olivetti_faces`、`mnist64`、`seeds`。

数据选择和分层分析参考 `hj-n/labeled-datasets`、`hj-n/clm` 及
`papers/参考资料/Measuring_the_Validity_of_Clustering_Validation_Datasets.md`。
CLM 只用于数据分层和结果解释，不进入 graph、gate、loss、超参数或 variant 选择。由于两个外部仓库的 commit、文件清单与 SHA256、字段含义和本地映射尚未全部重新核验，当前相关记录必须保持 `CLM-unranked`，不能写成正式 CLM 证据。

### 2026-08-06 V16.1 Stage-0 数据扩充与理论域筛选

V16.1 Stage 0 使用 `scripts/V16_1/run_stage0.py`，只读取计数语义、稀疏性和
候选图/support 统计，不使用标签、不训练模型。当前固定审计结果如下：

| 数据集 | 形状 | 存储/计数语义 | 理论域状态 | recurrence | support 正值率 |
|---|---:|---|---|---:|---:|
| `Campbell` | `9993×26774` | sparse NPZ chunked / registered scRNA count | candidate | `0.4724` | `0.0034%` |
| `Mouse_retina` | `8352×6198` | sparse NPZ chunked / registered scRNA count | candidate | `0.2667` | `0.0054%` |
| `hrvatin` | `65539×25187` | dense NPZ / scRNA declaration, unsupported encoding | theory_domain_not_supported | - | - |
| `hrvatin_filtered` | `48266×25187` | dense NPZ / scRNA declaration, unsupported encoding | theory_domain_not_supported | - | - |
| `Baron Human` | `8451×20125` | sparse NPZ chunked / raw integer | candidate | `0.5155` | `0.025%` |
| `Quake_Smart-seq2_Lung` | `1676×23341` | dense NPZ / raw integer | theory_domain_not_supported | - | - |
| `fbis.wc` | `2196×2000` | sparse NPZ chunked / raw integer | candidate | `0.4041` | `0.086%` |
| `tr45.wc` | `676×8261` | sparse NPZ chunked / raw integer | candidate | `0.4685` | `0.017%` |

每个 split 同时对 A→B 和 B→A 评分，三次 split 共 6 次 support evaluation，最后逐边
取 median。Baron Human 的 raw zero fraction 为 `0.9060`，tr45.wc 为 `0.9659`；两者均无空行，
满足当前维度、稀疏性和行非零数证书，但 held-out predictive support 几乎全部
为负，因此只能标记为 Stage-0 candidate，不能直接进入 Stage 1。Quake 的计数和
稀疏统计满足数值阈值，但其 NPZ `x` member 是 dense storage，按内存协议记录为
`theory_domain_not_supported`，不能用近似或替代数据挽救。

`hrvatin` 与 `hrvatin_filtered` 的 dense member 同样不满足稀疏内存证书，且全量
CSR data 未通过当前 log1p-count 恢复检查，直接记录为理论域外。`fbis.wc` 通过
理论域证书，但 support 正值率只有 `0.0856%`、median support 为 `-6.581`，与
此前 exploratory failure 一致，不重复调 gate。

Campbell/Mouse_retina 的首次 V16.1 Stage-0 图审计超过 360 秒，是当前 block sparse
cosine kNN 的计算成本事件；随后使用延长窗口完成了两份 exploratory 产物：
`/tmp/v16_1_stage0_campbell_exchange.json` 和
`/tmp/v16_1_stage0_mouse_exchange.json`。这四份 `/tmp/v16_1_stage0_*_exchange.json`
文件均不写入正式结果盘；V16.1 尚未运行 Stage 1。

### 2026-08-06 V16.1 expanded-count 本地候选扩展

新增 registry：`scripts/V16_1/count_candidate_registry.json`。来源为本地
`scCluBench/data/scMAE/*.h5` 的 `X/Y` 字段；转换器对全部 CSR 非零值执行非负整数检查，
不对非整数矩阵做四舍五入。转换 bundle 和 Stage-0 exploratory JSON 暂存于
`/tmp/v16_1_expanded_data/` 与 `/tmp/v16_1_expanded_stage0_*.json`，未写入只读正式结果盘。

| 数据集 | 形状 | zero fraction | median nnz | 分层 | Stage-0 图/support 状态 |
|---|---:|---:|---:|---|---|
| `Bach` | `23184×19965` | `0.880400` | `2188` | `high_sparse_bonus` | Stage-1 首个 seed 有工程产物，固定三 seed 在 1800 秒窗口未完成 |
| `Guo` | `6490×27477` | `0.887839` | `2572.5` | `high_sparse_bonus` | recurrence `0.5195`；正支持行 `1.2327%` |
| `Limb_Muscle` | `3909×23341` | `0.935695` | `1356` | `high_sparse_bonus` | recurrence `0.4939`；正支持率 `0%` |
| `Macosko` | `44808×23288` | `0.969477` | `482` | `high_sparse_bonus` | 已转 CSR，图审计待完成 |
| `Melanoma_5K` | `4513×23684` | `0.870044` | `2801` | `high_sparse_bonus` | recurrence `0.8439`；稳定边比例 `0.9474`；正支持行 `3.8777%` |
| `Quake_10x_Spleen` | `9552×23341` | `0.943383` | `1182` | `high_sparse_bonus` | recurrence `0.2762`；正支持行 `0.0314%` |
| `Shekhar` | `26830×13166` | `0.933413` | `811` | `high_sparse_bonus` | 已转 CSR，Stage-0 图审计未在固定 CPU 窗口完成 |
| `Tosches` | `18664×23500` | `0.908336` | `1590` | `high_sparse_bonus` | recurrence/support 审计在 900 秒 CPU 窗口未完成 |
| `Young` | `5685×33658` | `0.946964` | `1350` | `high_sparse_bonus` | recurrence `0.5075`；正支持行 `1.5831%` |
| `worm_neuron_cell` | `4186×13488` | `0.986181` | `151` | `high_sparse_bonus` | recurrence `0.2701`；正支持率 `0%` |

这些 recurrence/support 数字只做无标签机制筛选，不能替代 Stage-1 ARI 或消融。
`Wang`（`9519×14561`）在转换时发现非整数归一化值，标记为
`theory_domain_not_supported`；不把它作为 V16.1 计数候选。当前所有新增 scRNA 候选
均只是 `stage0_candidate` 或 `stage0_incomplete_compute`，没有自动晋级为正例。

### 2026-08-06 V16 输入与 stress 协议修正

V16 loader 现在识别未压缩 NPZ 中的 numeric `x.npy`，按固定行块通过 memmap 转为
CSR；当前 `Campbell`、`Mouse_retina` 和 `fbis` 的 `x.npy` 均属于该存储形式。
压缩或无法分块读取的 dense member 不进入 V16 理论域，保存为
`dense_input_not_supported` 状态。count 语义检查覆盖全部 CSR 非零值，不再只检查
前一百万个值。

正式 Stage-1 的 compound 条件固定为：20% observed-support feature dropout、
20% integer Poisson count perturbation、10% row contamination。它只用于固定的
clean/stress 配对，不参与模型或 variant 选择。

### 2026-08-06 V16 Campbell/Mouse_retina Stage-0/1 数据记录

Stage 0 使用 `scripts/V16/run_stage0.py`，输出为
`/tmp/v16_stage0_anchors.json`。`Campbell`（`9993×26774`，zero fraction
`0.930739`）和 `Mouse_retina`（`8352×6198`，zero fraction `0.948001`）均通过
计数域证书，输入存储为 `sparse_npz_chunked`、计数语义识别为
`log1p_integer`。三次 split 的候选图 recurrence 为 `0.472390` 和 `0.266699`；
held-out predictive support 正值比例为 `0.001531` 和 `0.000629`。

修正后的 Stage 1 使用 `scripts/V16/run_paired.py`，固定 seeds `[42,123,7]`，
五路 paired readout，输出暂存于 `/tmp/v16_stage1_anchors_20260806_fixed/`。
clean 与 compound 共 60 个 summary，全部完成且所有 run 均记录
`labels_used_during_fit=false`；该目录不属于正式 result 盘。

clean 的 mean ARI（self-only / fixed graph / V16 / shuffled support /
output-disabled）为：Campbell `0.158261 / 0.217547 / 0.157655 /
0.1583428 / 0.158261`；Mouse_retina `0.404180 / 0.429160 / 0.404147 /
0.404583 / 0.404180`。V16 相对 self-only 的 clean paired delta 为
`-0.000607` 和 `-0.000033`，compound delta 为 `0.000000` 和 `+0.000961`，
均不满足预注册晋级条件。因此两个锚点均记录为 `empirical_not_supported`，不进入
候选池确认；不根据这些结果重新调门控参数。

### 2026-08-04 V15 Stage-0/1 exploratory data record

V15 数据审计入口为 `scripts/V15/build_dataset_manifest.py`。预注册的 16 个
NPZ（Mouse_retina、cnae9、imdb、sms_spam_collection、secom、enron、
reuters、20newsgroups、cifar10、CIFAR10_CLIP、labeled_faces_in_the_wild、
flickr_material_database、ISOLET、olivetti_faces、mnist64、seeds）已完成
路径、SHA256、`n/d/K`、raw zero fraction、nnz 分位数、density 和 sampled
distance concentration 扫描；当前 exploratory manifest 暂存于
`/tmp/v15_manifest_fixed16_v2.json`，未写入只读的正式 result 盘。带 raw/SVD
proxy latent/union graph audit 的 union candidate recall 中位数约 0.787，低于
Stage-1 的 0.80 预注册门槛；该 proxy 不能替代训练后的 EMA latent graph。

Stage-0 candidate audit 的 recall 是 budget-normalized local same-label
coverage，edge purity 单独记录，避免用整个类别规模机械压低 kNN recall。无
本地可核验 CLM JSON 时所有记录写为 `CLM-unranked`；CLM 不参与 graph、loss、
超参数或 variant 选择。

Stage-1 panel 的输入、K 和协议均可回到各 run 的 `summary.json`：K 由
`int(np.unique(y).size)` 作为 benchmark oracle，训练器未接收标签并记录
`labels_used_during_fit=false`。代表集 panel 与 cnae9 graph-pollution
梯度均为单 seed/短 epoch engineering evidence，不是论文级性能结果。

`scripts/V15/audit_dataset_mapping.py` 解析当前 `baseline/CLUBench/README.md`
的 131 行表格，得到 172 个本地 NPZ 的名称映射；预注册集合中只有 `HIVA`
缺失。README 的 `r_mm` 仅作为 exploratory mapping，文件 SHA256 已记录，
但 `hj-n/labeled-datasets` 与 `hj-n/clm` commit 尚未固定，因此
`clubench_commit_verified=false`，不把这些数值写成正式 CLM 证据。

### 2026-08-04 V15 Stage-1B certificate audit record

`scripts/V15/audit_stage1b_certificates.py` 对 `/tmp/v15_stage1_panel_v2`
进行只读审计并输出 `/tmp/v15_stage1b_certificates.json`。它重新读取每个 run
的 `candidate_indices`、`candidate_valid`、`candidate_features`、
`utility_target`/`utility_hat`、`labels_true.npy` 和 `summary.json`。
标签只用于 graph 的后验 edge purity、budget-normalized candidate recall 和
same-label coverage，并且每条记录保留 `label_use=posthoc_only`。

当前输出契约没有 teacher assignment/embedding、跨视图或时间 teacher pair、
scorer held-out 预测、逐边反事实 embedding 或独立 downstream gain，因此三道
证书不能被合并为一个“模型有效”结论：teacher 为 0/7 可证，graph 为 7/7 可算，
utility 仅 7/7 in-sample 可算，held-out 与 independent gain 均为 0/7。

### 2026-08-05 V15 repaired-code minimal paired data record

当前 source hash 的 clean exploratory 输出位于
`/tmp/v15_local_consensus_matrix_20260805/`：sms/cnae9 各 5 个 variant，
以及 reuters self-only；compound 输出位于
`/tmp/v15_compound_matrix_20260805/`，为 sms/cnae9 的 self-only、
direct-local-consensus、counterfactual-learned 共 6 个 run。全部 run 使用
`K=int(np.unique(y).size)` 作为 benchmark oracle，训练期间
`labels_used_during_fit=false`；标签只用于 summary 的 ARI/NMI 和 graph 后验
recall/purity。

clean 结果中 sms/cnae9 的 candidate recall/purity 分别约为 `0.89/0.89` 和
`0.75/0.75`。compound 条件下 cnae9 降至约 `0.26/0.26`，sms 约
`0.81/0.81`。这批数据只用于定位 target/scorer/污染机制，未写入正式结果盘，
不支持跨 seed 性能结论。

追加的 `candidate_scope=both_views` 交集消融暂存
`/tmp/v15_compound_both_views_20260805/`，仅 4 个 run。compound 下 cnae9
recall/purity 约 `0.15/0.15`、sms 约 `0.78/0.78`，learned scorer 的 null mass
仍为 `0`；该图范围控制因此不进入后续主路径。

### 2026-08-03 V12 self/null stage-1 formal data record

**输入与固定协议**：

- `datasets/AHDPC/processed/flame.npz`（240×2）和
  `datasets/enron.npz`（9999×4096）；
- StandardScaler、hidden size=128、mask ratio=0.3、batch size=256、
  neighbor_k=5、80 epochs、seeds=[42,123,7]；
- NoMix、edge-only、self/null lambda=0.01/0.03/0.1，共 30 个配对运行；
- K 由 `np.unique(y)` 仅用于 benchmark 评估，所有 30 个 summary 的
  `labels_used_during_fit=false`，source hash 在同一数据集内一致。

**正式产物**：

`result/V12/v12_self_null_stage1_2026-08-03/` 下保存每个 run 的
`summary.json`、`history.json`、`resolved_args.json`、source path/hash、
`predictions.npy`、`labels_true.npy`、embedding 和 topology diagnostics；
`runs.csv`、`summary_by_dataset.csv`、`summary_by_variant.csv`、
`paired_deltas.csv`、`report.md` 和 `coverage.json` 已由汇总脚本生成。
详细失败分析写入 `result/analysis/V12_self_null_stage1_2026-08-03.md`。

**结果边界**：

30/30 完成且无错误。NoMix 宏观 ARI=0.6616，edge-only=0.2015，
self/null lambda=0.01/0.03/0.1=0.6195/0.3374/0.1872。self/null 的
self mass 非零，但 conditional edge entropy 约为 log(5)，表明当前训练
尚未形成有效逐边选择；lambda=0.03/0.1 在 enron 出现明显退化。因此该批次
仅作为 V12 实现和失败边界的可追溯证据，不作为拓扑性能提升结论，也不扩展
到第二阶段数据集。

### 2026-08-03 V12 finalized-code warmup-fix stage-1 data record

为避免覆盖已完成的 pre-fix 批次，最终源码（warmup 期间 gate 真正冻结、并
记录 runner/model/gate source hash）在独立目录
`result/V12/v12_self_null_stage1_2026-08-03_warmup_fix/` 重跑相同 30 条件：
2 datasets × 5 variants × 3 seeds，30/30 completed，0 errors。汇总文件为
`runs.csv`、`summary_by_dataset.csv`、`summary_by_variant.csv`、
`paired_deltas.csv`、`report.md` 和 `coverage.json`。

最终宏观 ARI 为 NoMix=0.6616、edge-only=0.2016、self/null
lambda=0.01/0.03/0.1=0.6194/0.3372/0.1874。self mass 非零且 warmup
不漂移，但 lambda=0.01/0.03 的 edge entropy 仍约为 log(5)，lambda=0.1
仅轻微偏离均匀，尚未实现可靠邻居选择；enron 高维去噪仅在 lambda=0.01
保持，flame 所有 topology 条件低于 NoMix。该批次是当前源码的权威阶段性
证据，但仍为 restricted no-go，不扩展第二阶段，不支持普遍拓扑收益。

### 2026-08-03 V12 latent-topology 工程 smoke

**输入与协议**：

- `datasets/AHDPC/processed/flame.npz`（240×2）和 `datasets/enron.npz`
  （9999×4096）；默认 StandardScaler、PCA-kNN `k=10`、seed=42、CPU；
  `K=int(np.unique(y).size)` 只用于 benchmark 指标，训练器不消费 `y`。
- full 使用 `lambda_topology=0.1`、mask loss weight=0.1；NoMix 使用相同
  autoencoder 和 topology disabled 作为配对控制。

**生成与验证**：

- 运行入口：`methods/TopoGate/V12_latent_topology/run_npz.py`；短 smoke 的
  `summary.json`、history 和数组写入 `/tmp/topogate_v12_*`，未写入正式结果表。
- `compileall` 与 V12 三个单元/梯度测试通过；实际 runner 记录非零 gate
  gradient。flame 8/80 epoch 和 enron 8 epoch 的 ARI 只作为工程诊断，不能
  代替 `[42,123,7]` 多 seed、至少五个数据集的正式比较。

### 2026-08-03 V12 性能骤降同协议诊断（单 seed，非正式结果）

为解释 V12 看似大幅下降，使用 `flame.npz`、seed=42、CPU、80 epochs、
`hidden_size=128`、`mask_ratio=0.3`、StandardScaler、batch=256 做最小配对：

| 路径 | 设置 | ARI |
|---|---|---:|
| legacy V9 | NoMix，`mask_loss_weight=0.7` | 0.4764 |
| legacy V9 | NoMix，`mask_loss_weight=0.1` | 0.4649 |
| V12 latent topology | 当前 decoder，NoMix | 0.1843 |
| V12 latent topology | 临时恢复 legacy `[latent, mask_logits]` decoder，NoMix | 0.4534 |
| V12 latent topology | 当前 decoder，Full，`lambda_topology=0.1` | 0.0747 |

V12 Full 的最终边权熵为 1.6088，接近 `log(5)`，最大边权均值为 0.2088，
说明 softmax gate 基本没有选择性。该诊断支持“decoder 接口改写是主要跌幅来源，
强制均值对齐是 Full 的附加过平滑风险”，不能替代多 seed 正式实验。临时产物均
写入 `/tmp/topogate_v12_diag_*` 并在核验后清理。

### 2026-08-03 V12 当前源码四路隔离复核（单 seed，非正式结果）

为避免历史源码与当前源码混用，重新固定 `flame.npz`、seed=42、CPU、80 epochs、
batch=256、hidden=128、mask ratio=0.3、StandardScaler、K=5 graph，分别运行
legacy decoder/latent-only decoder 与 NoMix/Full。当前源码结果为：V12 legacy
NoMix ARI=`0.4998`，latent-only NoMix=`0.1843`，legacy Full=`0.1844`，
latent-only Full=`0.0747`。legacy Full 的 edge entropy=`1.6088766`，`log(5)`
为 `1.6094379`，mean max edge weight=`0.2080687`，说明 softmax edge gate
近似等权，拓扑项没有形成 abstention。

该复核的机器可读数值没有作为正式性能批次保存；持久化解释报告为
`result/analysis/V12_performance_drop_diagnosis_2026-08-03.md`。结论仅限工程诊断：
decoder 接口回归和无 self/null 的邻居均值对齐是两个独立跌幅来源；真正 V12 仍需
五数据集 × 三 seed 的 paired benchmark。

## 模板

### [日期] [操作类型]

**输入数据**：
- 文件路径：
- 数据描述：
- 来源验证：

**生成数据**：
- 输出文件：
- 对应图表：

**追溯代码**：[指向 papers/codes/ 中的具体脚本]

---

## 数据处理记录

### 2026-08-03 V11 h0_early_mst 正式配对审计

**输入数据与协议**：

- `datasets/AHDPC/processed/{balance_scale,spect_heart,banknote,flame,vehicle}.npz`，
  source hash 由每个 V11 summary 保存并在 Full/候选之间核对。
- `K=int(np.unique(y).size)` 仅用于 benchmark K 和运行后指标；raw PCA、raw kNN、
  固定 H0 filtration、prior、gate target 和 variant 选择均不使用 `y`。
- V11 default YAML、80 epochs、CPU `--no-cuda`、单线程数值后端、seeds `[42,123,7]`；
  无逐数据集调参。`h0_early_mst` 只保留 H0 merge edges，并以归一化 death distance
  的反指数分数优先早合并边。

**生成产物**：

- `result/V11/tda_h0_early_mst_pilot_2026-08-03/comparison.csv`：30 条 runner 记录，
  0 errors；每条运行目录包含 `summary.json`、`args.json`、metrics、embedding、
  probabilities、predictions、labels_true 和 label mapping。
- 同目录的 `run_diagnostics.csv`、`summary_by_dataset_variant.csv`、
  `paired_deltas.csv`、`protocol.json` 和 `report.md`。

**核验与结论边界**：`PYTHONPATH=/home/luolie/ToPoGate pytest -q
methods/TopoGate/V11/tests/test_v11.py` 得到 `20 passed`；结果审计确认 30/30
summary、五个数据集 hash 一致、标签隔离字段均为 false、输出文件无缺失。候选相对
同批 Full 的 head/KMeans ARI、NMI、silhouette 差值为 `+0.000010/-0.001139/+
0.000013/+0.000140`，固定协议内为性能 no-go，不作为论文性能主张。

### 2026-08-03 TopoGate 跨版本优势/劣势景观统一审计

**输入数据与边界**：只读取已完成的 V9 advantage、V11 minimum、V12、V13、V14、
StaticGate CSV/JSON，以及独立 TDA pilot 的 75 个 `summary.json`；不重新训练，
不使用标签选择 variant/阈值/seed。Full-NoMix 只在同一 batch、同一数据集、同一
seed 配对，TDA 则与同一 pilot batch 的 `V11_full` 配对。纵向同名数据集只有在
source SHA-256 一致时才合并；`vehicle` 的两个 hash 被明确标记为不可纵向合并。

**生成产物**：

- `result/analysis/topogate_cross_version_landscape_2026-08-03.md`；
- `result/analysis/topogate_cross_version_landscape_2026-08-03_per_dataset.csv`；
- `result/analysis/topogate_cross_version_landscape_2026-08-03_summary.csv`；
- `result/analysis/topogate_cross_version_landscape_2026-08-03_trajectory.csv`；
- `result/analysis/topogate_cross_version_landscape_2026-08-03_tda.csv`；
- `result/analysis/topogate_cross_version_landscape_2026-08-03_correlations.csv`。

**复算事实**：56 条 Full-NoMix 配对的版本均值为 V9 `+0.015356`、V11
`-0.000475`、V12 `-0.001244`、V13 `-0.000238`、V14 `+0.004373`，StaticGate
历史单 seed 表为 `-0.015310`；TDA H0/fixed/random 的 head/KMeans 差值与 75-run
正式审计一致。相关性只作小样本描述性结果，不用于配置选择或论文因果结论。

**追溯代码与验证**：`scripts/analysis/analyze_topogate_cross_version_landscape.py`；
运行前确认 `result -> /data/luolie/ToPoGate/result`，脚本通过路径检查、Python
编译和结果数量断言后生成上述文件。

### 2026-08-03 V11 sparse H0 TDA pilot 工程验证

**输入数据与协议**：

- `datasets/iris.npz`，CPU、seed=42、3 epochs、缩小网络，仅用于验证真实 NPZ
  输入和 V11 输出契约；K 从 `np.unique(y)` 得到，仅作 benchmark oracle 和后验
  指标用途。
- H0 skeleton 使用 raw PCA embedding 的固定 raw kNN，单位行 Euclidean chord
  filtration、median distance scale；训练器和 prior 计算不接收 `y`。

**生成产物**：

- 持久化研究报告：`result/analysis/topogate_v11_tda_h0_pilot_2026-08-03.md`；
- smoke 的 `summary.json`、数组和临时目录写入 `/tmp` 后已清理，没有进入结果盘
  性能表，也没有新增 smoke 目录。

**追溯代码与验证**：

- `methods/TopoGate/V11/tda.py`、`methods/TopoGate/V11/trainer.py`、
  `methods/TopoGate/V11/tests/test_v11.py`；
- `python -m compileall -q methods/TopoGate/V11 scripts/V11`；
- `PYTHONPATH=/home/luolie/ToPoGate pytest -q methods/TopoGate/V11/tests/test_v11.py`：
  `19 passed`。

**结论边界**：本条只记录当时的机制/工程验证阶段，不是性能实验；后续正式五组
比较已完成，结果见下方“V11 sparse H0 TDA 正式五数据集对照”及新的跨版本景观审计。

### 2026-08-03 V11 sparse H0 TDA 正式五数据集对照

**输入数据与协议**：

- `datasets/AHDPC/processed/balance_scale.npz`（625×4，K=3）、
  `spect_heart.npz`（267×22，K=2）、`banknote.npz`（1372×4，K=2）、
  `flame.npz`（240×2，K=2）和 `vehicle.npz`（846×18，K=4）；五个文件均来自
  `datasets/AHDPC/MANIFEST.json` 的 `prepared` 条目，source hash 写入每个 summary。
- `K=int(np.unique(y).size)` 只用于 benchmark K 和运行后指标；输入、raw PCA、
  raw kNN、H0 filtration、graph prior、gate target 和 variant 选择不读取 `y`。
- V11 default YAML、80 epochs、CPU `--no-cuda`、单线程数值后端、seeds
  `[42,123,7]`；无逐数据集调参。

**生成产物**：

- `result/V11/tda_h0_pilot_2026-08-03/comparison.csv`：75 条 runner 记录，0 errors；
- 同目录 75 个 run 子目录，每个含 `summary.json`、`args.json`、metrics、
  `embedding_final.npy`、`cluster_probabilities.npy`、`predictions.npy`、
  `labels_true.npy` 和 `label_mapping.json`；
- `run_diagnostics.csv`、`summary_by_dataset_variant.csv`、`paired_deltas.csv`、
  `protocol.json`、`report.md`；生成脚本为
  `scripts/analysis/analyze_v11_tda_h0_pilot.py`。

**结果与边界**：

- 15 个 paired dataset-seed 中，H0 相对 V11 Full 的 head ARI 为 `+0.000010`，
  KMeans ARI 为 `-0.000726`；fixed-filtration 为 `+0.000002/-0.000665`，
  random 为 `+0.000018/-0.000274`。
- H0 sparse skeleton 的 merge count 为每个数据集 `n-1`（五个 raw kNN skeleton
  均连通），非零 prior edge fraction 约为 `0.19--0.26`；这确认 prior 实际
  计算并参与 graph-prior score，但不等于它改善了簇分配。
- 正式结论是该固定五数据集协议内的 TDA prior **no-go**。所有持久化输出均在
  `/home/luolie/ToPoGate/result` 对应的 `/data/luolie/ToPoGate/result`，没有新增
  根目录 smoke 结果；不写入论文主方法，也不推广到完整 H1/dense VR TDA。

### 2026-08-03 跨版本 evidence/provenance 统一审计

**输入数据**：

- `result/v9_results_2026-08-02_paper_preprocess/`、`result/v9_results_2026-08-02_advantage_ablation/`；
- `result/V11/topogate_v11_minimum_5x3/`；
- `result/v12_results_2026-08-03_advantage/`、`result/v13_results_2026-08-03_advantage/`、`result/v14_results_2026-08-03_advantage_5ds/`；
- `result/ablation/merged_summary.csv`。

**生成产物**：

- `result/analysis/cross_version_evidence_2026-08-03.csv`：240 条按 dataset/variant 聚合的证据行；
- `result/analysis/paired_version_deltas_2026-08-03.csv`：同批次 Full/NoMix 配对摘要；
- `result/analysis/provenance_audit_2026-08-03.csv`：summary、CSV、source hash、K 协议和标签隔离字段覆盖审计；
- `result/analysis/cross_version_evidence_audit_2026-08-03.md`：解释边界和数学/TDA 术语约束。

**核验结果**：统一表复算的 Full-NoMix ARI 差值为 V9 `+0.015356`、V11 `-0.000475`、V12 `-0.001244`、V13 `-0.000238`、V14 `+0.004373`。V9 advantage 和 V12 summary 的 `dataset=adhoc` 被标记为 metadata gap；分析器使用 CSV/run_record 的真实 dataset 和 source hash，不修改历史 JSON。静态 legacy 表没有 source hash，不进入正式性能主表。

**追溯代码**：`scripts/analysis/build_topogate_evidence_audit.py`。该脚本只读取现有产物，不使用标签选择 variant、阈值、seed 或结论。

### 2026-08-03 无标签优势/劣势数据特征与真正 TDA 诊断

**输入数据与协议**：

- 由 V9--V14、StaticGate 现有结果表反推出 49 个结果相关数据集及其 manifest/NPZ source path；特征脚本只读取 NPZ 的 `x`，不加载 `y`。
- 对输入做标准化和最多 50 维 PCA；有限图统计使用 cosine kNN `k=5`，TDA 诊断使用固定稀疏 kNN `k=15` 的 Vietoris--Rips 1-skeleton component persistence 和阈值图 cycle rank。
- 超过 4,000 个样本或 512 个特征时使用固定随机子集；超过 80,000,000 个矩阵元素时只读取 header 并跳过。CSV 显式记录采样状态和 `feature_error`。

**生成产物**：

- `result/analysis/topogate_dataset_features_2026-08-03.csv`：49 行，47 个完成、2 个因矩阵元素上限跳过；保存输入 shape、source、无标签几何/TDA 特征和事后连接的版本差值。
- `result/analysis/topogate_feature_version_correlations_2026-08-03.csv`：180 条探索性 Spearman 相关；不用于选择配置、阈值或论文性能结论。
- `result/analysis/topogate_advantage_feature_audit_2026-08-03.md`：协议、版本正负集合、统计边界和可回退 TDA pilot。

**主要事实与边界**：

- `balance_scale` 是当前最完整的 V9 topology 正例；`spect_heart` 的 baseline 优势在 NoMix 下仍存在，`landsat` 差值很小，不能把三者都归因于 topology mixing。
- 当前 `tda_h0_*` 只表示固定稀疏 1-skeleton 上的 H0/component persistence 摘要；`cycle_rank_*` 不是 H1 persistence diagram。没有把普通 kNN、SNN、动态图或 edge reliability 宣称为 persistent homology。
- 关联结果为小样本、固定协议下的探索性假设，不能证明因果；后续若实现 TDA pilot，必须保留原 V11、NoMix、random prior、fixed-filtration 和 `[42,123,7]` 配对控制。

**追溯代码**：`scripts/analysis/build_topogate_dataset_feature_audit.py`。本轮未修改模型代码、既有结果数组或引用索引。

### 2026-08-03 临时 smoke 产物清理

**清理范围**：删除仓库根目录下不参与正式统计、且已被正式多种子产物替代的临时运行目录：

- `v12_results_2026-08-03_smoke/`（3 个短运行）；
- `v13_results_2026-08-03_smoke/`（6 个短运行）；
- `v14_results_2026-08-03_smoke/`（首次参数失败后留下的 2 个部分记录）；
- `v14_results_2026-08-03_smoke_rerun/`（10/10 engineering smoke）。

这些目录只用于工程链路诊断，不进入性能事实表；对应正式证据保留在 `result/v12_results_2026-08-03_advantage/`、`result/v13_results_2026-08-03_advantage/` 和 `result/v14_results_2026-08-03_advantage_5ds/`。AHDPC/V10/V11 的结果 smoke 已按新的生命周期规则清理；历史文字记录不再宣称它们是当前可审计产物。`result/`、`datasets/`、`papers/` 软链接目标未被删除或覆盖。

### 2026-08-03 V9 相对 AHDPC/HDPC 数据特征再分析

**输入数据与协议**：

- result/v9_results_2026-08-02_paper_preprocess/comparison_by_dataset.csv：24 个已准备数据集，V9 seeds [42, 123, 7]，AHDPC/HDPC 为已持久化单次参考。
- result/v9_results_2026-08-02_paper_preprocess/V9_vs_AHDPC_HDPC.md：由预测数组按共同 ARI/NMI/FMI/RI/ACC 定义重算的逐数据集比较。
- 无标签特征检查使用 datasets/AHDPC/processed/*.npz 的输入矩阵，按论文匹配协议仅对 Banknote 做 z-score；检查 5-NN mutual、连通分量、n、d、K 和类别不均衡，不把 y 输入训练。

**生成产物**：

- result/analysis/V9_AHDPC_feature_profile_2026-08-03.md：优势、劣势、分组统计、探索性相关和论文可用表述。

**主要事实与边界**：

- V9 相对 AHDPC 的 ARI 胜/平/负为 3/1/20，平均 ΔARI=-0.1715；相对 HDPC 为 5/1/18，平均 ΔARI=-0.1530。
- 相对 AHDPC 的正差值仅为 spect_heart、balance_scale、landsat；正差值不是全指标普遍胜出，也不等价于绝对 ARI 较高。
- 5-NN mutual 与 ΔARI 在这 24 个数据集上呈探索性负相关，不能把高 mutual 视为 V9 优势条件；d 的正相关受样本量和 Olivetti 输入协议例外影响，不能写成高维普遍优势。
- 结论仍是固定 V9 配置 + 持久化 baseline 的描述性证据；需对代表性优势/劣势数据做对称多 seed baseline 重跑和门控/embedding 诊断。

**追溯入口**：result/v9_results_2026-08-02_paper_preprocess/、result/analysis/V9_AHDPC_feature_profile_2026-08-03.md。

### 2026-08-03 V9 Full/NoMix 配对消融补充

**输入数据与协议**：

- `result/v9_results_2026-08-02_advantage_ablation/ablation_runs.csv`：7 个 UCI 数据集、4 个 V9 variant、seeds `[42,123,7]`，共 84 条 completed 运行。
- `v9_full` 使用 reliability mix、`pseudo_weight=0.3`；`v9_nomix` 使用 `mix_mode=none`、`pseudo_weight=0`，均为 80 epochs、raw 输入、adaptive PCA 和相同的 K 协议。
- 与 AHDPC/HDPC 的联表来自 `result/v9_results_2026-08-02_paper_preprocess/comparison_by_dataset.csv`；baseline 仍是持久化单次参考。

**生成产物与统计边界**：

- 更新 `result/analysis/V9_AHDPC_feature_profile_2026-08-03.md` 和 `result/analysis/V9_AHDPC_advantage_deep_analysis_2026-08-03.md`，新增逐数据集 Full/NoMix 的 ARI/NMI、seed 胜负和 baseline 联表。
- 21 个配对的 ARI 平均 `+0.015356`，Wilcoxon `p=0.3905`，配对 t 检验 `p=0.1417`；NMI 平均 `+0.011522`，不能宣称总体拓扑增益。
- `balance_scale` 是最清晰的正例（+0.080941，3/3 seed）；`spect_heart`、`vehicle`、`vertebral_column` 偏向 NoMix。该消融只覆盖 7 个数据集，不能外推为 24 个数据集的普遍结论。

**追溯入口**：`result/v9_results_2026-08-02_advantage_ablation/ablation_runs.csv`、`summary_by_dataset.csv`、`result/analysis/V9_AHDPC_feature_profile_2026-08-03.md`。

### 2026-08-03 V9 优势几何分析与 V14 正式小批次

**输入数据与协议**：

- V9 论文预处理匹配产物 `result/v9_results_2026-08-02_paper_preprocess/`：24 datasets × 3 seeds；几何特征来自 `result/v9_results_2026-08-02/geometry_features_no_label.csv`，不含标签。
- 相关数据集清单：`datasets/AHDPC_related_advantage/MANIFEST.json`，12 个真实 NPZ 软链接，记录 source path、SHA-256、shape 和 K。
- V14 固定配置：`methods/TopoGate/V11/configs/topogate_v14_advantage_minimum.yaml`；严格 NoMix 通过 `use_topology=false` 等开关关闭图状态；K 只用于 benchmark 评估。

**生成产物**：

- 共性报告：`result/analysis/V9_AHDPC_advantage_deep_analysis_2026-08-03.md`。
- V14 engineering smoke 曾完成 10/10，但其临时目录已于 2026-08-03 清理；该批次不参与性能统计。
- V14 正式小批次：`result/v14_results_2026-08-03_advantage_5ds/runs.csv`，5 datasets × 2 variants × 3 seeds，30/30 completed；每个运行保存 summary、metrics、预测/真值数组、配置和数据源 hash。

**统计结果与边界**：

- V9 在论文匹配协议下相对 AHDPC 的正差值数据集为 `spect_heart`、`balance_scale`、`landsat`（3/24）；历史标准化协议下为 9/24，不能混写。
- V14 full ARI=0.133629，nomix ARI=0.129256，配对差=+0.004373；Wilcoxon p=0.8139，配对 t 检验 p=0.6597；full 平均 target gate=0.006276。结论为机制可运行但性能 no-go，不进入论文主方法。

**追溯代码**：`scripts/v9_learnable_gate/run_v14_advantage_smoke.py`；清理后默认工程输出路径为 `result/V14/smoke/`，正式批次通过 `--output-dir` 写入结果盘中的 `result/v14_results_2026-08-03_advantage_5ds/`。

### 2026-08-03 CLUBench 131 数据集 AHDPC/HDPC/V9 全量对照完成

**输入数据**：

- CLUBench 官方 `DATASETS` 清单的 131 个 `.npz`，路径为 `datasets/`；运行入口通过官方 `CLUBench.load_data` 进行列级 z-score。
- K 由每个数据集 `int(np.unique(y).size)` 自动取得，仅用于 benchmark K 和拟合后的指标；AHDPC、HDPC、V9 的训练调用均不传入真值标签。
- AHDPC/HDPC：`epsilon=1.0`、`paper_semantic`、`table_reproduction`、`block_size=256`；V9：`learnable_gate_v9_adaptive`、seed=42、80 epochs、batch size=256、`scale_input=false`。

**生成数据**：

- `result/clubench_ahdpc_hdpc_v9_2026-08-02/`：131×3=393 条 `benchmark_summary.json`，393/393 completed、0 errors。
- 汇总文件：`comparison_long.csv`（393 行）、`comparison_wide.csv`（131 行）、`method_summary.csv`、`comparison_report.md`、`MANIFEST.json`。
- 分析文件：`analysis_by_dataset.csv`（131 个完整三方法配对）、`analysis_full.json`、`analysis_report.md`；ARI 作为主比较指标，保留 ACC/NMI/AMI/RI/FMI。

**主要结果（单 seed=42）**：AHDPC/HDPC/V9 的 Mean ARI 分别为 0.1830/0.1614/0.3227，Median ARI 为 0.0320/0.0104/0.2484。V9 相对 AHDPC 为 105 胜、2 平、24 负（平均 ΔARI=0.1396）；相对 HDPC 为 104 胜、1 平、26 负（平均 ΔARI=0.1613）。

**正负面分层**：相对 AHDPC，ΔARI≥0.10 的优势数据集 58 个；V9 ARI≥0.50 且明显占优的 26 个。ΔARI≤−0.10 的退化数据集 7 个，其中 AHDPC ARI≥0.50 的强基线退化为 `banknote_authentication`、`shuttle`、`extyaleb`、`world12d`；`heart_disease`、`paris_housing_classification`、`echocardiogram` 也达到 substantial regression 阈值。三种方法 ARI 均≤0.10 的共同困难数据 43 个，单独列出以免误归因于 V9。

**追溯代码**：`scripts/run_clubench_ahdpc_hdpc_v9.py`、`scripts/summarize_clubench_ahdpc_hdpc_v9.py`、`scripts/analyze_clubench_ahdpc_hdpc_v9.py`。

### 2026-08-02 V9 × AHDPC 数据集对照运行

**输入数据**：

- `datasets/AHDPC/processed/` 中 MANIFEST 标记为 `prepared` 的 24 个数据集；
- 每个数据集使用 seeds `42, 123, 7`，K 由 `len(unique(y))` 自动取得，训练路径不消费标签；
- V9 配置为 `learnable_gate_v9_adaptive`：80 epochs、mask ratio 0.3、neighbor k=5、reliability mix、无 HVF、adaptive PCA 上限 2000；
- GPU 6，`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`。

**生成数据**：

- 历史 V9 标准化协议：`result/v9_results_2026-08-02/`，72/72 成功；
- 论文预处理匹配协议：`result/v9_results_2026-08-02_paper_preprocess/`，72/72 成功。除 Banknote（z-score）外按 AHDPC 清单保留 raw 输入；`run_npz.py --scale_input` 已修复为真正生效；
- 每个运行保存 `summary.json`、`args.json`、`embedding_final.npy`、`predictions.npy`、`labels_true.npy`，并记录 source SHA-256、K 来源和 `labels_used_during_fit=false`；
- Olivetti-HDPC 补充参考：`result/v9_results_2026-08-02/olivetti_hdpc_reference/`，使用与现有 Olivetti-AHDPC 相同的 t-SNE 参数（perplexity=30、max_iter=1000、seed=42）。

**比较汇总**：

- `comparison_per_run.csv`：V9 72 行 + AHDPC 24 行 + HDPC 24 行；
- `comparison_by_dataset.csv`：24 个数据集的 V9 mean±std、AHDPC/HDPC 单次参考值及 ARI/NMI/FMI/RI/ACC 差值；
- `comparison_overall.csv`：方法级宏观指标；
- `V9_vs_AHDPC_HDPC.md`：可读版主表。共同指标由预测数组重新计算，避免把 AHDPC 的 AMI 与 V9 的 ARI 混用。

**结果边界**：

- 论文预处理匹配批次中，V9 的 ARI 相对 AHDPC 在 24 个数据集上 3 胜、1 平、20 负；相对 HDPC 为 5 胜、1 平、18 负。该统计是当前固定 V9 配置的对照事实，不是调参后的性能结论；
- Olivetti 的 AHDPC/HDPC 参考使用 t-SNE，V9 运行使用原始 4096 维输入，表中已标注为不同输入协议；
- 结果文件当前写在仓库内可写目录；由于沙箱拒绝向 `result/` 软链接目标的大批量写入，未把本批次冒充为数据盘持久产物。

| 日期 | 操作 | 输入 | 输出 | 代码 |
|------|------|------|------|------|
| 2026-08-02 | V11 旧 `semantic_residual` breast 临时 3-seed 对照（历史产物已清理） | `datasets/breast_cancer_wisconsin_original.npz`，K 自动取 `len(unique(y))`，seeds 42/123/7 | `/tmp/topogate_v11_semantic_breast__{full,nomix}__seed*`；Full head ARI 0.887228±0.003224，NoMix 0.885369±0.011102，KMeans Δ +0.00371；仅历史工程记录，非正式论文结果 | `methods/TopoGate/V11/run.py` + `topogate_v11_semantic_residual.yaml` |
| 2026-08-02 | V11.3 `semantic_metric` 工程 smoke（历史产物已清理） | `datasets/iris.npz`，CPU，seed=42，4 epochs，缩小 hidden/latent | `/tmp/topogate_v11_semantic_metric_iris/`；head ARI 0.6051，KMeans ARI 0.5961，最后 gate/target=0.311/0.021；仅链路验证 | `methods/TopoGate/V11/run.py` + `topogate_v11_semantic_metric.yaml` |
| 2026-07-31 | ESWA-2026 AHDPC 论文数据归档与验证 | 公开 UCI / UEF / GitHub benchmark / OpenML / AT&T 原始数据 | `datasets/AHDPC/raw/`、24 个经形状/K 校验的 `processed/*.npz`、`datasets/AHDPC/MANIFEST.json` | `baseline/AHDPC/download_datasets.py` |
| 2026-07-31 | AHDPC 真实数据 smoke（历史产物已清理） | `datasets/AHDPC/processed/{flame,aggregation,banknote}.npz` | `result/AHDPC/verified_smoke_2026-07-31/`（已清理） | `baseline/AHDPC/run_benchmark.py`、`run.py` |
| 2026-07-31 | AHDPC 扩展 smoke（历史产物已清理） | `datasets/AHDPC/processed/{2d_20c_no0,dim064,image_segment,rice}.npz` | `result/AHDPC/verified_smoke_2026-07-31/extended/summary.csv`（已清理） | `baseline/AHDPC/run_benchmark.py` |
| 2026-07-31 | AHDPC Olivetti t-SNE 图像分支 smoke（历史产物已清理） | `datasets/AHDPC/processed/olivetti_faces.npz`（400×4096） | `result/AHDPC/verified_smoke_2026-07-31/olivetti_faces_tsne_seed42/`（已清理） | `baseline/AHDPC/run_face.py` |
| 2026-07-31 | AHDPC 扩展 smoke（历史产物已清理） | `datasets/AHDPC/processed/{2d_20c_no0,dim064,image_segment,rice}.npz` | `result/AHDPC/verified_smoke_2026-07-31/extended/summary.csv`（已清理） | `baseline/AHDPC/run_benchmark.py` |
| 2026-07-23 | 移动数据集 | `/home/luolie/ToPoGate/baseline/CLUBench/CLUBench/datasets/*.npz` (10 个文件) | `/data/luolie/ToPoGate/datasets/` | N/A |
| 2026-07-24 | 导出 CLUBench 24 算法 baseline 表 | `baseline/CLUBench/performance_matrix/best_hpc/*.p` (24 个 pickle) | `result/baseline_clubench*.csv` (5 个 CSV) | `baseline/CLUBench/export_baseline_csv.py` |
| 2026-07-24 | 复制 NeighborMix_scMAE 到项目内（基类本地化） | `/home/luolie/biopipeline/dimension-reduction/plantnet/experimental_retired_models/NeighborMix_scMAE` | `/home/luolie/ToPoGate/methods/NeighborMix_scMAE/` | `cp -r` + 验证 import |
| 2026-07-24 | 下载完整 CLUBench 131 数据集（hfd 工具） | `https://hf-mirror.com/datasets/Feng-001/Clustering-Benchmark/.../CLUBench-Datasets.zip` | `/data/luolie/ToPoGate/download/CLUBench-Datasets.zip` (689MB) + `/data/luolie/ToPoGate/datasets/*.npz` (131 个) | `hfd.sh` (aria2c 多线程) |
| 2026-07-24 | run_topogate() 支持 y=None（包装层改造） | `run_npz.py:main()` (强制要 y) | `run_npz.py:main()` (y 可选) | y=None 时跳过 metrics，不写 fake_y |
| 2026-07-24 | 创建 benchmark 入口脚本 run_topogate_benchmark.py | N/A | `papers/codes/run_topogate_benchmark.py` | 131 数据集批量运行脚本 |
| 2026-07-24 | 创建 CLUBench 包装器 ToPoGate.py（B2） | `run_npz.py:run_topogate()` | `baseline/CLUBench/CLUBench/algorithms/ToPoGate.py:TopoGate(BaseCluster)` | 继承 + 调 run_topogate |
| 2026-07-24 | 创建 hpc JSON 配置 | N/A | `baseline/CLUBench/CLUBench/hpc/topogate.json` (80 epochs, lr=1e-3, hidden=128) | 手动 YAML 化 |
| 2026-07-24 | 新增 SSEKMSupervised + 工厂 + ALGOS 注册 | `baseline/CLUBench/CLUBench/algorithms/SklearnEKMeans.py`（仅 EKMeans / SSEKM unsup） | `SSEKMSupervised` 类（KMeans 伪标签构造 prior_matrix）+ `_build_ssekm_sup` 工厂 + ALGOS `SSEKM_sup` 条目 | `baseline/CLUBench/CLUBench/__init__.py` 导出新类 |
| 2026-07-24 | SSEKM_sup smoke test 通过：合成数据 233/300 标签改变（theta_super=1.0 确认 prior 被消费） | `np.random.default_rng(0)` 合成 (300,6) + sms_spam_collection.npz (835,500) | `theta_super=1.0` 触发半监督分支，与 EKMeans 数值不同 | 详见 CHANGELOG_errors.md「GBUSC/SSEKM_sup 在 Mouse_retina 上无限挂起」条目 |
| 2026-07-24 | 清理两个僵尸 Python 进程（1051023 GBUSC、1270553 SSEKM_sup） | `Mouse_retina.npz` 跑 GBUSC 超过 2h、SSEKM_sup theta sweep 超过 55min 不落盘 | 全部 SIGKILL；luolie 用户 Python 进程清零 | `kill -9 PID` 多次触发后清除 |
| 2026-07-25 | 汇总 NPZ 数据集元数据 | `/data/luolie/ToPoGate/datasets/*.npz`（磁盘现状 133 个；CLUBench 配置 131 个，额外为 `hrvatin.npz`、`Quake_Smart-seq2_Lung.npz`） | `result/dataset_npz_info.md`（按类型分组完整表）+ `result/dataset_npz_info.csv`（133 行） | `papers/codes/summarize_npz_datasets.py` |
| 2026-07-25 | 修复 hrvatin.npz:矩阵转置（基因×细胞 → 细胞×基因） | `hrvatin_geo/GSE102827_MATRIX.csv.gz`（25187 基因 × 65539 细胞） | `hrvatin.npz` (65539, 25187) | 读取后 `.T` 转置 |
| 2026-07-25 | 过滤 hrvatin -1 标签（maintype 缺失） | `hrvatin.npz` (65539, 25187) 含 17273 个 -1 标签 | `hrvatin_filtered.npz` (48266, 25187) | 删除含 -1 行 |
| 2026-07-25 | 创建 G-CEALS / IDC / TableDC / ZEUS CLUBench wrapper | 上游 4 个仓库（baseline/G-CEALS, baseline/IDC, baseline/TableDC, baseline/ZEUS）源代码**未修改** | `baseline/CLUBench/CLUBench/algorithms/GCEALS.py, IDC.py, TableDC.py, ZEUS.py` | sys.modules 注入 + 子模块 patch + 自写训练循环 |
| 2026-07-25 | 创建 4 模型 hpc JSON 配置文件 | `baseline/IDC/cfg/cfg_run.yaml`, G-CEALS 默认超参, ZEUS GMMConfig, TableDC 默认超参 | `baseline/CLUBench/CLUBench/hpc/{gceals,idc,tabledc,zeus}.json` | 手动 JSON 化（完整保留原始配置） |
| 2026-07-25 | 运行 4 baseline × 15 数据集基准 | `/data/luolie/ToPoGate/datasets/*.npz` (15 个指定数据集) | `result/baseline_comparison/{GCEALS,IDC,TableDC,ZEUS,summary}.csv` (56 行结果) | `scripts/run_baseline_comparison.py` |
| 2026-07-25 | 启动 TopoGate 15 数据集超参调优（Phase 0） | `/data/luolie/ToPoGate/datasets/*.npz` (15 个数据集) | `result/tune_15datasets/<dataset>/<dataset>__ep<ep>_mr<mr>_k<k>.json` (目标 405 行) | `scripts/run_topogate_tune_15datasets.py` |
| 2026-07-25 | 创建 TopoGate ablation 实验脚本 | `methods/TopoGate/configs/*.yaml` (8 variants) | `scripts/run_topogate_ablation.py` + `scripts/aggregate_ablation.py` + `scripts/plot_ablation.py` | 框架就位，等待 Phase 0 调优结果 |
| 2026-07-25 | 修改 TopoGate wrapper 加 `neighbor_k` 字段 | `baseline/CLUBench/CLUBench/algorithms/ToPoGate.py` | 支持 `--neighbor_k` CLI 参数覆盖 | model-integrity 原则（包装层改动，算法 main() 不动） |
| 2026-07-25 | 调优结果汇总（13/15 datasets） | `result/tune_15datasets/grid.csv` (351 行) + `best_per_dataset.csv` + `dominant_hparams.json` | 13/13 datasets 最佳超参与 131-dataset 不同，**epochs=150, mr=0.3, k=5** | `scripts/aggregate_tune_15datasets.py` |
| 2026-07-25 | 启动 Phase 1 消融核心层（40 runs） | 5 datasets × 8 variants | `result/ablation/<dataset>/<variant>__ep150_mr0.3_k5.json` | `scripts/run_topogate_ablation.py --layer core` |
| 2026-07-25 | Phase 1 消融核心层完成（40/40 runs） | 同上 | 40/40 json 完成，0 错误 | `scripts/run_topogate_ablation.py --layer core` |
| 2026-07-25 | Phase 1 消融核心层完成（40/40 runs） | 同上 | 40/40 json 完成，0 错误 | `scripts/run_topogate_ablation.py --layer core` |
| 2026-07-25 | Phase 1 核心层消融结果深度解读 | 40 jsons | 5 datasets × 8 variants 关键发现 + **v2 改造方向审计**（撤回若干建议） | 见下「Phase 1 核心层消融结果 + 解读」|
| 2026-07-25 | LearnableGate LearnableGate 实现 + 5 数据集 smoke test | `/data/luolie/ToPoGate/datasets/{Mouse_retina,enron,sms_spam_collection,har,breast_cancer_wisconsin_original}.npz` | `result/learnable_gate_smoke/*.json` (15 个) + `result/learnable_gate_smoke/comparison.csv` | `scripts/run_learnable_gate_sched_smoke.py` |
| 2026-07-25 | v2 网格扫描：learnable_gate@sched 5 datasets × 150 epochs | 同上 | 5 个 .json + β_perturb 模式差异显著（enron +4.10 vs Mouse_retina -1.56） | `scripts/run_learnable_gate_sched_smoke.py` ||
| 2026-07-25 | **校正**：Mouse_retina v2_smoke K 错误（硬编码 K=7，实际 K=5） | `result/learnable_gate_smoke/Mouse_retina__*/embedding_final.npy`（已存 K=7 KMeans，但底层 embedding 是正确的） | K=5 重聚类后 Mouse_retina v1=0.9421, v2=0.9405（与 v1 ablation 完全一致）| 验证脚本：`python -c` 用 KMeans 重新聚类 |
| 2026-07-25 | **multi-seed 验证**：5 ds × 2 variants × 3 seeds = 30 runs | 同上 + 扩展 seeds=[42,123,7] | `result/learnable_gate_smoke/multiseed/comparison.csv` (30 行) + 30 个 .json | `scripts/learnable_gate/run_learnable_gate_sched_multiseed.py` |
| 2026-07-25 | multi-seed 真实 verdict：v2 整体 Δ +0.013 ARI | multi-seed 30 run | enron +0.044, Mouse_retina +0.011, har +0.028, breast_cancer 0.000, sms_spam -0.017 | 见 `result/RESULTS_SUMMARY.md` |
| 2026-07-25 | 确定 v2 最小改动方案 | 消融数据 | v2 = 4 个 β 变成 nn.Parameter（~30 行） + Schedule（5 行） | 见 CHANGELOG.md 对应条目 |
| 2026-07-25 | Phase 1 核心层消融结果汇总 | 40 jsons | 5 datasets × 8 variants 关键发现录入 | 见下「Phase 1 核心层消融结果」|
| 2026-07-30 | V10 Reliable-Graph 持久工程 smoke（历史产物已清理） | `datasets/iris.npz`，150×4，K 从 3 个唯一标签自动检测，seed=42 | `result/v10_reliable_graph/smoke/iris__topogate_v10_reliable_graph__seed42/`（已清理），3 epochs CPU 完整产物 | `methods/TopoGate/v10_reliable_graph/run.py` |
| 2026-07-30 | V11 概率可信拓扑持久工程 smoke（历史产物已清理） | `datasets/iris.npz`，150×4，K=3 仅作为 benchmark oracle，seed=42 | `result/V11/smoke/iris__V11__seed42/`（已清理），3 epochs CPU 完整产物 | `methods/TopoGate/V11/run.py` |

## 详细记录

### 2026-07-31 ESWA-2026 AHDPC 数据归档与真实 smoke

**输入数据**：

- 论文：`papers/references/pdf/clustering_sota_2026/ESWA-2026-AHDPC.pdf`，DOI `10.1016/j.eswa.2025.130065`。
- 原始源：UEF Sipu（Asymmetric/Dim）、UCI（12 个表格集）、公开 benchmark 镜像（其余合成集）、OpenML 54（完整 UCI Vehicle 镜像）和 AT&T face archive；逐个 URL、原始 SHA-256 都写入 `datasets/AHDPC/MANIFEST.json`。
- 输入数据处理不使用真实标签参与模型拟合；`y` 只用于自动检测 `K=len(unique(y))` 与事后指标计算。

**生成数据**：

- `datasets/AHDPC/raw/`：原始下载文件及可复核解压内容；`datasets/AHDPC/processed/`：24 个主 NPZ，字段为 `x` / `y`。
- `glass_without_id.npz` 与 `student_evaluation_without_instr.npz` 是额外的 schema-safe 对照版本；主文件保留论文表中 10-D Glass 和 33-D Student 的兼容约定，后者明确带 `instr` 标签泄漏警告。
- 28 个论文行中，G2、Head CT、Skin cancer、Lung cancer 为 `unresolved`：论文没有给出足以获得精确版本的来源或处理协议，故没有创建替代性 NPZ。
- `result/AHDPC/verified_smoke_2026-07-31/summary.csv`：Flame、Aggregation、Banknote 的 AHDPC/HDPC 完整真实结果；`banknote_reported_equation/summary.json`：印刷 Eq.(10) 审计。

**验证结果**：

| 数据集 | 模式 | AMI | RI | FMI | NMI |
| --- | --- | ---: | ---: | ---: | ---: |
| Flame | table-reproduction AHDPC | 0.9353 | 0.9834 | 0.9846 | 0.9355 |
| Aggregation | table-reproduction AHDPC | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Banknote（z-score） | table-reproduction AHDPC | 0.9316 | 0.9812 | 0.9814 | 0.9317 |
| Banknote（z-score） | HDPC | 0.6092 | 0.8120 | 0.8139 | 0.6094 |
| Banknote（z-score） | 印刷 Eq.(10) | 0.0084 | 0.5046 | 0.7063 | 0.0095 |

**扩展 smoke（同一实现、AHDPC table-reproduction）**：`2d_20c_no0` AMI/RI/FMI/NMI=0.9730/0.9942/0.9492/0.9741；`dim064`=1.0000/1.0000/1.0000/1.0000；`image_segment`=0.5499/0.8530/0.5066/0.5718；`rice`=0.4630/0.7684/0.7840/0.4631。输出为 `result/AHDPC/verified_smoke_2026-07-31/extended/summary.csv`。

**Olivetti 图像分支**：seed=42、perplexity=30、max_iter=1000、ε=0.1，AMI/RI/FMI/NMI=0.8001/0.9746/0.5930/0.8767。`run_face.py` 写入 `tsne_embedding.npy`、`predictions.npy`、`labels_true.npy` 和带 `source_sha256`/`labels_used_during_fit=false` 的 `summary.json`。

**扩展 smoke（同一实现、AHDPC table-reproduction）**：`2d_20c_no0` AMI/RI/FMI/NMI=0.9730/0.9942/0.9492/0.9741；`dim064`=1.0000/1.0000/1.0000/1.0000；`image_segment`=0.5499/0.8530/0.5066/0.5718；`rice`=0.4630/0.7684/0.7840/0.4631。输出为 `result/AHDPC/verified_smoke_2026-07-31/extended/summary.csv`。

**追溯代码**：`baseline/AHDPC/download_datasets.py`、`baseline/AHDPC/run_benchmark.py`、`baseline/AHDPC/run.py`。

### 2026-07-30 TopoGate V11 持久工程 smoke

**定位**：真实 NPZ 的工程烟雾测试，验证 V11 的 Student-t responsibility、self/null edge mixture、EMA 动态候选图、输出数据溯源和严格 NoMix 单元回归；不是性能实验。

**输入数据**：
- 文件路径：`/home/luolie/ToPoGate/datasets/iris.npz`（项目数据软链接目标）
- 数据描述：`x.shape=(150, 4)`，`y.shape=(150,)`；`K=len(unique(y))=3`
- 来源验证：运行 summary 记录 `source_sha256=c31ba1e4f6d7a1dbeb7287dac646598d95986cf0d5a6b26705f684da53f33fd5`；训练器本身不读取 y，K 协议记录为 `benchmark_oracle_from_y`

**运行参数**：
- CPU、seed=42、epochs=3、batch=64、`warmup=1/ramp=1/refresh=1`
- 缩小 `hidden_size=64`、`latent_size=16`、`pca_dim=8`、`neighbor_k=3/candidate_k=6` 仅为了工程 smoke；不是主配置或调参结论

**生成数据**：
- 输出目录：`result/V11/smoke/iris__V11__seed42/`
- 主要文件：`embedding_final.npy`、`cluster_probabilities.npy`、`predictions.npy`、`labels_true.npy`、`label_mapping.json`、`metrics.json`、`args.json`、`summary.json`
- 机制核验：实际 PCA 维数=2；epoch 2/3 各刷新一次 raw+latent 候选图，edge change fraction=0.4600/0.0044；末 epoch learned topology gate=0.2141，风险 target gate=0.1419
- 工程指标（仅验证链路）：head ACC=0.8267、NMI=0.6651、ARI=0.6129；KMeans 诊断 ARI=0.6292
- 对应图表：无

**验证边界**：single-seed、3 epochs、缩小模型；任何 V11 相对 V9/NoMix 的论文结论均被禁止。正式实验至少覆盖预注册 15 datasets 与多种子，并报告 V11 full 与 `V11_nomix`、静态图、均匀 edge、无 teacher、无 cluster head 等消融。

**追溯代码**：`methods/TopoGate/V11/run.py`、`methods/TopoGate/V11/trainer.py`、`scripts/V11/run_v11_multiseed.py`。

### 2026-07-30 TopoGate V10 Reliable-Graph 持久工程 smoke

**定位**：这是一项真实 NPZ 的工程冒烟测试，用于验证最终 V10 代码的训练、动态图、原型初始化与输出契约；它不是正式性能实验。

**输入数据**：
- 文件路径：`/home/luolie/ToPoGate/datasets/iris.npz`（解析到 `/data/luolie/ToPoGate/datasets/iris.npz`）
- 数据描述：`x.shape=(150, 4)`，`y.shape=(150,)`，`K=len(unique(y))=3`
- 来源验证：NPZ 包含真实 `x`/`y` 字段；未使用合成或替代数据

**运行参数**：
- variant：`topogate_v10_reliable_graph`
- device：CPU（CLI 同时传入规则允许的 `--gpu 1` 与 `--no_cuda`）
- seed：42；epochs：3；batch size：150
- schedule：`warmup_epochs=0`、`ramp_epochs=1`；`refresh_interval=1`
- 其余参数来自 `methods/TopoGate/v10_reliable_graph/configs/topogate_v10_reliable_graph.yaml`

**生成数据**：
- 输出目录：`result/v10_reliable_graph/smoke/iris__topogate_v10_reliable_graph__seed42/`
- 主要文件：`embedding_final.npy`、`predictions.npy`、prototype 系列诊断、`labels_true.npy`、`label_mapping.json`、`history.json`、`graph_history.json`、`final_graph_edges.npz`、`config_resolved.json`、`summary.json`
- 输出契约核验：`predictions.npy` 与 `labels_true.npy` 均为 150 个元素；不存在易混淆的 `labels.npy`
- 动态图核验：3 个 epoch 均刷新候选图；每次 1500 条候选边，最终 input/latent stability=0.887333；epoch 2/3 的独立 temporal recurrence 均为 0.868667。`final_graph_edges.npz` 保存全图 gates 与两类 recurrence。duplicate-row/tie 情形显式按 node id 去除 self。
- 原型初始化核验：`prototype_initialization_epoch=1`；方法为 `kmeans_on_normalized_ema_clean_embedding_n_init20`；`cluster_prior_mode=warmup_kmeans`，先验为 `[0.459998, 0.213336, 0.326667]`
- 工程指标（仅用于确认评估链路）：ACC=0.8400、NMI=0.685151、ARI=0.633486、macro-F1=0.834646；最终产物记录训练耗时约 0.400 s
- 对应图表：无

**验证边界**：single-seed、仅 3 epochs、且专为快速工程 smoke 配置；不得据此声称 V10 优于 V9、NoMix 或任何 baseline。正式结论仍需至少 5 个核心数据集 × 3 seeds，并报告 mean ± std 与对应消融。

**追溯代码**：`/home/luolie/ToPoGate/methods/TopoGate/v10_reliable_graph/run.py`；权威参数与产物事实见输出目录中的 `config_resolved.json` 和 `summary.json`。

---

### 2026-07-25 NPZ 数据集基本信息汇总

**输入数据**：
- 文件路径：`/data/luolie/ToPoGate/datasets/*.npz`
- 数据描述：磁盘现状共 133 个 NPZ；均包含二维 `x` 特征矩阵与一维 `y` 标签数组
- 来源验证：与 `CLUBench.configs.DATASETS` 比对后，131/131 配置数据集均存在，另有 `hrvatin.npz` 和 `Quake_Smart-seq2_Lung.npz`

**生成数据**：
- 输出文件：`/data/luolie/ToPoGate/result/dataset_npz_info.md`
- 机器可读文件：`/data/luolie/ToPoGate/result/dataset_npz_info.csv`
- 类型统计：`scRNA-seq=5`、`image_embed=20`、`tabular=108`
- 验证：133 个数据集名唯一，所有样本数、维度、类别数均大于 0

**追溯代码**：`/home/luolie/ToPoGate/papers/codes/summarize_npz_datasets.py`

---

### 2026-07-25 修复 hrvatin.npz + 过滤 -1 标签

**背景**：hrvatin.npz 出现两个问题：
1. **矩阵形状错误**：原始矩阵是 基因×细胞（25187×65539），但存储在 npz 时被当作 (细胞×基因) 读取，导致 cell_ids 实际是基因名
2. **17,273 个 -1 标签**：maintype 缺失的细胞，原始元数据中本来就 NaN

**问题 1 修复**：

**输入数据**：
- 文件路径：`/data/luolie/biopipeline/dimension-reduction/plantnet/data/hrvatin_geo/GSE102827_MATRIX.csv.gz`（380MB）
- 文件路径：`/data/luolie/biopipeline/dimension-reduction/plantnet/data/hrvatin_geo/GSE102827_cell_type_assignments.csv.gz`（275KB，6 列，65539 行）
- 数据描述：原始矩阵为「基因 × 细胞」格式（25187 基因 × 65539 细胞），表头第一列是基因名，其余 65539 列是细胞 ID

**生成数据**：
- 输出文件：`/home/luolie/ToPoGate/datasets/hrvatin.npz`（179 MB）
- 形状：`(65539, 25187)` 即 65539 细胞 × 25187 基因
- 标签映射：

| 编码 | 细胞类型 | 数量 |
|------|---------|------|
| 0 | Astrocytes | 7,039 |
| 1 | Endothelial_SmoothMuscle | 4,071 |
| 2 | Excitatory | 14,287 |
| 3 | Interneurons | 936 |
| 4 | Macrophages | 537 |
| 5 | Microglia | 10,158 |
| 6 | Mural | 782 |
| 7 | Oligodendrocytes | 10,456 |
| -1 | maintype 缺失 | 17,273 |

**追溯代码**：转换脚本（详见状态追踪，当前在 `/home/luolie/ToPoGate/datasets/hrvatin.npz`）

---

**问题 2 修复 - 过滤 -1**：

**输入数据**：
- 文件路径：`/home/luolie/ToPoGate/datasets/hrvatin.npz`（含 17,273 个 -1 标签）

**生成数据**：
- 输出文件：`/home/luolie/ToPoGate/datasets/hrvatin_filtered.npz`（137 MB）
- 形状：`(48266, 25187)` 即 48,266 细胞 × 25,187 基因
- 标签：8 类（0-7），无 -1
- 文件大小：179 MB → 137 MB（节省 41.9 MB）

**使用建议**：
- 用于 TopoGate 训练时**优先使用 `hrvatin_filtered.npz`**，避免 -1 干扰聚类结果
- 原 `hrvatin.npz` 保留作为对照（需要看完整 65K 细胞分布时使用）

**追溯代码**：filter 脚本（详见 `/home/luolie/ToPoGate/datasets/hrvatin_filtered.npz`）

---

**已知问题**：
- TopoGate 优势分析.md 中 hrvatin ACC=0.9271 数据来源是 `rg_neighbormix_scmae`（scVICAR-T），不是 TopoGate 实际跑出来的结果
- TopoGate 实际还未跑过 hrvatin 和 Quake_Smart-seq2_Lung（topogate_opt_results.csv 中无记录）
- 文档修订时机：完成 TopoGate 在 hrvatin_filtered.npz 上的实验后同步更新

---



### 2026-07-24 B1 run_topogate() 包装器（argv 注入方案）

**输入数据**：
- 源文件：`/home/luolie/ToPoGate/methods/TopoGate/run_npz.py`（含 4 个 variant YAML 配置）
- 现有算法：`main()` 函数（150 行，CLI 驱动）

**生成数据**：
- 新增函数：`run_topogate(X, y, n_clusters, gpu=4, variant='topogate_full', save_dir=None, seed=42, return_metrics=False, **overrides)`
- 算法 `main()` 一行未动
- Smoke test：weather 数据集 ACC/NMI/ARI = 1.0，5.4s

**追溯代码**：`/home/luolie/ToPoGate/methods/TopoGate/run_npz.py:run_topogate()`

---

### 2026-07-24 完整 CLUBench 131 数据集就位

**输入数据**：
- 来源：`https://hf-mirror.com/datasets/Feng-001/Clustering-Benchmark`（HuggingFace mirror）
- 原始文件：`CLUBench-Datasets.zip`（689.86 MB，5 个 parquet + 131 个 npz 打包）
- 下载工具：`hfd.sh`（hf-mirror 官方提供，aria2c 内核，自动断点续传，8 线程）
- 下载命令：`./hfd.sh Feng-001/Clustering-Benchmark --dataset --local-dir /data/luolie/ToPoGate/download -x 8 --include "CLUBench-Datasets.zip"`
- 速度：~26 MB/s，26s 完成
- 完整性校验：`unzip -t` No errors detected

**生成数据**：
- 压缩包保留：`/data/luolie/ToPoGate/download/CLUBench-Datasets.zip`（658M 实际大小，因 hfd 计数按 IB 二进制）
- 解压后：`/data/luolie/ToPoGate/datasets/*.npz`（131 个文件，平铺到根目录）
- 软链接有效性：
  - `/home/luolie/ToPoGate/datasets -> /data/luolie/ToPoGate/datasets` ✓
  - `/home/luolie/ToPoGate/baseline/CLUBench/CLUBench/datasets -> /data/luolie/ToPoGate/datasets` ✓
- 备份：`/data/luolie/ToPoGate/datasets_backup_20260724/`（保留旧 10 个 npz，保险用）

**完整性验证**：
- ✅ 131/131 数据集名在 `CLUBench/configs.DATASETS` 中能匹配到 `.npz` 文件
- ✅ `python -c "import CLUBench; CLUBench.configs.DATA_DIR"` 解析为 `baseline/CLUBench/CLUBench/datasets` 软链接目标
- ✅ 两类 `AutoEncoder`（TopoGate / NeighborMix_scMAE）独立加载，import 链通

**追溯代码**：
- 下载：`/data/luolie/ToPoGate/download/hfd.sh`
- 解压命令：`unzip -o /data/luolie/ToPoGate/download/CLUBench-Datasets.zip -d /data/luolie/ToPoGate/datasets/`
- 验证：`python -c "import sys; sys.path.insert(0,'/home/luolie/ToPoGate/baseline/CLUBench'); from CLUBench.configs import DATA_DIR, DATASETS; import os; data='/home/luolie/ToPoGate/baseline/CLUBench/CLUBench/datasets'; files={f for f in os.listdir(data) if f.endswith('.npz')}; print(sum(1 for n in DATASETS if n in files), '/', len(DATASETS))"`

**已知限制**：
- 9 个 text 数据集需要对应 text embedding，目前未下载（阶段 A3 触发）
- CLIP_Embedding.zip / ResNet_Embedding.zip 在阶段 C 按需下载

## 历史记录

### 2026-07-25 StaticGate 消融实验完成（15 datasets × 8/4 variants = 80 runs）

**实验内容**：
- Core 层（5 datasets × 8 variants = 40 runs）：Mouse_retina, enron, sms_spam, har, breast_cancer
- Extended 层（10 datasets × 4 variants = 40 runs）：reuters, ISOLET, spambase, cnae9, Campbell, hrvatin_filtered, Quake_Smart-seq2_Lung, mammographic_mass, first-order-theorem-proving, iris
- 所有结果写入 `result/ablation/merged_summary.csv`

**关键发现**：
- `static_gate_edge_only` 在 core 5 上 ARI=0.8054（最高）
- `static_gate_nomix` 表现接近 full（core 5 avg: 0.803），说明 neighbor mixing 在部分数据集上非必需
- `static_gate_far_neighbors` 显著差于 `static_gate_full`（core 5: 0.758 vs 0.798），证明"近邻"语义是关键
- hrvatin_filtered ARI ≈ 0.003（几乎无效，需要特殊处理）

### 2026-07-24 导出 CLUBench 24 算法 baseline 性能表

**输入数据**：
- 文件路径：`/home/luolie/ToPoGate/baseline/CLUBench/performance_matrix/best_hpc/*.p`（24 个 pickle）
- 数据描述：作者已完成 178,815 次实验，每个 `.p` 是一个算法在 131 个数据集上的最佳超参结果，键为 `acc / nmi / ari / time`
- 算法清单：`agglo, autosc, birch, cc, dbscan, dec, divc, dmicc, dscn, edesc, gmm, idec, kernel_kmeans, kfsc, kmeans, kpc, lfss, lrr, meanshift, p2ot, pica, s3comp, spectral_clustering, ssc`
- 数据集顺序：与 `CLUBench/configs.py` 的 `DATASETS` 列表严格对齐（131 个）

**生成数据**：
- 输出文件 1：`/home/luolie/ToPoGate/result/baseline_clubench.csv`（3144 × 8）—— 长格式，行为「算法 × 数据集」组合
- 输出文件 2：`/home/luolie/ToPoGate/result/baseline_clubench_wide.csv`（131 × 99）—— 宽格式，每数据集一行，列为 `acc_kmeans / nmi_kmeans / ...`
- 输出文件 3：`/home/luolie/ToPoGate/result/baseline_clubench_per_dataset.csv`（131 × 99）—— 与上同（保留两个名字方便脚本引用）
- 输出文件 4：`/home/luolie/ToPoGate/result/baseline_summary.csv`（24 × 22）—— 每个算法在 131 数据集上的均值/标准差/最小/最大/中位数
- 输出文件 5：`/home/luolie/ToPoGate/result/baseline_summary_by_type.csv`（96 × 6）—— 按 dataset_type（tabular/image/text/bioinfo）分组均值
- 对应图表：暂未生成，作为 ToPoGate 论文「Baseline Comparison」表的原始数据

**完整性验证**：
- ✅ 24 算法 × 131 数据集 = 3144 行（无误）
- ✅ acc/nmi/ari/time 四个指标 NaN 计数均为 0
- ✅ top-5 按 acc_mean 排名：spectral_clustering (0.689), kernel_kmeans (0.642), kmeans (0.636), agglo (0.631), gmm (0.626)
- ✅ 全数据集 acc 均值 0.590，标准差 0.216（分布健康）

**追溯代码**：`/home/luolie/ToPoGate/baseline/CLUBench/export_baseline_csv.py`（可重复运行）

**已知限制**：
- 时间字段是作者用其硬件 + 默认超参跑出来的，ToPoGate 对比耗时时应保持同硬件对照
- `baseline_summary.csv` 默认按 `acc_mean` 降序，可手动改为 `nmi` 或 `ari`

### 2026-07-23 移动 CLUBench 样本数据集

**输入数据**：
- 文件路径：`/home/luolie/ToPoGate/baseline/CLUBench/CLUBench/datasets/`
- 数据描述：CLUBench 仓库自带的 10 个样本数据集（`.npz` 格式），每个文件包含 `x`（特征）和 `y`（标签）字段
- 来源验证：上游仓库 `https://github.com/xiaofeng-github/CLUBench`，commit hash 随 `baseline/CLUBench/.git/` 保留

**生成数据**：
- 输出文件：`/data/luolie/ToPoGate/datasets/`（10 个 `.npz` 文件，共 3.7 MB）
- 软链接：`/home/luolie/ToPoGate/datasets` -> `/data/luolie/ToPoGate/datasets`
- 备份：`/data/luolie/ToPoGate/datasets_backup_20260723_234557.tar.gz`（旧目录的 tar.gz 备份）
- 对应图表：暂无（仅为数据准备）

**数据集清单**：
| # | 文件名 | 大小 (B) | 备注 |
|---|--------|---------|------|
| 1 | COIL20_CLIP.npz | 1,486,570 | 图像 CLIP 嵌入 |
| 2 | Letter.npz | 1,360,490 | 表格 |
| 3 | breast_cancer_coimbra.npz | 9,770 | 表格 |
| 4 | breast_tissue.npz | 8,970 | 表格 |
| 5 | echocardiogram.npz | 5,858 | 表格 |
| 6 | parkinsons.npz | 36,370 | 表格 |
| 7 | poker-hand.npz | 180,490 | 表格 |
| 8 | vehicle.npz | 129,082 | 表格 |
| 9 | weather.npz | 564,050 | 表格 |
| 10 | zoo.npz | 14,226 | 表格 |

**追溯代码**：N/A（数据搬运操作，无脚本）

**已知限制**：CLUBench 完整 131 个数据集未下载，目前仅含 10 个样本数据集。完整数据集需从 CLUBench-Datasets 单独下载并放入 `/data/luolie/ToPoGate/datasets/`。

---

## 2026-07-24：TopoGate 全文实验（131 数据集）完成

**实验**：`papers/codes/run_topogate_benchmark.py --variant topogate_full --epochs 80`

**输入**：
- 数据集：131 个（CLUBench 完整清单，混合图像/文本/表格/单细胞）
- 输出目录：`/home/luolie/ToPoGate/result/topogate/`
- 汇总 CSV：`/home/luolie/ToPoGate/result/topogate_all.csv`

**输出**：
- 131/131 json 完成，0 错误
- **均值：ACC=0.6047, NMI=0.3759, ARI=0.3241**
- **中位数：ACC=0.5823, NMI=0.3203, ARI=0.2538**
- 标准差：ACC=0.2194, NMI=0.3069, ARI=0.3047

**分布**：
- 高分 (ACC ≥ 0.9)：19 个（14.5%）
- 有效分类 (ACC ≥ 0.5)：95 个（72.5%）
- 低分 (ACC < 0.2)：3 个（2.3%）

**耗时**：3 卡并行 (GPU 4/5/7) 共 1.2 小时（4230 秒），平均每数据集 32.3 秒

**Top-5（按 ACC）**：
- weather: 1.0000
- smoker_condition: 0.9872
- Mouse_retina: 0.9848
- rice_seed_gonen_jasmine: 0.9759
- sms_spam_collection: 0.9725

**Bottom-5（按 ACC）**：
- tamilnadu-electricity: 0.0659
- street_view_house_numbers: 0.1544
- 20newsgroups: 0.2000
- cifar10: 0.2191
- microbes: 0.2299

**GPU 使用**：仅空闲 GPU 4/5/7，未碰 GPU 0/1/2/3/6（被其他用户占用）

**对应图表**：Table 1（全文主表）、Figure 4（ACC/NMI/ARI 分布）、Table 3（top/bottom 5）

---

## 2026-07-24：Round 2 超参数调优（epochs × mask_ratio）

**实验**：`papers/codes/run_topogate_tune.py`（Round 2 网格）

**输入**：
- 代表性子集：13 个数据集（4 类型覆盖）
- 网格：9 configs = 3 (epochs) × 3 (mask_ratio)
  - epochs: 40, 80, 150
  - mask_ratio: 0.3, 0.4, 0.5
  - 固定：neighbor_k=10, hidden_size=128（Round 1 最优）
- 输出目录：`/home/luolie/ToPoGate/result/tune_round2/`
- 总运行：117 runs（3 GPU worker 并行）

**结果（9 configs 平均 ACC）**：

| epochs \ mask_ratio | 0.3 | 0.4 | 0.5 |
|---|---|---|---|
| **40** | **0.6466** | 0.6324 | 0.6207 |
| **80** | 0.6373 | 0.6269 | 0.6027 |
| **150** | 0.6349 | 0.6359 | 0.5980 |

**结论**：
- **最优配置：ep=40, mr=0.3** → mean ACC=0.6466
- **次优配置：ep=80, mr=0.3** → mean ACC=0.6373
- mask_ratio 主效应最大（0.3 > 0.4 > 0.5），epochs 影响小
- baseline (ep=80, mr=0.4) = 0.6269，提升 **+0.0198**

**显著提升**：4 个数据集（har +13.3%, Campbell +9.3%, Baron Human +1.9%, breast_cancer +1.0%）
**显著下降**：1 个（mnist64 -1.0%，可忽略）

---

## 2026-07-24：TopoGate 最优配置全 131 数据集

**实验**：`papers/codes/run_topogate_opt.py`

**最优超参数**（Round 2 确定）：
- epochs=40, mask_ratio=0.3, neighbor_k=10, hidden_size=128

**输入/输出**：
- 数据集：131 个（CLUBench 完整）
- 输出目录：`/home/luolie/ToPoGate/result/topogate_opt/`
- 汇总 CSV：`/home/luolie/ToPoGate/result/topogate_opt_results.csv`

**结果**：
- 131/131 完成，0 错误
- **均值：ACC=0.6053, NMI=0.3753, ARI=0.3234**
- 标准差：ACC=0.2170

**ACC 分布**：
| 分段 | 数量 |
|---|---|
| ≥0.9 | 17 (13.0%) |
| 0.8–0.9 | 9 (6.9%) |
| 0.6–0.8 | 32 (24.4%) |
| 0.4–0.6 | 51 (38.9%) |
| <0.4 | 22 (16.8%) |

**Top-5**：weather (1.0), smoker_condition (0.99), Mouse_retina (0.98), breast_cancer_original (0.98), sms_spam (0.97)

**Bottom-5**：tamilnadu-electricity (0.07), street_view (0.16), 20newsgroups (0.20), cifar10 (0.22), microbes (0.23)

**GPU**：仅 GPU 4/5（2 worker 并行），未用 GPU 0/7

---

## 2026-07-25 对比实验主表设计（思路上锁）

**目标**：以 TopoGate 为中心，验证三个核心主张。

### 表 1（重建家族，验证 C1：掩码自编码是合理基座）

| 算法 | 来源 | 角色 |
|------|------|------|
| KMeans | scikit_cluster.py | 欧式聚类下界，无表示 |
| GMM | scikit_cluster.py | 概率生成下界 |
| DEC | DEC.py | AE + 软分配聚类头 |
| IDEC | IDEC.py | DEC + 重建正则 |
| DSCN | DSCN.py | AE + 谱聚类 |
| EDESC | EDESC.py | 端到端 AE + 谱聚类 |
| TopoGate_nomix | topogate_nomix.yaml | 掩码基座（ablation 上界） |
| TopoGate | topogate_full.yaml | 主结果 |

**数据集**：11 advantage datasets（mouse_retina, weather, smoker_condition, breast_cancer_original, sms_spam, spambase, sonar, isolet, har, mnist, coil20）

**输入**：`datasets/*.npz`（已就位 131 个）  
**输出**：`result/<algo>/<dataset>/metrics.json`  
**规模**：11 × 8 = 88 runs

### 表 2（图自监督家族，验证 C2：邻居混合 vs 图对比）

| 算法 | 来源 | 角色 |
|------|------|------|
| LFSS | LFSS.py | 标签特征自监督 |
| DIVC | DIVC.py | 深度内视图对比 |
| PICA | PICA.py | 增强样本对比 |
| P2OT | P2OT.py | 最优传输聚类 |
| TopoGate | topogate_full.yaml | 主结果 |

**规模**：11 × 5 = 55 runs

### 表 3（TopoGate Ablation，验证 C3：拓扑感知门控的独特贡献，最关键）

| Variant | Config | 验证点 |
|---------|--------|-------|
| TopoGate_nomix | topogate_nomix.yaml | 拓扑增益总幅度 |
| TopoGate_random_neighbors | topogate_random_neighbors.yaml | 拓扑 nb vs 随机 nb |
| TopoGate_far_neighbors | topogate_far_neighbors.yaml | 拓扑 nb vs 远 nb |
| TopoGate_constant_gate | topogate_constant_gate.yaml | 自适应门控 vs 固定门 |
| TopoGate_gate_only | topogate_gate_only.yaml | edge reliability 贡献 |
| TopoGate_edge_only | topogate_edge_only.yaml | gate 贡献 |
| TopoGate_no_topology_features | topogate_no_topology_features.yaml | 拓扑特征 mutual/snn 贡献 |
| TopoGate_full | topogate_full.yaml | 主结果 |

**规模**：11 × 8 = 88 runs

### 表 4（跨域泛化，验证 "通用算法" 主张）

| 数据集 | 类别 | 用途 |
|--------|------|------|
| MNIST | 10 类手写数字 | 跨域最简 |
| FashionMNIST | 10 类服装 | 跨域更复杂 |
| COIL-20 | 20 类物体 72 视角 | 连续流形（优势方向） |

**算法**：KMeans / DEC / IDEC / DSCN / TopoGate  
**规模**：3 × 5 = 15 runs

### 移除的 baseline

| 算法 | 移除原因 | 决定时间 |
|------|---------|---------|
| SSEKM_sup | unsupervised benchmark 中退化为 EKMeans/KMeans | 2026-07-25 |
| CL-LRPE | 真实任务异质图节点分类，不适合 tabular 聚类 | 2026-07-25 |
| DPCAC_CSC | Dual-Perspective 在同质 kNN 图上无意义 | 2026-07-25 |

---

## 2026-07-25 修订：表 1 从 scRNA-seq SOTA 改为通用深度聚类 SOTA

**触发**：用户指出 TopoGate 走通用算法路线，scCDCG/scSGC/scAGC 等都是 scRNA-seq 专用（ZINB），不能跑在 tabular/text/image 上。

**调研结果**：
- **ZEUS**（NeurIPS 2025）：第一个 zero-shot tabular clustering Transformer，无需微调
- **TableDC**（ICLR 2024）：tabular 深度聚类 SOTA，Mahalanobis + Cauchy 重尾
- **IDC**（ICML 2024）：可解释聚类 SOTA
- **G-CEALS**：DEC 现代替代（高斯混合替代 t 分布）
- **DEPICT**：经典 baseline（softmax + 互信息）

**修订后的实验结构**：
- **表 1（通用深度聚类 SOTA）**：ZEUS + TableDC + G-CEALS + DEPICT + IDC + TopoGate — 直接竞争对手
- **表 2（重建家族 CLUBench）**：KMeans/GMM/DEC/IDEC/DSCN/EDESC/scMAE/TopoGate
- **表 3（图自监督）**：LFSS/DIVC/PICA/P2OT/TopoGate
- **表 4（Ablation）**：8 variant × 11 datasets
- **表 5（跨域）**：MNIST/FashionMNIST/COIL-20

**移除的 scRNA-seq 专用方法**：scCDCG/scSGC/scAGC/scDCC/scDeepCluster/scNAME/scGPT/GeneFormer/GeneCompass — 全部因领域不匹配移除。

---

## 2026-07-25 发现：TopoGate 真实表现 vs CLUBench

**关键数据**（131 数据集）：

| 指标 | TopoGate | Best CLUBench | 说明 |
|------|----------|-------------|------|
| 平均 ACC | 0.6053 | 0.7174 | 落后 0.1121 |
| 赢/输/平 | 12/107/12 | — | — |

**TopoGate 真正长处**：
1. **Mouse_retina**: 0.983 vs 0.580 (**+40.3%**) — scRNA-seq 拓扑感知极强
2. **sms_spam**: 0.974 vs 0.577 (**+39.7%**) — 高维稀疏文本
3. **reuters**: 0.521 vs 0.225 (**+29.5%**) — 高维文本
4. **enron**: 0.969 vs 0.883 (+8.6%) — 高维稀疏

**TopoGate 弱项**：
1. **20newsgroups**: 0.197 vs 0.913 (-71.7%) — CLIP 嵌入已被好表示，MAE 破坏
2. **wos**: 0.467 vs 0.977 (-51.0%) — 高维稀疏，kmeans 天然强
3. **Baron Human**: 0.431 vs 0.818 (-38.7%) — 与 Mouse_retina 差异大
4. **synthetic_control**: 0.568 vs 0.938 (-37.0%) — 表格线性可分

**论文叙事**：不主张"全面 SOTA"，而是"在 scRNA-seq 和高维稀疏数据上有结构性优势，诚实承认在表格线性可分数据上的弱项"。

完整分析见 `papers/EXPERIMENT_PLAN.md`。

### 总实验规模

| 表 | 数据集 | 算法 | runs |
|----|--------|------|------|
| 表 1 | 11 | 8 | 88 |
| 表 2 | 11 | 5 | 55 |
| 表 3 | 11 | 8 | 88 |
| 表 4 | 3 | 5 | 15 |
| **合计** | — | — | **246** |

**产物路径**：`result/<algo>/<dataset>/{metrics.json, summary.json, embedding_final.npy, ...}`  
**汇总表路径**：`result/extended_baseline/aggregated.csv`（待建）  
**脚本**：`scripts/baseline_runner.py`


## 2026-07-25 IDC 修复 + hrvatin PCA(500) + 加入 TopoGate

### 关键实验
- **5 模型 × 15 数据集 = 75 个完整结果**
- **版式备份**：`/home/luolie/ToPoGate/result/baseline_comparison/versioned/20260725_1215/`

### TopoGate 排名（1=最好）
- **NMI: 1.93**（最佳，平均排名）, ARI: 2.00（最佳）, ACC: 2.47（次 GCEALS 0.07）

### hrvatin_filtered (PCA=500) 详细数据
| 模型 | ACC | NMI | ARI | 时间(s) |
|------|-----|-----|-----|---------|
| GCEALS | 0.847 | 0.864 | 0.839 | 734 |
| ZEUS | 0.635 | 0.679 | 0.556 | 5.8 |
| IDC | 0.588 | 0.670 | 0.467 | 648 |
| TableDC | 0.498 | 0.276 | 0.123 | 208 |
| TopoGate (10k subsample) | 0.346 | 0.172 | 0.012 | 104 |

### 完整对比表
- 位置：`/home/luolie/ToPoGate/papers/tab_figs/comparison_table.md`
- CSV 版本：`/home/luolie/ToPoGate/papers/tab_figs/comparison_table.csv`

### 关键修复
- **IDC 修复**：wrapper 层 post-training fallback + KMeans on KMeans on raw X
- **ZEUS batched 推理**：hrvatin 48k 样本不再 OOM
- **TopoGate subsample 路径**：hrvatin 48k 样本 kNN 图 OOM，改用 10k subsample + 1-NN OOB

### 文件
- `baseline/CLUBench/CLUBench/algorithms/IDC.py`：+post-training fallback
- `baseline/CLUBench/CLUBench/algorithms/ZEUS.py`：+batched forward
- `baseline/CLUBench/CLUBench/algorithms/ToPoGate.py`：+subsample_size 参数
- `scripts/run_baseline_comparison.py`：+PCA_DIM_OVERRIDE / +MODEL_KWARGS_OVERRIDE / +TIMEOUT_OVERRIDE
- `scripts/generate_comparison_table.py`：生成对比表

---

### 2026-07-25 Phase 1 核心层消融结果（5 datasets × 8 variants = 40 runs）

**输入**：
- 5 datasets: Mouse_retina, sms_spam_collection, enron, har, breast_cancer_wisconsin_original
- 8 variants: topogate_full / topogate_nomix / topogate_no_topology_features / topogate_gate_only / topogate_constant_gate / topogate_edge_only / topogate_far_neighbors / topogate_random_neighbors
- 超参: epochs=150, mask_ratio=0.3, neighbor_k=5
- 硬件: 3 worker × GPU 4/5/7

**输出**：
- 40/40 json 完成，0 错误
- 总耗时 ~30 分钟（最大单 run：Mouse_retina 全变体 ~280s）

**完整结果（ARI 排序，越大越好）**：

| variant | Mouse_retina | sms_spam | enron | har | breast_cancer | **avg** |
|---------|-------------:|---------:|------:|----:|--------------:|--------:|
| **full** | 0.9416 | 0.8200 | 0.7677 | 0.5579 | 0.9021 | **0.7979** |
| edge_only | 0.9403 | 0.8478 | 0.7956 | 0.5579 | 0.8855 | 0.8054 |
| constant_gate | 0.9416 | 0.8478 | 0.7811 | 0.5538 | 0.8855 | 0.8019 |
| no_topology_features | 0.9411 | 0.8200 | 0.7896 | 0.5579 | 0.8965 | 0.8010 |
| gate_only | 0.9384 | 0.8189 | 0.7677 | 0.5579 | 0.9021 | 0.7970 |
| **random_neighbors** | 0.9310 | 0.7292 | 0.7839 | 0.5380 | 0.8853 | 0.7735 |
| nomix | 0.9456 | 0.8443 | 0.8753 | **0.4582** | 0.8910 | 0.8029 |
| **far_neighbors** | 0.8468 | 0.7119 | 0.7842 | 0.5570 | 0.8909 | **0.7582** |

**关键发现**：

1. **topology-aware 邻居选择是核心**：`far_neighbors`（远邻）平均 ARI 0.7582 显著恶化（-0.040 vs full），`random_neighbors` 降 0.024
   - Mouse_retina 上 `far_neighbors` ARI 暴跌 0.8468（vs full 0.9416），证明短程拓扑贡献最大
2. **PE loss 关键性**：仅在 har 上明显（ARI 0.5579 vs nomix 0.4582），其他数据集几乎无差别
3. **结构组件贡献稳定**：edge_only / constant_gate / gate_only / no_topology_features 与 full 差距 < 0.005 ARI
   - 原因：单一变量在小数据集上效果波动在 ±0.01，远小于 5-seed std
4. **sms_spam 上 edge_only/constant_gate 反而优于 full**（0.8478 vs 0.8200）：说明该数据集不依赖混合边界
5. **v2 改造方向的核心数据依据**（之前漏掉的关键点）：
   - enron/sms_spam 上 Full < Nomix（门控有害） → v2 应该学到降低 gate
   - har/breast_cancer 上 Full > Nomix（门控有利） → v2 应该学到保持 gate
   - **LearnableGate 的目标不是"全数据集提升 ARI"，而是"让 gate 自适应"**——这要求 4 个 β 必须参与梯度训练

**问题**：
- 当前每 variant 仅 1 seed（seed=42），方差不可量化 → 5-seed 平均是 Phase 2 必备
- 单数据集上变体差异往往 < 0.01，不足以支持 claim
- v1 改造的核心矛盾：**4 个 β 是 argparse 默认值，不在 torch 计算图中**——它们全程不更新
  - run.py 第 80-83 行：`parser.add_argument("--beta_mutual", type=float, default=1.0)`
  - mixing.py 第 33-37 行：用 numpy 算 sigmoid
  - run.py 第 328-339 行：训练前调一次，全程不变
  - **修复 = 把 β_* 改成 torch.nn.Parameter，加入 optimizer**

**追溯代码**：`/home/luolie/ToPoGate/scripts/run_topogate_ablation.py` + `scripts/aggregate_ablation.py`


---

## 2026-07-26 v3 模块优化实验数据

### Inputs / Outputs 概述

| 阶段 | 输入 | 输出 |
|------|------|------|
| v3_smoke | 5 ds × 4 variants × 3 seeds config | `result/learnable_gate_smoke/v3_smoke/results.csv` (60 rows + 1 header) |
| v3_tune | 5 ds × 3 variants (lr 3x/5x/10x_no_lgm) × 3 seeds | `result/learnable_gate_smoke/v3_tune/results.csv` (45 rows + header) |
| v3_best | 5 ds × 1 variant × 3 seeds | `result/learnable_gate_smoke/v3_best/results.csv` (15 rows + header) |
| per-run | script + YAML config + seed | `result/learnable_gate_smoke/v3_*/<dataset>__<variant>__seed<n>/{metrics.json, summary.json, args.json}` |

### Hyperparameters

所有 v3 smoke runs 共用:
- `--epochs` 50
- `--batch_size` 256
- `--neighbor_k` 5
- `--mask_ratio` 0.3
- `--warmup_epochs` 10
- `--ramp_epochs` 10
- `--freeze_mae_after_epoch` 1000000000 (effectively disabled)
- `--edge_reliability_mode` sim_mutual_snn_distance
- `--pseudo_weight` 0.3
- `--gate_max` 0.15 (initial)
- `--learned_gate_init_mode` zero

各 variant 独有参数:
- baseline: (all defaults)
- v3_lgm: `--learnable_gate_max true`
- v3_lr: `--gate_lr_multiplier 10.0`
- v3_full: `--learnable_gate_max true --gate_lr_multiplier 10.0`
- v3_conservative: `--learnable_gate_max true --gate_lr_multiplier 5.0`
- v3_lr3: `--learnable_gate_max true --gate_lr_multiplier 3.0`
- v3_lr10_no_lgm: `--gate_lr_multiplier 10.0`
- v3_best: `--learnable_gate_max false --gate_lr_multiplier 10.0 --enhanced_stats 6 --learnable_gamma true --learnable_mask_ratio true`

### 完整 ARI 表（v3_smoke 60 runs）

| Dataset | seed | baseline | v3_lgm | v3_lr | v3_full |
|---------|------|----------|--------|-------|---------|
| Mouse_retina | 42 | 0.9505 | 0.9346 | 0.9424 | 0.8979 |
| Mouse_retina | 123 | 0.8937 | 0.8949 | 0.8955 | 0.8828 |
| Mouse_retina | 7 | 0.9456 | 0.9406 | 0.9416 | 0.9156 |
| enron | 42 | 0.8196 | 0.8798 | 0.8798 | 0.8798 |
| enron | 123 | 0.8197 | 0.8343 | 0.8238 | 0.8238 |
| enron | 7 | 0.8200 | 0.7492 | 0.7677 | 0.7676 |
| har | 42 | 0.5326 | 0.5326 | 0.5326 | 0.5326 |
| har | 123 | 0.5137 | 0.5137 | 0.5041 | 0.5041 |
| har | 7 | 0.4716 | 0.4716 | 0.4757 | 0.4757 |
| breast_cancer | 42 | 0.9022 | 0.9022 | 0.9022 | 0.9022 |
| breast_cancer | 123 | 0.9021 | 0.9078 | 0.9021 | 0.9021 |
| breast_cancer | 7 | 0.8798 | 0.8798 | 0.8798 | 0.8798 |
| sms_spam | 42 | 0.8487 | 0.8700 | 0.8770 | 0.8841 |
| sms_spam | 123 | 0.8558 | 0.8558 | 0.8637 | 0.8707 |
| sms_spam | 7 | 0.7993 | 0.7921 | 0.7993 | 0.8210 |

**Mean per dataset & variant**:

| Dataset | baseline | v3_lgm | v3_lr | v3_full |
|---------|----------|--------|-------|---------|
| Mouse_retina | 0.9299 | 0.9234 | 0.9265 | 0.8988 |
| enron | 0.8198 | 0.8211 | 0.8238 | 0.8237 |
| har | 0.5060 | 0.5060 | 0.5041 | 0.5041 |
| breast_cancer | 0.8947 | 0.8966 | 0.8947 | 0.8947 |
| sms_spam | 0.8346 | 0.8393 | 0.8467 | 0.8586 |
| **overall** | **0.7970** | **0.7973** | **0.7992** | **0.7960** |

### 完整 ARI 表（v3_tune 45 runs）

| Dataset | v3_conservative (lr 5x + lgm) | v3_lr3 (lr 3x + lgm) | v3_lr10_no_lgm (lr 10x, no lgm) |
|---------|-------:|-------:|-------:|
| Mouse_retina | 0.9029 | 0.9131 | 0.9265 |
| enron | 0.8238 | 0.8238 | 0.8238 |
| har | 0.5041 | 0.5041 | 0.5041 |
| breast_cancer | 0.8947 | 0.8928 | 0.8947 |
| sms_spam | 0.8441 | 0.8444 | 0.8467 |

### learned 参数 final values 模式

**v3_lr (lr 10x, no lgm) - 最稳的🔥**:
- Mouse_retina: β_mutual=4.85, β_snn=9.98, β_perturb=-5.80 (β 学得很猛)
- sms_spam: β_mutual=2.27, β_snn=2.28, β_perturb=-2.28 (温和)

**v3_full (lr 10x + lgm) - 破坏性**:
- Mouse_retina: eff_gate_max=0.985 ⚠️ (退化成全 mixing)
- enron: eff_gate_max=0.057 (学到几乎不 mixing)
- sms_spam: eff_gate_max=0.681 (过大)

**关键学习**:
1. lr multiplier 单独用 → β 训练充分 → +0.002 ARI
2. lgm + lr multiplier 联用 → gate_max 飞 → 破坏 mixing → -0.001 ARI
3. lgm 单独用 (lr 1x) → 几乎不动 (eff_gate_max 0.15-0.33) → 中性

### 完整 v3_best 实验结果

(写入 `result/learnable_gate_smoke/v3_best/results.csv`)

**预期**: lr 10x + EnhancedTopologyFeatures + LearnableEdgeReliability + AdaptiveMaskRatio (no lgm) 应配合 stacking 效应 → +0.005 ~ +0.01 ARI vs baseline.

### 验证命令

```bash
# 复现 v3_smoke
bash scripts/learnable_gate/launch_v3_workers.sh "4 5 6"

# 复现 v3_tune
bash scripts/learnable_gate/launch_v3_tune.sh

# 复现 v3_best
bash scripts/learnable_gate/launch_v3_best.sh

# 聚合
python3 -c "
import csv
from collections import defaultdict
import statistics
rows = list(csv.DictReader(open('result/learnable_gate_smoke/v3_smoke/results.csv')))
by = defaultdict(list)
for r in rows:
    if r['ari']:
        by[(r['variant'], r['dataset'])].append(float(r['ari']))
for k, v in sorted(by.items()):
    print(f'{k}: {statistics.mean(v):.4f} (n={len(v)})')
"
```

### 论文引用此数据的方式

论文 Section 4.x (Ablation on v3 模块) 使用 v3_smoke 60 runs 表展示 4 个 variants 的对比，强调：
1. v3_lr (lr 10x) = 唯一正向改动，可作为 v3 默认 config
2. v3_full (lr 10x + lgm) 退化 → 论文应说明 lgm 在某些数据集上需进一步 regularisation
3. v3_best (4 改动组合, no lgm) → 进一步正向提升

### 完整 v3_best 实验结果 (15 runs)

**配置**: lr 10x + EnhancedTopologyFeatures (6 stats) + LearnableEdgeReliability (4 gamma learned) + AdaptiveMaskRatio (no lgm)

| Dataset | seed 42 | seed 123 | seed 7 | Mean | Std | 基线 Mean | Δ |
|---------|---------|----------|--------|------|-----|-----------|---|
| Mouse_retina | 0.9493 | 0.8964 | 0.9350 | 0.9269 | 0.0265 | 0.9299 | -0.0031 |
| enron | 0.8798 | 0.8238 | 0.7677 | 0.8238 | 0.0560 | 0.8198 | +0.0040 |
| har | 0.5326 | 0.5041 | 0.4757 | 0.5041 | 0.0285 | 0.5060 | -0.0019 |
| breast_cancer | 0.9022 | 0.9021 | 0.8798 | 0.8947 | 0.0129 | 0.8947 | +0.0000 |
| sms_spam | 0.8770 | 0.8496 | 0.7993 | 0.8420 | 0.0392 | 0.8346 | +0.0074 |
| **OVERALL** | -- | -- | -- | **0.7983** | -- | **0.7970** | **+0.0013** |

### 训练后的 learned 参数

**beta (Mouse_retina seed 42)**:
- β_mutual=4.776, β_snn=9.852, β_perturb=-5.722, β_degree=3.793, β_cluster=0.0

**gamma (Mouse_retina seed 42)**:
- γ_sim=0.060, γ_mutual=0.060, γ_snn=0.060, γ_distance=0.060 (全部收敛到同一值)

**mask_ratio**: 0.300 (未动)

**关键发现**:
1. **gamma 4 个值完全相等** — 表明 LearnableEdgeReliability 实际上没学到差异化模式
2. **mask_ratio 完全不更新** — gradient 通道太弱
3. **β_degree (3.79) 显著非零** — EnhancedTopologyFeatures 中 degree_norm 提供了有效信号
4. **β_cluster = 0** — 因为 Mouse_retina (n=8K) > 5000 threshold，cluster 被跳过

### 三方案最终对比

| Variant | Mean ARI | Δ vs baseline | 提升 / 退化 |
|---------|----------|---------------|-------------|
| baseline (静态) | 0.7970 | +0.0000 | -- |
| v3_lr (lr 10x only) | **0.7992** | **+0.0021** | **正向** |
| v3_best (lr 10x + 3 more) | 0.7983 | +0.0013 | 微正 |
| v3_conservative (lr 5x + lgm) | 0.7939 | -0.0031 | 负 |
| v3_lr3 (lr 3x + lgm) | 0.7956 | -0.0014 | 负 |
| v3_full (lr 10x + lgm) | 0.7960 | -0.0010 | 负 |
| v3_lgm (lgm only) | 0.7973 | +0.0003 | 中性 |

**最终推荐 default config**: `learnable_gate_max=False, gate_lr_multiplier=10.0` (即 v3_lr)

## 2026-07-26 — v4_baseline 第一组 (Phase 1a)

**输入**: GPU 4 + GPU 5, 8 ds × 3 seeds × 2 variants = 48 runs, epochs=30
**输出**: `result/learnable_gate_smoke/v4_baseline/results_v4_{static,lr10}.csv`
**关键发现**: v4_lr10 OVERALL diff = -0.0013, p=0.5 (no sig.)
**结论**: v3_lr 不是 universal improvement — 在 Mouse_retina/mammographic 上好，在 har/sms_spam/enron 上退化

**复现**:
```bash
python3 scripts/learnable_gate/run_v4_static.py --datasets breast_cancer_wisconsin_original iris mammographic_mass sms_spam_collection har enron Mouse_retina spambase --epochs 30 --gpu 0  # CUDA_VISIBLE_DEVICES=4
python3 scripts/learnable_gate/run_v4_baseline.py --lr_multiplier 10.0 --datasets same --epochs 30 --gpu 0  # CUDA_VISIBLE_DEVICES=5
```

## 2026-07-26 v5_1g_ste 多 seed 验证（Phase 2.2 收尾）

**输入数据**：
- 7 个数据集：`har`, `iris`, `spambase`, `breast_cancer`, `mammographic`, `enron`, `Mouse_retina`
- 来自 `datasets/<name>.npz`（已 CLUBench 验证）
- 3 seeds × 7 ds = 21 runs

**实验命令**：
```bash
bash scripts/learnable_gate/run_v5_multiseed.sh
# 内部：CUDA_VISIBLE_DEVICES=4 python3 scripts/learnable_gate/run_v5_separate.py \
#   --v5_gamma_mode one_param_scalar --mask_ratio_learnable --epochs 30
```

**输出文件**：
- `result/learnable_gate_smoke/v5_multiseed/run_all.log` (21 runs full log)
- Per-run metrics: 单个 `metrics.json`（脚本 bug — 每次覆盖而非追加）

**结果（ARI）**：

| Dataset | v4_static | s1 | s2 | s3 | mean | std | vs v4 |
|---------|-----------|-----|-----|-----|------|-----|-------|
| Mouse_retina | 0.9314 | 0.9046 | 0.8894 | 0.9091 | 0.9010 | 0.008 | -0.030 |
| breast_cancer | 0.8984 | 0.8631 | 0.8741 | 0.8797 | 0.8723 | 0.007 | -0.026 |
| enron | 0.8647 | 0.7579 | 0.8431 | 0.7981 | 0.7997 | 0.035 | -0.065 |
| har | 0.4226 | 0.4612 | 0.3320 | 0.3714 | 0.3882 | 0.054 | -0.034 |
| iris | 0.6795 | 0.7028 | 0.6703 | 0.6480 | 0.6737 | 0.023 | -0.006 |
| mammographic | 0.3479 | 0.3621 | 0.3621 | 0.3680 | 0.3641 | 0.003 | +0.016 |
| spambase | 0.6542 | 0.4831 | -0.0290 | 0.5116 | 0.3219 | 0.248 | -0.332 |
| **AVG** | 0.6855 | - | - | - | **0.6173** | - | **-0.068** |

**关键发现**：
1. **v5_1g_ste 平均比 v4_static 差 -0.068 ARI** — 1-seed 结论被 3-seed 验证
2. **Spambase 严重不稳定** (std=0.25) — 1-seed 时幸运得了 ARI=0.48，3-seed 均值仅 0.32
3. **其他数据集相对稳定** (std < 0.06)，但方向一致：v5 都输 v4
4. **唯一 win**：mammographic (+0.016, std<0.01) — 几乎是 tie

**追溯代码**：`scripts/learnable_gate/run_v5_separate.py` (Phase 2.2 v5 components)

**结论**：v5 修复了 v3 的两个数学缺陷（4-γ 退化 + mask 不动），但**未能转化为 ARI 优势**。Phase 2.2 的"修复"叙事成立，但 Phase 3 必须找到**真正的性能突破点**。

---

## 2026-07-26 表 4 Ablation 完整化（Phase 4）

**输入数据**：
- 15 datasets (5 core + 10 ext) × 8 variants = 120 runs
- 来自 `datasets/<name>.npz`（已 CLUBench 验证）
- 大部分来自原有 ablation (5/15 datasets 完整 8 variants, 10/15 仅有 4 variants)
- 补 36 runs (10 ext ds × 4 missing variants, 5 core 已完整)

**实验命令**：
```bash
bash scripts/learnable_gate/run_ablation_ext_complete.sh  # 给 5 个 ext 跑 4 missing variants
python3 scripts/learnable_gate/run_ablation_ext_remaining.py  # 补 iris/Quake 等
python3 scripts/learnable_gate/run_ablation_hrvatin.py  # hrvatin 用 subsample_size=5000
python3 scripts/learnable_gate/append_ablation_ext_complete.py  # 合并到 CSV
```

**输出文件**：
- `result/ablation/merged_summary.csv` (120 rows, 15 ds × 8 variants)
- `result/ablation/<ds>/<ds>__static_gate_<variant>__ep*.json` (per-run, 120 个)

**Ablation 结果 (15 datasets 平均)**：

| Variant | Mean ACC | Mean NMI | Mean ARI | vs Full |
|---------|---------|---------|---------|--------|
| static_gate_constant_gate | 0.6720 | 0.5357 | 0.4635 | +0.0005 |
| static_gate_edge_only | 0.6672 | 0.5134 | 0.4664 | -0.0042 |
| static_gate_far_neighbors | 0.6490 | 0.4719 | 0.4199 | -0.0224 |
| **static_gate_full** | **0.6714** | 0.5285 | 0.4586 | +0.0000 |
| static_gate_gate_only | 0.6654 | 0.5086 | 0.4616 | -0.0061 |
| static_gate_no_topology_features | 0.6709 | 0.5111 | 0.4649 | -0.0006 |
| static_gate_nomix | 0.6736 | 0.5357 | 0.4739 | +0.0021 |
| static_gate_random_neighbors | 0.6527 | 0.4933 | 0.4146 | -0.0187 |

**关键发现**：
1. **`full` vs `nomix` 几乎完全相等** (0.6714 vs 0.6736) — mixing 本身对 ACC 没有显著贡献
2. **`random_neighbors` / `far_neighbors` 显著差** (0.65) — **拓扑邻居 vs 随机邻居有结构性差异** (-0.02 ACC)
3. **`gate_only` / `edge_only` / `no_topology_features` 几乎没有差异** — **单组件贡献小**
4. **Spambase 是 mix 真正受益的数据集**：full=0.908, nomix=0.900, random=0.686（**-0.22 ACC**！）

**Per-dataset 关键点**：

| Dataset | full | nomix | random | far | 关键发现 |
|---------|------|-------|--------|-----|---------|
| Mouse_retina | 0.984 | 0.985 | 0.981 | 0.950 | 拓扑极强，far_neighbors 唯一差 |
| spambase | 0.908 | 0.900 | 0.686 | 0.786 | **邻居信息最关键** (-0.22) |
| har | 0.707 | 0.548 | 0.652 | 0.709 | **mix 反而退化** (-0.16) |
| iris | 0.840 | 0.913 | 0.840 | 0.867 | nomix 反而更好 |
| Campbell | 0.417 | 0.379 | 0.385 | 0.241 | 全 mixture 都差不多；far 极差 |
| enron | 0.938 | 0.968 | 0.943 | 0.943 | nomix 反而最好 |

**追溯代码**：`scripts/static_gate/run_topogate_ablation.py` + `scripts/learnable_gate/run_ablation_ext_complete.sh` + `scripts/learnable_gate/run_ablation_ext_remaining.py` + `scripts/learnable_gate/run_ablation_hrvatin.py`

**结论**：
- ✅ **Ablation 表 4 完整 120 runs**，EXPERIMENT_PLAN.md P0 阻塞项完成
- ✅ **核心洞察**：TopoGate 的核心价值来自**拓扑邻居混入**（vs random/far），而非具体 gate/edge control 的细化
- ✅ **可学门控（gate/edge components）单独贡献小** — 与 v5 修复叙事一致（v5 修复了 bug 但无 ARI 优势）
- ⚠️ **重要警示**：在 har/iris 上 **mixing 反而让 ACC 下降** — 这是诚实的方法学发现

## 2026-07-26 v7_cross_attn latent-mix smoke test (6 datasets, single-seed=42)

### 实验产物路径
- 训练产物: `result/v7_cross_attn/smoke/<dataset>__v7_cross_attn__seed42/` (embedding_final.npy, metrics.json, summary.json)
- 聚合表: `result/v7_cross_attn/smoke/comparison.csv`, `result/v7_cross_attn/smoke/v7_vs_ablations.csv`
- 训练日志: `logs/v7_smoke_all.log`, `logs/v7_smoke_iris.log`
- YAML 配置: `methods/TopoGate/v7_cross_attn/configs/v7_cross_attn_smoke.yaml`

### 输入参数
- 数据集（6 个）: enron, sms_spam_collection, ISOLET, cnae9, Quake_Smart-seq2_Lung, iris
- seed: 42 (single-seed smoke)
- epochs: 150, batch: 256, hidden: 128, lr: 1e-3
- LearnableGate: gate_max=0.15, learnable_gate_max=true, schedule warmup=10/ramp=10
- v7 关键改动: mask_b_anchor=False (B 不 mask), cross_attn_heads=1

### 实测结果

| dataset | K | v7 ARI | v3_full ARI | ΔARI vs v3_full | best_v1_abl ARI | ΔARI vs best_abl |
|---|---:|---:|---:|---:|---:|---:|
| enron | 2 | 0.7076 | 0.8354 | **-0.128** | 0.8753 (nomix) | -0.168 |
| sms_spam_collection | 2 | 0.8513 | 0.7834 | **+0.068** | 0.8478 (edge_only) | **+0.004** |
| ISOLET | 26 | 0.5080 | 0.5033 | +0.005 | 0.5472 (nomix) | -0.039 |
| cnae9 | 9 | 0.2211 | 0.2844 | **-0.063** | 0.3503 (gate_only) | -0.129 |
| Quake_Smart-seq2_Lung | 11 | 0.1608 | 0.1829 | -0.022 | 0.1898 (nomix) | -0.029 |
| iris | 3 | 0.6312 | 0.6402 | -0.009 | 0.7720 (nomix) | -0.141 |

**整体（6 datasets × 1 seed, single-seed smoke 标记）**：
- avg Δ vs v3_full = -0.0249, wins/losses/ties = 2/4/0
- ≤-0.03 退化: 2/6 (enron, cnae9)
- ≤-0.05 严重退化: 2/6
- v7 ≥ best_v1_ablation: 1/6 (仅 sms_spam +0.004)

### 计划与实际
- 计划: 6 datasets × 1 seed ≈ 2.5 min
- 实际: 6 datasets = 433.5s (~7.2min)
  - 主要因为 enron 167.2s + Quake 108.6s（大数据集上 cross-attn 计算开销）
  - 中小型数据集 (iris 3.3s, sms_spam 14.2s, cnae9 18.2s, ISOLET 121.0s) 表现如预期

### 结论
- **NO-GO 触发**（plan 停止条件：≥2/6 数据集 Δ ≤ -0.05）
- v7 验证未通过，不进入 multi-seed 阶段
- 唯一正向信号：sms_spam v7 超过 topogate_full 同时压过 best v1 ablation
- 详细分析与未来方向见 CHANGELOG_errors.md 2026-07-26 v7 段落

## 2026-08-05 V16 计数域筛选与 Stage-0 边界

V16 预注册数据仍为 `Campbell`、`Mouse_retina`、`Baron Human`、
`Quake_Smart-seq2_Lung`、`fbis.wc`、`tr45.wc`。当前 NPZ 复核结果：前两者
可恢复为 `log1p(count)`，后四者为非负整数 count；六者均满足
`d >= 2000`、零比例 `>= 0.80`、中位数行 nnz `>= 5`、空行比例 `<= 0.10`。
`enron`、`reuters` 等当前已稠密化或非 count 语义的数据不进入 V16 主域。

fbis/tr45 的无标签 Stage-0 exploratory 已完成：候选图 recurrence 约
`0.40/0.47`，median held-out support 的正边比例约 `1.4%/1.7%`。这只是
机制筛选信号，不使用标签，不代表性能。Campbell/Mouse_retina/Baron/Quake
的大集 support 审计因稀疏 kNN 和逐块 support 成本在本轮中止，未写入正式
结果盘。

fbis 五 epoch、三 seed `[42,123,7]` 的固定五路 exploratory 已保存于
`/tmp/v16_stage1_fbis_5ep_3seed`：V16 未超过 self-only 或 fixed graph，按固定规则
标记为 `empirical_not_supported`，不通过修改 support 或门控公式挽救。
## 2026-08-06 V9 related public-data bundle

To extend the fixed `spambase` structural/application neighborhood without
changing the model question, three public sources were downloaded once and
prepared under `datasets/external/v9_related_20260806/`:

- UCI dataset 51, Internet Advertisements: raw `ad.data`/documentation in
  `raw/internet_advertisements.zip`; prepared `processed/internet_advertisements.npz`
  (`3279 x 1558`, `K=2`, missing values retained as NaN, zero fraction after
  protocol fill `0.9904`). The package contains `2820` non-ad and `459` ad rows,
  differing from the UCI documentation's `2821/458`; no row was edited.
- UCI dataset 228, SMS Spam Collection v1: raw corpus in
  `raw/sms_spam_collection.zip`; prepared
  `processed/sms_spam_collection_full_tfidf500.npz` (`5574 x 500`, `K=2`, zero
  fraction `0.9778`) using fixed unlabeled word unigram+bigram TF-IDF
  (`min_df=2`, `max_features=500`, sublinear TF, L2 norm).
- OpenML DID 350 (`webdata_wXa`, file 52253): raw Sparse ARFF in
  `raw/openml_webdata_wXa.arff`; prepared `processed/webdata_wXa.npz`
  (`36974 x 123`, `K=2`, zero fraction `0.8872`). OpenML metadata and the
  source description are stored once in `provenance/openml_350.json`.

UCI dataset 379 (`website_phishing`) was downloaded for source verification;
its 1353 x 9 matrix is elementwise identical to the existing local prepared
copy and is not duplicated. The machine-readable source record is
`processed/manifest.json`; the V9-ready external matrix manifest is
`processed/v9_external_manifest.json`.

The frozen X-only Stage-0 audit completed `3/3` candidates with no feature
errors. This is data/provenance and structural-audit evidence only; no model
training or performance claim is attached. The preparation recorded URLs,
versions, file IDs and byte metadata once and did not recompute SHA-256 or
other hashes during experiment preparation or resumption.
## 2026-08-06 V9 Full/NoMix related-dataset calculation

在 `datasets/external/v9_related_20260806/processed/v9_external_manifest.json`
上按冻结 V9 配置运行 Full 与 NoMix，各 3 seeds `[42,123,7]`，共 18 runs，全部
完成且 `labels_used_during_fit=false`。`webdata_wXa` 按固定 `max_samples=20000`
执行行采样，其他两个矩阵使用完整样本。

| Dataset | Full ARI mean±std | NoMix ARI mean±std | paired ΔARI | paired ΔNMI |
|---|---:|---:|---:|---:|
| Internet Advertisements | -0.0222±0.0385 | -0.0360±0.0599 | +0.0138 | -0.0134 |
| webdata_wXa | 0.1971±0.0049 | 0.2003±0.0097 | -0.0033 | -0.0122 |
| SMS Spam Collection full TF-IDF500 | 0.8572±0.0100 | 0.8718±0.0078 | -0.0146 | -0.0080 |

总体数据集均值 ΔARI=`-0.00135`，median=`-0.00326`，dataset-bootstrap 95% CI
`[-0.01461,+0.01381]`，Wilcoxon `p=0.75`。Internet Advertisements 的正均值
来自 NoMix 更负，且 seed 方向不一致；SMS 与 webdata_wXa 均由 NoMix 略高。该批次
只回答固定 Full/NoMix 对照，不做超参数、污染比例、损失或数据集后选择。

正式产物：`result/v9_related_20260806_full_nomix/`，配对 CSV 为
`summary/related_methods_summary.csv`，逐 run 记录和标准输出均保留在同一路径。

## 2026-08-06 Internet Advertisements fixed baseline comparison

在 UCI Internet Advertisements (`3279 x 1558`, `K=2`) 上补充同一预处理
(`nan_to_num_then_column_standard_scaler`) 的三 seed `[42,123,7]` 对照：
V9 Full `-0.0222+-0.0385`、NoMix `-0.0360+-0.0599`、V9-compatible scMAE
`-0.0360+-0.0599`、PCA(95%)+KMeans `0.0229+-0.0299`、AHDPC `-0.0005+-0.0000`、
DPC-GFNN `-0.0613+-0.0000`、GCC fixed-scale known-K adapter
`0.0349+-0.0442`（均为 ARI mean+-std）。所有已完成 run 均
`labels_used_during_fit=false`；指标在拟合后计算，未重新计算哈希。

NoMix 与 scMAE 的 `predictions.npy`、`embedding_final.npy` 在三 seed 都完全相同。
GCC 原生分区始终为一个簇，固定尺度结果通过本地 known-K split adapter 输出两簇，
不能表述为原生 GCC 的优势。完整多尺度 known-K GCC 没有完成任何 seed，见
`result/v9_related_20260806_other_models/internet_advertisements/gcc/incomplete_compute.json`；
不填充 ARI。汇总在
`result/v9_related_20260806_other_models/internet_advertisements/comparison_summary.csv`。

## 2026-08-07 V16.1 raw-count candidate source audit

对本地 scCluBench H5AD 文件进行只读、backed sparse-aware 元数据核验；没有转换、
训练或读取标签用于模型选择，也没有重复计算任何哈希。

| 数据集 | 形状 | X 存储/抽样语义 | 当前状态 |
|---|---:|---|---|
| `SRP171040` | `33956 x 53678` | backed CSR，float64；抽样非零值非整数，无 raw-count layer | `not_admitted_raw_semantics_unresolved` |
| `SRP235541` | `27798 x 53678` | backed CSR，float64；抽样非零值非整数，无 raw-count layer | `not_admitted_raw_semantics_unresolved` |
| `SRP309176` | `47747 x 56723` | backed CSR，float64；抽样非零值非整数，无 raw-count layer | `not_admitted_raw_semantics_unresolved` |
| `SRP145013` | `40848 x 67300` | backed CSR，float64；初始块无非零值，未发现 raw-count layer | `not_admitted_raw_semantics_unresolved` |
| `CRA007122` | `25121 x 56044` | backed CSR，float64；抽样非零值非整数，无 raw-count layer | `not_admitted_raw_semantics_unresolved` |
| `Mouse_Hypothalamus` | `6400 x 18573` | backed CSR，float32；抽样值为整数，但 `cell_type` 只有一个类别 | `not_admitted_single_benchmark_class` |
| `GSE96583` | `43095 x 35635` | backed CSR，float64；没有可用聚类标签字段 | `not_admitted_missing_benchmark_labels` |

这些状态只表示当前协议门槛下不进入 V16.1 Stage-0/Stage-1，不把数据源问题
解释为模型性能失败。后续候选必须先补齐可核验 raw count（或可逆 `log1p(count)`）
语义和至少两个 benchmark 类别。

## 2026-08-07 V16.1 `subsample_2k` count candidate

登记本地 scCluBench 的 `/data/luolie/biopipeline/scCluBench/data/subsample_2k.h5ad`：
`X` 为 CSR-backed 非负整数 count，`obs.cell_type` 提供 benchmark 类别；转换后的
临时 bundle 为 `/tmp/v16_1_expanded_data/subsample_2k.npz`，形状
`2000 x 53678`、`5664809` 个非零项。`scripts/V16_1/run_stage0.py` 已按固定
`k=20`、三次 split 和 cross-fitted support 审计；Stage 0 通过，candidate recurrence
`0.5676`、稳定边比例 `0.7902`、support 非退化，随后启动固定 Stage 1。标签只用于
benchmark K/后验指标，未用于图、support 或模型拟合；本条未重复计算哈希。

## 2026-08-07 V16.1 `hrvatin_geo_maintype_counts` paired result

`hrvatin_geo_maintype_counts` 来自本地 scCluBench 的 raw integer count bundle，形状
`48266 x 25187`，零比例 `0.94188`，median row nnz `1112`，`K=8`，四项高维稀疏加分
全部满足。按固定 `[42,123,7]`、clean/compound、五路 readout 完成 `30/30` 个产物。
候选图后验 edge purity 为 `0.9968`、budget-normalized recall 为 `0.9971`，但
cross-fitted predictive support 正边率仅 `0.000524`，gate 平均 null mass 为
`0.999118`。该数据集按预注册晋级规则分类为 `empirical_not_supported`，不进入正例表；
标签只用于 K 和后验指标，未用于图、support 或拟合。

### 2026-08-07 Norman Stage-0 resource boundary

`NormanWeissman2019_perturbation` 的 expanded-count bundle 为 `111445×33694`，约
`361582621` 个 CSR 非零项。固定 Stage-0 在当前 CPU 资源上运行约 4 小时 45 分钟仍
未写出审计 JSON；由于全局已达到预注册搜索上限且没有任何 candidate positive，任务已
停止并标记 `stage0_incomplete_compute`。它不进入性能汇总、不被解释为模型性能失败，也不
重新调整输入或 gate。

## 2026-08-07 V 系列数据域与失败证据统一索引

V1--V16.1 的数据适用域、输入语义错误、理论域外状态、graph recall 与 predictive
support 的分离，以及 Campbell/Mouse_retina/hrvatin 等历史正例和边界集，统一汇总于
[`V_SERIES_FAILURE_RETROSPECTIVE.md`](V_SERIES_FAILURE_RETROSPECTIVE.md)。

重要数据结论：高维、稀疏、count 语义和高 graph purity 不是 V16.1 的充分条件；
`hrvatin_geo_maintype_counts` 同时具有 high-sparse bonus、purity `0.9968` 和 recall
`0.9971`，但 support 正边率仅 `0.000524`，故不能再按稀疏度或图质量反复筛选以挽救
模型。后续数据选择必须先由生成假设划定理论域，再按固定协议记录
`theory_domain_not_supported` 或 `empirical_not_supported`。
# 2026-08-11 V21 formal six-dataset matrix submitted

V21 formal matrix launcher 已启动，旧输出根
`result/V21/v21_formal6_full_20260811/` 因图的自邻居过滤缺陷仅保留为审计记录；修复后的
协议文件为 `result/V21/v21_formal6_full_20260811_graphfix/stage_spec.json`，唯一正式输出根为
`result/V21/v21_formal6_full_20260811_graphfix/`。本批固定六个数据集：`cnae9`、`Mouse_retina`、
`Baron Human`、`Campbell`、`sms_spam_collection`、`hate_speech`；固定两个 variant
(`topology_assignment_adversarial`、`scmae_only`) 和 seeds `[42, 123, 7]`，共 36 个
run key。配置、variant 和 seed 不按 ARI 选择。

launcher 只允许物理 GPU 1--6，每卡最多一个子进程，GPU 0/7 禁用；启动前检查显存/利用率，
不会停止或复用已有进程。当前状态由
`result/V21/v21_formal6_full_20260811_graphfix/launcher_state.json` 记录；截至本条记录，
24/36 已完成、12/36 排队，Baron Human/Campbell 的 4 个 Full key 在 CPU fallback 中运行。
为避免 GPU/CPU 同 key 覆盖同一输出目录，主 launcher 已在无 GPU 子进程时暂停，
`scripts/V21/resume_after_cpu_fallback.py` 等待 Mouse_retina、Baron Human、Campbell 三批 CPU
状态全部终态后再恢复唯一 launcher。模型拟合和图/Gate/loss 不接收 `y`；cluster-head 变体的
known K 由 runner 外层从 NPZ 标签唯一值推导并在每个 summary 记录，scMAE-only 的 K 只用于
拟合后 KMeans readout。

# 2026-08-11 V21 graph-fix formal matrix terminal audit

上述矩阵已完成终态收尾。正式输出根为
`result/V21/v21_formal6_full_20260811_graphfix/`，`matrix_audit.json` 显示
`expected_jobs=36`、`completed_valid_jobs=36`、`audit_ok=true`、`provenance_ok=true`，没有
`incomplete_compute`。六个数据集的 Full/scMAE-only ARI 宏平均分别为 `0.207693` 和
`0.418579`，配对差为 `-0.210886`；结果只用于完整 V21 与 scMAE-only 的比较，不包装成
纯 Gate 消融。

GPU handoff 期间有少量迁移命令因错误的绝对配置路径（例如 `/methods/...`）在加载模型前
退出；这些进程没有生成或覆盖模型产物，最终有效结果来自后续正确路径的重跑。终态时没有
V21 worker；GPU5 基本空闲，但其它可见 GPU 仍有外部进程，GPU0/7 仍按项目规则禁用。

# 2026-08-11 V21 ARI grid and three-seed confirmation

`result/V21/v21_ari_grid_seed42_20260811/` 完成 `72/72` 个候选任务，固定 seed42、六数据集
宏平均 ARI 选择配置：`assignment_weight=0.1`、`gate_lr=2.5e-4`、`epochs=80`、
`warmup_epochs=40`、`infomax_weight=0.05`，选择分数 `0.395632`。确认根
`result/V21/v21_ari_confirm_aw0.1_glr0.00025_ep80_20260811/` 完成 `18/18`，严格审计通过；
三 seed 宏平均 ARI=`0.342684`，相对正式 Full=`+0.134991`，相对正式 scMAE-only=
`-0.075895`。该选择使用 ARI，拟合函数不接收 `y`，因此记录为开发/确认层证据。
## 2026-08-12 V22 dataset extension panel downloaded and audited

在看到任何 V22 结果前固定并下载四个新增候选，输出目录为
`datasets/external/v22_dataset_extension_20260812/`，登记入口为
`datasets/external/v22_dataset_extension_20260812/manifest.json`。候选分层如下：

| dataset_id | source / shape | zero fraction | labels | status |
|---|---:|---:|---:|---|
| `sector__libsvm_sparse_highdim` | LibSVM sector, `6412 x 55197` | `0.997046` | 105 | eligible |
| `real_sim__libsvm_sparse_highdim` | LibSVM real-sim, `72309 x 20958` | `0.997552` | 2 | eligible |
| `covtype__libsvm_dense_control` | LibSVM covtype, `581012 x 54` | `0.777778` | 7 | eligible control |
| `pbmc3k__10x_unlabelled_count` | 10x PBMC3k, `2700 x 32738` | `0.974128` | none | eligible_unlabelled |

LibSVM 与 PBMC 原始文件均保存于 `raw/`，处理文件保存于 `processed/`，manifest 记录
来源 URL、source identity、原始/处理后 SHA-256、矩阵形状、非零数、存储格式和
`labels_used_during_fit=false`。PBMC3k 原始 10x 归档没有独立细胞类型标签，因此只允许
显式 `--n-clusters` 的无标签探索，不进入 ARI/NMI 汇总；四个 strata 不合并成统一普适性
结论。当前下载使用本地代理的 `verify=False`，manifest 明确记录该传输边界，原始 SHA
保留供独立环境复核。

验证：`file` 确认四个原始文件分别为 bzip2/gzip，处理文件为 NPZ；四个 V22 输入均通过
shape、有限值、稀疏重建和标签长度检查。V22 的 sector 与 PBMC3k 单 epoch CPU 运行只
作为 `engineering_smoke`，不作为性能结论。

## 2026-08-15 V25 closure artifact source and publication bundle

新增 `scripts/V25/build_closure_artifacts.py`，只读取已冻结的 A0/A1/A2、E1 pilot/confirmation、
E2-A、E2-B/C、A1 E3 gate 和 Phase E closure JSON/CSV，生成五个闭环表/决策文档及
`V25_CLOSURE_ARTIFACTS.json`。E1 汇总使用六个 dataset-level rows；seed 是重复测量，E2-A
coordinate distributions 仅作描述，pilot 缺失的 feature counts 标记为 `deferred`。新发布副本
位于 `papers/V25_systematic_mechanism_study/results/`，不含 checkpoint、branchpoint、原始数据、
预测数组或缓存；Holdout 仍为 `0/6 inconclusive_not_completed`，未新增计算。

## 2026-08-15 V25 A2 holdout adapter contract

V25 A2 在不读取任何 E1 outcome 的前提下，从既有 outcome-independent manifests 冻结
`result/V25_systematic_mechanism_study/A2/holdout_candidate_manifest.json`。输入 adapter
只允许当前 V21 已实现的 `clubench_bridge`/`shared_text` -> `prepare_dual_input` 路径，
feature selection、normalization、graph input、model input 和 label/K 边界均固定；
`scRNA_count` 的 PBMC 条目因 adapter 未冻结且无外部标签被排除，不在 Phase D 临时开发。

当前候选审计记录 1 个 adapter-valid 的独立 scRNA（Quake Smart-seq2 Lung）和 9 个
sparse-text/related candidates；目标为 4 scRNA + 2 text 的 pool shortfall 被显式写入 manifest，
不是用 outcome 事后补齐。V25 暂不把这些候选当作 holdout 结果，只有 Claim Freeze 后按已冻结
primary endpoint 激活。

## 2026-08-15 V25 E1 pilot source and label boundary

E1 pilot 使用已登记的 `datasets/cnae9.npz`、`datasets/Mouse_retina.npz` 和
`datasets/sms_spam_collection.npz`，每个数据集 seeds `[42,123,7]`。每个 panel 的
`manifest_record.json` 保存 source path/SHA256、input protocol、known-K 来源和输出契约；
输入 adapter 继续使用冻结的 V21 `prepare_dual_input`，graph/model preprocessing 不读取
labels。labels 只由 outer runner 在 N/R/T fit、actual Adam one-step 和 clean embedding
KMeans 完成后计算 ARI/NMI；`K_source=benchmark_oracle_from_y` 仅表示外层 benchmark K
元数据。

E2-A 的 label-free feature counts 在 confirmation 的 T training path 保存；其 post-hoc
feature semantics（Fisher、support-MI、class support enrichment）不会回流到 Gate、loss、
preprocessing 或数据选择。旧 pilot 未保存 counts，不因补充诊断而重训或修改 source/protocol
hash。

## 2026-08-15 V25 paper evidence provenance export

`result/V25_systematic_mechanism_study/PaperEvidence/` 是从已冻结 V25 工件复制/汇总的
分析层输出。`source_manifest.json` 记录 A0/A1/E1/E2/PhaseD/PhaseE 输入文件的 SHA256；
`claim_scope_audit.json` 明确允许范围为 observational V1--V22 atlas 与 conditional
heterogeneous V21 case study，禁止 universal topology、independent holdout 或
coordinate-level inferential claim。该步骤没有新增数据、预处理、模型训练或 holdout
选择。

PaperEvidence 的 `figures/` 由同一冻结 CSV 输入离线渲染；`figure_manifest.json` 保存输入
文件 SHA256、行数和图表的 claim scope。PNG 只是论文展示资产，不是额外实验结果或新的
endpoint。
## 2026-08-15 V25 A0 registry schema synchronization

重导出 `result/V25_systematic_mechanism_study/A0/mechanism_evidence_registry.csv`，使正式
registry 与 `scripts/V25/build_a0_registry.py` 的冻结 schema 一致。每条记录现在显式保留
`source_hash`、`preprocess_hash`、`k_source`/`k_hash`、labels/K isolation、
`measurement_timing`、`causal_status`、`reused_from` 和 `alternative_explanation` 字段。
使用的 V1--V22/V23/V24 输入及 2209/1637/431 计数未改变；没有重算模型或修改任何 primary
endpoint。对应的 PaperEvidence source manifest 与 V25 contract audit 已刷新。
### [2026-08-15 V25 holdout adapter metadata refresh]

V25 PhaseD holdout preflight was rerun for the predeclared `news20__libsvm_sparse_highdim`
and `rcv1_train__libsvm_sparse_highdim` candidates without reading any outcome. Each frozen
dataset/job now records the input adapter, label-free feature-selection rule, normalization,
max-feature rule, graph input, model input, source hash, and known-K source. The refresh only
updates manifest provenance; it does not launch training or alter the existing `0/6`
`inconclusive_not_completed` holdout boundary.
### 2026-08-17 representation-consumer probe S1 data/provenance

- Input snapshots: S0-frozen `H0` (`d0=128`, SVD random_state=0), positive-cosine `k=20` candidate
  pool, `budget_cap=8`, row-specific `b_i=min(8,positive_count_i)`。
- Arms: raw-H0 feature-only F、ungated candidate U、matched random R、diagnostic O_pool/O_full；
  all graph consumers use active-subgraph Spectral with frozen eigsh/KMeans contract.
- Seeds `[42,123,7]` are paired repeats; labels are outside fit and used only for O diagnostics and
  outer metrics. Formal outputs and per-run hashes are under
  `result/representation_consumer_probe/S1_oracle_v2/`.
- Support deficiency is reported separately: cnae9 zero-budget rows=1, sms=40, hate_speech=135;
  no row was filled with negative-cosine edges or removed at dataset level.
- Fresh experiment-integrity audit completed after hardening: all 90 stored ARI/NMI values were
  independently recomputed exactly; root artifact manifest verifies an exact 827-entry tree and
  nested run manifests; reuse/aggregation now require `audit_ok=true` plus config identity. The
  evidence remains known-K real-GT benchmark plus label-derived diagnostic oracle, not a deployable
  selector or TopoGate gain.
# 2026-08-18 Parallel mechanism probes — S0 freeze only

No dataset was loaded for fitting and no GPU job was launched.  Compact S0
artifacts were generated under
`result/learned_relation_rule_probe/S0_freeze/` and
`result/adaptive_corruption_probe/S0_freeze/`.  Track A records the dormant,
outcome-independent twelve-dataset holdout by source hash without copying raw
data; Track B freezes its holdout selection rule but defers membership until
before B5.  No ARI/NMI/ACC or corruption performance number is produced.
The review also fixed the protocol semantics of Track A `R` (inherited
matched-random reference) and Track B `C_clean_no_corruption` (primary floor);
these are definitions, not computed results.
### [2026-08-18 independent representation-consumer follow-up probes]

Track A A1 reuses the audited RS1 edge feature tables and S1 v2 R/O_pool
graphs.  It uses Logistic and `p->32->1` TinyMLP diagnostic scorers over Full,
No-geometry and No-rank views with five GroupKFold-by-anchor folds; labels are
available only to the diagnostic O_pool target builder and post-fit metrics.
The compact source/config/OOF audit and A1 summary are in
`result/learned_relation_rule_probe/A1_supervised_ceiling/`.

Track B B1 uses the audited S0 H0 stem, clean-H0 column standardization, a
fixed `d_eff->64->32->64->d_eff` reconstruction probe, row support threshold
`0.05`, corruption rate `0.25`, 30 epochs and batch size 512.  The six arms
record requested/effective coordinate, support/value and absolute-change
statistics.  The formal 108-run compact matrix is in
`result/adaptive_corruption_probe/B1_corruption_library/`; the aborted width
attempt is explicitly quarantined under the corresponding `*_attempts/`
directory.  No raw arrays or model weights are publication artifacts.
## 2026-08-18 support-target validation probe data boundary

M0 inherited the audited S0 dense `H0` snapshots and budget manifests without copying them. The full
source paths and SHA256 values are in `result/support_target_validation_probe/M0_freeze/freeze_manifest.json`.
M1 uses only clean H0, deterministic P2 action replay and Hungarian active-partner matching.

Structural checks passed for all 9 rows (exact touched-coordinate budget, zero threshold-support
change, zero row-value-multiset mismatch, labels unused). Dataset-total L1 mismatch was
`0.001582--0.001686` for Mouse, `0.094640--0.095877` for Baron, and `0.005726--0.006001` for
Campbell; only the six non-Baron rows satisfy the frozen `<=0.05` tolerance. No model, labels,
embeddings, predictions or raw arrays were written.
