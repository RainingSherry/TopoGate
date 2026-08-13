# V11 sparse H0 TDA pilot: formal comparison

**Status**: 75/75 completed; this is a fixed-protocol performance comparison, not a universal claim.

## Protocol

- Datasets: `balance_scale, spect_heart, banknote, flame, vehicle`; seeds: `[42, 123, 7]`; variants: `V11_full, V11_nomix, V11_tda_h0_mst, V11_tda_fixed_filtration, V11_tda_random`.
- Input: AHDPC processed `x/y` NPZ files; `K=int(np.unique(y).size)` only for benchmark K and post-fit metrics.
- Training: V11 default YAML, 80 epochs, CPU `--no-cuda`, one thread per numerical backend; no per-dataset tuning.
- TDA object: fixed raw-PCA kNN sparse 1-skeleton, unit-row Euclidean chord filtration, exact H0 union-find; H1/dense VR are not computed.
- Controls: `fixed_filtration` is distance-only; `random` is deterministic edge-shared random prior; all prior values are detached.
- Evidence: every run has `summary.json`, resolved config, source hash, predictions, labels_true, and `labels_used_during_fit=false`.

## Aggregate metrics

Values are mean +/- sample standard deviation over 5 datasets x 3 seeds.

| Variant | Head ARI | KMeans ARI | NMI | Silhouette | Final gate | Mean graph loss |
|---|---:|---:|---:|---:|---:|---:|
| `V11_full` | 0.1468 +/- 0.1419 | 0.1629 +/- 0.1522 | 0.1418 +/- 0.1306 | 0.3308 +/- 0.0566 | 0.0160 +/- 0.0153 | 0.0444 +/- 0.0431 |
| `V11_nomix` | 0.1493 +/- 0.1429 | 0.1643 +/- 0.1495 | 0.1433 +/- 0.1306 | 0.3313 +/- 0.0590 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| `V11_tda_h0_mst` | 0.1468 +/- 0.1418 | 0.1622 +/- 0.1526 | 0.1418 +/- 0.1306 | 0.3308 +/- 0.0567 | 0.0178 +/- 0.0169 | 0.0508 +/- 0.0416 |
| `V11_tda_fixed_filtration` | 0.1468 +/- 0.1419 | 0.1622 +/- 0.1525 | 0.1418 +/- 0.1306 | 0.3307 +/- 0.0567 | 0.0179 +/- 0.0167 | 0.0483 +/- 0.0423 |
| `V11_tda_random` | 0.1468 +/- 0.1418 | 0.1626 +/- 0.1521 | 0.1418 +/- 0.1306 | 0.3308 +/- 0.0566 | 0.0172 +/- 0.0168 | 0.0492 +/- 0.0427 |

## Paired tests

`mean_delta` is left minus right over the 15 paired dataset-seed runs. P-values are descriptive checks, not a license to select a method after seeing labels.

| Comparison | Metric | Delta | Wins/Ties/Losses | Wilcoxon p | Paired t p |
|---|---|---:|---:|---:|---:|
| `V11_full` - `V11_nomix` | `head_ari` | -0.002555 | 4/3/8 | 0.2721 | 0.3903 |
| `V11_full` - `V11_nomix` | `kmeans_ari` | -0.001397 | 8/1/6 | 0.9250 | 0.7239 |
| `V11_full` - `V11_nomix` | `nmi` | -0.001518 | 4/3/8 | 0.2094 | 0.1374 |
| `V11_full` - `V11_nomix` | `silhouette` | -0.000570 | 6/0/9 | 0.3591 | 0.8425 |
| `V11_tda_h0_mst` - `V11_full` | `head_ari` | 0.000010 | 2/12/1 | 1.0000 | 0.7172 |
| `V11_tda_h0_mst` - `V11_full` | `kmeans_ari` | -0.000726 | 1/10/4 | 0.1380 | 0.0998 |
| `V11_tda_h0_mst` - `V11_full` | `nmi` | 0.000009 | 2/12/1 | 1.0000 | 0.7601 |
| `V11_tda_h0_mst` - `V11_full` | `silhouette` | -0.000022 | 10/0/5 | 0.6788 | 0.8810 |
| `V11_tda_fixed_filtration` - `V11_full` | `head_ari` | 0.000002 | 3/11/1 | 0.7150 | 0.9667 |
| `V11_tda_fixed_filtration` - `V11_full` | `kmeans_ari` | -0.000665 | 1/10/4 | 0.0796 | 0.1079 |
| `V11_tda_fixed_filtration` - `V11_full` | `nmi` | 0.000011 | 3/11/1 | 0.7150 | 0.7886 |
| `V11_tda_fixed_filtration` - `V11_full` | `silhouette` | -0.000090 | 8/0/7 | 0.6387 | 0.4307 |
| `V11_tda_random` - `V11_full` | `head_ari` | 0.000018 | 1/14/0 | 0.3173 | 0.3343 |
| `V11_tda_random` - `V11_full` | `kmeans_ari` | -0.000274 | 2/10/3 | 0.5002 | 0.3204 |
| `V11_tda_random` - `V11_full` | `nmi` | 0.000015 | 1/14/0 | 0.3173 | 0.3343 |
| `V11_tda_random` - `V11_full` | `silhouette` | 0.000072 | 9/0/6 | 0.3028 | 0.2267 |

## Dataset-level head ARI

| Dataset | Full | NoMix | H0 MST | Fixed filtration | Random |
|---|---:|---:|---:|---:|---:|
| `balance_scale` | 0.1121 +/- 0.0614 | 0.1146 +/- 0.0604 | 0.1121 +/- 0.0614 | 0.1121 +/- 0.0614 | 0.1121 +/- 0.0614 |
| `spect_heart` | 0.1370 +/- 0.0421 | 0.1476 +/- 0.0657 | 0.1370 +/- 0.0421 | 0.1370 +/- 0.0421 | 0.1370 +/- 0.0421 |
| `banknote` | 0.0081 +/- 0.0038 | 0.0076 +/- 0.0034 | 0.0081 +/- 0.0037 | 0.0080 +/- 0.0033 | 0.0082 +/- 0.0038 |
| `flame` | 0.3943 +/- 0.0791 | 0.3943 +/- 0.0791 | 0.3943 +/- 0.0791 | 0.3943 +/- 0.0791 | 0.3943 +/- 0.0791 |
| `vehicle` | 0.0823 +/- 0.0103 | 0.0825 +/- 0.0095 | 0.0823 +/- 0.0104 | 0.0824 +/- 0.0104 | 0.0823 +/- 0.0103 |

## TDA diagnostics

- The H0 prior is structurally active: its nonzero edge fraction and merge count are recorded in `run_diagnostics.csv`; `fixed_filtration` and `random` have different score distributions by construction.
- Compare `mean_graph_loss` and `final_gate` jointly with clustering metrics. A lower graph loss or larger gate is not a clustering improvement by itself.
- If H0, fixed-filtration, and random produce similar ARI while graph diagnostics differ, the result supports a no-go for this prior as a validated clustering mechanism, not a claim that TDA is generally ineffective.

## Reproducibility inputs

- Raw run outputs: this directory's `*__*__seed*/summary.json` and arrays.
- Aggregation script: `scripts/analysis/analyze_v11_tda_h0_pilot.py`.
- Source manifest: `datasets/AHDPC/MANIFEST.json`.
- Existing mathematical boundary audit: `result/analysis/TopoGate_whole_project_math_TDA_audit_2026-08-03.md`.
