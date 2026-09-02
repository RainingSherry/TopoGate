# TopoGate V0 formal reproducibility rerun

This directory is the canonical result root for the 2026-09-01 V0 rerun. It contains
36 completed-valid cells: six fixed PlantNet inputs, two V0 parameterizations (fixed/F
and topology/T), and paired seeds `[42, 123, 7]`. No cell was reused, and no cell was
left incomplete or protocol-mismatched.

## Frozen protocol

- Input source: `/home/luolie/biopipeline/dimension-reduction/plantnet/result/scvicar_attribution_v2/datasets/*.h5ad`.
- Preprocessing: V0 h5ad loader, automatic raw/count input selection, Seurat HVG 1000,
  `normalize_total(target_sum=10000)`, `log1p`, and input scaling.
- Training: current V0 YAMLs, 80 epochs, one shared scMAE backbone/trainer. Fixed/F uses
  `neighbor_k=5`; topology/T uses `neighbor_k=10` and the analytic reliability/node-gate path.
- K protocol: `K=int(np.unique(y).size)` from the outer `resolved_label` benchmark column.
  K is used only by the post-fit KMeans readout and metrics; it is not passed into the
  label-free fit boundary.
- GPU protocol: physical GPU 6 was the only idle legal card available. Physical GPUs 0
  and 7 were forbidden; each cell used one worker with controlled numerical threads.

| Dataset | K |
| --- | ---: |
| `Blood_BoneMarrow` | 30 |
| `Human_Pancreas_1` | 6 |
| `Human_Pancreas_3` | 13 |
| `Mouse_Pancreas_1` | 10 |
| `PRJNA895163` | 12 |
| `TabulaSapiens_Pancreas` | 16 |

The manifest records the source and configuration SHA256 anchors for every cell. The
independent `matrix_audit.json` rechecks those anchors against the current files and
also verifies canonical keys, required artifacts, array shapes, finite values, K source,
and all label-isolation flags.

## Metrics

The CSV files contain all six benchmark metrics for every seed. The table below reports
ARI and NMI as mean +/- sample standard deviation over the three paired seeds; it is a
descriptive known-K benchmark, not a claim of unsupervised deployment superiority.

| Dataset | Fixed ARI | Topology ARI | Fixed NMI | Topology NMI |
| --- | ---: | ---: | ---: | ---: |
| `Blood_BoneMarrow` | 0.418699 +/- 0.012852 | 0.436933 +/- 0.011185 | 0.720427 +/- 0.005993 | 0.731938 +/- 0.007351 |
| `Human_Pancreas_1` | 0.878527 +/- 0.000955 | 0.880978 +/- 0.001383 | 0.855246 +/- 0.000859 | 0.857834 +/- 0.002682 |
| `Human_Pancreas_3` | 0.775411 +/- 0.001173 | 0.859935 +/- 0.073995 | 0.823593 +/- 0.001612 | 0.848855 +/- 0.024646 |
| `Mouse_Pancreas_1` | 0.884923 +/- 0.007399 | 0.862244 +/- 0.041974 | 0.861885 +/- 0.003335 | 0.846122 +/- 0.026733 |
| `PRJNA895163` | 0.177385 +/- 0.013274 | 0.168517 +/- 0.020170 | 0.371692 +/- 0.018618 | 0.357035 +/- 0.022564 |
| `TabulaSapiens_Pancreas` | 0.487205 +/- 0.006509 | 0.512820 +/- 0.015274 | 0.721740 +/- 0.003970 | 0.734451 +/- 0.004563 |

Across the 18 paired seed comparisons, topology minus fixed is ARI `+0.016546 +/-
0.047936` and NMI `+0.003609 +/- 0.022245`; this variation includes both positive and
negative dataset-level changes and is not a pre-registered superiority claim.

## Files

- `manifest.json`: frozen protocol, per-cell command/status, source/config hashes, and execution counts.
- `run_level_metrics.csv`: one row per dataset/parameterization/seed.
- `summary_metrics.csv`: dataset/parameterization means and standard deviations.
- `aggregate_summary.json`: aggregate row count and status.
- `matrix_audit.json`: independent final integrity and protocol audit (`audit_ok=true`).
- Each cell directory contains resolved config, status/run records, predictions, true labels,
  clean embedding, graph/gate diagnostics, history, metrics, summary, and model weights.

The earlier `result/v0/formal_v0_repro/` attempt is excluded: it passed explicit K and is
retained only as protocol-mismatch evidence. Historical `legacy_attribution_v2` staging is
also not part of this current-YAML rerun.

Claude review was requested through `auto-review-loop`. The private-data request was
privacy-rejected and the subsequent code-only request timed out after 1200 seconds without
a response; therefore there is no Claude acquittal or reviewer score for this run. Local
focused tests passed (`32 passed`), and that local verification is kept separate from the
benchmark metrics.
