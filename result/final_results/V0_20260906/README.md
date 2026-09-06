# ToPoGate V0 result snapshot (2026-09-06)

This directory contains metadata-only result tables for the V0-T experiments.
Raw datasets, labels, embeddings, predictions, model checkpoints, topology
arrays, and worker logs are intentionally excluded.

## Included artifacts

- `frozen_ablation/`: 600 completed frozen-weight operator-ablation cells
  (8 datasets, 15 variants, 5 seeds), with per-seed rows, aggregate summaries,
  paired deltas, and the reconciled `T_full` selection records.
- `t_plus_ablation/`: the independently trained T+ ablation panel and its
  audit tables.
- `baselines/`: the recorded 5-seed comparison tables.
- `biological/`: the corrected biological V0-T summary and the biological
  baseline-completion audit.
- `paper_tables/`: the compact comparison workbook used for manuscript
  table preparation.

## Protocol boundary

The frozen ablation evaluator uses a trained Full V0-T representation and
freezes its weights before applying each operator variant. No optimizer is
created, no backward pass is called, and `training_enabled=false` is recorded
for every evaluation row. Labels are opened only after embedding export for
the oracle-K KMeans readout and metric calculation.

The reconciled `T_full` summary chooses the complete record with maximum
`ARI_mean` per dataset, using `NMI_mean`, `ACC_mean`, and lower `ARI_std` as
deterministic tie-breakers. The exact source record for each dataset is listed
in `t_full_reconciliation_manifest.json`. This reconciliation is a summary
operation; source per-seed files are unchanged.

These tables are evidence for the stated protocols, not a universal SOTA
claim. Some older summary records do not contain per-seed provenance; those
records remain marked by their source path and should be replaced by a
reproducible rerun before making seed-level claims.
