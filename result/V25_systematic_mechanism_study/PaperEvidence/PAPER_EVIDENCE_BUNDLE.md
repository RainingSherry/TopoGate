# V25 Paper Evidence Bundle

Generated: `2026-08-15T02:30:03.914322+00:00`

This bundle is an analysis-only export from frozen V25 artifacts. It does not retrain
a model and does not treat rows, coordinates, or seeds as independent population units.

## Primary facts

- A0: `2209` registry rows, `1637` paired rows, `431` units.
- A1: `194` material positive, `680` material negative, `763` observed-small.
- E1 confirmation: `9/9` panels audited successfully.
- E1 primary interpretation: conditional/heterogeneous V21 case study; not universal topology superiority.
- E1 evaluation boundary: real dataset ground truth is used after fitting, while benchmark-known K can size the cluster head during fitting; this is not fully label-free fitting.
- Independent holdout: `0/6` panels completed; status `inconclusive_not_completed`.

## Scope firewall

- Atlas rows are observational and stratified by protocol/readout.
- E2-A coordinate distributions are descriptive; inference is dataset x seed.
- Post-hoc Fisher/MI/class-support metrics were not fit inputs.
- Holdout CUDA OOM is incomplete compute, not a performance result.

## Files

- `atlas_version_family.csv`, `atlas_rows.csv`, `structural_opportunity_summary.csv`, `magnitude_gain_summary.csv`, `failure_localization_taxonomy.csv`, `local_global_boundary.csv`
- `e1_dataset_effects.csv`, `e1_seed_effects.csv`, `e1_pair_effects.csv`
- `e2_semantic_dataset_seed.csv`, `e2_semantic_dataset_summary.csv`, `e2_gradient_geometry.csv`
- `a2_decision.json`, `a2_claim_evidence_matrix.csv`, `measurement_schema.json`, `frozen_claim.json`
- `holdout_activation_manifest.json`, `holdout_e1_manifest.json`, `closure.json`, `closure_audit.json`
- `figures/` and `figure_manifest.json` (generated separately by `build_paper_figures.py`)
- `claim_scope_audit.json`, `source_manifest.json`, `paper_evidence_summary.json`
