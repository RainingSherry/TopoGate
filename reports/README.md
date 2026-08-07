# Results and Reports

This directory is a curated documentation snapshot of the TopoGate research
repository, updated on 2026-08-07. It contains the current fact table,
aggregate result tables, and selected experiment/provenance reports.

## Contents

- `RESULTS_SUMMARY.md`: current result fact table, including explicit
  `empirical_not_supported`, `no-go`, and `incomplete_compute` boundaries.
- `analysis/`: cross-version, V9, V11, V12, and V13 analysis reports.
- `tables/`: aggregate per-dataset and paired-result tables. These are not raw
  model outputs.
- `V_SERIES_FAILURE_RETROSPECTIVE.md`: cross-version failure taxonomy and
  current research decision.
- `EXPERIMENT_PHASES.md`: experiment-stage history and protocol boundaries.
- `CHANGELOG.md`, `CHANGELOG_data.md`, and `CHANGELOG_errors.md`: method,
  data-provenance, and error-audit records.
- `ESWA_BASELINES.md`: baseline eligibility and fairness registry.

## Evidence boundary

No input datasets, prediction arrays, checkpoints, batch logs, paper PDFs, or
temporary run directories are included. Paths in copied documents are
normalized to portable placeholders; `unpublished-temp/` refers to artifacts
that remain outside this repository and are not recreated here.

Historical documents retain their original dates and claims. They must be read
with the status and evidence-tier fields in `RESULTS_SUMMARY.md`; a historical
smoke, single-seed result, adapter result, or incomplete computation is not a
current paper-level performance claim.
