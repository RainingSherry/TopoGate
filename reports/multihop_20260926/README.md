# ToPoGate multihop research snapshot

This directory publishes compact, weight-free result summaries for the V0_RG
multihop studies. The corresponding training and analysis code is under
`experiments/v0_rg_validation_tuned_multihop_20260922/`.

- `validation/`: completed validation-only tuning (19 datasets), Top5 H1–H8
  search, H9+ adaptive-depth search, and the selected configurations' posthoc
  full-X evaluation across 60 datasets. Full-X labels include validation and
  test rows, so those scores are descriptive, not held-out estimates. The summaries include selected
  configurations; checkpoint paths have been replaced with placeholders.
- `mechanism/` and `multihop_mechanism_20260924/`: completed source and
  intervention analyses. Full-X labels were read **after** training for
  evaluation and interpretation, never for fitting or selection.
- `aggregation/` and `aggregation_mechanism_20260924/`: completed three-case
  aggregation/readout analysis. Repeated KMeans fits are readout checks, not
  independent training seeds.
- `source_coverage_in_progress/`: completed portions of the separate 153-task
  source-coverage study. At snapshot time, 140 tasks had completed, 10
  official scHNTL CPU controls were running, and 3 tasks were pending.
  **These files are interim, not a final baseline comparison.**

Dataset arrays, predictions, embeddings, checkpoints, sampling traces,
per-cell records, and worker logs are intentionally omitted. Historical
validation search and selected-case posthoc analyses are exploratory evidence;
they are not a harmonized SOTA benchmark or proof that every dataset benefits.
