# Published result tables

`v0_20260904/dataset_aggregate.csv` reports the five-seed mean and standard
deviation for ACC, NMI, and ARI for each completed dataset. The companion
`per_seed_metrics.csv` contains the 765 seed-level rows.

`v0_20260904/selection_manifest.json` records the candidate selected for each
dataset using only held-out masked loss, view consistency, and input/latent
neighbour overlap. It records `labels_read: false` for tuning. The full
candidate table is included as `all_candidate_metrics.csv` for auditability.

`baseline_comparison/` contains the available GCEALS, IDC, TableDC, ZEUS, and
legacy ToPoGate comparison CSVs. They are retained as comparison evidence and
are not silently converted into five-seed estimates.
