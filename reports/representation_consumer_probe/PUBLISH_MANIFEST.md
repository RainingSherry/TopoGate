# Representation-consumer probe — GitHub publication scope

This commit publishes the canonical, weight-free evidence for the independent
`representation_consumer_probe` study. It does not publish raw data or large run outputs.

## Included

- Canonical protocol, execution plan, pre-registration, stage definitions, decision, and S1/S2
  result reports under `reports/representation_consumer_probe/`.
- The four implementation scripts and focused contract tests under
  `scripts/representation_consumer_probe/` and `tests/representation_consumer_probe/`.
- S0 contract metadata, S1-v2 opportunity summaries/manifests, and S2 SimpleCut summaries/manifests
  under `result/representation_consumer_probe/`.
- The current S2 integrity audit (`EXPERIMENT_AUDIT.md/.json`) with known-K/oracle qualifications and
  the training-history timing warning.
- The project fact-table and data/protocol history updates in the tracked `reports/` changelog and
  results files.

## Explicitly excluded

- `embedding.npy`, `predictions.npy`, `labels_true.npy`, sparse graph arrays, checkpoints, optimizer
  state, raw datasets, caches, and temporary launch logs.
- The invalid first S1 matrix, dormant stage result directories, review traces, `AUTO_REVIEW` HTML,
  reviewer memory/state, acquittal logs, and timestamped working snapshots.

All published path fields use `<local_data_root>`, `<local_result_root>`, or
`<local_project_root>` placeholders; no workstation path is exposed. The
`artifact_hashes.json` files retain hashes and relative member names for the
audited local result-disk trees before publication redaction. The excluded
large files remain local and are not part of this GitHub publication.
