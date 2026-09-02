# TopoGate V0 Result Snapshot

This directory contains the current V0 implementation and the compact,
metadata-only artifacts for its final audited experiments.

## Included

- `methods/TopoGate/V0/`: current V0 model, trainer, graph/corruption code,
  configs and focused tests.
- `clubench_single_seed_v1/`: 131 CLUBench datasets, fixed/F and topology/T,
  seed 42, 80 epochs, known-K outer readout, and the label-free tuning record.
- `formal_v0_repro_v2/`: six PlantNet datasets, two parameterizations and
  three paired seeds, including the matrix audit and aggregate metrics.
- `legacy_attribution_v2/` and `legacy_parity_plantnet_a1/`: historical
  provenance/parity evidence, explicitly separated from current V0 efficacy.
- `RESULTS_SUMMARY_20260902.md`: the current project result fact table.

## Excluded by design

Embedding/prediction arrays, model checkpoints, h5/h5ad inputs, topology
caches, temporary logs and worker directories are excluded. They are large
runtime artifacts rather than compact result records, and some are outside
GitHub's 100 MB per-file limit. The original local results remain at
`/home/luolie/ToPoGate/result/v0`; hashes and protocol metadata are retained
in the included manifests.

The CLUBench single-seed result is descriptive only. The formal matrix uses
three paired seeds and known-K benchmark readout; it does not establish
unlabelled deployment performance or universal superiority.
