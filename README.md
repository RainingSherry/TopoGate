# ToPoGate

ToPoGate is an auditable research repository for unsupervised clustering on
high-dimensional and sparse single-view data. The repository contains the
TopoGate/scMAE implementations, isolated exploratory versions, experiment
launchers, contract tests, and compact evidence bundles.

## What is included

- `methods/TopoGate/`: TopoGate and scMAE code, configurations, and version
  boundaries from the legacy gate through V26 support-oracle studies.
- `scripts/` and `tests/`: reproducible launch/audit utilities and regression
  tests.
- `reports/`: preregistrations, protocol decisions, integrity audits, and
  bounded result reports.
- `result/`: compact JSON/CSV/Markdown summaries and hashes for the audited
  result panels. V22 metadata and selected V25 evidence tables are included.
- `papers/V25_systematic_mechanism_study/`: the V25 protocol, paper source,
  figures, and generated PDF.
- `datasets/iris.npz`: a tiny public smoke-test fixture only; it is not used
  as research evidence.

## Result boundaries

The published summaries preserve the project's label-isolation and benchmark
protocol boundaries. Labels are used only for post-fit benchmark metrics (or
explicitly marked diagnostic oracles), and unlabelled runs record an explicit
cluster count. The reports distinguish engineering smoke runs, incomplete
computations, bounded development findings, and evidence suitable for a
scientific claim. They do not claim universal generalisation.

## Deliberately excluded

Raw datasets, dataset symlinks, checkpoints, branchpoints, model weights,
embeddings, prediction arrays, topology caches, per-step logs, Python bytecode,
and nested upstream repositories are not part of this snapshot. Source paths
and SHA-256 provenance for local inputs remain in the compact manifests and
reports; the corresponding local files are required for full reproduction.

## Quick checks

```bash
python -m compileall methods/TopoGate scripts
python -m pytest -q methods/TopoGate/V11/tests/test_v11.py -k 'not frozen_v9_reference_manifest'
python -m pytest -q tests/v10_reliable_graph
```

The full V11 test file currently reports one known warning: the checked-in
legacy `v9_reference_manifest.json` records a SHA-256 for an older
`learnable_gate/run_npz.py`. The snapshot preserves both files unchanged so
that this historical mismatch is visible rather than silently re-anchored.

The exact dataset, seed, K protocol, and output contract for each study are
documented next to its implementation and report. This snapshot is a
publication/reproducibility boundary, not a packaged Python distribution.
