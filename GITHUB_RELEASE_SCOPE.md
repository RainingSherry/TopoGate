# GitHub Release Scope

This repository is a reproducibility snapshot of the ToPoGate research project.
This release is scoped to the V22 implementation and its auditable result
metadata. It does not publish the paper or any paper source bundle.

## Included

- Core scMAE dependencies and the V22 source code, configurations, tests, and
  runnable scripts needed to inspect or rerun the V22 protocol.
- Key V22 result metadata only: the V22 manifests, aggregate summaries,
  aggregate reports, selected run summaries, resolved configurations, metrics,
  histories, and incomplete-compute records where available.
- Dataset provenance manifests for the V22 panel. No dataset binary is part of
  this release.

## Excluded

- Model checkpoints, optimizer state, topology memmaps, prediction/embedding
  arrays, caches, and temporary logs.
- Raw download archives and all dataset binaries.
- All paper files, manuscript sources, and literature PDFs.

The excluded artifacts remain on the project data volume. Their original
source URLs, dataset profiles, labels-used-during-fit flags, and SHA-256 values
are retained in `result/V22/dataset_manifests/` and the result records. Local
machine paths are sanitized in this public snapshot. A result record marked
`incomplete_compute` is intentionally not treated as a completed performance
result.

## Reproduction boundary

The repository does not infer cluster counts from uploaded labels during model
fitting. Labeled benchmark runs use labels only for the outer K protocol and
post-fit metrics; unlabelled runs require an explicit `n_clusters` argument.
Single-seed or short engineering runs are retained as engineering evidence,
not performance claims.
