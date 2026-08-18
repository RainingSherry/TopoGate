# GitHub Release Scope

This repository is a reproducibility snapshot of the ToPoGate research project.
This release includes the main versioned implementations (V9-V22), the
independent post-V25 ACCG implementation, their
direct scMAE dependencies and runners, plus curated final result tables and
audit summaries. It does not publish the paper or any paper source bundle.

## Included

- Core TopoGate/scMAE dependencies, versioned source code, configurations,
  tests, and runnable scripts already present in the repository, including
  V18 latent-gate, V19 RG adapter, V20 adversarial mask, V21 assignment-gate,
  V22 discriminator/Keep-Gate implementations, and ACCG's joint-action
  constrained selection policy.
- `reports/ACCG/`: frozen synthetic contract, method protocol, and sanitized
  weight-free real-panel audit. The real clustering-improvement claim is
  retained as an auditable No-Go result rather than omitted.
- `reports/representation_consumer_probe/` and
  `result/representation_consumer_probe/`: frozen S0-S2 protocol, reports,
  terminal decision, and weight-free opportunity diagnostics. The study's
  selector and new-backbone routes remain locked and are not presented as
  performance claims.
- `reports/sparse_corruption_principle_probe/` and
  `result/sparse_corruption_principle_probe/`: compact C0-C2 protocol,
  static corruption-principle result tables, and independent integrity audit.
  C3 holdout, adaptive policy, GAN, and learned generator remain locked.
- `reports/support_target_validation_probe/` and
  `result/support_target_validation_probe/`: compact M0 replay audit, M1
  magnitude-estimability preflight, and terminal decision. The M1 GPU matrix
  was not authorized after the frozen control failed estimability; M2-M4 and
  adaptive routes remain locked.
- `result/final_results/`: one curated final result table or summary set for
  each audited version with non-smoke evidence, including V09-V14, V16.1, and
  V18-V22. The directory README records versions with no promotable result.
- Aggregate reports, manifests, protocol metadata, and incomplete-compute
  records needed to interpret the retained tables. No dataset binary is part
  of this release.

## Excluded

- Model checkpoints, optimizer state, topology memmaps, prediction/embedding
  arrays, caches, and temporary logs.
- Raw download archives and all dataset binaries.
- All paper files, manuscript sources, and literature PDFs.

The excluded artifacts remain on the project data volume. Where retained,
source URLs, dataset profiles, labels-used-during-fit flags, and SHA-256 values
are recorded in the corresponding final-result manifests and summaries. Local
machine paths are sanitized in this public snapshot. A result record marked
`incomplete_compute` is intentionally not treated as a completed performance
result.

## Reproduction boundary

The repository does not infer cluster counts from uploaded labels during model
fitting. Labeled benchmark runs use labels only for the outer K protocol and
post-fit metrics; unlabelled runs require an explicit `n_clusters` argument.
Single-seed or short engineering runs are retained as engineering evidence,
not performance claims.
