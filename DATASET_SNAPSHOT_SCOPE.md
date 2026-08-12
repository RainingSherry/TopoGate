# Dataset Snapshot Scope

The GitHub release stores no dataset binary. It stores only the V22 dataset
provenance manifests under `result/V22/dataset_manifests/`.

The manifests include source URLs, raw and processed SHA-256 values,
shape/sparsity profiles, label availability, and the `labels_used_during_fit`
boundary. Raw archives and processed matrices remain outside GitHub.

The V22 unlabelled PBMC records intentionally have no ARI/NMI/ACC without an
external label source. Explicit K values used for engineering execution are
recorded in the corresponding result metadata and are not inferred from the
uploaded matrices.
