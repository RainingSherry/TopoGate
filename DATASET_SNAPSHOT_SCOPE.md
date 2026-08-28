# Dataset Snapshot Scope

Only the 6.5 KB `datasets/iris.npz` fixture is included so the CPU smoke tests
can run from a clean clone. It is a convenience test input, not a research
result or a replacement for the study datasets. No other dataset files are
included.

The compact manifests and reports retain source URLs, local source paths,
shapes, sparsity profiles, label availability, SHA-256 values, and the
`labels_used_during_fit` boundary needed to identify the inputs used locally.

Reproduction therefore requires obtaining the source datasets separately and
placing them at the paths expected by the study-specific manifests. Raw
archives and the repository's local `datasets` symlink are intentionally
excluded from publication.
