# GitHub Release Scope

This snapshot publishes the auditable code and compact result evidence for the
current ToPoGate research workspace.

Included are `methods/TopoGate/`, the related `scripts/` and `tests/`, public
protocol/result reports, compact `result/` evidence bundles, and the V25 paper
source and figures. V22 is represented by metadata-only summaries; newer
probes are represented by their explicitly marked `FINAL` or compact audit
artifacts.

The only dataset file is the tiny public `datasets/iris.npz` smoke-test
fixture; it is not research evidence.

Excluded are raw or processed datasets, dataset symlinks, model and optimizer
weights, checkpoints, branchpoints, embeddings, prediction arrays, topology
caches, logs, bytecode, temporary review material, and nested upstream Git
repositories. See `RELEASE_MANIFEST.md` for the exact publication boundary.

The release does not infer cluster counts from uploaded labels during model
fitting. Labeled benchmark runs use labels only for the outer K protocol and
post-fit metrics; unlabelled runs require an explicit `n_clusters` argument.
Engineering smoke and incomplete runs remain clearly marked and are not
promoted to performance claims.
