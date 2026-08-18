# C1 mechanism localization

C1 consumes the closed B1 compact summary and audited S0 H0. It does not retrain an encoder and does
not load labels. For each development dataset, old B1 arm and paired seed it computes:

- support transitions (`0→nonzero`, `nonzero→0`), cell nnz and gene prevalence shifts, support
  Jaccard, binary support entropy and co-occurrence distortion;
- value distribution/rank/absolute-change diagnostics and high-expression distortion;
- cosine kNN preservation, neighbor rank stability, local-density change and sampled pairwise
  distance distortion.

The old B1 ARI/L_rec values are joined only as post-fit provenance columns. They are not recomputed,
retrained or relabeled as new results. The residual score used for C4 structural replay is a
label-free column-median/MAD proxy and is explicitly not the B1 warm-up residual artifact.

Interpretation is restricted to mechanism quadrants (`support`, `value`, `geometry/difficulty`);
there is no causal claim that one diagnostic explains all B1 utility.
