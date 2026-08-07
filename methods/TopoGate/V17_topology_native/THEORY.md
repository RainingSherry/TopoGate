# V17 Topology-Native Reference Solver

## Status

This directory is the first, non-deep V17 mechanism implementation. It is not
yet a paper-performance claim and it is not presented as a novel replacement
for SSC, robust SSC, or DSC. Its purpose is to test whether a single relation
object can close the TopoGate objective before an unfolded network is added.

V1--V16.1 and external baselines are not imported or modified.

## Closed objective

Samples are stored as rows. For fixed structure-preserving views
`H^(1), ..., H^(V)` and a label-free candidate support `E0`, V17-reference
solves

```text
min_C mean_v group_huber(H^(v) - C H^(v); lambda_outlier)
      + lambda_l1 * ||C||_1
      + 0.5 * lambda_l2 * ||C||_F^2

subject to diag(C) = 0 and supp(C) subset E0.
```

`group_huber` is the exact envelope obtained by minimizing a quadratic
residual plus a row-wise L2 outlier penalty. The proximal L1 update creates
exact zeros. The reference solver uses FISTA with a per-row spectral-norm
Lipschitz bound to solve this same convex objective; this changes only
convergence speed, not the solution being defined. Consequently, `supp(C)` is the topology gate; there is no second
utility score, teacher, gate probability, forced neighbor, or latent readout.

The sole affinity and output path are

```text
A = abs(C) + abs(C.T)
normalized spectral embedding(A)
KMeans on that spectral embedding
```

The final KMeans is only the standard spectral discretization. It never reads
an encoder embedding.

## Input and candidate semantics

- Count input: source-declared exact non-negative integer observations,
  `log1p`, then row L2. Numeric integrality alone is not a count certificate.
- Non-negative or signed continuous input: row L2 without count recovery.
- Fixed sparse random projections form multiple geometry-preserving views.
- Each view contributes a small blockwise cosine neighbor set; their union is
  only the allowed support and never forces a non-zero coefficient.
- Neither labels nor cluster count K enter the adapter, views, candidates, or C.

Degree-zero nodes are explicit topology abstentions and receive prediction
`-1`. They are not silently assigned through a second feature-space model.

## Deliberate first-stage boundary

The current solver does not yet add the alternating spectral feedback term
`Tr(F.T L(A(C)) F)` and does not learn per-layer thresholds. Those additions
are allowed only after the reference solver demonstrates all of the following
under a fixed protocol:

1. candidate union recalls useful same-structure edges;
2. proximal C retains non-zero edges without fragmenting most rows;
3. retained C improves edge purity over the ungated candidate set;
4. the same-C spectral readout improves over ungated and shuffled controls.

If these conditions fail, unrolling or adding an encoder would hide the failed
relation hypothesis rather than repair it.

## Implementation invariants

- no full `n x n` distance matrix is materialized;
- no coefficient outside `E0` can be created;
- `diag(C) = 0`;
- soft-thresholding creates exact zeros;
- `A` is symmetric and non-negative;
- K is used only by spectral readout;
- labels are post-hoc benchmark metadata only.
