# TopoGate V16 Predictive Graph Gate

V16 is an isolated two-stage experiment for high-dimensional, naturally
sparse count matrices.

1. Stage A trains a topology-disabled sparse count MAE.  The reconstruction
   objective is a masked Poisson likelihood on observed nonzero counts plus a
   small sampled-zero background term.  A KMeans-initialized spherical
   prototype readout is then frozen before topology is evaluated.
2. Stage B splits every count into independent binomial-thinned views.  View A
   builds a sparse cosine `k`NN candidate graph.  View B scores each candidate
   donor by held-out predictive support against a global background profile.
   Repeated splits are median-aggregated.
3. The support scores enter only an assignment-space abstaining sparsemax.
   Its first component is the self/null branch, so a row with no positive
   support is exactly unchanged.

The formal paired launcher also exposes one fixed `compound` condition.  It
uses count-domain-preserving feature dropout, integer Poisson perturbation on the
observed support, and row contamination; its parameters are recorded by the
runner and are not selected from labels or downstream metrics.

The supported input certificate is intentionally narrow: nonnegative integer
counts or exactly recoverable `log1p(count)`, `d >= 2000`, zero fraction at
least 0.80, median row nnz at least 5, and at most 10% empty rows.  Inputs
outside the certificate are reported as `theory_domain_not_supported` and are
not silently transformed into a different problem.

The formal NPZ loader reads the repository's uncompressed dense NPY members by
row-block memmap into CSR.  A compressed or otherwise dense-only member is
marked `dense_input_not_supported` before Stage A.  Because topology is an
assignment-space readout, `embedding_final.npy` remains the frozen Stage-A
latent, while `cluster_probabilities.npy` stores the propagated assignment.

Variants are `self_only`, `fixed_predictive_graph`, `V16_predictive_gate`,
`shuffled_support`, and `output_disabled`.  Labels are accepted only for
benchmark metrics and saved as post-hoc artifacts; they never enter Stage A,
the graph, support, or readout.
