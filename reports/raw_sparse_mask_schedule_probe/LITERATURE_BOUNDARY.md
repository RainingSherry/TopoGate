# Literature boundary

The protocol uses only decision-relevant motivation, not reproduction claims.

- R²MAE motivates asking whether a fixed mask ratio is unnecessarily narrow;
  this study does not reproduce its architectures or reported benchmarks.
- Proportionally Masked Autoencoders and CACTI motivate respecting empirical
  missingness/statistical structure; exact raw non-zero support here is an
  operational mask target, not a claim that zeros are missing values.
- RaTab motivates a later representation-space localization branch when the
  input-level mechanism is sensitive but fails strong simple baselines.
- TabPFN v2 analysis and Dynamic Sparsity are counterpoints against assuming
  feature identity or a global sparsity prior is universally necessary.
- Masked graph work motivates awareness of structured masking, but no graph,
  topology selector, learned difficulty policy, GAN, attention, or Transformer
  is implemented in this project.

These references provide motivation only. The present evidence can support at
most a bounded statement about the six frozen sentinel matrices under the
small-AE protocol; it cannot support a universal sparse representation claim,
a biological support mechanism, an R²MAE replication, SOTA, or holdout
generalization.
