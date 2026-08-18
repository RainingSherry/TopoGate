# support_target_validation_probe

This is an independent mechanism-validation study. It does not create a new V
version and does not reopen `sparse_corruption_principle_probe`.

The completed C2 matrix found one simple static intervention, `P2_SupportTarget`,
with material `Delta_ARI = ARI(P2) - ARI(P0_Random)` on Mouse_retina, Baron
Human and Campbell. The new study asks whether that observation remains in a
matched support-role contrast when the frozen threshold-defined support of
dense `H0` is preserved, rather than treating the contrast as a strict causal
isolation. The active-active control may be easier for reconstruction than P2's
active-inactive swap, so a positive `Delta_cross` is conservative/downward-biased
for a support-crossing interpretation.

Current authorization is limited to M0 and M1:

```text
M0 new-project freeze
  -> deterministic C2 action replay audit
M1 H0 matched support-role contrast
  -> one new magnitude-matched support-preserving control
```

M1 GPU execution is conditional on a no-training full-epoch magnitude
preflight. A tolerance failure is recorded as `magnitude_match_not_estimable`
and does not become a negative performance result.

M2 raw-X validation, M3 holdout, M4 full-backbone transfer, adaptive policy and
GAN remain locked until the pre-registered M1 gate is satisfied.

The C2 project is read-only evidence. M1 reuses its P2 summaries and never
reruns or retunes P2. Raw data, labels, arrays, embeddings, predictions,
weights, checkpoints and logs remain local and are excluded from publication.

> Support in C2/M1 denotes threshold-defined support of dense H0, not raw-X
> zero/nonzero support; raw sparse-support claims require a separate validation.
