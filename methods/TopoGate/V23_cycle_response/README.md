# V23 Cycle-Response Geometry

V23 does not improve or retrain MAE with a cycle-consistency objective. It
tests whether a frozen perturb-repair-reencode probe exposes cross-sample
relations in high-dimensional matrices.

## Hypotheses

- **H1, response geometry:** samples from the same latent cluster have more
  similar responses to a shared dictionary of feature perturbations.
- **H2, recovery contribution:** post-repair cycle fingerprints contain
  information beyond pre-cycle response and support/effective-mask statistics.
- **H3, dependency conditionality:** the incremental signal is strongest when
  clusters differ in conditional dependency, and weakens under a
  within-cluster dependency-destroyed control.

`C_cycle` is the primary scientific fingerprint. `G_gain = A_pre - C_cycle`
is a separate, secondary recoverability fingerprint. Neither may replace the
other, and labels may not select between them per dataset.

## Innovation Ledger

1. **Tested in V23: perturbation-response geometry.** Shared feature
   perturbations produce per-sample cycle-repair fingerprints and an unlabeled
   cross-sample relation signal.
2. **Deferred and not implemented: functional/conditional feature
   redundancy.** Future work may ask whether individual features or feature
   sets functionally substitute for one another under controlled masking.
3. **Deferred and not implemented: recoverability-guided masked learning.** If
   the first two ideas are validated, a later method may mask informative but
   recoverable information instead of using random masks.

V23 contains no Gate, graph, discriminator, prototype head, contrastive loss,
feature-redundancy graph, or trainable redundancy-aware mask.

## Semantic-Space Mask Contract

Mask selection, donor replacement, zero corruption, support statistics, and
effective-mask auditing are defined before mean centering. For scRNA this is
selected log1p expression; for text and general tables it is the selected
feature matrix. Only after corruption or repair is constructed is the fitted
model transform applied. Numeric zero in centered model space is never used as
the semantic definition of a missing/zero feature.

## Fingerprints and Controls

The primary protocol saves raw and robustly standardized versions of:

- pre-cycle response `A_pre`;
- repair-only cycle response `C_cycle`;
- recovery gain `G_gain`;
- support/effective-mass response;
- untrained-network cycle response;
- fixed low-rank recovery response;
- frozen-encoder latent-only linear-decoder response;
- clean-drift-adjusted full-decoder response.

The primary response distance is preregistered as cosine. Euclidean and
correlation distances may be reported only as secondary diagnostics.

## Label Boundary

`fit` and `profile` accept matrix-only NPZ files and have no label or K
argument. `evaluate` is a separate outer process that may read `labels_true.npy`
or an explicit external K. With neither labels nor K, profiling still completes
and K-dependent readout is skipped.

## Minimal CLI

```bash
python -m methods.TopoGate.V23_cycle_response.fit \
  --matrix MATRIX_ONLY.npz --input-protocol shared_text \
  --output-dir result/V23_cycle_response/example/fit/seed42 --seed 42

python -m methods.TopoGate.V23_cycle_response.profile \
  --matrix MATRIX_ONLY.npz \
  --fit-dir result/V23_cycle_response/example/fit/seed42 \
  --output-dir result/V23_cycle_response/example/profile/seed42

python -m methods.TopoGate.V23_cycle_response.evaluate \
  --fingerprints result/V23_cycle_response/example/profile/seed42/fingerprints.npz \
  --labels LABELS_TRUE.npy \
  --output-dir result/V23_cycle_response/example/evaluate/seed42

# Inspect the fixed 4-world x 3-seed M0 matrix without starting compute.
python scripts/V23/run_m0_synthetic.py --dry-run

# Formal multi-GPU execution. Check occupancy first; physical GPUs 0 and 7
# remain forbidden. The launcher assigns one serial queue per listed GPU.
python scripts/V23/run_m0_synthetic.py \
  --device cuda --gpus 1 2 3 4 5 6 \
  --output-root result/V23_cycle_response/m0_synthetic_protocol_a_v1
```

The M0 runner launches fit, profile, and outer evaluation as separate
processes. It uses stage-level protocol equivalence to reuse completed
artifacts, invalidates downstream stages after any upstream mismatch, and
never passes labels to fit or profile. Its aggregate is descriptive and does
not make an automatic Go/No-Go decision.

Pairwise evaluation uses bounded balanced sampling and is never treated as
independent sample size. Benchmark-validity output is evaluation-only. Its
permutation-adjusted silhouette is an explicit local proxy, not an
implementation of Jeon et al.'s Adjusted IVM.

## Formal M0 Outcome

The fixed four-world, three-seed M0 matrix completed `12/12` jobs and `36/36`
stages on GPUs 2--6. It triggered the preregistered **No-Go**: `C_cycle`
improved over `A_pre` but did not consistently beat the support control, and
the dependency-destroyed conditional null retained a cycle increment. M1 and
Protocol B must not be launched under this protocol. See
`result/V23_cycle_response/m0_synthetic_protocol_a_v1/m0_decision.md`.
