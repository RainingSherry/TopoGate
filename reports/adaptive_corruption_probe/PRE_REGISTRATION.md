# Adaptive-corruption probe — pre-registration

This document freezes the corruption taxonomy and promotion gates before B1
performance runs.  It does not assert that adaptive corruption or GAN is
needed.

## Hypotheses

- **H-B1:** corruption principle affects clustering in the fixed development
  panel.
- **H-B2:** if effects are material and role-dependent, learning where to
  corrupt beats frozen static rules.
- **H-B3:** only if adaptive location is useful, generating replacement values
  adds material utility beyond a learned masker.

## Frozen arm contract

```yaml
base_commit: c80877cf904e41950315d37b95374825c33a7362
development_panel: [sms_spam_collection, hate_speech, Mouse_retina, Baron Human, cnae9, Campbell]
arms: [C_clean_no_corruption, C0_MatchedRandom, C1_ValueOnly, C2_SupportOnly, C3_MixedMatched, C4_StaticHard]
material_delta_ari: 0.03
primary_seeds: [42, 123, 7]
holdout_seeds: [42, 123, 7, 3032, 3033]
legal_gpu_pool: [1, 2, 3, 4, 5, 6]
forbidden_gpu_ids: [0, 7]
labels_used_during_fit: false
positive_control: synthetic_support_value_sensitivity_before_B1_null_decision
cross_track_holdout_disjointness_required: true
```

The current audited six-dataset panel is intentionally recorded as two
text-like sparse inputs, three registered scRNA count inputs and one generic
non-expression sparse high-dimensional control.  It has no verified dense
control; no dataset is relabeled as dense after the fact.  Adding a dense
control requires a new, outcome-independent manifest and protocol revision
before B1.

Support/value matching is audited per run.  Requested mask ratio alone is not
accepted as proof of matching; effective changed coordinates and magnitude
must be recorded.
The common pair-feasible budget is frozen as
`m_i=min(ceil(0.25*active_i), floor(active_i/2), inactive_i)` and every arm
changes `2*m_i` coordinates.  If an arm cannot meet this exact budget it is
`protocol_mismatch`/`incomplete_compute`, never a negative result.

## Estimands and gates

```text
Delta_clean(C)  = ARI(C) - ARI(C_clean_no_corruption)
Delta_random(C) = ARI(C) - ARI(C0_MatchedRandom)
H_corr          = max_C Delta_clean(C)   (tested-library diagnostic only)
```

The three-level decision hierarchy is frozen before any B1 result is
inspected:

1. **Corruption matters (Level 1).** Report
   `ARI(C0_MatchedRandom)-ARI(C_clean_no_corruption)` and the full-library
   `Delta_clean` values.
2. **A structured principle beats random (Level 2).** For each structured arm
   `C1--C4`, report `Delta_random(C)=ARI(C)-ARI(C0_MatchedRandom)`.
3. **Adaptation is necessary (Level 3).** Only if at least two role classes
   each have a valid winner with mean `Delta_random >= 0.03`, and those winners
   are different structured principles, may B2 start.

If Level 1 is positive but Level 2/3 does not establish a structured advantage,
the terminal label is `random_corruption_sufficient`: corruption has an
observable effect, but the evidence does not justify a domain-aware/adaptive
corruption model.

```yaml
B1_sensitivity: a pre-registered unlabeled synthetic support/value fixture must detect its known perturbation; otherwise protocol_insensitive
B1_no_go: after sensitivity passes, abs(Delta_clean(C)) < 0.03 for every valid arm/dataset role and random-vs-clean is < 0.03
B1_random_sufficient: corruption is material relative to C_clean but no structured C1--C4 has mean Delta_random >= 0.03 on the required role comparison
B1_simple: one fixed structured principle has mean Delta_random >= 0.03 on >=2 development datasets; stop before B2
B1_adaptive: at least two predeclared role classes each have a material valid Delta_random winner and their winning structured arms differ
B2_gate: LearnedHard beats StaticHard by >= 0.03 on pre-registered primary comparison
B3_gate: Generator beats LearnedMasker by >= 0.03; otherwise generator_not_necessary
```

The phrases “consistently” and “differs across roles” are descriptive gate
conditions resolved at the dataset-role level with the full three-seed panel;
no threshold or role may be added after B1 output is visible.  A failed or
incomplete run cannot be used to satisfy a gate.

The positive-control fixture is an apparatus sensitivity check, not a
clustering result: it uses no labels and verifies that the frozen corruption
library can expose a known support/value perturbation before a real-data null
decision is interpreted.  B5 must also prove dataset-ID disjointness from
Track A's final holdout and from this B1 development panel; overlap is
`protocol_mismatch`.

For every C1--C4 arm, `Delta_clean` is paired against the same C_clean
no-corruption run and `Delta_random` against the same C0 run.  ARI is the clustering endpoint and
`L_rec` is a degradation-monitor diagnostic, not a promotion criterion.  The
per-dataset three-seed standard deviation and range are reported as the
descriptive noise floor around `0.03`, not as inferential intervals.

Track-A/Track-B development overlap is allowed only as pre-recorded context
and must be audited separately; final A5/B5 holdout overlap is forbidden.

## Label and hardness firewall

The corruption and model code accepts only input matrix, frozen corruption
configuration, seed and (where applicable) reconstruction residuals.  `y`,
ARI, NMI and ACC are post-fit evaluation inputs only.  K is an outer
benchmark-known readout parameter and never a training target.  `L_rec` is
reported alongside ARI and cannot authorize promotion by itself.

## Publication and compute

Formal runs may use the legal GPU pool and should use GPU when the matched
training probe benefits from it, but resource abundance does not authorize
skipping the cheap B1 library.  Only compact protocols, manifests, decisions,
audits and important aggregate results are publishable; raw inputs,
checkpoints, embeddings, predictions, graphs and logs are excluded.

## Allowed terminal labels

```text
completed_valid
protocol_insensitive
corruption_not_current_bottleneck
simple_corruption_principle_sufficient
adaptive_corruption_opportunity_present
random_corruption_sufficient
adaptive_model_not_necessary
generator_not_necessary
incomplete_compute
protocol_mismatch
```
