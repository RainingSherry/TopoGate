# Learned-relation-rule probe — pre-registration

This file freezes the decision logic before any A1 performance run.  It is a
mechanism probe, not a claim that a learned selector will improve clustering.

## Hypotheses

- **H-A1:** a diagnostic supervised scorer can capture material opportunity in
  the inherited candidate pool.
- **H-A2:** the actionable relation rule transfers across the fixed
  leave-one-dataset-out folds.
- **H-A3:** a label-free rule can recover actionable ranking information after
  the diagnostic ceiling is established.
- **H-A4:** only if simple label-free proxies fail, a minimal learned rule adds
  material capture without labels.

## Frozen inputs

- Base commit: `c80877cf904e41950315d37b95374825c33a7362`.
- Development datasets and sentinel roles are those in `PROTOCOL.md`.
- The burned primary development set is exactly
  `[cnae9, Campbell, sms_spam_collection]`; `Mouse_retina`, `Baron Human` and
  `hate_speech` are fixed sentinels and are not exposed to A1--A4 fitting or
  selector tuning.
- Candidate graph, H0, edge weights, row budgets, Spectral configuration and
  K protocol are inherited read-only.
- A5 holdout membership is the dormant twelve-dataset manifest selected using
  label-free characteristics only.  It was frozen before this project's
  outcomes and is not used for A1--A4 tuning.
- A5 must be disjoint from any future Track B holdout at dataset-ID level;
  overlap is a protocol mismatch, not a result.
- Primary seeds `[42,123,7]`; A5 seeds `[42,123,7,3032,3033]`.

## Estimands

For a selector `S`,

```text
Delta(S) = ARI(S) - ARI(R)
H_pool   = ARI(O_pool) - ARI(R)
Capture  = Delta(S) / H_pool, when H_pool >= 0.03
```

Here `R` means the inherited matched-random selector/reference artifact, and
`S_A4` is compared to that same `R` on each development dataset.  This
reference is frozen before A1 and is never selected by ARI.

ARI is the primary benchmark endpoint with known-K clean Spectral readout.
AP/AUROC and rank diagnostics are secondary and cannot authorize a later
stage by themselves.

## Frozen gates

```yaml
material_delta_ari: 0.03
minimum_primary_datasets: 2
minimum_median_capture: 0.25
a1_gate: Delta_sup >= 0.03 on >=2/3 and median Capture_sup >= 0.25
a2_gate: fixed transfer folds show non-zero material evidence; no universal
  claim is allowed if all three held-out deltas are <= 0
a3_gate: simple proxy passing the material criterion stops learned-model work
a4_development_gate: Delta_A4 >= 0.03 on >=2/3 and median Capture_A4 >= 0.25
a4_direction_consistency: no material negative Delta_A4 <= -0.03 among the three development rows
a4_holdout_gate: learned rule must beat best simple proxy on A5; otherwise no-go
```

The exact A2 “reasonable transfer” descriptive threshold is not an additional
effect-size gate: all three fold outputs are reported, and the all-nonpositive
rule is the pre-registered stop condition.  The `0.03` margin is inherited
from the frozen RS/V25 material-effect contract and is descriptive, not a
p-value claim.  `Delta_A4` is `ARI(S_A4)-ARI(R)` and per-dataset seed spread
is the descriptive noise-floor report.  No threshold may be changed after
looking at results.

## Label and leakage firewall

The feature extractor/scorer interface accepts only `X/H0`, candidate pool,
relation features, seed and frozen configuration.  `y`, class labels,
`O_pool`, ARI, NMI and ACC are available only to the diagnostic target builder
or post-fit evaluator.  A1 is explicitly marked
`diagnostic_supervision=true`, `deployable_method=false`.

## Resource and publication contract

New runners must declare `CUDA_VISIBLE_DEVICES` from legal pool
`[1,2,3,4,5,6]` and reject physical devices `0` and `7`; CPU execution is
allowed for A1 because it does not change the estimand.  Formal runs save
resolved configuration and audit metadata locally.  GitHub receives only
protocols, decisions, compact summaries and tests—never raw data, model
weights, embeddings, predictions, graphs or caches.

## Pre-registered outcome labels

The only allowed terminal labels are:

```text
completed_valid
predictable_reference_not_actionable_for_selection
relation_rule_is_dataset_conditional
simple_label_free_proxy_sufficient  # reserved for A3 proxy result
learned_rule_no_holdout_gain
incomplete_compute
```

`incomplete_compute` is not a performance negative.  Any run with that state
is excluded from the corresponding gate and recorded separately.
