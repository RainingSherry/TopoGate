# ACCG Synthetic Contract

**Status:** frozen implementation contract; no formal synthetic or real ACCG
training has started.

## Purpose

The synthetic stage tests whether action-level joint structural energy adds
observable information beyond sample hardness, donor magnitude, and singleton
residual changes. It is not a preliminary leaderboard and cannot be rescued by
a larger model if the frozen contract fails.

## Generator contract

Two generator families are frozen:

```text
lognormal_sparse
count_sparse
```

Within each family and seed, W0-W5 share sample count, feature count, labels,
cluster sizes, exact sparse support, per-feature nonzero multisets, zero rate,
and feature marginals. World differences are created by rank coupling before
the same values are reassigned. Support is shared inside declared feature
modules in every world, so W5's sparse pair relation is observable and is not
created only in the positive world.

Oracle masks mark observed nonzero coordinates only. They are stored outside
the matrix-only model input and are unavailable to fitting or selection.

| World | Frozen role | Required interpretation |
|---|---|---|
| W0 matched null | false-positive reference | no intervention oracle |
| W1 isolated corruption | repair target | selected observed corruption can be repaired |
| W2 rare coherent signal | protection target | rare coherent coordinates should not be replaced |
| W3 coherent nuisance | boundary | coherence can protect task-irrelevant nuisance |
| W4 observational alias | impossibility | identical X supports incompatible task partitions |
| W5 joint interaction | action-level counterexample | a same-donor pair can be safe jointly while unsafe as singletons |

W5 freezes an `interaction_pair` of two coordinates inside a coherent module.
Positive probe actions replace both pair coordinates with the same donor;
negative actions replace only one while preserving the same total budget.

## Shortcut audit

For every family and seed:

1. support arrays must be exactly identical across W0-W5;
2. sorted values for every feature must be identical across worlds;
3. maximum per-column mean/std/zero-rate gap must be numerical noise only;
4. support-summary and marginal-summary world classifiers must have macro AUC
   no greater than `0.60`.

A failure is a generator failure. No selector or clustering result from that
panel is admissible evidence.

## Action probe

Each probe record freezes row, donor row, exact budget, selected mask, sample
hardness, donor magnitude, mean singleton delta, full joint delta, epsilon, and
an outer-only oracle target. Repeated actions from the same row remain grouped
in cross-validation and bootstrap resampling.

The baseline probe uses:

```text
sample hardness + donor magnitude + mean singleton delta
```

The full probe adds:

```text
-joint structural delta = kappa_i(M)
```

Baseline and full models use identical grouped folds. Fold-local scaling and
logistic regression are fit only on training folds.

For W1, W2, and W5, a required record passes when:

```text
joint AUC >= 0.65
group-bootstrap lower 95% bound of delta AUC > 0
delta PR > 0
```

The pooled leave-one-generator-family-out probe must also have joint AUC at
least `0.65`, positive delta AUC, and positive delta PR. W3 is reported as the
coherent-nuisance boundary and is not required to become a positive
identifiability result.

## Joint selector audit

Every saved hard action must satisfy exact agreement between the selector's
incremental bookkeeping and a full recomputation of `R_i^M(x_i^M) - R_i^M(x_i)`.
The deterministic W5 pair must be recovered as jointly admissible even when
both singleton prefixes are inadmissible. Small random W5 instances are also
compared with the brute-force exact constrained action and report feasibility
and hardness gaps.

The primary method keeps the exact requested budget. If no admissible exact
budget is found, it records least-violation fallback and constraint failure;
it does not silently shrink the dose. Abstention is a separate sensitivity.

## Promotion rule

End-to-end ACCG training is not scientifically authorized until all of the
following are true:

```text
shortcut audit valid
action-probe identifiability decision passes
W5 joint selector contract passes
source/config/resource preflight passes
```

If the joint feature adds no held-out-family information, if W5 breaks the
joint action definition, or if shortcut classifiers identify the worlds, the
ACCG route stops. No MLP, attention layer, new backbone, donor change, or
outcome-selected epsilon may be introduced as a rescue inside this protocol.

