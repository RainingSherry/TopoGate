# Adaptive-corruption probe — protocol

## 1. Scope and frozen starting point

| field | frozen value |
|---|---|
| project | `adaptive_corruption_probe` |
| protocol | `adaptive_corruption_probe_b0_v1` |
| base commit | `c80877cf904e41950315d37b95374825c33a7362` |
| backbone | matched small reconstruction probe, fixed before B1 |
| readout | clean embedding + benchmark-known-K KMeans |
| primary seeds | `[42, 123, 7]` |
| confirmatory seeds | `[42, 123, 7, 3032, 3033]` |
| materiality | `delta_ari=0.03` |

The exact implementation must reuse the current audited input/preprocessing
adapter and record its source/config hash.  B1 may not introduce a new
preprocessing choice to make a corruption arm favorable.

### Frozen small reconstruction probe

Before B1, the common input is the audited S0 `H0` (the frozen per-dataset
`d_eff`, at most 128 dimensions),
standardized once per dataset using clean-H0 column means and standard
deviations.  Every arm uses the same
`d_eff -> 64 -> 32 -> 64 -> d_eff` ReLU autoencoder, Adam (`lr=1e-3`), 30
epochs, batch size 512, and corruption rate
`0.25`.  The benchmark embedding is the clean-H0 encoder output after fitting;
known-K KMeans is an outer readout only.  No labels enter standardization,
corruption, optimization or encoder evaluation.

Because the S0 stem is a dense SVD representation, support is defined once and
without labels as
`abs(H0_ij) >= max(1e-6, 0.05 * max_j abs(H0_ij))` within each row.  This
threshold and the requested corruption rate are frozen constants, not tuned
per dataset or arm.  The run audit reports the effective changed-coordinate,
support-change and value-change rates; a requested rate alone is not accepted
as a match.  To enforce matched budgets, each row uses
`m_i=min(ceil(0.25*active_i), floor(active_i/2), inactive_i)` pairs and every
arm changes the same `2*m_i` coordinate budget.  Rows without a feasible pair
have zero requested changes in every arm.

## 2. Development panel

B1 uses six pre-declared structural roles selected from the frozen evidence
panel, not from ARI or historical gain.  The audited local registry identifies
two text-like sparse inputs, three registered scRNA count inputs, and one
generic high-dimensional sparse control in this six-dataset panel.  It does
not contain a verified dense control; we therefore do not mislabel one as
"dense" merely to satisfy a proposed stratum.  A future dense control would
require a separately frozen input manifest before B1.

| role | dataset |
|---|---|
| sparse text 1 | `sms_spam_collection` |
| sparse text 2 | `hate_speech` |
| registered scRNA count 1 | `Mouse_retina` |
| registered scRNA count 2 | `Baron Human` |
| registered scRNA count 3 / boundary control | `Campbell` |
| generic non-expression sparse high-dimensional control | `cnae9` |

These are a development/mechanism panel.  They are not an independent
generalization claim.  B5 must freeze a separate holdout using only dataset
characteristics and before any B1 result is inspected.

## 3. Frozen corruption library

Every arm receives the same input, backbone, decoder, optimizer, epochs,
readout, K protocol, seed and requested budget.  The implementation records
effective changed coordinates, support-change ratio, value-change ratio and
total absolute change.

- **C_clean NoCorruption:** an uncorrupted floor/control;
- **C0 MatchedRandom:** the current fixed/random corruption semantics;
- **C1 ValueOnly:** modify non-zero values only, preserving support;
- **C2 SupportOnly:** reassign a row’s non-zero positions while preserving its
  non-zero value multiset as far as the adapter permits;
- **C3 MixedMatched:** change support and values while matching C0’s effective
  change budget and auditing both components;
- **C4 StaticHard:** after a frozen warm-up model, target high residual
  coordinates with a deterministic, non-learned rule.

If an exact matched-budget implementation cannot be achieved for an input
adapter, the arm is `incomplete_compute`/`protocol_mismatch`, not a negative
result.

## 4. Stage contracts

### B1 — opportunity and random-reference test

**Question.** Does corruption itself change downstream clustering relative to
the uncorrupted floor, and is the best principle dataset-dependent?

The decision is deliberately hierarchical. Level 1 asks whether corruption
matters at all, using `Delta_clean(C) = ARI(C) - ARI(C_clean)` and the explicit
random-vs-clean contrast `ARI(C0)-ARI(C_clean)`. The tested-library envelope
`H_corr=max_C ARI(C)-ARI(C_clean)` is not a universal oracle.  Reconstruction
loss, effective change counts and support/value audit fields are secondary
mechanism diagnostics.

For every structured arm C1--C4, `Delta_clean` is paired against the same
uncorrupted C_clean run, and `Delta_random(C)=ARI(C)-ARI(C0)` is paired against
the same matched-random reference. `Delta_clean` is the Level-1 clustering
endpoint; `Delta_random` is the Level-2 endpoint. `L_rec` is a degradation
monitor only: an arm cannot claim a useful corruption principle by breaking
reconstruction while its ARI changes.

Decision:

1. first require the pre-registered positive-control sensitivity check; if it
   fails, classify the stage as `protocol_insensitive` rather than a null;
2. Level 1: if the valid library has no material `Delta_clean` and the
   random-vs-clean contrast is also below `0.03`, classify
   `corruption_not_current_bottleneck`;
3. Level 2: compare each structured arm against C0 using
   `Delta_random(C)`. If one fixed structured principle is material on at
   least two development datasets, it is the consistently material simple
   principle; stop before adaptation with
   `simple_corruption_principle_sufficient`;
4. Level 3: authorize B2 only when at least two pre-declared role classes each
   have a valid winner with mean `Delta_random >= 0.03` and the winning
   structured arm differs between those classes;
5. if corruption matters relative to C_clean but no structured arm is
   consistently material on two datasets and the Level-3 contrast is absent,
   terminate with `random_corruption_sufficient`.

The B2 unlock is frozen before B1. A singleton generic-control class cannot
trigger B2 by itself. `Delta_clean` therefore answers “does corruption
matter?”, while `Delta_random` answers the separate question “does a
domain-aware principle add value beyond random corruption?”. If both the simple
and Level-3 conditions hold, the Level-3 adaptive authorization takes
precedence.

### B2 — adaptive location necessity

Only after B1 case 3.  A scorer may choose where to corrupt, but replacement
values remain frozen.  Compare random-matched, static-hard and learned-hard
under the same budgets.  A learned rule must beat static-hard materially;
otherwise `adaptive_model_not_necessary`.

Record both reconstruction hardness and clustering utility.  A rise in
`L_rec` with no ARI gain is explicitly a negative utility finding.

### B3 — generator/GAN necessity

Only after B2 passes.  Compare Random, StaticHard, LearnedMasker and a
constrained generator.  The generator may not trivially destroy support or
magnitude; support, budget and magnitude constraints are part of the frozen
contract.  A generator is necessary only if it beats the frozen learned
masker by `0.03` ARI on the pre-registered primary comparison.

### B4/B5 — conditional model and holdout

Implement the smallest model authorized by the preceding gate and freeze it
before B5.  B5 uses a new outcome-independent holdout and five paired seeds;
architecture, loss, budget, learning rate, epoch count and dataset membership
cannot change after holdout results.

## 5. Explicit no-go rules

- B1 has no material opportunity: stop all adaptive/GAN work.
- B1 shows corruption impact but no structured arm exceeds C0 materially:
  `random_corruption_sufficient` and stop all adaptive/GAN work.
- A fixed support/value principle is sufficient: do not add a learner.
- Learned location does not beat static-hard: stop before a generator.
- A generator does not beat learned location: `generator_not_necessary`.
- Any support/value/budget mismatch, label leakage or result-dependent panel
  change invalidates the stage and is not recoded as performance.

`incomplete_compute` remains a resource/protocol status, never a negative
performance outcome.

For each dataset and arm, the three-seed spread (standard deviation and
range) is reported as a descriptive noise floor around the `0.03` margin; it
is not treated as an inferential confidence interval.

Development overlap with Track A is permitted as pre-recorded mechanism
context only and is audited independently.  Final A5/B5 holdout dataset-ID
overlap is forbidden.
