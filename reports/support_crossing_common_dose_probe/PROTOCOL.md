# Protocol: D0/D1 common-dose feasibility

## Scope

The closed C2 `P2_SupportTarget` action is an active-to-inactive value swap.
The closed M1 control showed that reproducing its perturbation magnitude with
an active-to-active swap is not estimable on Baron Human. This project reverses
the construction: first find a dose that both operation classes can attain,
then compare them only if the dose contract passes.

## Frozen inputs

| item | value |
|---|---|
| datasets | Mouse_retina, Baron Human, Campbell |
| seeds | 42, 123, 7 |
| input | audited dense SVD/H0 from the representation-consumer S0 freeze |
| support | fixed clean-row threshold, ratio 0.05 |
| pair budget | inherited C2 `row_budgets`, corruption rate 0.25 |
| Cross | swap one active value with one inactive value per pair |
| Preserve | swap unequal values between two active coordinates per pair |
| dose | `sum(abs(corrupted-clean))` within one row |
| labels | never loaded; no clustering consumer in D0/D1 |
| GPU | zero runs; physical GPUs 0 and 7 forbidden |

Both arms use exactly `2*m_i` changed coordinates for a row with `m_i` pairs.
Cross must change support on the swapped endpoints; Preserve must leave the
fixed clean-reference support unchanged. Swapping preserves the row value
multiset by construction, and the audit checks it independently.

## Constructive feasibility map

For each positive-budget row, D1 builds four deterministic matching witnesses:

- Cross minimum and maximum over a greedy disjoint active/inactive edge order;
- Preserve minimum and maximum over a greedy disjoint unequal active/active edge order.

The reported interval is the intersection of those witness ranges. It is named
`constructive_min_max_witnesses`, not an exhaustive feasible set. The target is
the midpoint of the intersection; both arms then select disjoint pairs nearest
to the target per-pair dose and are audited after construction.

Seeds only change deterministic tie ordering. They are replayed rows, not
independent statistical observations.

## D1 gate (frozen before execution)

Every dataset×seed must satisfy all of the following:

- at least 95% of positive-budget rows have a nonzero constructive common interval;
- all those common rows can be constructed;
- dataset-total Cross/Preserve dose relative mismatch is at most 5%;
- median row mismatch is at most 10%;
- changed-coordinate counts are exact;
- Cross support change is positive and Preserve support change is zero;
- both row value multisets are unchanged.

If any dataset×seed fails, the state is `common_dose_not_estimable` under this
frozen constructive contract and D2 is stopped. This is a protocol feasibility
decision, not a negative clustering result or a theorem that no other matching
algorithm could work.

## D2 lock

D2 would use the new estimand `ARI(Cross)-ARI(Preserve)` at the matched dose,
not the original C2 P2-vs-random effect. D2, raw-X validation, holdout,
adaptive policy and GAN are locked until D1 passes.
