# Relation-selection probe — pre-registration

**Project:** `relation_selection_probe`  
**Authorized stages:** RS0, RS1, RS2, RS3  
**Locked:** RS4 learned selector, any GNN/Transformer/TopoCut/DCGC transfer,
new reconstruction objective, and hyperparameter search.

## Frozen estimands

The RS1 thresholds (`Delta AP >= 0.10`, `Lift@b >= 1.5`, and at least two of
the three primary datasets) were frozen in this pre-registration before any
formal RS1 result was inspected. They are descriptive decision margins, not
significance thresholds.

RS1 estimates information about two diagnostic targets, not utility:

```text
t_class(i,j) = 1[y_i = y_j]
t_ref(i,j)   = 1[(i,j) belongs to O_pool]
```

RS2 estimates actual downstream capture:

```text
Delta_S(d) = ARI(S_d) - ARI(R_d)
Capture_S(d) = Delta_S(d) / H_pool(d), when H_pool(d) >= 0.03
```

The statistical unit is dataset. The three seeds are paired repeats, not
independent datasets.

## Frozen Go/No-Go rules

1. **Information stop:** if no feature family satisfies the RS1 materiality
   rule on two primary datasets, record
   `current_relation_evidence_not_sufficient` and stop selector design.
2. **Simple-rule stop:** if one fixed selector satisfies the RS3 primary gate,
   record `simple_relation_rule_sufficient` and do not train a learned Gate.
3. **Conditional RS4 go:** only if RS1 passes while every fixed selector fails
   the primary gate, record `learned_decision_rule_justified`; this authorizes
   a new protocol proposal, not execution in this run.
4. **Candidate-family split:** if failure is isolated to hate_speech where
   `O_expanded` materially exceeds `O_pool`, record
   `candidate_family_problem` and do not rescue the current selector.
5. **Sentinel contradiction:** any unexpected large Mouse_retina gain triggers
   a contract audit before interpretation.

## Leakage boundary

Feature extraction, stability views, selector scoring, graph construction, and
Spectral fit do not receive labels or K. Labels are loaded only by RS1's
diagnostic probe and by the outer known-K evaluation. `O_pool` is a diagnostic
reference, never a deployable method.

`pool_reference_membership` is intentionally a label-informed target derived
from the diagnostic `O_pool` artifact; it is not a label-free target and must
not be described as one. Its prevalence and `Lift@b` base rate are likewise
label-derived diagnostic quantities. Neither may tune a selector, feature
family, budget, consumer, or future protocol.

The three primary datasets are report-only for this project. Any future
learned-selector proposal must freeze and use a separate holdout before
evaluating a learned rule; RS1–RS3 primary rows cannot be reused as its
confirmatory evaluation set.
