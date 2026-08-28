# RS3 failure map and terminal decision

RS3 joins the completed RS2 selector rows with the closed project's audited
S1 opportunity aggregates. It is analysis-only: it does not fit a learned
selector, rebuild a graph, or access labels. The statistical unit remains the
dataset; seeds are paired repeats.

## Failure map

| dataset | frozen role | H_pool | H_full | H_full-H_pool | best selector mean Delta_S | diagnosis |
|---|---|---:|---:|---:|---:|---|
| cnae9 | primary opportunity | +0.2157 | +0.1062 | -0.1095 | +0.0144 (B4) | opportunity present, fixed selectors miss material capture |
| Campbell | primary opportunity | +0.1914 | +0.0390 | -0.1525 | -0.0298 (B4) | opportunity present, selectors harmful on this consumer |
| sms_spam_collection | primary + candidate boundary | +0.3671 | +0.5423 | +0.1752 | -0.3002 (B0/B1) | opportunity present; selector failure plus material candidate-family gap |
| Baron Human | consumer-sensitive boundary | +0.0143 | -0.1695 | -0.1838 | +0.0042 (B4) | low observed opportunity; not a primary gate row |
| Mouse_retina | low-opportunity sentinel | +0.0274 | -0.0064 | -0.0338 | +0.0193 (B4) | no material contradiction sentinel |
| hate_speech | candidate-family sentinel | +0.0022 | +0.6365 | +0.6343 | +0.0163 (B0/B1/B3) | extreme expanded-reference gap; candidate construction sentinel |

`H_full-H_pool` is retained as a candidate-family diagnostic, not as a
monotone upper-bound claim. Negative values on cnae9, Campbell, Baron, and
Mouse are possible because the reference graph is consumed by a non-monotone
Spectral/readout pipeline.

Mouse_retina has low observed `H_pool` (`+0.0274`) and no material selector
gain under the primary materiality rule; its best selector mean is `+0.0193`,
about 70% of the small observed reference opportunity. This is useful
descriptive capture, but it is not a primary gate row and does not overturn the
heterogeneous result. Baron Human remains a low-opportunity/consumer-sensitive
boundary (`H_pool` `+0.0143`). These sentinels are not counted as primary
successes or failures.

## Terminal decision

1. RS1 passes only for pool-reference membership; semantic same-class
   information does not pass the complete diagnostic gate.
2. RS2 has zero fixed selectors satisfying the primary material capture rule.
3. hate_speech triggers the pre-registered candidate-family sentinel, and
   sms_spam_collection also has a material expanded-reference gap. The gap is
   therefore not isolated to hate_speech; the current selector route is not
   rescued by a learned rule.
4. Mouse_retina does not trigger a material contradiction audit, and Baron
   Human remains a consumer-sensitive low-opportunity boundary.

The authorized project therefore terminates with
`candidate_family_problem_and_learned_rule_only_proposal`. This means a
future learned-selector protocol could be drafted only as a new, separately
frozen proposal. It does **not** authorize RS4 execution, a GNN/Transformer,
TopoCut/DCGC transfer, a new reconstruction objective, a holdout run, or a
backbone swap in this project.

“Learned-rule-only” is a deliberate scope choice, not a claim that the evidence
uniquely selects learning over richer candidate construction or another
consumer. The three primary datasets are burned as report-only evidence; any
future learned-selector evaluation requires a new holdout.

The compact machine-readable outputs are
`result/relation_selection_probe/RS3_decision/rs3_summary.json`,
`rs3_failure_map.csv`, and `rs3_selector_capture.csv`. Full RS1/RS2 run trees
remain local and are not publication artifacts.
