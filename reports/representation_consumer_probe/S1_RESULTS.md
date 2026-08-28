# S1 opportunity-only results

有效协议：`representation_consumer_probe_s1_opportunity_spectral_v2`。本报告只分析
`result/representation_consumer_probe/S1_oracle_v2/` 的 90/90 completed-valid jobs；统计单位是
dataset，三个 seed 是 paired repeats。`H_pool`、`H_full` 和 `C` 都是 label-derived diagnostic
quantities，不是可部署方法性能，也不估计 `S_graph`。

## Raw dataset table

| Dataset | ARI(R) | H_pool | H_full | C = H_full − H_pool | within-pool | candidate gap | zero-budget fraction |
|---|---:|---:|---:|---:|---|---|---:|
| Baron Human | 0.672484 | +0.014306 | −0.169519 | −0.183825 | absent | absent | 0.000000 |
| Campbell | 0.238075 | +0.191444 | +0.038981 | −0.152463 | present | absent | 0.000000 |
| Mouse_retina | 0.942599 | +0.027426 | −0.006358 | −0.033784 | absent | absent | 0.000000 |
| cnae9 | 0.474525 | +0.215720 | +0.106188 | −0.109533 | present | absent | 0.000926 |
| hate_speech | −0.007068 | +0.002176 | +0.636495 | +0.634319 | absent | present | 0.041912 |
| sms_spam_collection | 0.229800 | +0.367108 | +0.542305 | +0.175197 | present | present | 0.047904 |

Materiality uses the frozen descriptive margin `delta=0.03` plus at least 2/3 positive seeds. It is
not a significance test.

## Key findings

1. **Within-pool opportunity is heterogeneous but present.** `H_pool` is material-positive on
   cnae9 (`+0.215720`), Campbell (`+0.191444`) and sms_spam_collection (`+0.367108`), so the frozen
   candidate family is not uniformly opportunity-free.
2. **Candidate restriction is data-dependent.** The matched-budget candidate gap is material-positive
   on sms_spam_collection (`+0.175197`) and hate_speech (`+0.634319`), while it is negative on Baron
   Human, Campbell, cnae9 and Mouse_retina. `C` is therefore an identity-gap diagnostic, not a total
   support-capacity or recall-loss estimate.
3. **Spectral-negative datasets remain conditional.** Baron Human and Mouse_retina do not reach the
   material `H_pool`/`H_full` margin. Their correct status is `S2 conditional`, not a topology No-Go.
4. **The strongest positive result is not a TopoGate result.** The oracle arms demonstrate diagnostic
   opportunity under the frozen relation family, but no T selector is estimable and no selector claim
   follows. A future selector study must be a new `relation_selection_probe` project.
5. **Support deficiency is visible and preserved.** zero-budget rows are 1 for cnae9, 40 for sms and
   135 for hate_speech; no negative-cosine edge was added to repair them.

## Decision and next experiment

S1 does not authorize S3/S4/S5/S6, TopoCut or a new selector. Run S2 SimpleCut only for the two
Spectral-negative/near-threshold datasets (Baron Human and Mouse_retina), using the same R/O_pool/O_full
graphs and frozen seeds, if the goal is to distinguish a Spectral relaxation miss from absent opportunity.
For the three within-pool-positive datasets, S1 already establishes opportunity existence; S2 is optional
robustness confirmation. Do not use the S1 outcome to tune budgets, consumers, datasets or a future selector.
