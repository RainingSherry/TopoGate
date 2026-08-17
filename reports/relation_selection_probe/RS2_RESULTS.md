# RS2 formal results

RS2 evaluated the five deterministic selectors on the identical S0 candidate
pool, row budget, cosine weights, symmetrization, Spectral consumer, known-K
KMeans readout, and paired seeds `[42, 123, 7]`. The matrix contains 90
completed-valid rows (6 datasets × 5 selectors × 3 seeds). No selector or
graph builder received labels.

## Frozen primary gate

No selector satisfies the pre-registered simple-rule gate: at least two of the
three primary datasets must have `Delta_S >= 0.03`, with median capture at
least `0.25` over material opportunities. The number of qualifying selectors
is zero, and the RS2 decision is `fixed_simple_selectors_not_sufficient`.

| selector | cnae9 Delta_S / capture | Campbell Delta_S / capture | sms Delta_S / capture |
|---|---:|---:|---:|
| B0 cosine | -0.0533 / -0.212 | -0.0473 / -0.259 | -0.3002 / -2.122 |
| B1 mutual-first | -0.0126 / -0.032 | -0.0306 / -0.161 | -0.3002 / -2.122 |
| B2 SNN/Jaccard | -0.1272 / -0.544 | -0.0542 / -0.286 | -0.3138 / -2.185 |
| B3 stability | -0.0392 / -0.144 | -0.0396 / -0.260 | -0.3010 / -2.125 |
| B4 equal-rank fusion | **+0.0144 / +0.094** | -0.0298 / -0.157 | -0.3080 / -2.158 |

The table uses dataset means for `Delta_S` and the median paired-seed
`Capture_S`; capture is shown only where `H_pool >= 0.03`. B4 is the best
descriptive row on cnae9, but it remains below the material threshold and does
not generalize to Campbell or sms.

The three primary datasets are consumed as report-only development evidence by
this study. They are not a confirmatory evaluation set for any future learned
selector; such a selector would require a newly frozen holdout.

## Sentinel behavior

Mouse_retina has low observed `H_pool` (`+0.0274`) and no material selector
gain; its best selector mean is `+0.0193`, below the contradiction sentinel.
Baron Human remains a low-opportunity/consumer-sensitive boundary (`H_pool`
`+0.0143`). These sentinels are not counted as primary successes or failures.

Raw embeddings, predictions, graphs, and per-run logs remain local under
`result/relation_selection_probe/RS2_simple_selectors/`. The compact table and
RS3 map are the publishable evidence layer.
