# C1 mechanism localization result

Status: `completed_valid`; `54` structural replays, `fit_runs=0`, `labels_loaded=false`.

The replay used the closed B1 corruption function on audited S0 `H0` for the three burned
development datasets and three paired seeds. Closed B1 ARI values are included only as post-fit
provenance columns; this C1 run did not retrain or recompute ARI. The complete compact rows are in
`result/sparse_corruption_principle_probe/C1_mechanism_audit/c1_structural_rows.csv`.

## Descriptive signatures

| dataset / old arm | closed-B1 ARI mean | support Jaccard | value rank Spearman | cosine-kNN Jaccard | local-density relative change |
|---|---:|---:|---:|---:|---:|
| Mouse / C0 random | 0.8286 | 0.8898 | 0.7520 | 0.1090 | 0.2543 |
| Mouse / C2 support-only | 0.8308 | 0.5905 | 1.0000 | 0.0609 | 1.0381 |
| Baron / C0 random | 0.1293 | 0.9519 | 0.8759 | 0.4600 | 1.9152 |
| Baron / C2 support-only | 0.3833 | 0.5401 | 1.0000 | 0.0923 | 10.8950 |
| Campbell / C0 random | 0.0838 | 0.9292 | 0.9158 | 0.2905 | 0.4647 |
| Campbell / C4 static-hard | 0.2445 | 0.9220 | 0.6356 | 0.1552 | 0.7375 |

The rows are mechanism diagnostics, not a new performance table. The visible patterns are compatible
with the three predeclared questions: Mouse has a material random-vs-clean clue in the closed B1
summary; Baron C2 is accompanied by strong support/geometry redistribution; Campbell C4 is
accompanied by a residual-targeted value/rank change and a smaller but visible geometry shift.
They do **not** establish that support or difficulty caused the ARI differences.

## Important limitation

S0 `H0` is a dense truncated-SVD representation. The support used here is the frozen threshold proxy
`abs(H0_ij) >= max(1e-6, 0.05*row_max)`, not the original count-matrix zero pattern. In addition,
the C4 structural replay uses a label-free column-median/MAD residual proxy rather than the old B1
warm-up residual artifact. Therefore this report localizes measurable structural changes only; it
does not upgrade the B1 findings into a causal support-semantics result.
