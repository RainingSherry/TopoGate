# Learned-relation-rule probe — protocol

## 1. Scope and frozen starting point

| field | frozen value |
|---|---|
| project | `learned_relation_rule_probe` |
| protocol | `learned_relation_rule_probe_a0_v1` |
| base commit | `c80877cf904e41950315d37b95374825c33a7362` |
| old project | `relation_selection_probe` (terminal, read-only) |
| candidate pool | inherited S0/S1 frozen pool, unchanged |
| consumer | normalized Spectral + benchmark-known-K KMeans |
| primary seeds | `[42, 123, 7]` |
| confirmatory seeds | `[42, 123, 7, 3032, 3033]` |
| budget | inherited row-specific `b_i=min(8, positive_count_i)` |
| materiality | `delta_ari=0.03` |

Only relation membership is allowed to change.  H0 construction, candidate
membership, cosine weights, symmetrization, isolates, Spectral settings,
readout and budget are not tunable in this protocol.

`R` is the read-only matched-random selector/reference from the terminal
`relation_selection_probe` artifacts.  Every A1--A4 Delta is
`ARI(S)-ARI(R)` on the same development rows; it is not relative to the A1
supervised ceiling, an A3 proxy, or an empty graph.

## 2. Roles and denominator

| dataset | role |
|---|---|
| `cnae9` | development opportunity panel |
| `Campbell` | development opportunity panel |
| `sms_spam_collection` | development panel and candidate-family boundary |
| `Mouse_retina` | low-opportunity falsification sentinel |
| `Baron Human` | consumer-sensitivity boundary |
| `hate_speech` | extreme candidate-family sentinel |

The three development datasets may be used to decide whether the mechanism
is worth pursuing, but cannot serve as future confirmatory generalization.
The A5 holdout is the twelve-dataset, label-free-characteristics manifest
frozen at A0.  No holdout membership may be changed after A1/A2 outcomes.

## 3. Stage contracts

### A1 — supervised actionable ceiling

**Question.** Can a deliberately diagnostic supervised scorer turn frozen
relation features into a useful edge ranking?

**Allowed scorers.** Exactly logistic regression and a one-hidden-layer Tiny
MLP (`p -> 32 -> 1`, ReLU).  The target is
`1[(i,j) is in O_pool]`; this is diagnostic supervision, not a method.

**Split.** Five-fold `GroupKFold` by anchor `i`.  Every edge from an anchor
stays in one fold and predictions used for graph construction are fully
out-of-fold.  No random edge split is legal.

**Shortcut audit.** Each scorer is evaluated with full features, no-geometry
(remove cosine, distance, rank and percentile), and no-rank (retain numeric
geometry but remove ordinal ranks).  A strong full/no-rank gap is reported as
reference reconstruction, not as learned reliability.

**Primary endpoint.** For each development dataset:

```text
Delta_sup = ARI(S_sup) - ARI(R)
Capture_sup = Delta_sup / (ARI(O_pool) - ARI(R))
```

Capture is reported only when the denominator is at least `0.03`; no clipping
is applied.  AUPRC, AP lift, AUROC, precision@b and NDCG@b are secondary
diagnostics.

**Gate.** A1 passes only when at least two of the three development datasets
have `Delta_sup >= 0.03` and their median `Capture_sup >= 0.25`.  Otherwise
the terminal decision is
`predictable_reference_not_actionable_for_selection` and A2--A5 are locked.

### A2 — cross-dataset transfer ceiling

Only after A1 passes.  The three leave-one-dataset-out folds are fixed:

```text
Campbell + sms -> cnae9
cnae9 + sms -> Campbell
cnae9 + Campbell -> sms
```

Feature normalization is fit on training datasets only.  If within-dataset
success does not transfer on the fixed folds, the decision is
`relation_rule_is_dataset_conditional`; no universal Gate is built.

### A3 — label-free solvability

Only after A1 and a non-zero transfer signal.  Two pre-registered simple
proxies are tested without labels: (i) cross-view relation agreement over
eight deterministic 75%-dimension views and (ii) context-matched,
null-corrected relation surprise.  Each proxy uses the frozen row budget and
Spectral readout.  If either proxy passes the same material capture criterion,
the simple principle is sufficient and A4 is not authorized.

### A4 — minimal learned relation rule

Only after A1/A2 provide evidence and A3 simple proxies are insufficient.  A
single `p -> 32 -> 1` scorer is trained only against the A3 pseudo-ranking;
labels, oracle targets and clustering metrics are forbidden.  Abstention is
permitted through a threshold fixed from the null score distribution or a
pre-registered quantile, never from ARI.

Before A5, the label-free candidate must itself reach `Delta_A4 >= 0.03` on
at least two of the three development datasets and median capture at least
`0.25` over material rows.  ARI is post-fit evaluation only and never feeds
the scorer.  If this gate fails, the terminal decision is
`learned_rule_not_actionable` and A5 is locked; the diagnostic A1 ceiling is
not treated as label-free evidence.  Here `Delta_A4` is explicitly
`ARI(S_A4)-ARI(R)` on the same development datasets, not a comparison with
the supervised A1 ceiling or an A3 proxy.  A passing A4 gate also requires no
material opposing-sign development row (`Delta_A4 <= -0.03`); per-dataset
seed spread is reported as a descriptive noise floor around the frozen
`0.03` margin.

### A5 — independent holdout

Only after A4 is frozen.  Compare `R`, the best simple proxy and the frozen
learned rule on the twelve-dataset holdout with five paired seeds.  Report
dataset-level effects, median/mean, win/tie/loss and bootstrap intervals; do
not treat seeds or edges as independent datasets.

## 4. Artifact and audit contract

Every authorized stage must write a compact `resolved_config.json`,
`source_manifest.json`, `run_manifest.json`, `decision.json`, `RESULTS.md`
and `audit.json`.  A decision must contain `next_stage_authorized`; runners
must refuse to enter a later stage when it is false.  Raw arrays, graphs,
weights, embeddings, predictions and logs are diagnostic-only local files and
are excluded from GitHub.

## 5. Explicit no-go rules

Stop the project if any of the following frozen conditions occurs:

1. A1 actionable ceiling fails.
2. A1 passes but transfer is consistently non-positive.
3. A3 simple proxy meets the material gate (no learned model needed).
4. A4 does not pass its label-free development capture gate, or does not beat
   the frozen best simple proxy on independent holdout.
5. Any label leakage, random edge split, candidate-pool change, budget change,
   or outcome-dependent holdout change is detected.

The `0.03` margin is inherited from the frozen RS/V25 material-effect
contract as a descriptive practical margin, not a significance claim or a
post-outcome threshold.  Failure is a scientific decision, not
`incomplete_compute`; resource failures remain explicitly
`incomplete_compute` and cannot enter negative summaries.

Development overlap with Track B is permitted only as pre-recorded shared
mechanism context; each track has its own fit, outcomes and code path and
audits that overlap.  Final A5/B5 holdout dataset-ID overlap is forbidden.
