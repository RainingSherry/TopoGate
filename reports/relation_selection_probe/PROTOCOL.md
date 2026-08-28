# Relation-selection probe — frozen protocol

## Scope

This project owns a new relation-selection question. It reads, but does not
rewrite, the audited artifacts under
`result/representation_consumer_probe/S0_freeze/` and
`result/representation_consumer_probe/S1_oracle_v2/`.

The old names `O_pool` and `O_full` are not used as theoretical upper bounds.
The new names are:

- `O_pool`: **Pool Label-Informed Reference**;
- `O_expanded`: **Expanded Label-Informed Reference**.

`H_pool = ARI(O_pool) - ARI(R)` is an observed within-pool reference
opportunity. `O_expanded` is a candidate-family diagnostic only.

## Frozen data roles

| Dataset | Role |
|---|---|
| `cnae9` | primary opportunity development |
| `Campbell` | primary opportunity development |
| `sms_spam_collection` | primary opportunity + candidate boundary |
| `Baron Human` | consumer-sensitive boundary |
| `Mouse_retina` | low-opportunity sentinel |
| `hate_speech` | candidate-family sentinel |

Primary gate denominators use only `{cnae9, Campbell, sms_spam_collection}`.
The other three roles cannot be reclassified after seeing selector results.

## Common graph contract

Every selector uses the S0 `H0`, the frozen positive-cosine candidate pool,
the row budget `b_i=min(8, positive_count_i)`, the original cosine edge
weight, the original symmetrization, isolate policy, Spectral consumer, and
known-K KMeans readout. The only changed object is directed edge membership.
The S1 `R` and `O_pool` artifacts are read-only reused references.

## RS1 relation features

All features are edge-local and label-free:

- geometry: cosine, row rank/percentile, margins at `b_i` and `b_i+1`, and
  distance normalized by local mean/median;
- topology: mutual-kNN, shared-neighbor count/Jaccard, common-neighbor ratio,
  target in-degree, degree asymmetry, and local hubness;
- stability: recurrence over eight deterministic 75%-dimension H0 views and
  the standard deviation of view-wise cosine.

The eight view seeds are `[17,31,47,61,73,89,101,113]`; each view uses 96 of
the 128 H0 dimensions. Candidate-edge membership is evaluated against each
view's deterministic k=20 ranking.

The diagnostic targets are `same_class` and `pool_reference_membership`.
They are never called true edge utility. Primary diagnostic evaluation is
five-fold `GroupKFold` by anchor sample, using fixed standardized logistic
regression (`C=1`, `class_weight=balanced`, `max_iter=200`, `random_state=0`).
Feature families are exactly `G`, `T`, `S`, `G+T`, `G+S`, `T+S`, and `G+T+S`.

Primary RS1 metrics are AUPRC, `Delta AP = AP - prevalence`, and `Lift@b`;
AUROC, NDCG@b, precision@b, and recall@b are secondary. The information gate
is descriptive and frozen at `Delta AP >= 0.10` and `Lift@b >= 1.5` for a
family on at least two of the three primary datasets.

These thresholds and the primary denominator were committed before formal RS1
evaluation. `pool_reference_membership` is a label-informed diagnostic target
constructed from `O_pool`; its prevalence and Lift base rate are diagnostic
uses of labels, not label-free training inputs.

## RS2 selectors

The fixed selectors are:

```text
B0 cosine
B1 mutual-first, cosine tie-break
B2 Jaccard/SNN, cosine tie-break
B3 stability recurrence, cosine tie-break
B4 equal rank fusion of cosine, Jaccard, and stability (1/3 each)
```

No fusion-weight, threshold, budget, consumer, or seed sweep is allowed.

## RS3 decision margins

For selector `S`, `Delta_S = ARI(S)-ARI(R)`. Where `H_pool >= 0.03`,
`Capture_S = Delta_S/H_pool` is reported without clipping. A simple selector is
sufficient if one frozen selector reaches `Delta_S >= 0.03` on at least two of
the three primary datasets and has median capture at least `0.25` over those
material rows. If RS1 information passes but all simple selectors fail this
criterion, RS4 may be proposed; it is not automatically started.

The phrase “learned-rule-only proposal” is a scope decision for this project,
not a claim that learned selectors are the uniquely supported remedy. Richer
candidate construction or another consumer could be reasonable future routes,
but neither is authorized here. The three primary datasets are burned as
report-only evidence; any future learned-selector evaluation requires a new,
separately frozen holdout.

Mouse_retina is a contradiction sentinel, hate_speech a candidate-family
sentinel, and Baron Human a consumer-sensitivity boundary. They cannot be
used to tune the selector.
