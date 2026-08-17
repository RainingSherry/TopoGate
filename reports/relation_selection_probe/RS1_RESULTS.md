# RS1 formal results

RS1 completed the six frozen datasets with the label-free relation feature
extractor and five-fold anchor-grouped probes. The feature table contains 17
fixed features in the seven pre-registered families (`G`, `T`, `S`, `G+T`,
`G+S`, `T+S`, `G+T+S`). Labels were loaded only after feature extraction to
form the two diagnostic targets.

The thresholds and the three-dataset primary denominator were committed in the
pre-registration before formal RS1 evaluation. `pool_reference_membership` is
an intentionally label-informed target read from the diagnostic `O_pool`
artifact; its prevalence and Lift base rate are not label-free quantities and
were never supplied to feature extraction or selector scoring.

## Decision

`relation_information_present` is supported only for the
`pool_reference_membership` diagnostic target. Every feature family passes
the two-threshold rule (`Delta AP >= 0.10`, `Lift@b >= 1.5`) on the three
primary datasets. No family passes both thresholds for the `same_class`
target. Therefore RS1 establishes reference-selection solvability, not
semantic same-class utility.

| target | families passing on at least two primary datasets | interpretation |
|---|---:|---|
| `pool_reference_membership` | 7/7 (all on cnae9, Campbell, sms) | selector-score information is present |
| `same_class` | 0/7 | stricter semantic relation gate not established |

Selected primary diagnostic values:

| dataset | target | best family | Delta AP | Lift@b |
|---|---|---|---:|---:|
| cnae9 | pool reference | G | 0.4910 | 2.156 |
| Campbell | pool reference | G | 0.5201 | 2.288 |
| sms_spam_collection | pool reference | G+S | 0.5498 | 2.354 |
| cnae9 | same class | G+T+S | 0.1341 | 1.075 |
| Campbell | same class | G+T+S | 0.1104 | 1.032 |
| sms_spam_collection | same class | G+T+S | 0.0575 | 1.028 |

The same-class rows illustrate why the full gate requires both metrics: AP
can increase while budget-level lift remains close to one.

This is a statement about clearing a pre-specified diagnostic gate, not a claim
that the relation features contain no class identity below that threshold.

## Boundary observations

The three non-primary datasets remain descriptive sentinels. The pool-reference
probe is also strong on them, but this does not change their frozen roles or
the primary denominator. RS1 did not inspect selector clustering outcomes when
defining its information gate.

The complete per-family rows and feature artifacts remain on the result disk
under `result/relation_selection_probe/RS1_information/`; only the compact
summary is a publication candidate.
