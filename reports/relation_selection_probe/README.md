# Relation-selection probe

`relation_selection_probe` is an independent, non-V-series mechanism study.
It starts after the formal closure of `representation_consumer_probe` and
does not continue scMAE, TopoCut, or any existing Gate implementation.

The finite question is:

> Can small, label-free relation evidence identify sample-sample relations
> worth retaining and capture part of the already observed reference
> opportunity?

Only RS0–RS3 are authorized initially:

```text
RS0 Freeze
  → RS1 information / solvability audit
  → RS2 fixed simple selectors
  → RS3 clustering capture and failure map
```

RS4 learned selectors, GNNs, Transformers, new reconstruction objectives,
TopoCut, DCGC transplantation, and large hyperparameter searches are locked.

The six stress datasets retain fixed roles. `cnae9`, `Campbell`, and
`sms_spam_collection` are the primary opportunity-development set; Baron Human,
Mouse_retina, and hate_speech are boundary/falsification sentinels, not extra
success opportunities selected after results.

All relation features are computed from the frozen S0 `H0` and candidate pool.
Labels may appear only in RS1 diagnostic targets and post-fit benchmark
metrics. They never enter feature construction or selector fitting.

## Current status

RS0–RS3 are complete. The result is heterogeneous and terminal under this
project scope: pool-reference information is detectable, but no fixed simple
selector captures a material share of the primary opportunity, and hate_speech
exposes an extreme candidate-family boundary (sms_spam_collection also has a
material expanded-reference gap). RS4 is only a possible future proposal;
it was not run. The three primary datasets are report-only evidence and are
burned for any future learned-selector confirmation; a new protocol must freeze
a separate holdout before evaluating such a rule.
