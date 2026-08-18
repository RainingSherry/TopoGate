# Learned-relation-rule probe

`learned_relation_rule_probe` is an independent, non-V-series mechanism
study.  It starts from the audited relation-selection release at frozen
commit `c80877cf904e41950315d37b95374825c33a7362`; it does not modify or
reopen `relation_selection_probe`, `representation_consumer_probe`, or any
V-series implementation.

## Finite question

> Within the already frozen candidate pool, is there a transferable,
> label-free relation decision rule that captures material clustering
> opportunity?

The project separates three claims that must not be conflated:

1. relation features can predict a diagnostic target;
2. a diagnostic supervised ceiling can rank useful edges;
3. a deployable, label-free rule can reproduce that action.

RS1 only established the first kind of evidence for the old project.  It is
not evidence for this project's clustering utility.

## Stage order and status

```text
A0/S0 Freeze
  -> A1 Supervised actionable ceiling
  -> A2 Cross-dataset transfer ceiling
  -> A3 Label-free solvability
  -> A4 Minimal learned rule (conditional)
  -> A5 Independent holdout (conditional)
```

At creation, only A1 is authorized.  A2 is unlocked only by the frozen A1
gate; A3 requires A1 and reasonable transfer; A4 requires A1/A2 plus failure
of the simple label-free proxies; A5 uses the holdout frozen at A0 only after
the label-free A4 candidate passes its own development capture gate.

## Data roles

The three old RS primary datasets (`cnae9`, `Campbell`,
`sms_spam_collection`) are development/mechanism evidence only and are
burned for future learned-selector confirmation.  `Mouse_retina`, `Baron
Human`, and `hate_speech` retain sentinel roles and cannot be promoted into a
success denominator after seeing results.  Confirmatory generalization uses
the separate twelve-dataset manifest frozen before this project’s outcomes;
the manifest is inherited only because it remains dormant and outcome-free.

## Non-negotiable boundaries

- The candidate pool, H0, row budgets, edge weights, consumer, K protocol and
  seeds are inherited read-only from the frozen evidence.
- Feature extraction and scoring never receive `y`, `O_pool`, ARI, NMI or ACC.
- A1 diagnostic supervision targets pool-reference membership only and is
  explicitly non-deployable.
- Grouped out-of-fold evaluation is by anchor sample; random edge splits are
  forbidden.
- No GNN, Transformer, new backbone, candidate construction change, or
  outcome-driven tuning is authorized in this project.
- GPU use is explicit (`[1, 2, 3, 4, 5, 6]`); physical GPUs `0` and `7` are
  forbidden.  Cheap A1 probes may remain CPU-first without changing the
  protocol.
- Raw data, weights, embeddings, predictions, graphs, caches and logs stay
  outside the publication bundle.

The current repository contains only the freeze contract.  No A1 performance
run is represented by this README.
