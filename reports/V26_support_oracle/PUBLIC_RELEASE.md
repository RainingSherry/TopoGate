# V26 Public Release Scope

This directory releases the weight-free, path-sanitized evidence for the
completed V26 Support Oracle Study v1.

The formal matrix comprised 11 datasets, 5 arms, and 3 paired seeds for a
total of 165 completed cells. The original formal implementation digest was
`8bb750e215d9b159aa0ee33319e4048880fabdc549adda9a720271cee5c9eb4b`.
The public code changes only local input paths and interpreter selection; it
is therefore not presented as byte-identical to the formal result code.

Included:

- `SUPPORT_ORACLE_STUDY_V1_FINAL.md`: formal protocol, compact per-dataset
  ARI table, decision rule, and bounded interpretation.
- `support_oracle_decision_public.json`: machine-readable compact result
  table, coverage, and decision boundary.
- `methods/TopoGate/V26_support_oracle/`, `scripts/V26/`, and
  `tests/V26_support_oracle/`: path-sanitized source and contract tests.

Excluded:

- Dataset files, labels, original data locations, model weights, optimizer
  states, predictions, embeddings, caches, per-run summaries, and worker logs.

`O_LABEL_ORACLE` is a class-conditional diagnostic ceiling. It is not a
label-free method and is never a deployable comparison arm. The final result
does not support a general support-target method: the four non-biological
datasets are all Case C under the frozen rule, and no dataset exhibits a
positive material oracle gap over `P2_SUPPORT_TARGET`.
