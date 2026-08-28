# GitHub Release Manifest

Snapshot prepared on 2026-08-28 from the local ToPoGate workspace.

## Included

- TopoGate source, configurations, version notes, and tests under
  `methods/TopoGate/`.
- Experiment and audit scripts under `scripts/`, plus the corresponding tests.
- Public-facing protocol, decision, result, and integrity reports under
  `reports/`.
- `result/RESULTS_SUMMARY.md` and compact evidence bundles for the current
  relation-selection, corruption, support, representation-consumer, and V26
  studies.
- Compact V22 JSON/CSV/Markdown/YAML metadata only.
- V25 paper source, figures, PDF, and non-weight result tables.
- The small public `datasets/iris.npz` smoke-test fixture.

## Excluded

Datasets and dataset symlinks; checkpoints, branchpoints, model weights,
embeddings, predictions, topology caches, per-step logs, bytecode, temporary
review material, and nested upstream Git repositories. The excluded artifacts
remain on the local result/data volumes and are not implied to be available
from GitHub.

## Verification boundary

Every included file is smaller than GitHub's 100 MB single-file limit. The
compact result files are copied from audited local artifacts without changing
their values. Absolute local paths in provenance fields are intentionally
retained as source-location evidence; they are not download links.

The V11 legacy reference manifest is retained unchanged. Its existing hash
check reports a mismatch against the current `learnable_gate/run_npz.py`; this
is recorded as a validation warning and is not repaired as part of publication.
