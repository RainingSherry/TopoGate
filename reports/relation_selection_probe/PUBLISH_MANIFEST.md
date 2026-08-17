# relation-selection probe publication manifest

This publication layer contains only the independent RS0–RS3 protocol, reports,
source scripts/tests, and the weight-free compact bundle under
`result/relation_selection_probe/FINAL/`.

Included:

- `reports/relation_selection_probe/*.md`;
- `scripts/relation_selection_probe/*.py`;
- `tests/relation_selection_probe/*.py`;
- `result/relation_selection_probe/FINAL/RESULTS_SUMMARY.json`;
- `result/relation_selection_probe/FINAL/EVIDENCE_HASHES.json`;
- `result/relation_selection_probe/FINAL/PUBLICATION_MANIFEST.json`.

Excluded by design: all RS1 feature tables/OOF arrays, RS2 graph NPZ files,
embeddings, predictions, per-run logs, caches, checkpoints, model weights,
input data, and Python bytecode. The hashes in `EVIDENCE_HASHES.json` bind the
compact report to the local audited result trees without publishing those raw
trees.
