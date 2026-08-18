# C0 implementation and inventory result

Status: `completed_valid` for the protocol/toy/inventory contract. No C2 model matrix was launched.

- Focused tests: `16 passed` after small-sample geometry-kNN and dense-proxy P2 regression fixes.
- Toy S/V/M apparatus: `completed_valid`; all support/value role checks passed and no labels were
  supplied to corruption.
- Holdout inventory: 14 valid candidates, 12 selected by the pre-registered label-free maximin rule;
  `shortfall=0`, `development_overlap=[]`, `audit_ok=true`.
- Holdout runs: **not authorized**. The manifest is frozen before any C2 performance result.
- Source hashes and compact label-column provenance are recorded in
  `result/sparse_corruption_principle_probe/C0_holdout_inventory/holdout_manifest.json`; source
  matrices and labels remain outside the project publication tree.

The inventory is a protocol artifact, not generalization evidence.

The implementation fixes change only C2 static-library semantics: legal `n_neighbors` on small
fixtures and P2 source/destination swapping for dense H0 proxy support. They do not launch or alter
the locked C2 performance matrix.
