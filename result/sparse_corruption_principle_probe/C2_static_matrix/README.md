# C2 curated result metadata

This directory contains only the compact, weight-free metadata needed to
inspect the completed C2 static corruption-principle matrix:

- `C2_RESULTS.md` and `c2_*_summary.csv`: final aggregate result tables;
- `decision.json`: frozen terminal decision and locked follow-up stages;
- `audit.json` and `C2_INTEGRITY_AUDIT.*`: per-root and independent integrity
  checks;
- `run_manifest.json` and `positive_control.json`: compact protocol/run
  manifests.

The matrix is `54/54` completed-valid with `audit_ok=true`. Raw inputs,
labels, H0, score arrays, corruption arrays, per-run histories/configs,
embeddings, predictions, checkpoints and logs remain outside GitHub.

Support in C2 is the threshold-defined support of dense H0, not raw-X
zero/nonzero support.
