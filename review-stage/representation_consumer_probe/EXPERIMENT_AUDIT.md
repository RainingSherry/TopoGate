# Experiment Audit Report — S2 SimpleCut

**Date**: 2026-08-17 17:24 +08:00
**Auditor**: fresh GPT-5.6-Sol ultra reviewer, same-family provisional
**Project**: `representation_consumer_probe`
**Scope**: `result/representation_consumer_probe/S2_simple_cut/`

## Overall verdict: WARN (no FAIL)

The fresh audit verified all 18 S2 runs, recomputed all stored metrics, checked labels and K
provenance, validated exact S1 graph reuse, verified root and per-run hashes, and found no phantom,
normalization, or label-in-fit violation. The WARN is limited to a training-log semantic mismatch:
the last CSV loss is captured before the optimizer step, whereas `fit_metadata.final_loss` is
recomputed after it. This is a metadata timing issue, not a performance or integrity failure, but the
two values must not be conflated.

## Checks

### A. Ground-truth provenance: PASS with known-K/oracle qualification

`s2_simple_cut.py:263-290` reads `y` from the dataset archive and derives K; `:157-223` shows that
SimpleCut fit accepts H0/W/seed/device/epochs only, and `:395-410` runs KMeans and metrics after fit.
R is a real-GT known-K post-fit benchmark. O_pool/O_full graphs are inherited label-derived
diagnostic artifacts and are marked `labels_used=true`, `purpose=diagnostic_only`,
`method_claim=false`. They are not label-free or deployable method results.

### B. Score normalization: PASS

Independent recomputation of ARI/NMI/optimal-mapping ACC from all 18 `predictions.npy` and
`labels_true.npy` pairs matched stored metrics within `1e-12`; H_pool/H_full/C aggregates also match.
No self-referential normalization was found.

### C. Result existence, hashes, and graph reuse: PASS

The S2 root exact-tree manifest verifies 197 entries. Exactly 18 expected run directories are present;
all are `completed_valid`, and all required arrays, graphs, JSON audits/configs, histories, and O-arm
oracle manifests exist. Each run hash verifies. Source archive labels and H0 hashes match. Selected
and directed graph files are hash/structure-identical to the corresponding S1 v2 sources for all 18
runs. S1 root and S0 root manifests verify with 827 and 27 entries respectively.

### D. Contract and training artifacts: WARN

All runs use physical GPU 3, within legal pool `[1,2,3,4,5,6]` and outside forbidden IDs `[0,7]`;
fit metadata contains no label/K path. All 18 histories have 80 finite rows, all embeddings are
finite, and artifact diagnostics show no collapse (`low_variance_dimension_ratio=0`; effective rank
Baron `16.7312–18.1273`, Mouse `20.5364–22.4333`; minimum dimension standard deviations > 0).

The real audit caveat is that `training_metrics.csv` logs the pre-step loss (`s2_simple_cut.py:181–197`)
while `fit_metadata.final_loss` is a post-step recomputation (`:199–203`). Example: Baron Human
seed42 R has `94686.3516` in the last CSV row versus `92718.7891` in `final_loss`. This does not
invalidate embeddings or primary opportunity metrics, but it is a known metadata timing gap.

After the fresh audit, the focused contract test was minimally hardened to assert a nonconstant
embedding on its tiny graph in addition to finiteness. This changed no S2 performance artifact and
was revalidated with `python -m pytest -q tests/representation_consumer_probe/test_s2_contract.py`
(`2 passed`).

### E. Scope: PASS (bounded)

Scope is exactly 2 datasets × 3 arms (`R/O_pool/O_full`) × 3 paired seeds = 18. Dataset is the
statistical unit; seed is a paired repeat. This supports only conditional SimpleCut opportunity
diagnostics under the frozen S0 H0/positive-cosine/k=20/budget=8 family. It does not support
generalization, universal topology, TopoGate, selector, backbone, or S_graph claims.

### F. Evaluation type: PASS with qualification

R is a real-GT known-K post-fit benchmark. O_pool/O_full are real-data, pre-fit label-derived
diagnostic oracles; their outer metrics are not deployable method performance. S2 does not estimate T
or `S_graph`.

## Claim impact

- Supported: 18/18 completed-valid S2 runs; exact metric/hash/graph-reuse checks; no-collapse artifact
  diagnostics; Baron Human conditional opportunity confirmation; Mouse_retina observed-small result.
- Supported with qualification: known-K real-GT benchmark and label-derived oracle semantics.
- Unsupported: TopoGate performance, selector utility, `S_graph`, new-backbone superiority, universal
  topology claims, and generalization.
- WARN to preserve: training history last-row loss and post-step `final_loss` are different time
  points; do not use this metadata field for a convergence claim without clarification.

## Decision impact

Terminal decision remains `opportunity_status=heterogeneous_with_spectral_relaxation_caveat`,
`selector_status=not_estimable`, and `representation_consumer_promotion=not_authorized`. S3/S4/S5/S6,
TopoCut, and any selector remain locked.

## Verification trace

Raw fresh reviewer response (local-only; intentionally excluded from GitHub):
`.aris/traces/experiment-audit/2026-08-17_s2_simplecut_run01/001-s2-integrity.response.md`.
