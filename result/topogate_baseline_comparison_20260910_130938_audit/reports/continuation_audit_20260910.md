# Continuation Audit (2026-09-10)

## Scope

This continuation re-audited the remote persistent assets before launching the remaining requested baselines.

## Verified evidence

- Remote scMAE control assets exist under `/data/luolie/ToPoGate/result/topogate_unified_archive_20260910/summaries/scmae_control/`.
- The directory contains exactly five dataset subdirectories: `20newsgroups`, `Human_Pancreas_1`, `Mouse_Pancreas_1`, `banknote_authentication`, and `cifar10`.
- Each discovered scMAE protocol file records seeds `[42, 123, 7, 2025, 3407]`, `fit_scope=train_validation`, `score_scope=test`, and `labels_used_during_fit=false`.
- These assets therefore do not establish 131-task coverage or the required three-seed/8-candidate protocol. They remain background evidence and are not promoted into the matched-budget main table.
- Remote constant-gate assets under `result/ablation/` and `result/v0t_ablation_20260905/` are historical V0/V0-T ablations. Their filenames/configuration do not establish the current V16.1 strict split, fixed indices, or 8-candidate selection protocol.
- A separate remote bundle, `/data/luolie/ToPoGate/result/baseline_5seed_20260905/`, contains completed `run.json` records for EDESC, TableDC, and ZEUS on selected datasets. The records report only aggregate metrics and fit metadata; the inspected run directories contain no predictions, sample-index files, split manifest, preprocessing fingerprint, or checkpoint. Consequently these rows cannot be recomputed or proven to use the frozen 60/20/20 protocol.
- The inspected EDESC and ZEUS records explicitly report `labels_used_during_fit=false`, which is useful provenance, but this alone is insufficient for protocol reuse because the fitting scope and sample partition are absent.

## Current decision

No new formal metric rows were fabricated from these assets. The main table remains limited to KMeans, PCA+KMeans, and ToPoGate_strict8 until protocol-matched adapters/runs for scMAE, constant gate, IDEC, EDESC, TableDC, ZEUS, scNAME, scDeepCluster, and scCDCG are completed and independently verified.

## Next execution gate

Before any launch, each remaining model must have: a native entry point, fixed-index loader, label-free fit path, reproducible prediction output, and atomic completion marker. A model lacking these items is recorded as `not_run`/`audit_only_reuse`, not silently substituted with KMeans or an unrelated historical result.
