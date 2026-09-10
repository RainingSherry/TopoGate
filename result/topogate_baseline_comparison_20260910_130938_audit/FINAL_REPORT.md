# FINAL_REPORT

## Status
This archive is a partial-completion checkpoint. The strict current-protocol KMeans/PCA+KMeans panel is complete; the requested scMAE, constant-gate, external-model, and biology panels remain incomplete.

## Completed
- Audited the 2026-09-10 protocol and existing cross-model TSV/XLSX/CSV assets.
- Recorded checksums and reuse decisions in `reuse_audit.csv`.
- Generated and executed the low-cost strict-split KMeans/PCA+KMeans plan; historical assets are not counted as formal training.
- Preserved historical metrics as background only with `direct_comparable=false`, including the 129-dataset V0-RG bundle and fixed-gate/scMAE-only arms.
- Generated all requested CSV deliverables plus `manifest.json`, `protocol_audit.md`, and an explicit empty formal-run ledger.

## Comparability
Existing baseline rows are generally single-seed, oracle-K and/or full-dataset results. They cannot be used as the main matched-budget comparison without verified sample indices, preprocessing, split isolation and native prediction semantics. IDEC, EDESC and biology-specific native adapters remain unverified. Historical ToPoGate V0-RG is also background only and must not stand in for current V16.1.

## Next gate
Freeze the required manifest, exact train/validation/test indices, model adapters and 8-candidate proposal rules. Then run the ordered low-cost panel beginning with KMeans/PCA+KMeans, followed by scMAE and the constant-gate control.

## New strict baseline run
A separate resumable KMeans/PCA+KMeans runner completed on the remote persistent path `/data/luolie/ToPoGate/result/topogate_baseline_strict_kmeans_20260910/`. It consumes the existing split/experiment metadata, fits only on train plus validation, selects PCA dimension from validation only, and evaluates the test split for seeds 42/123/7. Its outputs are isolated from the active ToPoGate run. All 131 datasets completed with zero failures. The archive contains 786 finite seed-level metric rows, 262 dataset/model summaries, 131 PCA selection records, and a compact source JSON snapshot under `remote_snapshot/strict_kmeans_summaries.json`.

Across the 131-task common set, KMeans has macro means ARI 0.3039, NMI 0.3563, and ACC 0.6106. PCA+KMeans has macro means ARI 0.2350, NMI 0.3079, and ACC 0.5724. These values are current-protocol direct baselines, not evidence about the still-incomplete external-model panel.

A separate strict ToPoGate run was launched after patching the remote tuner to use exactly 8 candidates, refine only the top 2 on seeds 123/7, and evaluate final seeds 42/123/7. It is isolated at `/data/luolie/ToPoGate/result/topogate_strict8_20260910/`.
The strict8 search and final stages are now complete for all 131 datasets. The final panel has 393 finite seed-level rows (3 seeds per dataset), 131 dataset summaries, and 100% coverage. Its macro means are ARI 0.3392, NMI 0.3979, and ACC 0.6481 (see `panel_summary.csv`).
An earlier intermediate snapshot had 114/131 search states and active shards; that snapshot is superseded by the completed final panel above.

The strict search completed 131/131 selections. The first final launch failed uniformly because the dispatcher final output directory did not contain the frozen `selected.json` files; this is retained as a launcher failure. The files were then copied into the isolated final directory and the controlled retry completed successfully. No related dispatcher remains running at the time of this report.

## Omissions and failures
The strict KMeans/PCA+KMeans panel is complete. No current-protocol scMAE, constant-gate, IDEC, EDESC, TableDC, ZEUS, or biology-specific formal panel was completed. The remote ToPoGate run completed 131/131 higher-budget CLUBench cells, but those results use 32 trials and 5 seeds and remain background-only. Git branch creation/push is unavailable because this workspace contains no `.git` repository.

## Remote higher-budget snapshot
The pre-existing remote unified schedulers were observed without restarting or stopping them. They have now terminated cleanly; the formal dataset directory contains 131/131 completed CLUBench summaries and no failed final states, plus 3 biology formal summaries. These runs use the observed 32-trial/5-seed protocol, so they are retained under `remote_snapshot/` as background evidence only and are excluded from the matched-budget main tables. No 8-candidate proposal manifest has been verified.

The dispatcher child commands display `--gpu 0`, but their inherited `CUDA_VISIBLE_DEVICES` values are 1, 2, and 3 (with the fourth shard mapped similarly), and `nvidia-smi` confirms the corresponding physical devices are being used. This is recorded as an implementation detail, not treated as a new run or altered.
