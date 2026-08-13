# Final V-Series Result Snapshot

This directory contains the final, largest auditable result table or summary
available for each version. It is a metadata-only publication snapshot:
raw inputs, checkpoints, embeddings, predictions, labels, caches, worker logs,
and smoke/debug outputs are excluded.

## Evidence rules

- A directory is included only when a formal or explicitly audited result
  artifact exists in the local result volume.
- `single_seed`, `coarse_screen`, `restricted`, `no-go`, and
  `incomplete_compute` labels are preserved from the source artifacts.
- No result is promoted from an engineering smoke run.
- Labels are not used during model fitting; where the source protocol derives
  benchmark K from labels, that boundary remains in the published summary.

## Version index

| Version | Published artifact | Scope and boundary |
|---|---|---|
| V01-V08 | none | No current, non-smoke final result table was found in the audited result volume. |
| V09 | `V09/` | CLUBench 131-dataset single-seed comparison, multi-seed advantage ablation, and paper-preprocess comparison. |
| V10 | `V10/` | Available comparison CSVs only: 3-dataset ablation plus one multiseed record; no larger reliable-graph aggregate was present. |
| V11 | `V11/` | Five-dataset, three-seed TDA-H0 pilot with per-run diagnostics and paired comparisons. |
| V12 | `V12/` | 144-run stage-3 topology grid with coverage and aggregate tables; source report marks the signal-amplification result no-go. |
| V13 | `V13/` | Five-dataset, two-variant, three-seed hard-gate batch and paired tables. |
| V14 | `V14/` | Five-dataset, two-variant, three-seed advantage batch. |
| V15 | none | No non-smoke formal performance table; retained locally as restricted/no-go evidence. |
| V16 | none | No promotable final table in the audited result volume; V16.1 is listed separately because its protocol and evidence boundary differ. |
| V16.1 | `V16_1/` | Expanded-count Stage-1 promotion summaries; all rows retain `empirical_not_supported`. |
| V17 | none | Reference implementation only; no performance evidence. |
| V18 | `V18/` | Complete v2.2 matrix summary: 4,470/4,470 runs over 149 eligible datasets. |
| V19 | `V19/` | Post-freeze 198-run matrix, PlantNet transfer aggregate, sparse extension table, and goal audit. |
| V20 | `V20/` | Eight-dataset seed-42 coarse screen represented by the eight final run summaries; no matched control aggregate was produced. |
| V21 | `V21/` | Complete six-dataset graph-fix matrix: 36/36 valid jobs with aggregate and paired tables. |
| V22 | `V22/` | Cooperative Keep-Gate 16-job single-seed audit plus hard-gate controls; two cooperative jobs remain `incomplete_compute`. |

The files are copied from the local result volume without recomputing metrics.
The original source paths and hashes remain in the local audit records; this
public snapshot deliberately omits machine-specific paths where possible.
