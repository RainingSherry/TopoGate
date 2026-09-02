# Final V-Series Result Snapshot

This directory is the metadata-only publication snapshot of the final, auditable
results currently available in the local result volume. It contains source
provenance, resolved protocols, aggregate tables, audits and result summaries.

Raw inputs, labels, checkpoints, embeddings, predictions, topology caches,
worker logs and other large runtime artifacts remain outside GitHub. The local
result volume is about 241 GB; GitHub's normal repository limit is not suitable
for an unchanged copy. Every excluded artifact remains represented by its
source path and/or SHA-256 in the local manifests where the source protocol
provides one.

## Evidence rules

- Completed and audited results are included; smoke, prepared-but-not-started,
  incomplete and protocol-mismatch runs are not promoted to final evidence.
- Single-seed and coarse-screen labels are preserved and are not presented as
  multi-seed robustness claims.
- Labels are not used during model fitting. When a benchmark derives K from
  labels, that is recorded as an outer readout-only protocol.
- The V0 files below are copied from the current V0 implementation and current
  result artifacts; they are not the retired `-f/-t` runner.

## Version index

| Version | Published artifact | Scope and boundary |
|---|---|---|
| V0 | `V0/` | Current V0 source plus CLUBench single-seed (131 datasets), formal 6-dataset × 3-seed matrix, label-free tuning record, and PlantNet legacy provenance/parity metadata. |
| V01-V08 | none | No current, non-smoke final result table was found in the audited result volume. |
| V09 | `V09/` | CLUBench 131-dataset single-seed comparison and audited ablations. |
| V10 | `V10/` | Available comparison CSVs and audited records. |
| V11 | `V11/` | TDA-H0 pilot and paired comparisons. |
| V12 | `V12/` | Stage-3 topology grid and its recorded no-go boundary. |
| V13 | `V13/` | Five-dataset, two-variant, three-seed batch. |
| V14 | `V14/` | Five-dataset, two-variant, three-seed advantage batch. |
| V16.1 | `V16_1/` | Expanded-count Stage-1 promotion summaries. |
| V18 | `V18/` | Complete v2.2 matrix summary. |
| V19 | `V19/` | Post-freeze matrix and transfer aggregates. |
| V20 | `V20/` | Eight-dataset seed-42 coarse screen. |
| V21 | `V21/` | Complete six-dataset graph-fix matrix. |
| V22 | `V22/` | Cooperative Keep-Gate single-seed audit and controls. |

The files are copied from the local result volume without recomputing metrics.
