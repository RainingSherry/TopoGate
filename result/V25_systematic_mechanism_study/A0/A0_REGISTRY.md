# V25 A0 Evidence Registry

Generated at `2026-08-14T17:18:56.282490+00:00` from the audited V1-V22 long table.

This is a registry, not a pooled inferential analysis. Rows are repeated records; the primary historical unit is a dataset/protocol/readout unit.

## Coverage

- V1-V22 rows: `2209`; completed: `2175`; reported/unpromoted: `32`; incomplete: `2`.
- V1-V22 paired Delta ARI rows: `1637`.
- V1-V22 dataset/protocol/readout units: `431`; unique dataset IDs: `342`.
- V23 and V24 are recorded as boundary evidence and are not included in the quantitative intervention atlas.

## Statistical boundary

Seed is a repeated measurement. Variant is an intervention condition. Coordinate, row, and pair counts are never treated as independent experiments.

## Version summary

| Version | Rows | Completed | Reported | Incomplete | Units | Unique datasets |
|---|---:|---:|---:|---:|---:|---:|
| V09 | 291 | 291 | 0 | 0 | 177 | 154 |
| V10 | 21 | 21 | 0 | 0 | 4 | 4 |
| V11 | 75 | 75 | 0 | 0 | 5 | 5 |
| V12 | 48 | 48 | 0 | 0 | 4 | 4 |
| V13 | 30 | 30 | 0 | 0 | 5 | 5 |
| V14 | 30 | 30 | 0 | 0 | 5 | 5 |
| V16.1 | 32 | 0 | 32 | 0 | 8 | 8 |
| V18 | 1490 | 1490 | 0 | 0 | 149 | 149 |
| V19 | 108 | 108 | 0 | 0 | 32 | 24 |
| V20 | 8 | 8 | 0 | 0 | 8 | 8 |
| V21 | 36 | 36 | 0 | 0 | 6 | 6 |
| V22 | 40 | 38 | 0 | 2 | 28 | 16 |

## Boundary evidence

| Version | Status | Evidence | Boundary |
|---|---|---|---|
| V23 | no_go | formal_no_go | Protocol A does not identify dependency-specific information beyond support statistics and therefore does not justify M1 or a cycle-guided training mechanism. |
| V24 | calibration_no_go | formal_calibration | calibration power is zero; no efficacy conclusion |

## Replay gate

A0 deliberately leaves `replay_eligible_rows=0` for the historical summary table. A1 replay must pass a separate artifact-complete gate with exact embeddings/predictions/labels provenance; missing public artifacts remain descriptive-only.
