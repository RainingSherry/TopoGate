# ACCG Real-Panel Audit Bundle

**Protocol:** `accg_action_conditional_joint_v2`  
**Decision:** real clustering promotion **No-Go**  
**Execution date:** 2026-08-16  
**Scope:** weight-free, label-isolation and artifact-integrity evidence

## What completed

- Main matrix: `30/30` panel runs.
- Confirmatory main panels: `27/27` labeled panels from 9 datasets and 3 seeds
  `[42, 123, 7]`.
- Operational panels: `3/3` PBMC3k runs with explicit `K=8`; no ground-truth
  labels and no ARI/NMI contribution.
- Development ablations: `48/48` arms across 4 datasets, 4 variants, and 3
  seeds. Canonical `N/R/T_s` controls were reused from the matching main panel.
- Confirmatory artifact count: `75/75` (`27` labeled main plus `48` ablations).

## Primary result

The locked labeled-panel endpoint is the dataset-level mean of
`ARI(T_c) - ARI(T_s)`, with seeds treated as repeated measurements:

| Quantity | Value |
|---|---:|
| Mean | `+0.007492` |
| Median | `+0.000363` |
| Dataset bootstrap 95% CI | `[-0.000879, +0.018889]` |
| Datasets positive for all 3 seeds | `4/9` |
| Datasets negative for all 3 seeds | `1/9` |

The development comparison does not support a joint-policy advantage:

| Policy | Mean paired effect |
|---|---:|
| Joint (`T_c - T_s`) | `+0.010751` |
| Coordinate control | `+0.015689` |
| Joint minus coordinate | `-0.004938` |
| Joint wins among paired seed rows | `1/12` |

These numbers support a bounded negative decision about the current ACCG
clustering-improvement claim. They do not show that the implementation failed:
the structural, matched-schedule, source/config identity, branchpoint reuse, and
label-isolation audits passed.

## Evidence boundary

The 9 labeled datasets use benchmark-known `K` for the clustering protocol, but
their labels are loaded only for outer evaluation. PBMC3k uses explicit `K=8`
and is operational-only. No labels or outcomes are used by preprocessing,
feature graph construction, Gate selection, loss, or training.

The v3 synthetic contract is retained separately in
`../ACCG_synthetic_v3_audit/`. It passed the action-level contract, which
authorized the real panel; it does not override the real clustering No-Go.

The files in this directory are summaries/manifests only. Checkpoints, model
weights, predictions, raw datasets, memmaps, caches, logs, and temporary
absolute paths are intentionally excluded.
