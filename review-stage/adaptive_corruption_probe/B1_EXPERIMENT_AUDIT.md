# Adaptive-corruption probe B1 — deterministic integrity audit

**Date:** 2026-08-18
**Scope:** `result/adaptive_corruption_probe/B1_corruption_library/`
**Audit class:** local deterministic contract audit; no external result service was used.

## Overall verdict: PASS with bounded-scope qualification

- Expected matrix: `6 datasets × 6 arms × 3 seeds = 108`.
- Completed-valid run summaries: `108/108`.
- Every run has `summary.json`, `audit.json` and `resolved_config.json`.
- Every run records `labels_used_during_fit=false`, a legal GPU visibility
  assignment, finite embedding/metrics and all five corruption audit fields.
- For every dataset×seed, the five non-clean arms have exactly the same
  `effective_changed_coordinate_rate_mean`; the clean arm is the zero-change
  floor. This is the pair-feasible matched-budget check, not an assumption from
  the requested rate alone.
- The root compact result tree contains only `.json` and `.csv` files.  Raw
  inputs, labels, corruption arrays, embeddings, predictions, checkpoints and
  logs are absent from this publication-facing tree.
- The synthetic positive-control fixture is `completed_valid` and uses no
  labels.

## Ground-truth and evaluation boundary

The six real-data runs use dataset labels only after fitting, for
benchmark-known-K KMeans and ARI/NMI.  They are real-GT, known-K benchmark
readouts, not label-free training evidence.  Corruption, standardization,
optimizer and encoder paths receive no `y`, ARI, NMI or ACC.

## Decision boundary

The aggregate recomputes Level 1 (`Delta_clean`) and Level 2
(`Delta_random` against C0) from the 108 stored summaries.  Level 3 requires
distinct material winners across at least two coarse role classes.  The
machine-readable decision is `simple_corruption_principle_sufficient`; it does
not authorize B2.  This is a six-dataset development/mechanism panel and does
not support a holdout or generalization claim.

## Quarantined attempts

An earlier launch was stopped after discovering that `hate_speech` has frozen
S0 width `d_eff=99` rather than 128.  Its partial files are preserved under
`result/adaptive_corruption_probe/B1_corruption_library_attempts/aborted_input_width_protocol_mismatch_20260818/`
and are not included in the 108-run aggregate.

A second complete-looking attempt was then quarantined because support-changing
arms had a different effective coordinate rate from C0 on several datasets (for
example, `cnae9` C0=`0.33864294` versus C2=`0.31196470`).  It is preserved at
`result/adaptive_corruption_probe/B1_corruption_library_attempts/aborted_support_budget_mismatch_20260818/`;
its metrics are not evidence. The formal rerun froze
`m_i=min(ceil(0.25*active_i), floor(active_i/2), inactive_i)` and changed
`2*m_i` coordinates in every non-clean arm, after which the exact rate audit
passed for all 18 dataset×seed groups.
