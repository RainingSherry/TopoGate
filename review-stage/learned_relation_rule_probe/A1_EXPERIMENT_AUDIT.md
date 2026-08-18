# Learned-relation-rule probe A1 — deterministic integrity audit

**Date:** 2026-08-18
**Scope:** `result/learned_relation_rule_probe/A1_supervised_ceiling/`
**Audit class:** local deterministic contract audit; A1 is diagnostic supervision.

## Overall verdict: PASS with diagnostic qualification

- Expected scorer/view rows: `3 datasets × 2 scorers × 3 views = 18`.
- Fold rows: `18 × 5 = 90`.
- Every fold is anchor-disjoint and OOF prediction coverage is exactly 100%.
- The source edge tables, inherited R/O_pool artifacts and row-budget capacity
  match the frozen RS1/S1 sources.
- The scorer target is the inherited O_pool reference-membership diagnostic;
  it is not a semantic same-class target and not a deployable label-free rule.
- The compact output tree contains only JSON/CSV/report metadata; raw scores,
  selected graphs, embeddings and predictions are not publication artifacts.

## Decision

The best frozen diagnostic configuration reaches `Delta_sup` means of
`-0.015740` (cnae9), `+0.024503` (Campbell) and `-0.300232`
(sms_spam_collection).  Therefore the pre-registered two-of-three material
ceiling fails and the terminal decision is
`predictable_reference_not_actionable_for_selection`.  A2--A5 remain locked.

The result is a supervised actionable-ceiling negative, not evidence that a
label-free rule was tested and failed.
