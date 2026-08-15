# V25 Paper Claim Audit

**Status:** `audit_ok`  
**Protocol:** `v25_paper_claim_audit_v1`  
**Generated:** `2026-08-15T02:30:12.306037+00:00`  
**Scope:** deterministic recheck of frozen artifacts; not an external reviewer verdict.

## Checks

| Check | Result |
|---|---|
| `a0_v1_v22_rows` | PASS |
| `a0_v1_v22_paired_rows` | PASS |
| `a0_v1_v22_units` | PASS |
| `a0_v23_v24_boundary_records` | PASS |
| `a1_paired_rows` | PASS |
| `a1_positive_rows` | PASS |
| `a1_negative_rows` | PASS |
| `a1_small_rows` | PASS |
| `a0_a1_summary_reconciles` | PASS |
| `a1_observational_boundary` | PASS |
| `a2_retained_without_e4` | PASS |
| `claim_endpoint_frozen` | PASS |
| `e1_complete_audited_phase` | PASS |
| `e1_effect_rows_complete` | PASS |
| `e1_effect_states_recomputed` | PASS |
| `holdout_firewall` | PASS |
| `paper_scope_audit_passes` | PASS |

## Claim Ledger

| ID | Status | Evidence | Allowed wording |
|---|---|---|---|
| C1 | `supported_with_observational_scope` | `PaperEvidence/paper_evidence_summary.json` | observational atlas of heterogeneous outcomes |
| C2 | `supported_with_case_study_scope` | `PaperEvidence/e1_dataset_effects.csv` | dataset-conditional effect in the audited V21 case study |
| C3 | `diagnostic_only` | `PaperEvidence/paper_evidence_summary.json` | diagnostic/localization evidence |
| C4 | `inconclusive_not_completed` | `PhaseE/closure.json` | independent replication not established; holdout inconclusive_not_completed |

## Scope Firewall

- E1 is a real-ground-truth, benchmark-known-K evaluation, not fully label-free fitting.
- E2/E3 remain diagnostic/post-hoc evidence; coordinate counts are not inferential sample sizes.
- The holdout is `inconclusive_not_completed`, not a negative performance result.
- No universal topology-superiority or pooled historical causal claim is permitted.

See `V25_PAPER_CLAIM_AUDIT.json` and `V25_PAPER_CLAIM_LEDGER.csv` for machine-readable details.
