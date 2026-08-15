# V25 A2 Mechanism Triage

Decision: **`retain_e1`**

V21 has auditable sign heterogeneity and the missing matched N/R/T counterfactual is identifiable; preserve the historical label-selected status as a limitation.

A2 has veto authority. If the decision is not `retain_e1`, no E4 or replacement prospective experiment is permitted.

## Gate checks

| Check | Result |
|---|---|
| `a0_rows_present` | `True` |
| `a1_paired_rows_present` | `True` |
| `v21_heterogeneity` | `True` |
| `matched_counterfactual_missing` | `True` |
| `historical_artifact_audit_ok` | `True` |
| `historical_v21_artifact_complete` | `True` |

## Claim-evidence matrix

| Hypothesis | Identifiability | Required compute | Status | Fatal falsifier |
|---|---|---|---|---|
| `H1_structural_quality_not_utility` | high_retrospective_only | none | retain_as_retrospective_claim | audited atlas loses the sign heterogeneity after unit/protocol stratification |
| `H2_topology_selectivity_conditional` | high_with_new_NRT_protocol | E1 only after A2 retain_e1 | provisionally_reserved | no material, seed-stable I or S effect in all pilot datasets or branchpoint matching fails |
| `H3_generic_intervention_effect` | high_with_E1_only | E1 shared with H2 | secondary_E1_quantity | N/R effect is observed-small in all predeclared cases |
| `H4_local_global_conversion` | moderate_offline_replay; prospective only if artifacts exist | none unless frozen claim requires holdout replay | retain_as_candidate_claim | all label-free local improvements co-occur with positive global gain |

## Frozen measurement schema

Primary readout is clean embedding plus known-K KMeans; Student-t is secondary. E1 uses real
dataset ground truth after fitting, while benchmark-known K can size the cluster head during
fitting; this is not fully label-free fitting. Holdout activation is claim-dependent and may not
add a new endpoint after Claim Freeze.

- `delta_threshold`: `0.03` (primary; sensitivity values `0.02` and `0.05` are descriptive only).
- E1 selection endpoint: `S_full_ARI = ARI_T - ARI_R`; generic intervention endpoint: `I_full_ARI = ARI_R - ARI_N`.
- E3 endpoint: `local_positive_and_global_nonpositive`; kNN label purity is post-hoc supervised geometry.
- E2-C endpoint: sign agreement between `S_1step` and `S_full`.

## Holdout contract

- Candidate pool is frozen before E1 outcomes: `True`.
- Eligible candidates by domain: `{"scRNA": 1, "sparse_text": 9}`; target counts: `{"scRNA": 4, "sparse_text": 2}`.
- Pool shortfall is recorded, not silently filled: `{"scRNA": 3, "sparse_text": 0}`.
- Adapter validity checks inspect source existence and frozen input protocol only; they never inspect ARI or other outcomes.

## Scope boundary

V1-V22 support a retrospective observational claim. V23/V24 remain boundary evidence. Any E1 result is a V21 case study and cannot be promoted to a universal population claim without a separate predeclared replication.
