# V25 Closure

V25 is closed as `V25_systematic_mechanism_study`; no new TopoGate architecture,
V26, Gate/loss/selector, DCBoost, or V18/V22/V24 rescue route is opened.

## Evidence status

- The retrospective V1--V22 atlas remains observational. V23/V24 remain isolated
  boundary evidence.
- The audited V21 N/R/T confirmation completed `9/9` panels with `audit_ok=9/9`.
  Its primary selection effect was heterogeneous: Baron Human `+0.044617`,
  Campbell `-0.065332`, and hate_speech `-0.033410` for `S_d`.
- The frozen independent holdout endpoint was `S_full_ARI = ARI_T - ARI_R`.
  The claim-bound pool produced no completed panel: `0/6` panels are evaluable
  and the holdout audit contains no performance rows.

The holdout outcome is therefore `inconclusive_not_completed`, not a negative
model result and not evidence that the frozen claim is false. The resource
boundary is recorded in `PhaseD/E1/queue_state.json` and the per-attempt logs;
three news20 seeds reached the same CUDA OOM during Adam state initialization.

## Allowed wording

The paper may report the failure atlas and the conditional V21 case-study result.
It must not report independent holdout replication, a universal topology claim,
or a causal explanation for the holdout resource failure. No further prospective
training is authorized by this closure.
