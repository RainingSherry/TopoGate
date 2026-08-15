# V25 Frozen Paper Claim

- Claim family: `selection`
- Primary endpoint: `S_full_ARI = ARI_T - ARI_R`
- Activation subset: `E1_NRT`
- Delta threshold: `0.03`; sensitivity: `[0.02, 0.03, 0.05]`
- Frozen at: `2026-08-14T20:31:50.912860+00:00`

## Frozen Wording

Topology-dependent selection has conditional incremental utility in the audited V21 case study; this is not a universal population claim.

## Falsifier

No predeclared holdout dataset has a seed-stable material S_full_ARI effect, or the T/R matching audit fails.

## Governance

This file freezes one predeclared endpoint for holdout validation. Secondary metrics cannot replace it, and an unattractive result cannot reopen claim selection or create V26.

## Evaluation Boundary

E1 is a real-ground-truth, benchmark-known-K evaluation: the full label vector is
isolated from preprocessing, graph construction, Gate, and loss, but K may be
derived from the benchmark labels to size the cluster head and readout. It is
not a fully label-free fitting claim.
