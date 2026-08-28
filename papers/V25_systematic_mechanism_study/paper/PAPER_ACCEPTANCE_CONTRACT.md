# V25 Paper Acceptance Contract

**Status:** checked against the compiled PDF and frozen evidence; final audit `audit_ok`

Each assertion is checkable from the final PDF, the local evidence bundle, or
the deterministic audit outputs. A failed assertion blocks a submission-ready
label; it does not authorize a new experiment.

1. The title, abstract, and conclusion identify V25 as a failure-localization
   study rather than a new TopoGate architecture.
2. The abstract states both the observational scope of V1--V22 and the
   conditional case-study scope of E1.
3. Every occurrence of the A0 counts (2,209 rows, 1,637 paired rows, 431 units)
   matches `registry_summary.json`.
4. The paper states the layer-specific units: A0/A1 use
   dataset/protocol/readout units, E1/E2 summaries use dataset-by-seed units,
   and coordinate/seed/variant records are repeated measurements rather than
   independent population samples.
5. V23/V24 are described as boundary evidence and do not enter the V1--V22
   quantitative atlas.
6. The methods define the N/R/T losses, the shared warmup/head branchpoint and
   exact model/head/Adam/RNG restore state, donor/eligible/budget/batch/topology
   statistics matching, the identical frozen Gumbel tensor, and the None
   no-assignment/no-JS contract.
7. The primary E1 endpoint is exactly per-seed `S_full_ARI = ARI_T - ARI_R`;
   `S_d` is explicitly its dataset mean. The paper states `delta=0.03`, the
   Positive/Negative/Observed-Small/Inconclusive mean/sign rule, and that no
   three-seed bootstrap establishes equivalence.
8. The three confirmation `S_d` values match the audited CSV: Baron Human
   `+0.044617`, Campbell `-0.065332`, and `hate_speech` `-0.033410`; pilot and
   confirmation are reported separately and never pooled.
9. The paper calls E1 a real-ground-truth, benchmark-known-K evaluation. It
   states that the full label vector is excluded from preprocessing, graph,
   Gate, loss, optimizer/model updates, and enters only for post-fit ARI/NMI;
   only benchmark-known `K` may size the head/readout. It does not call fitting
   fully label-free.
10. E2-A coordinate distributions are explicitly descriptive and all inference
    is dataset-by-seed; post-hoc label metrics are not fit inputs. A0/A1 use
    dataset/protocol/readout units, E1/E2 summaries use dataset-by-seed units,
    and gradient diagnostics retain their timepoint nesting.
11. E2-B/C and E3 are labeled diagnostics/boundary evidence, not universal
    causal explanations. The paper states that E3's replay gate found zero
    artifact-complete candidates and that V23 rows remain separate.
12. The holdout candidate pool, adapters, preprocessing, hashes, and
    claim-dependent measurement schema were frozen before outcomes, including
    the recorded domain shortfall. The holdout is reported as `0/6`,
    `inconclusive_not_completed`, incomplete compute, and never as a negative
    performance result or independent replication.
13. The five figure captions identify their evidence scope and source-bound
    interpretation; all 15 PNG/PDF/SVG assets exist.
14. Tables are generated from the frozen evidence CSVs/JSON and the static
    evidence-layer schema; their source hashes are recorded in the LaTeX asset
    manifest.
15. Every citation used by the final LaTeX manuscript is present in the formal
    `references.bib`, has a verified PDF and `INDEX.md` lifecycle record, and
    passes the formal citation audit; the missing scMAE PDF boundary is
    disclosed.
16. The final LaTeX compilation log contains no undefined citation/reference,
    unresolved `??`/`[?]` marker, or `Overfull \\hbox` warning under the exact
    audit policy recorded by `audit_final_paper.py`.
17. The final PDF numeric-claim audit passes against the frozen JSON/CSV
    artifacts, and the final source/PDF text contains no unsupported universal,
    fully-label-free, causal, negative-holdout, or positive independent-
    replication wording; qualified non-claims remain explicit.
18. Limitations state that the broad atlas and narrow V21 case study do not
    support a universal population claim.
19. A2's `retain_e1` decision, veto authority, no-E4 rule, and closure without
    V26/new Gate/loss/selector/rescue route are stated and auditable.
20. The final audit verifies all five figure environments/captions, all 15
    PNG/PDF/SVG assets, the frozen figure-manifest hashes, and the table source
    hashes before any submission-ready label is used.

## Disputed

None at proposal time. Any reviewer objection must be recorded here with the
exact assertion number and disposition; silently weakening the contract after
writing is not allowed.
