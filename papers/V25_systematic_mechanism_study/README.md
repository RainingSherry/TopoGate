# V25 Paper Materials

This directory is the publication-facing entry point for
`V25_systematic_mechanism_study`. It contains no new model or experiment.

The frozen paper-level plan is [`../V25_SYSTEMATIC_MECHANISM_STUDY_PLAN.md`](../V25_SYSTEMATIC_MECHANISM_STUDY_PLAN.md).

## Current sources

- Working manuscript: [`refine-logs/V25_MANUSCRIPT_WORKING_DRAFT.md`](../../refine-logs/V25_MANUSCRIPT_WORKING_DRAFT.md)
- Protocol narrative: [`methods/TopoGate/V25_systematic_mechanism_study/PROTOCOL.md`](../../methods/TopoGate/V25_systematic_mechanism_study/PROTOCOL.md)
- Frozen evidence bundle: [`result/V25_systematic_mechanism_study/PaperEvidence/`](../../result/V25_systematic_mechanism_study/PaperEvidence/)
- Figure generator: [`scripts/V25/build_paper_figures.py`](../../scripts/V25/build_paper_figures.py)
- Numerical claim audit: [`review-stage/V25_PAPER_CLAIM_AUDIT.md`](../../review-stage/V25_PAPER_CLAIM_AUDIT.md)
- Formal manuscript citation audit: [`CITATION_AUDIT.md`](CITATION_AUDIT.md)
- Compiled manuscript and final audit: [`paper/main.pdf`](paper/main.pdf), [`paper/FINAL_PAPER_AUDIT.md`](paper/FINAL_PAPER_AUDIT.md)
- Evidence map: [`refine-logs/V25_RELATED_WORK_EVIDENCE_MAP.md`](../../refine-logs/V25_RELATED_WORK_EVIDENCE_MAP.md)

## Scope

The defensible paper has two central claims only:

1. V1--V22 form an observational atlas of heterogeneous structural-intervention
   outcomes, with repeated rows kept distinct from independent dataset units.
2. In the audited V21 N/R/T case study, topology-dependent selection has a
   conditional, sign-heterogeneous effect relative to matched random selection.

E2/E3 remain localization diagnostics. The Phase D holdout is
`inconclusive_not_completed`, not a negative result or an independent replication.
E1 is a real-ground-truth, benchmark-known-K evaluation, not fully label-free
fitting.

## Citation boundary

`CITATION_AUDIT.md` and `references.bib` cover only locally verified references.
The scMAE source is intentionally not cited until its PDF lifecycle is complete.
No external benchmark number is merged into the V25 evidence tables.

## Publication status

The formal LaTeX manuscript is compiled and the final deterministic audit is
`audit_ok`. It verifies the frozen numeric anchors, five figure captions and 15
hash-matched assets, table source hashes, formal citation lifecycle, unresolved
reference markers, and scope wording. The remaining scientific boundary is
unchanged: the holdout is `0/6` and independent validation is not established.
Do not reopen V25 training solely to improve the narrative.
