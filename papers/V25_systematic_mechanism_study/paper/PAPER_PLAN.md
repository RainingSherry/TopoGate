# V25 Paper Plan

## Working title

**When Structure Is Not Utility: Localizing Failure Modes of Structural
Interventions in Unsupervised Clustering**

## Target and stance

This is a mechanism/benchmark paper. The paper does not present a new
TopoGate architecture. It uses a broad retrospective atlas and one narrow,
matched V21 case study. The strongest supported conclusion is conditional and
sign-heterogeneous, not universal topology superiority.

## Central question

Why does apparently useful structural information fail to translate reliably
into clustering utility?

## Claims-evidence map

| Claim | Evidence | Scope firewall |
|---|---|---|
| V1--V22 show heterogeneous structural-intervention outcomes | `PaperEvidence/paper_evidence_summary.json`, `atlas_rows.csv` | observational; dataset/protocol/readout units |
| Topology selection has conditional incremental utility in the audited V21 case study | `e1_dataset_effects.csv`, confirmation phase audit | matched prospective case study; known-K benchmark |
| Feature/gradient/local-global records localize possible failure stages | E2 summaries, gradient geometry, V23 boundary table | diagnostic/post-hoc; no universal causal law |
| Independent replication | Phase E closure | not established; holdout is incomplete compute |

## Section order

1. Introduction: structural opportunity versus intervention utility.
2. Study design: evidence layers, statistical units, and claim firewall.
3. Methods: A0/A1/A2, matched N/R/T protocol, diagnostics, and endpoints.
4. Results: atlas, E1 decomposition, localization diagnostics, and holdout
   boundary.
5. Discussion: what the decomposition changes and why another Gate is not the
   contribution.
6. Limitations and conclusion.

## Figures and tables

- Figure 1: observational V1--V22 Failure Atlas.
- Figure 2: frozen Opportunity -> Selection -> Intervention -> Representation
  -> Readout chain.
- Figure 3: E1 generic intervention/selectivity decomposition `(I_d,S_d)`.
- Figure 4: gradient and one-step diagnostic geometry; no causal fit.
- Figure 5: V23 local/global boundary evidence, kept outside the atlas; the E3
  replay gate itself had zero eligible artifact-complete rows.
- Table 1: confirmation dataset means for `I_d` and `S_d`.
- Table 2: evidence layers, units, and causal status.
- Supplement: per-job provenance, hashes, audit states, and holdout resource
  boundary.

All figures are generated from frozen CSV/JSON artifacts. Tables are generated
by `scripts/V25/build_latex_assets.py` and retain source hashes in
`tables/latex_assets_manifest.json`.

The compiled manuscript is checked by
`scripts/V25/audit_final_paper.py`, which also verifies formal citation keys,
five figure environments, all 15 copied assets, table/figure source hashes,
numeric anchors, and the no-overclaim scope firewall.

## Non-claims

The paper must not claim universal topology superiority, pooled historical
causality, fully label-free E1 fitting, independent holdout replication, a
proven objective-conflict law, or a universal local-to-global theorem. No new
Gate, loss, selector, DCBoost comparison, or V26 iteration is part of this
paper.
