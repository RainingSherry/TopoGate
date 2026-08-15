# V25 Protocol Narrative

V25 is a systematic study of the audited V1--V24 history. It is not a new
TopoGate architecture and it does not authorize V26, a new Gate, a new loss,
DCBoost, or a rescue experiment. The paper question is:

> Why does apparently useful structural information fail to translate reliably
> into clustering utility?

The frozen localization chain is:

```text
Opportunity -> Selection -> Intervention -> Representation -> Readout
```

## Evidence layers

### A0: registry

V1--V24 are registered with source, preprocessing, readout, K, timing, causal
status, artifact, label-isolation, reuse, and alternative-explanation fields.
V1--V22 enter the quantitative atlas. V23/V24 are boundary evidence only and
are not pooled with structural-intervention rows.

The row count is not the sample size. The A0 registry contains repeated rows
from the same dataset/protocol/readout units. A1 summaries use
dataset/protocol/readout as the retrospective unit; seeds and variants are
repeated measurements. The 1,637 paired rows are never treated as 1,637
independent datasets.

### A1: atlas and replay

Only artifact-complete cases can enter offline replay. Metadata-only records
remain provenance records and cannot become replay evidence. Raw Delta ARI is
the primary descriptive quantity. Headroom, embedding displacement, effective
corruption, and final-neighborhood descriptors are sensitivity or post-treatment
measurements, not causal covariates. Label-free geometry is kept separate from
post-hoc label-aware geometry.

### A2: triage

A2 has real veto authority. Its only valid decisions are `retain_e1`,
`cancel_e1`, and `no_prospective_compute`. If E1 is not retained, no replacement
E4 may be invented. A2 also freezes the holdout candidate/adapter manifest,
measurement schema, delta threshold, primary endpoint, and falsifier before any
E1 outcome is used.

## E1: matched V21 case study

The three arms are:

```text
N = matched None
R = matched Random assignment policy
T = topology-dependent selection policy
```

The estimands are:

```text
I_d = Q(R) - Q(N)
S_d = Q(T) - Q(R)
Q(T) - Q(N) = I_d + S_d
```

The primary readout is clean embedding plus known-K KMeans. Student-t cluster
head metrics are secondary diagnostics. The primary selection endpoint is
`S_full_ARI = ARI_T - ARI_R`.

The loss contract is:

```text
L_N = L_scMAE + lambda_i * L_InfoMax
L_R = L_scMAE + lambda_i * L_InfoMax + lambda_a * L_JS^R
L_T = L_scMAE + lambda_i * L_InfoMax + lambda_a * L_JS^T
```

N keeps the Student-t head, InfoMax term, warmup initialization, and optimizer
state, but performs no assignment forward and no JS term. Donor, eligible,
budget, batch-order, topology-statistics, and selection-noise schedules are
replayed only as shadow/audit quantities for N.

All arms branch after warmup and head initialization, before the first
assignment update. The branchpoint stores model/head/optimizer/Gate state,
Python/NumPy/Torch/CUDA RNG state, batch state, K provenance, and data,
preprocessing, graph, and statistics hashes.

For R and T, the policy difference is restricted to topology logits:

```text
s_T = f_theta(phi_ij) + epsilon_ij
s_R = 0 + epsilon_ij
```

The same frozen Gumbel tensor, `gumbel_scale`, exact top-k rule, donor schedule,
eligible set, and effective budget are used in both arms. Gate auxiliary state
belongs to the treatment policy and is audited separately from shared
backbone/head state.

The full label vector is unavailable to preprocessing, graph construction, Gate,
loss, and model updates. Benchmark-known K may be derived from the outer labels
to size the trainable head and the final KMeans readout. E1 is therefore a
real-ground-truth, known-K benchmark, not fully label-free fitting.

Effects are classified independently for I and S at delta=0.03 as
`Positive`, `Negative`, `Observed-Small`, or `Inconclusive`. Three seeds are
repeated measurements, not an equivalence sample. No three-seed bootstrap or
cluster-robust population claim is used.

## E2 and E3 diagnostics

E2-A aggregates feature semantics at dataset-by-seed level. Coordinate-level
distributions are plots only; millions of sample-feature coordinates are not
independent observations. Fisher, MI, and class-support enrichment are post-hoc
descriptors only.

E2-B records base, assignment, and InfoMax gradient geometry at T0, T1, and T2.
E2-C clones model, head, and Adam state and executes actual N/R/T Adam one-step
counterfactuals. These diagnostics may localize a failure stage but cannot by
themselves prove an objective-conflict law.

E3 applies an artifact-complete replay gate. In the frozen V25 evidence it found
zero eligible rows (`candidate_rows=0`), so no replay was run; the retained V23
local/global rows remain separate boundary evidence. Label-free geometry and
post-hoc supervised neighborhood metrics are separate. A local increase with
non-positive ARI is a bounded boundary example, not a universal theorem.

## Holdout and closure

The holdout measures exactly the subset required by the frozen claim. For the
selection claim this is the N/R/T panel and `S_full_ARI`; secondary metrics cannot
replace it. The current six-panel holdout produced no evaluable endpoint because
the frozen dense decoder/Adam resource path did not complete. Its state is
`inconclusive_not_completed`, not a negative model result and not independent
replication.

## Required artifacts and checks

Formal outputs live under `result/V25_systematic_mechanism_study/`. Each job must
retain its resolved config, source/preprocessing hashes, K audit, labels-after-fit
audit, relationship artifacts, and one of `audit_ok`, `invalid_design`, or
`incomplete_compute`. The required checks are:

```bash
python -m compileall -q methods/TopoGate/V25_systematic_mechanism_study scripts/V25
pytest -q scripts/V25/tests
python scripts/V25/audit_v25_contract.py
python scripts/V25/audit_paper_claims.py \
  --v25-root result/V25_systematic_mechanism_study \
  --draft refine-logs/V25_MANUSCRIPT_WORKING_DRAFT.md \
  --output-dir review-stage
python scripts/V25/audit_final_paper.py
```

The paper may claim an observational V1--V22 atlas and a conditional,
sign-heterogeneous V21 case-study effect. It may not claim universal topology
superiority, pooled historical causality, fully label-free E1 fitting, or
independent holdout replication.
