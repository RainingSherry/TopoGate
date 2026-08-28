# V25 Systematic Mechanism Study

V25 is a protocol and diagnostic layer over the audited V1--V24 evidence. It is not a new model
architecture. The implementation is intentionally small and keeps the V21 model/loss semantics in
their existing module.

## Modules

- `e1_protocol.py`: matched V21 N/R/T panel, shared warmup branchpoint, frozen donor/eligible/budget/
  Gumbel schedules, clean KMeans readout, and actual Adam one-step counterfactuals.
- `e2_metrics.py`: streaming feature-audit reducers whose inferential unit is dataset x seed rather
  than sample-feature coordinates.

The orchestration, registries, audits, manifests, and paper exports live under `scripts/V25/`.
Formal outputs live under `result/V25_systematic_mechanism_study/` and must retain resolved configs,
hashes, labels/K audits, relationship artifacts, and incomplete-compute markers.

The compact protocol narrative is in [`PROTOCOL.md`](PROTOCOL.md). It is the
single reader-facing description of the evidence layers, N/R/T loss semantics,
matching contract, label/K boundary, diagnostics, and closure rules.

The formal runner `scripts/V25/run_e1_matched_protocol.py` requires the frozen A2 decision file and
rejects every decision other than `retain_e1`; it records the decision hash in `runner_profile.json`.
Engineering smoke tests call the library on toy data and are not formal E1 jobs. Holdout manifests
repeat the frozen input adapter, feature-selection, normalization, max-feature, graph-input, and
model-input fields at dataset/job level so an adapter cannot be changed after preflight.

## Arm semantics

`N` keeps the scMAE reconstruction and InfoMax terms, but does not execute assignment corruption or
the JS term. `R` uses zero topology logits with the same frozen Gumbel tensor as `T`. `T` uses the
topology-dependent FeatureGate logits. Donor, eligible set, effective budget, batch order, graph
statistics, and readout settings are shared or explicitly audited.

The primary estimands are:

```text
I_d = Q(R) - Q(N)
S_d = Q(T) - Q(R)
Q(T) - Q(N) = I_d + S_d
```

For each seed, `Q` is the clean-embedding, known-K KMeans ARI. The primary
seed-level selection endpoint is `S_full_ARI = ARI_T - ARI_R`; `S_d` is the mean
of those three seed-level endpoints for one dataset. Pilot and confirmation
panels remain separate and are not pooled.

The full label vector is isolated from preprocessing, graph construction, Gate, loss, and model
updates. E1 nevertheless uses benchmark-known `K` (derived from `y` when no explicit `--n-clusters`
is supplied) to size the cluster head and to run the primary clean-embedding KMeans readout. It is
therefore a real-GT, known-K benchmark, not fully label-free fitting. A2 must authorize E1 before
any formal E1 runner is launched. A completed engineering smoke is not a performance result, and an
incomplete holdout is not a negative result.

## Verification

From the repository root:

```bash
python -m compileall -q methods/TopoGate/V25_systematic_mechanism_study scripts/V25
pytest -q scripts/V25/tests
python scripts/V25/audit_v25_contract.py
python scripts/V25/audit_paper_citations.py
```

Do not add V26, a new Gate/loss/selector, DCBoost, or a rescue experiment under this directory.
