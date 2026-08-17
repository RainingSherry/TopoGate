# S1 opportunity-only formal result

Protocol: `representation_consumer_probe_s1_opportunity_spectral_v2`.

This directory contains the frozen CPU-only Spectral matrix:

```text
6 datasets × {F,U,R,O_pool,O_full} × 3 paired seeds = 90 jobs
```

All 90 jobs completed with valid artifacts. `H_pool`, `H_full`, and
`C_matched_budget_candidate_gap` are label-derived diagnostic quantities; they
are not deployable method performance and do not estimate `S_graph`.

The `F` arm uses raw `H0 → known-K KMeans`. The old `S1_oracle/` directory is
retained separately as `invalid_design` provenance because its F arm used the
wrong row-L2 carrier.
