# D1 Results

D1 completed all `3 datasets × 3 seeds = 9` CPU/no-training rows. The
computation audit is `audit_ok=true`, but the pre-registered estimability gate
is false, so D2 GPU runs were not started.

| Dataset | Common positive-budget rows | Dataset-total mismatch | Median row mismatch | Gate |
|---|---:|---:|---:|---|
| Mouse_retina | 100.000% | 3.134% | 0.420% | pass |
| Baron Human | 93.098% | 8.981% | 12.188% | fail |
| Campbell | 100.000% | 8.492% | 9.643% | fail |

The three seed rows are deterministic tie-break reproductions and therefore
have identical aggregate values within each dataset. Baron has 579 positive-
budget rows without a constructive interval in each seed row. Campbell has a
nonzero interval for every positive-budget row, but the constructed midpoint
arms still miss the frozen 5% dataset-total tolerance.

Final state:

```text
status = common_dose_not_estimable
audit_ok = true
d1_gate_pass = false
d2_authorized = false
d2_gpu_runs_started = 0
```

This result is bounded to the frozen constructive matching contract. It does
not prove universal impossibility, does not refute C2 P2's clustering result,
and does not isolate a causal support effect. No labels, ARI, NMI, ACC,
embeddings or model outputs were used or produced.
