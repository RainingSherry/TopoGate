# Validation of the unified implementation

Executed locally on 2026-09-09. These checks establish implementation behavior,
not performance on CLUBench or biological benchmarks.

- Nine portable unittest tests passed, including constant/analytic gate
  equivalence, zero edge modulation, exact old aggregation/RNG replay,
  Monte Carlo/full aggregation consistency, training-only preprocessing,
  split disjointness, invalid-config rejection, selection provenance/budget
  guards, and training-only readout.
- The training test changed held-out input features while preserving training
  features, and verified identical training graph and loss trajectory. It also
  checked prediction when the held-out sample count is smaller than K.
- End-to-end CPU smoke runs completed for two synthetic datasets: signed
  general numerical features and nonnegative integer biological counts.
  Each ran five candidates for one epoch, refined the top four over three
  selection seeds, froze the winner, and completed five final seeds.
  Biological preprocessing exercised training-only feature selection and
  count normalization. These are synthetic fixtures, not biological results.
- A repeat search after test evaluation began was rejected for both datasets.
- Python compilation and git diff whitespace checks passed.

The historical test files in this directory reference scripts/V0_RG,
V0_RG_rg_adapter, and server-local manifests that are absent from this public
snapshot. They were not represented as passing regression tests. Run the new
portable suite with:

```bash
python -m unittest methods.TopoGate.V0_RG.tests.test_unified -v
```

No full-library experiment, server GPU run, historical raw-run replay, or
publication-level performance claim was completed in this change. Before the
formal matrix, fill the verified dataset manifest and perform the three-task
runtime/memory check described in TUNING_PLAN_ZH.md.
