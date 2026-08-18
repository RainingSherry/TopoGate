# TopoGate

Repository for the TopoGate research implementations and their reversible
variant configurations. The current public snapshot includes the core V9-V22
code paths, the independent post-V25 ACCG route, the independent
`representation_consumer_probe` S0-S2 terminal study, the independent
`sparse_corruption_principle_probe` C0-C2 terminal study, curated auditable
result tables, and concise evidence summaries; it
does not publish papers or dataset binaries.

This repository intentionally excludes raw datasets, model outputs,
checkpoints, papers, logs, caches, and local machine paths. Curated result
metadata is retained under `result/final_results/`; runners still expect the
caller to provide input data and an output directory.

## Contents

- `methods/TopoGate/`: TopoGate variants and configuration files from V9
  through V22, including the independent V17 relation-native reference, the
  V22 discriminator/Keep-Gate prototype, and ACCG's joint-action constrained
  policy.
- `scripts/V11/` through `scripts/V22/`, plus `scripts/ACCG/`: version-specific runners, matrix
  preparation, audits, and summarizers.
- `methods/NeighborMix_scMAE/model.py`: the scMAE backbone used by several
  TopoGate variants.
- `methods/DeepLearning/scMAE_family.py` and `methods/shared_utils.py`: direct
  runtime dependencies retained for the included code paths.
- `reports/`: curated result facts, aggregate tables, and experiment reports;
  it contains no raw model outputs.
- `result/final_results/`: the largest final result table or summary retained
  for each audited V version. Check its README for coverage and evidence
  boundaries.
- `reports/ACCG/`: ACCG protocol, synthetic contract, and weight-free real-panel
  audit. The current real clustering promotion decision is No-Go.
- `reports/representation_consumer_probe/` and
  `result/representation_consumer_probe/`: the frozen S0-S2 protocol, terminal
  decision, and weight-free opportunity diagnostics. The study does not
  promote a selector or new backbone.
- `reports/sparse_corruption_principle_probe/` and
  `result/sparse_corruption_principle_probe/`: the C0-C2 protocol, static
  corruption-principle result, compact aggregate tables, and independent
  integrity audit. C3 holdout, adaptive policy, GAN, and learned generator
  remain locked.

The public release is code-first. Result files are metadata-only summaries;
raw arrays, checkpoints, worker logs, and smoke/debug directories remain
outside Git.

The versioned directories under `methods/TopoGate/` are research variants.
They are kept as separate implementations for traceability; choose a specific
variant and configuration explicitly for an experiment.

## Environment

Python 3.10 or newer is recommended. Install the runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

PyTorch installation may need a platform-specific command when GPU support is
required.

## Basic checks

```bash
python -m compileall -q methods
python -m methods.TopoGate.V11.run --help
python methods/TopoGate/v10_reliable_graph/run.py --help
python -m methods.TopoGate.V17_topology_native.run --help
python scripts/V17/run_reference.py --help
```

Data-dependent runs require a real input file supplied by the caller; no sample
dataset is committed here.
