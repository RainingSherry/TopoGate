# TopoGate

Repository for the TopoGate research implementations and their reversible
variant configurations. The current public snapshot adds auditable V22 result
metadata; it does not publish papers or dataset binaries.

This repository intentionally excludes raw datasets, model outputs,
checkpoints, papers, logs, caches, and local machine paths. V22 result metadata
is retained under `result/V22/`; runners still expect the caller to provide
input data and an output directory.

## Contents

- `methods/TopoGate/`: TopoGate variants and configuration files.
- `scripts/V17/`: V17 reference and sparse-input audit entrypoints.
- `methods/NeighborMix_scMAE/model.py`: the scMAE backbone used by several
  TopoGate variants.
- `methods/DeepLearning/scMAE_family.py` and `methods/shared_utils.py`: direct
  runtime dependencies retained for the included code paths.
- `reports/`: curated result facts, aggregate tables, and experiment reports;
  it contains no raw model outputs.
- `result/V22/`: V22 aggregate audits, per-run JSON metadata, and dataset
  provenance manifests. Check `result/V22/README.md` for the evidence boundary.

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
