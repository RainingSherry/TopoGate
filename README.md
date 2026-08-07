# TopoGate

Code-only repository for the TopoGate research implementations and their
reversible variant configurations.

This repository intentionally excludes datasets, experiment results,
checkpoints, papers, logs, caches, and local machine paths. Runners expect the
caller to provide input data and an output directory.

## Contents

- `methods/TopoGate/`: TopoGate variants and configuration files.
- `methods/NeighborMix_scMAE/model.py`: the scMAE backbone used by several
  TopoGate variants.
- `methods/DeepLearning/scMAE_family.py` and `methods/shared_utils.py`: direct
  runtime dependencies retained for the included code paths.

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
```

Data-dependent runs require a real input file supplied by the caller; no sample
dataset is committed here.
