# ToPoGate V0 results release (2026-09-04)

This branch is a compact reproducibility snapshot for the ToPoGate V0
topology parameterization. It contains the runnable V0 implementation and
lightweight result tables from the five-seed final evaluation.

## Contents

- `methods/TopoGate/V0/`: V0 model, graph, corruption, trainer, configs, and
  tests.
- `methods/NeighborMix_scMAE/` and `methods/DeepLearning/scMAE_family.py`:
  direct runtime dependencies used by the V0 implementation.
- `results/v0_20260904/`: 153-dataset aggregate results (765 completed seed
  rows), per-seed metrics, and the label-free tuning selection manifest.
- `results/baseline_comparison/`: compact comparison CSVs for the available
  baseline runs. These are point estimates unless a file says otherwise.
- `results/selected_panels/`: the previously prepared exploratory comparison
  panel, retained for traceability and not promoted as a universal SOTA claim.

## Reproduction boundary

The V0 fit and tuning protocol is X-only: labels are not read for graph
construction, corruption, training, or candidate selection. Labels and the
number of clusters are used only by an external post-fit benchmark readout
(oracle-K KMeans, ACC/NMI/ARI). The final V0 matrix uses seeds
`42, 123, 7, 2025, 3407`; all 153 registered datasets completed and selected
the topology parameterization in this snapshot.

The published tables do not contain raw datasets, H5/H5AD/NPZ files, model
weights, embeddings, predictions, caches, worker logs, or local machine paths.
Download the input data from the sources recorded by the experiment protocol
and pass an explicit input path to the runner.

## Example

```bash
python -m methods.TopoGate.V0.run \
  --data-path /path/to/example.npz \
  --save-dir /tmp/topogate_v0 \
  --config methods/TopoGate/V0/configs/topogate_v0_topology.yaml \
  --parameterization T \
  --device cuda \
  --epochs 80 \
  --n-clusters 3
```

Run the V0 tests with:

```bash
python -m pytest methods/TopoGate/V0/tests -q
```

## Interpretation

The 153-dataset aggregate is an auditable V0 result table, not evidence that
ToPoGate is universally superior to every clustering method. The baseline
files were produced under an older, single-run comparison protocol and must
be described as such in a paper. Results marked unavailable or blocked are
not replaced with proxy values.
