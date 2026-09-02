# TopoGate V0 legacy equivalent evidence

This directory stages the completed PlantNet `scVICAR` attribution-v2 matrix as
legacy equivalent evidence for the unified TopoGate V0 model identity. It is a
data reuse/staging artifact, not a new V0 training run.

## Scope

- Source project: `/home/luolie/biopipeline/dimension-reduction/plantnet`
- Source protocol: `result/scvicar_attribution_v2/formal` and
  `papers/scVICAR/experiments/attribution_v2`
- Matrix: 6 datasets x 7 variants x 3 seeds = 126 completed runs
- Copied payload: all formal run artifacts (about 4.07 GB), source freeze and
  run-status metadata, aggregate CSVs, and the original protocol README
- Input `.h5ad` files are intentionally not duplicated. Their frozen source
  paths and SHA256 values remain in `source_freeze.json`.

## V0 mapping

The historical `fixed` arm is the equivalent F parameterization
(`scVICAR-F`), and `topology_full` is the equivalent T parameterization
(`scVICAR-T`). The other five arms are retained so the paired controls remain
auditable; they are not silently renamed as V0 variants.

This mapping is about model identity only. The historical formal runs were
produced by the PlantNet attribution-v2 runner, so every copied run has
`direct_v0_run: false`. In particular, the historical topology arm used
`neighbor_k=5` and `gate_max=0.1`, whereas the current V0 topology YAML uses
`neighbor_k=10` and `gate_max=0.15`. These differences are recorded in
`manifest.json` and prevent the data from being presented as current V0 formal
GPU results.

## Labels and K

The source runner records labels and known-K benchmark metrics after fitting.
The source summaries report `label_leakage: false`; labels were not supplied to
the scMAE optimizer, graph, gate, corruption, or loss. The `n_clusters` value
is benchmark/readout metadata (derived from the source label set), not a
training signal. This directory therefore cannot support a claim of a new
label-free V0 run or of V0 clustering efficacy.

## Verification

`manifest.json` records the source/destination paths, variant mapping, parameter
mismatches, and copy counts. `file_hashes.sha256` contains destination hashes
for every copied formal artifact. A checksum-aware `rsync --dry-run` and
independent source/destination file-count and byte audits are required before
this bundle is used. The corresponding Claude review trace and raw response
are under `review-stage/V0_data_staging/` and `.aris/traces/`.

