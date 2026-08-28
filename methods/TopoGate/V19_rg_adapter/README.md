# TopoGate V19 RG Adapter

V19 is an independent NPZ adapter for the original Reliability-Gated
NeighborMix-scMAE path. It does not import or modify V9, V18, the retired RG
source tree, or CLUBench baseline implementations.

The only model variants are:

- `scmae_only`: the same scMAE backbone, with no graph, gate, mixing, or pseudo loss.
- `rg_full`: PCA-cosine kNN, analytic edge reliability, topology node gate,
  reliability-weighted NeighborMix, and `real_loss + 0.3 * pseudo_loss`.

The core API is label-free:

```python
fit_predict(X, *, n_clusters=None, config, seed, device)
```

`n_clusters` is optional and is consumed only by the final KMeans readout. The
formal benchmark runner loads labels outside model fitting for K and posterior
metrics; the independent tuning entry point passes `n_clusters=None` and never
loads benchmark labels.

Build the fixed 11-stratum manifest:

```bash
python scripts/V19/build_manifest.py
```

Run one formal seed batch at a time:

```bash
python scripts/V19/run_matrix.py \
  --manifest result/V19/v19_rg_dataset_manifest_20260808.json \
  --seed 42 --gpu 4
```

After seed 42 completes, repeat with seed 123 and then seed 7. Native biological
and CLUBench-bridge results are separate protocols. Archived SOTA values may be
compared with `clubench_bridge` and the deduplicated, bridge-equivalent
`shared_text` records, but not with biological `rg_native` records.

## Label-free RG tuning

The independent tuning entry point reads only the feature matrix from each NPZ. It
does not access `y`, derive `K`, run KMeans, or write label metrics:

```bash
python scripts/V19/tune_unsupervised.py \
  --manifest result/V19/v19_rg_dataset_manifest_20260808.json \
  --output-dir result/V19/v19_rg_unsup_tuning_v1 \
  --seeds 42 123 7 --gpu 4

python scripts/V19/summarize_unsupervised_tuning.py \
  --manifest result/V19/v19_rg_dataset_manifest_20260808.json \
  --output-dir result/V19/v19_rg_unsup_tuning_v1 \
  --seeds 42 123 7
```

Selection uses an equal-weight, per-dataset/seed X-only rank score over masked
recovery, latent-view stability, and input-neighbor preservation. The selected
configuration is an engineering configuration, not an ARI-selected oracle.
