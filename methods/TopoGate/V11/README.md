# TopoGate V11

V11 is an independent implementation; it does not import the mutable V9
training runner.  The central change is to formulate topology selection as a
conditional mixture of experts rather than as unrelated global beta/gamma
heuristics.

`v9_reference_manifest.json` freezes hashes of the current V9 comparison path
after the beta-scale compatibility fix.  It is explicitly not presented as a
byte-identical reconstruction of the original July 29 source; the historical
run predates a complete source snapshot.

For sample `i`, the mixer outputs one probability for the self/null expert and
one probability for every candidate neighbour:

```text
a_i = softmax(l_self, l_i1, ..., l_ik)
g_i = 1 - a_i,self
x_i,mix = a_i,self x_i + sum_j a_ij x_j
```

Thus `g_i` is the node-level topology gate and `a_ij` are edge reliabilities.
The null expert lets the model close topology completely on datasets where
neighbour mixing is harmful.

## Training stages

1. Masked-autoencoder warmup with a data-appropriate likelihood.
2. Multi-start KMeans initialisation of a diagonal Student-t mixture head.
3. EMA teacher creation and construction of `raw-kNN union EMA-latent-kNN`
   candidates.
4. Joint optimisation with periodic graph refresh.  Candidate selection is an
   alternating discrete step; within each refresh interval, edge weights and
   the null expert remain fully differentiable.

The named objective is intentionally compact:

```text
L = L_rec + lambda_cls L_cls + lambda_graph L_graph
```

- `L_rec`: real-view likelihood plus the topology-mixture reconstruction view.
- `L_cls`: confidence-filtered teacher/student soft responsibility KL plus a
  weak learned-mixture-prior term.
- `L_graph`: KL from the learned self/edge expert posterior to a target built
  from the raw graph prior, teacher assignment agreement, and measured
  reconstruction-risk improvement.

No training function accepts ground-truth labels.  In benchmark mode `K` may
be obtained from `len(unique(y))`, but this is recorded as
`benchmark_oracle_from_y`; label-free use must pass `--n_clusters` explicitly.

## Run

```bash
python -m methods.TopoGate.V11.run \
  --data_path datasets/iris.npz \
  --save_dir result/V11/iris__seed42 \
  --config methods/TopoGate/V11/configs/topogate_v11.yaml \
  --seed 42 --gpu 1
```

CPU smoke test:

```bash
python -m methods.TopoGate.V11.run \
  --data_path datasets/iris.npz \
  --save_dir /tmp/topogate_v11_iris \
  --no_cuda --seed 42 \
  --set epochs=4 --set warmup_epochs=1 --set ramp_epochs=1
```

## Required ablations

All mechanisms are reversible configuration switches:

- no topology/null-only model: `use_topology=false`
- static candidates: `use_dynamic_graph=false`
- uniform candidate edges: `use_edge_reliability=false`
- no EMA teacher: `use_teacher=false`
- no end-to-end cluster head: `use_cluster_head=false`
- no topology reconstruction view: `use_mixed_reconstruction=false`
- no graph-posterior regularisation: `use_graph_prior=false`

The reversible `topogate_v11_semantic_metric.yaml` candidate additionally
enables a target-guided latent geometry term. It uses the detached
counterfactual edge distribution as a soft contrastive target over EMA
neighbour embeddings; the learned null/edge posterior is still trained by the
graph KL and remains the only source of topology mass.

V11.4 also exposes `topogate_v11_semantic_minimum.yaml`.  It uses the same
counterfactual probes but combines reconstruction-help and assignment-help by
their elementwise minimum.  This is a strict two-channel abstention rule:
strong assignment evidence cannot open a topology edge when the paired
reconstruction evidence is weak.  The older geometric and harmonic
combiners remain available through `semantic_help_combiner` for reversible
comparisons.

The main configuration must be fixed before final multi-seed evaluation; it
must not be selected separately on each reported dataset using labels.

## Sparse H0 TDA pilot

`tda_prior_mode=none` is the default and leaves the original V11 path
unchanged. The reversible `h0_mst` pilot computes exact H0 component deaths on
the fixed raw-kNN Vietoris--Rips 1-skeleton using a unit-row Euclidean
filtration and union-find. Only finite merge edges receive a bounded,
detached persistence prior; EMA-latent candidate edges are not silently
relabelled as TDA edges. `h0_early_mst` uses the same merge-edge mask but
reverses the distance ordering so early component merges receive more prior
mass than late bridge-like merges. `fixed_filtration` is a distance-only
control and `random` is a deterministic edge-shared control.

The prior is added to the existing detached graph-prior score through
`tda_prior_weight`. It does not change the encoder, mixture-head dimension,
loss definitions, or label boundary. This is sparse H0 persistence, not a
dense VR complex and not an H1 persistence diagram. Compare it against
`V11_full`, `V11_nomix`, `V11_tda_fixed_filtration`, `V11_tda_random`, and
`V11_tda_h0_early_mst` with the same seeds before considering any performance
claim.

## Output contract

- `embedding_final.npy`: final EMA/student embedding.
- `cluster_probabilities.npy`: Student-t mixture responsibilities.
- `predictions.npy`: primary cluster prediction.
- `labels_true.npy` and `label_mapping.json`: benchmark truth, stored under a
  distinct name and never consumed by the trainer.
- `metrics.json`, `args.json`, `summary.json`: metrics, resolved configuration,
  source SHA256, environment, loss/gate history, and graph-refresh history.
