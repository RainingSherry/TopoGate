# TopoGate V10 Reliable Graph

`topogate_v10_reliable_graph` is the new canonical V10 implementation. It is
independent of the historical `learnable_gate_v10_nomix_init` experiment and
does not modify V1--V9 or any external baseline.

## Implemented training flow

1. Preprocess the feature matrix. Optional variance selection is performed
   before standardization. PCA is used only to initialize the input kNN graph.
2. Warm up a deterministic low-rank masked autoencoder with two independently
   corrupted views. Both views reconstruct the same clean sample and are
   aligned by latent cosine consistency.
3. At the first graph-training epoch, run KMeans on normalized clean EMA
   embeddings and use its centroids to initialize both the online and EMA
   prototype heads.
4. Build a latent-space kNN graph from the EMA encoder. The input and latent
   graphs form a candidate graph; edge recurrence is retained as evidence
   rather than treated as ground truth.
5. Predict one differentiable reliability value per candidate edge from five
   non-redundant features: cosine similarity, mutual-kNN, SNN, local-density
   compatibility, and input/latent recurrence stability.
6. Train the representation and gate jointly with assignment JS consistency,
   entropy balance, an upper-bound gate budget, and independent temporal
   recurrence supervision. Current input/latent recurrence remains a gate
   feature, but the BCE target comes only from the previous latent graph. A
   top-fraction curriculum uses a clean EMA encoder/prototype teacher, starting
   with the most confident edge pairs and gradually admitting more candidates.
7. Refresh the latent graph periodically. The final primary readout is KMeans
   on normalized EMA embeddings; EMA-prototype diagnostics are saved
   separately. Thus the prototype objective is a representation regularizer;
   V10 must not claim that `predictions.npy` is the prototype argmax readout.

The graph objective has one schedule only: zero during warmup and one linear
ramp afterwards. V10 does not reconstruct an anchor from a neighbor-mixed
input, and it does not claim a VAE, ELBO, Bayesian posterior, or unified
generative likelihood.

For graph construction, `knn_backend: auto` keeps deterministic exact cosine
search for up to `knn_exact_max_nodes` nodes (default 5000) and uses optional
FAISS HNSW above that threshold. The actual backend is recorded in the graph
profile. Approximate-search recall and end-to-end performance still require a
formal exact-versus-HNSW ablation before any scalability claim.

## Objective

The implemented objective is a deterministic multi-term objective:

```text
L = L_reconstruction
  + lambda_view * L_view_consistency
  + graph_scale * (
        lambda_edge * L_edge_assignment_JS
      + lambda_entropy * L_entropy_balance
      + lambda_gate_budget * L_gate_budget
      + lambda_gate_temporal * L_gate_temporal
    )
```

`graph_scale` is applied exactly once. By default, entropy balance matches a
smoothed cluster-size prior estimated from the unlabeled warmup KMeans
partition; `cluster_prior_mode: uniform` is available as an ablation. The edge
gate remains in the PyTorch autograd graph; it is not converted to NumPy or
detached into sample weights.

## Variants

- `configs/topogate_v10_reliable_graph.yaml`: full EMA dynamic-graph V10.
- `configs/topogate_v10_fixed_graph.yaml`: same objective with one fixed input
  graph, isolating the effect of dynamic refresh.
- `configs/topogate_v10_feature_only.yaml`: no graph or prototype objective;
  the strict feature-only/NoGraph control.

## Run

```bash
python methods/TopoGate/v10_reliable_graph/run.py \
  --data_path datasets/iris.npz \
  --save_dir result/v10_reliable_graph/example \
  --no_cuda
```

Programmatic API:

```python
from methods.TopoGate.v10_reliable_graph import run_v10

predictions, elapsed = run_v10(
    X,
    y=y,
    save_dir="result/v10_reliable_graph/example",
    no_cuda=True,
)
```

Multi-seed entry point:

```bash
python scripts/v10_reliable_graph/run_v10_multiseed.py \
  --datasets iris enron har \
  --seeds 42 123 7 \
  --worker_id 0
```

The project GPU pool is `[1, 4, 5]`; GPUs 0 and 7 are forbidden.
The runner writes per-run rows, `mean_std.csv`, and paired ARI deltas for full
minus fixed-graph / feature-only controls. Use `--resume` only when the saved
configuration is known to match the current code.

## Output contract

- `embedding_final.npy`: clean EMA embedding.
- `predictions.npy`: primary KMeans prediction.
- `labels_true.npy`: encoded ground truth, only when labels were supplied.
- `label_mapping.json`: mapping from encoded labels to original classes.
- `cluster_probabilities.npy`: EMA-prototype probabilities when the prototype
  head was initialized; absent in the feature-only control.
- `final_graph_edges.npz`: final candidate sources, targets, full-graph gates,
  current input/latent stability, and independent temporal targets.
- `history.json`: losses, schedule, confidence acceptance, and gate diagnostics.
- `graph_history.json`: graph refresh and recurrence diagnostics.
- `summary.json`: resolved protocol, known-K source, parameter profile, metrics,
  and output paths.

## Validation boundary

Unit/integration tests and short real-data smoke tests verify the computation
and artifact contract. They do not establish a performance improvement. Any
paper claim comparing V10 with V9, fixed graph, or feature-only controls must
use at least five core datasets and at least three seeds, reporting mean and
standard deviation.
