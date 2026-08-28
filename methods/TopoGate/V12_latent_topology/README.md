# TopoGate V12 latent topology

V12 is an isolated architecture experiment. It does not change the V9
`learnable_gate` runner or the V10/V11 implementations.

## Training path

1. Build a fixed, label-free PCA-kNN graph.
2. Apply ordinary feature-mask corruption to an anchor and reconstruct the
   clean anchor with `AutoEncoder` (`mask_loss_weight=0.1` by default). The
   default decoder preserves the original scMAE
   `[latent, mask_logits] -> reconstruction` contract; `latent_only` is an
   explicit decoder ablation rather than the V12 default.
3. Encode the clean full dataset once per epoch as detached neighbour targets.
4. Predict a self/null mass plus one softmax weight per candidate edge with
   `LearnableGate`, then align the corrupted anchor latent to the weighted
   clean self/neighbour target. `edge_only` is retained as a registered
   ablation.

The default objective is additive,
`reconstruction_loss + 0.1 * mask_loss`; `legacy_weighted` remains available
only as a loss ablation. The edge-weight path remains differentiable. Only
clean self/neighbour latent values are detached; detaching the weights would
prevent the gate from learning. No NumPy sampling or input-space neighbour
mixing is used.

## Protocol boundary

Labels are used only to derive benchmark `K` and compute post-fit metrics.
They are never passed to graph construction, gate training, topology targets,
or model selection. This implementation is a latent alignment experiment,
not a claim of persistent homology, a VAE, or a probabilistic posterior.

Short smoke runs belong in `/tmp` and are engineering evidence only. Formal
claims require the project's multi-seed protocol and paired controls.

The registered stage-1 launcher covers flame and enron with NoMix,
edge-only, and three self/null topology strengths:

```bash
python scripts/V12/run_stage1.py
python scripts/V12/summarize_stage1.py \
  --input-dir result/V12/v12_self_null_stage1_2026-08-03_warmup_fix
```

The launcher default points to the current warmup-fixed evidence directory;
the earlier `v12_self_null_stage1_2026-08-03` directory is retained as a
pre-fix audit batch and should not be overwritten.

## Example

```bash
python methods/TopoGate/V12_latent_topology/run_npz.py \
  --data_path datasets/AHDPC/processed/flame.npz \
  --save_dir /tmp/topogate_v12_flame \
  --seed 42 --no_cuda --epochs 8
```
