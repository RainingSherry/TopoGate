# TopoGate V21: assignment-adversarial feature Gate

V21 is an independent successor to V20. It preserves the original random-donor
scMAE reconstruction branch and moves the adversarial objective from unbounded
reconstruction MSE to bounded cluster-assignment divergence.

## Frozen variants

| Variant | scMAE reconstruction | Student-t head + InfoMax | Assignment consistency | Topology Gate |
|---|---:|---:|---:|---:|
| `scmae_only` | yes | no | no | no |
| `random_assignment_control` | yes | yes | random effective mask | no |
| `topology_assignment_adversarial` | yes | yes | topology adversarial mask | yes |

The two assignment variants require `n_clusters` as a protocol input. They do
not receive labels. In benchmark mode the outer runner may derive known K from
`y`; that fact is recorded as `K_source=benchmark_oracle_from_y` and
`K_used_during_fit=true`. Truly unlabeled inputs must pass `--n-clusters`.

## Losses

The model minimizes random scMAE reconstruction, clean/adversarial Jensen-Shannon
assignment divergence, and IMSAT-style information maximization. The Gate
maximizes the same bounded assignment divergence while a coverage penalty
discourages repeatedly selecting the same feature subset.

The assignment mask budget is 40% of positions where the selected donor value
actually differs from the anchor. This makes every selected assignment-attack
position effective by construction. The implementation separately records its
rate over all features; it never reports 40% as a global effective-change rate
when sparse inputs cannot support that value.

## Literature basis

- IMSAT, ICML 2017: virtual adversarial assignment consistency and InfoMax.
- ALRDC, NeurIPS 2020: attack clustering outputs while preserving embedding reconstruction.
- Chhabra et al., NeurIPS 2022: soft cluster-membership attack target.
- AR-DMVC, ICML 2024: clean/adversarial assignment consistency in a multi-view setting.
- scAGCL, Briefings in Bioinformatics 2025: graph and feature perturbations for scRNA-seq.

The first three are direct assignment-level motivation. AR-DMVC and scAGCL are
adjacent evidence and are not claimed as implementations reproduced by V21.

The CLI requires an explicit physical `--gpu` for CUDA and accepts only GPUs
1--6. After `CUDA_VISIBLE_DEVICES` isolation, V21 seeds only the selected
logical CUDA device; CPU runs do not touch CUDA RNGs.

## Readout-fix protocol (v3)

The original v2 protocol keeps the Student-t head as its primary final readout.
The formal six-dataset audit showed that this head often produced empty clusters;
on the same saved Full embeddings, a fresh known-K KMeans readout increased the
macro ARI from `0.2077` to `0.3841` without using labels for fitting or readout.

The v3 configuration therefore keeps the Student-t head as a differentiable
training surrogate and diagnostic, but uses `kmeans_embedding` on the final clean
embedding as the primary label-free readout. It writes both
`predictions.npy` and `student_t_predictions.npy`, plus `readout_profile.json`.
The v2 configs and result roots remain historical and are not overwritten.

## Extension panel

`scripts/V21/build_extended_manifest.py` freezes the 13-dataset transfer panel at
`result/V21/v21_extended13_readoutfix_manifest_20260811.json`. The panel reuses
audited local sources and is selected independently of V21/V19 outcomes. Run
keys, source provenance, label isolation, and incomplete-compute state are
managed by `scripts/V21/run_extended_matrix.py`; its default matrix is 13
datasets x 2 variants x 3 seeds = 78 runs. Use an explicit `--gpu` in 1--6 or
`--cpu`; `--dry-run` expands the fixed job set without training.

`scripts/V21/summarize_extended_matrix.py` rejects missing artifacts, non-finite
values, empty primary clusters, protocol mismatches, and label-boundary
violations. It accepts `--datasets` and `--seeds` for bounded engineering-smoke
audits; a full matrix summary remains the only source for multi-seed extension
claims.
