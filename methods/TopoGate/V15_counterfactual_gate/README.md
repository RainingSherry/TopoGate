# TopoGate V15 Counterfactual Gate

TopoGate is prototyped from `scMAE`. The shared research goal is reliable
clustering on single-view data that are simultaneously high-dimensional,
feature-noisy, and naturally sparse. V15 is one exploratory attempt toward
that goal, not a model tied to a separate application scenario or a
preselected final paper identity. The V-series are kept as traceable
research branches; only one generation will eventually be selected as the
paper's external TopoGate method.

V15 is an isolated exploratory implementation. It treats the topology gate as
an abstaining edge utility model rather than as a hand-composed distance
reliability function. Dataset stratification may consult `hj-n/labeled-datasets`,
`hj-n/clm`, and
`papers/参考资料/Measuring_the_Validity_of_Clustering_Validation_Datasets.md`;
these references must not enter fitting, graph construction, gate/loss
optimization, or variant selection. Until the external repositories and local
mapping are fixed and reverified, CLM evidence remains `CLM-unranked`.

The trainer does not accept labels. A benchmark runner may derive `K` from
labels and may compute external metrics after fitting, but every run records
`labels_used_during_fit: false`.

The canonical path is:

1. sparse-aware anchor masked autoencoder;
2. raw sparse kNN union EMA-latent kNN candidate graph;
3. detached single-edge counterfactual utility evaluated on the exact clean
   assignment intervention used by the output;
4. sparsemax over one null/self branch plus edges;
5. the same assignment-space intervention for utility estimation and final
   readout.

The six-feature utility scorer is used only by the explicit
`counterfactual_learned` amortisation mode. It is not part of the exact
`direct_counterfactual` decision because the detached counterfactual utility is
already available at inference time.
The earlier masked-probe reference remains available through
`direct_utility_source=masked_probe` as a feature-noise mechanism ablation.

`gate_mode=direct_counterfactual` with `final_prediction_source=gate_readout`
is the exact-readout mechanism control. For a trainable counterfactual gate,
use `gate_mode=counterfactual_learned` with a positive `lambda_gate`: it keeps
the detached ExactCF utility as the target, fits the amortized utility scorer
only on the utility-training split, and reports a held-out utility loss without
using labels. The exact mode remains available to separate utility quality from
scorer generalisation. The formal paired controls are `self_only`
(V15 NoMix), `union_uniform`, `forced_topk`, and `shuffled_utility`.
The launcher also separates `direct_local_consensus` (exact
leave-one-candidate-out readout) from `counterfactual_learned` (the same
detached target amortised by the six-feature scorer), so target quality and
scorer generalisation are not conflated.
`output_disabled` is redundant with `self_only` when distillation is disabled,
so it is a unit-level contract check rather than a formal benchmark variant.
`graph_replacement_fraction` is a stress-test hook and is zero by default.

Stage-0/Stage-1 tooling is under `scripts/V15/`:

- `build_dataset_manifest.py` records sparse/density/graph audit fields and
  SHA256 provenance;
- `audit_dataset_mapping.py` maps local NPZ names to the unverified local
  CLUBench table;
- `run_stage1.py` and `analyze_stage1.py` run and audit the mechanism panel;
- `audit_stage1b_certificates.py` independently audits teacher, candidate-graph,
  and utility certificates without using labels for fitting;
- `run_stage2.py` launches the eligible single-seed corruption exploration;
- `run_corruption_diagnostics.py` covers feature mask, heavy-tail noise, row
  contamination, random graph replacement, and compound stress;
- `run_formal.py` is the explicit Stage-3 multi-seed launcher and supports
  `--dry-run` before any large execution; `summarize_formal.py` computes paired
  deltas and aggregate metrics.

The current short Stage-1 panel is restricted evidence rather than a go
signal: utility/candidate thresholds were not met, so the formal matrix is
paused until the mechanism is revised or a longer preregistered Stage-1 run
changes that conclusion.

The Stage-1B audit deliberately distinguishes three claims. Existing V15 runs
can provide post-hoc graph purity/recall and same-run utility diagnostics, but
they do not persist teacher assignments/embeddings, held-out scorer
predictions, or independent per-edge downstream gains. The audit therefore
reports those certificates as `not_available` rather than inferring them from
EMA existence or in-sample AUROC.

All outputs belong under `result/`; this directory must not be used to alter
historical V2-V13 implementations.
