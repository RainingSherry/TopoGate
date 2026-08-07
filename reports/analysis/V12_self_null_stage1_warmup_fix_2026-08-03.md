# V12 finalized-code stage-1 report

## Provenance

The immutable finalized-code benchmark is
`result/V12/v12_self_null_stage1_2026-08-03_warmup_fix/`. The earlier
`result/V12/v12_self_null_stage1_2026-08-03/` directory is preserved as a
pre-warmup-fix audit batch and is not overwritten. This report uses only the
finalized-code directory.

The matrix is flame/enron x NoMix/edge-only/self-null lambda
`{0.01,0.03,0.1}` x seeds `[42,123,7]`: 30/30 completed, with no OOM,
timeout, or training exception. All rows share one runner/model/gate source
hash, and all summaries record `labels_used_during_fit=false`. Labels were
used only for benchmark K and post-fit metrics.

## Fixed protocol

StandardScaler, hidden size 128, mask ratio 0.3, batch size 256, neighbor k=5,
80 epochs, additive `reconstruction + 0.1 * mask`, compatible
`[latent, mask_logits] -> Linear` decoder, topology warmup 20 epochs and
ramp 10 epochs. The training path is clean input plus mask corruption, Torch
gather and Torch gate aggregation; it does not call `make_pseudo_batch` or
NumPy sampling.

## Finalized aggregate

| variant | ARI mean +/- std | NMI mean +/- std | self mass | edge entropy | effective neighbours |
|---|---:|---:|---:|---:|---:|
| NoMix | 0.6616 +/- 0.2103 | 0.6130 +/- 0.2082 | 0 | n/a | n/a |
| edge-only | 0.2016 +/- 0.1739 | 0.1957 +/- 0.1438 | 0 | 1.2439 | 3.8527 |
| self/null lambda=0.01 | 0.6194 +/- 0.2592 | 0.5766 +/- 0.2495 | 0.8083 | 1.6094 | 4.9998 |
| self/null lambda=0.03 | 0.3372 +/- 0.3288 | 0.3194 +/- 0.2986 | 0.8114 | 1.6094 | 4.9997 |
| self/null lambda=0.1 | 0.1874 +/- 0.1848 | 0.1816 +/- 0.1535 | 0.8244 | 1.5858 | 4.8860 |

Dataset-level ARI means are:

- flame NoMix 0.4729; edge-only 0.3556; self/null 0.3916, 0.3806 and
  0.3521 for lambda 0.01, 0.03 and 0.1;
- enron NoMix 0.8502; edge-only 0.0476; self/null 0.8472, 0.2937 and
  0.0226 for lambda 0.01, 0.03 and 0.1.

## Interpretation

The warmup fix prevents gate drift before the topology ramp, but it does not
solve the main mechanism limitation. Self mass is nonzero, while conditional
edge entropy remains essentially `log(5)` for lambda 0.01/0.03 and is only
modestly lower for lambda 0.1. The gate therefore learns an abstention
probability more readily than a reliable per-edge ranking.

The high-dimensional enron result is stable only at lambda 0.01. Larger
topology strengths produce severe seed-sensitive degradation. Flame is below
NoMix for every topology condition. This is a restricted no-go for default
lambda 0.1 and does not justify a second-stage five-dataset expansion.

The code and tests satisfy the requested interfaces and gradient contracts;
the performance acceptance criterion does not. No claim is made about strict
TDA, probabilistic inference, or universal topology superiority.
