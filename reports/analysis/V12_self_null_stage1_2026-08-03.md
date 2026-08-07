# V12 self/null stage-1 implementation and failure analysis

## Scope and provenance

This report audits the registered V12 latent-topology implementation after the
flame/enron stage-1 matrix. The exact output directory is
`result/V12/v12_self_null_stage1_2026-08-03/`, with 30 run summaries, source
SHA-256 values, resolved arguments, predictions, true labels, histories and
graph diagnostics. The matrix is 2 datasets x 5 variants x 3 seeds:

- NoMix;
- edge-only topology;
- self/null topology with lambda 0.01, 0.03 and 0.1.

All runs completed (`30/30`) without OOM or training exceptions. Labels were
used only to derive benchmark K and calculate post-fit metrics. Every summary
records `labels_used_during_fit=false`; labels were not passed to graph
construction, gate, topology loss, or variant selection.

## Protocol

The runner used the clean V12 path: StandardScaler, hidden size 128, mask ratio
0.3, batch size 256, 80 epochs, graph k=5, seeds `[42, 123, 7]`, and the
legacy mask-conditioned decoder. Mask loss used the additive objective
`reconstruction + 0.1 * mask`. The topology schedule was 20 warmup epochs and
10 ramp epochs. No `make_pseudo_batch` call or NumPy neighbour sampling occurs
in the training path.

## Aggregate metrics

The machine-readable tables are `runs.csv`, `summary_by_dataset.csv`,
`summary_by_variant.csv`, `paired_deltas.csv`, and
`summary_by_dataset_variant.csv` in the stage-1 directory. Mean ARI over the
six runs in each variant was:

| variant | ARI mean | ARI std | conditional edge entropy | effective neighbours |
|---|---:|---:|---:|---:|
| NoMix | 0.6616 | 0.2103 | not applicable | not applicable |
| edge-only | 0.2015 | 0.1739 | 1.2376 | 3.8517 |
| self/null lambda=0.01 | 0.6195 | 0.2592 | 1.6094 | 4.9999 |
| self/null lambda=0.03 | 0.3374 | 0.3293 | 1.6094 | 4.9998 |
| self/null lambda=0.1 | 0.1872 | 0.1849 | 1.6094 | 4.9998 |

The self/null conditional entropy is essentially `log(5)=1.60944`. Thus the
self branch learns a nonzero refusal mass, but the edge branch remains almost
uniform rather than selecting reliable neighbours. This satisfies the
self-fallback shape and gradient contracts but fails the stronger
selective-neighbour acceptance criterion.

## Dataset-specific outcome

On flame, NoMix ARI was `0.4729 +/- 0.0401`. Edge-only and self/null were lower:
`0.3556 +/- 0.0565`, `0.3916 +/- 0.0884`, `0.3806 +/- 0.0799`, and
`0.3521 +/- 0.0538` for edge-only and lambda 0.01/0.03/0.1 respectively.

On enron, NoMix was `0.8502 +/- 0.0463`. Self/null lambda 0.01 retained
`0.8475 +/- 0.0657`, within the pre-registered 0.03 ARI degradation boundary
in mean. Lambda 0.03 (`0.2942 +/- 0.5091`) and lambda 0.1
(`0.0224 +/- 0.0325`) were unstable or collapsed for one or more seeds.
Edge-only was `0.0475 +/- 0.0351`.

The paired deltas therefore do not support a default lambda=0.1 claim. The
registered lambda sweep is useful as a failure boundary: 0.01 is the only
tested self/null strength that preserves enron mean ARI, while it still
reduces flame relative to NoMix and does not produce selective edge entropy.

## Diagnosis

1. The decoder and additive mask-loss contracts are now correct and covered by
   tests. The old latent-only decoder regression is not present in this batch.
2. Self/null mass is nonzero and clean target tensors are detached, while gate
   parameters receive finite nonzero gradients. The gradient path is therefore
   operational.
3. The edge branch is not learning useful selection under the current
   objective. Self/null can increase its self mass, but the conditional edge
   distribution remains nearly uniform. This is a mechanism limitation, not
   evidence that topology is beneficial.
4. Larger topology strengths make the consistency target dominate enough to
   produce seed-sensitive embedding geometry on enron. The current schedule
   prevents an immediate topology update but does not prevent later
   representation drift.

## Decision boundary

The implementation is complete and reversible, but this stage-1 evidence is a
restricted no-go for the default self/null lambda=0.1 setting. No five-dataset
extension or paper performance claim should be made until a follow-up
ablation addresses edge selection and high-lambda instability. The edge-only,
self/null and NoMix variants remain available for controlled repair work.

This report makes no claim of persistent homology, a probabilistic model, or
universal topology superiority.
