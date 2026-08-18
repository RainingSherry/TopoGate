# B1 corruption library results

Formal matrix: 6 datasets × 6 arms × 3 paired seeds = 108/108 completed-valid; positive-control status=`completed_valid`. The published matrix is the fresh pair-feasible rerun; the earlier support-budget-mismatch attempt is quarantined and not aggregated.

Terminal decision: `simple_corruption_principle_sufficient`. B2 is authorized: `False`.

This is a bounded mechanism panel under the frozen S0 H0 small reconstruction probe. It does not support a holdout/generalization claim.

| dataset | ARI(clean) | ARI(C0) | Δclean(C0) | best structured arm | Δrandom(best) |
|---|---:|---:|---:|---|---:|
| sms_spam_collection | 0.161677 | 0.150478 | -0.011199 | C3_MixedMatched | +0.011397 |
| hate_speech | 0.002692 | 0.013448 | +0.010756 | C4_StaticHard | -0.005331 |
| Mouse_retina | 0.754456 | 0.828638 | +0.074182 | C3_MixedMatched | +0.019242 |
| Baron Human | 0.144447 | 0.129276 | -0.015171 | C2_SupportOnly | +0.253983 |
| cnae9 | 0.038605 | 0.039600 | +0.000995 | C3_MixedMatched | +0.000395 |
| Campbell | 0.042775 | 0.083814 | +0.041038 | C4_StaticHard | +0.160710 |

Level 1 asks whether corruption changes clustering relative to clean; the primary C0 contrast is material for Mouse_retina (`+0.074182`) and Campbell (`+0.041038`), while the best structured corruption also produces a material change for Baron Human. Level 2 compares structured arms with matched-random C0: C2, C3 and C4 each have material positive Δrandom on two registered-scRNA datasets. Level 3 requires distinct material winners across at least two coarse role classes before B2; only the registered-scRNA role supplies a material role winner, so adaptive location/generator work is not justified.

The fresh rerun also audited the matched budget directly: for every dataset×seed, C0/C1/C2/C3/C4 have identical `effective_changed_coordinate_rate_mean` (the clean arm is the zero-change floor). All 108 runs record `labels_used_during_fit=false`; labels are used only for the outer benchmark-known-K readout and post-fit metrics.

All six arms record requested/effective changed-coordinate, support-change, value-change and total-absolute-change fields. Reconstruction loss is a monitor only. Raw H0-derived arrays, embeddings, predictions, checkpoints and logs remain local and are excluded from publication.
