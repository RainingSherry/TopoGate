# V13 Gumbel-Top-k experiment report

- Expected runs: 30
- Completed: 30
- Failed: 0
- Datasets: balance_scale, enron, flame, spect_heart, vehicle
- Variants: nomix, topk2

## Headline diagnostic: effective_neighbor_count vs top_k=2

The hard gate should produce `effective_neighbor_count ≈ 2.0` at inference.
Values significantly below 2.0 indicate the gate is not selecting;
values above 2.0 indicate the gate is using soft relaxation at eval time.

| dataset | variant | eff_neigh mean | eff_neigh std | topology_loss |
|---|---|---:|---:|---:|
| balance_scale | nomix | 0.0000 | 0.00000 |
| balance_scale | topk2 | 2.0000 | 0.02107 |
| enron | nomix | 0.0000 | 0.00000 |
| enron | topk2 | 2.0000 | 0.07633 |
| flame | nomix | 0.0000 | 0.00000 |
| flame | topk2 | 2.0000 | 0.01963 |
| spect_heart | nomix | 0.0000 | 0.00000 |
| spect_heart | topk2 | 2.0000 | 0.09160 |
| vehicle | nomix | 0.0000 | 0.00000 |
| vehicle | topk2 | 2.0000 | 0.07088 |

## ARI mean ± std (across all seeds)

| dataset | nomix ARI | topk2 ARI | delta |
|---|---:|---:|---:|
| balance_scale | 0.1163 ± 0.0392 | 0.1394 ± 0.0157 | +0.0231 |
| enron | 0.8026 ± 0.1041 | 0.0716 ± 0.0061 | -0.7310 |
| flame | 0.3897 ± 0.1092 | 0.3058 ± 0.0721 | -0.0839 |
| spect_heart | -0.0264 ± 0.0302 | -0.0106 ± 0.0152 | +0.0158 |
| vehicle | 0.0780 ± 0.0017 | 0.0761 ± 0.0026 | -0.0019 |

## Interpretation

Use `paired_deltas_vs_nomix.csv` for seed-matched ARI comparisons.
Positive delta > 0.03 ARI is evidence for a real improvement;
values in [-0.03, 0.03] are within the documented noise band.
effective_neighbor_count far from 2.0 indicates the gate has not
learned to make hard selections.
