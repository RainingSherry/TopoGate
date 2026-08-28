# C2 Static Corruption Principle Matrix

Status: `simple_static_principle_sufficient`; completed-valid runs: `54/54`.

> Support in C2 denotes the frozen threshold-defined support of dense H0, not raw-X zero/nonzero support; raw sparse-support claims require a separate validation.

Primary endpoint: `Delta_P = ARI(P) - ARI(P0_Random)`; seeds are paired repeats and the dataset is the analysis unit.

## Dataset-level summary

| Dataset | Principle | ARI mean | Delta vs P0 | Support change | Value change | Sum abs delta | L_rec |
|---|---|---:|---:|---:|---:|---:|---:|
| Mouse_retina | P0_Random | 0.439834 | 0.000000 | 0.111820 | 0.157530 | 152650.087952 | 0.778043 |
| Mouse_retina | P1_SupportPreserve | 0.758878 | 0.319044 | 0.000000 | 0.269350 | 165309.486096 | 0.767260 |
| Mouse_retina | P2_SupportTarget | 0.834732 | 0.394898 | 0.269350 | 0.000000 | 212575.268988 | 0.811638 |
| Mouse_retina | P3_FrequencyAware | 0.827868 | 0.388034 | 0.000000 | 0.269350 | 97468.948344 | 0.747281 |
| Mouse_retina | P4_ResidualHard | 0.824561 | 0.384727 | 0.000000 | 0.269350 | 139642.260757 | 0.766597 |
| Mouse_retina | P5_GeometryHard | 0.556810 | 0.116976 | 0.000000 | 0.269350 | 221277.777918 | 0.771608 |
| Baron Human | P0_Random | 0.214700 | 0.000000 | 0.050331 | 0.028517 | 2538284.808991 | 0.785783 |
| Baron Human | P1_SupportPreserve | 0.088413 | -0.126288 | 0.000000 | 0.078848 | 5018453.603989 | 0.743769 |
| Baron Human | P2_SupportTarget | 0.340769 | 0.126069 | 0.078848 | 0.000000 | 7236594.662907 | 0.855477 |
| Baron Human | P3_FrequencyAware | 0.106423 | -0.108278 | 0.000000 | 0.078848 | 2094817.864469 | 0.740145 |
| Baron Human | P4_ResidualHard | 0.111966 | -0.102735 | 0.000000 | 0.078848 | 3774607.880347 | 0.743362 |
| Baron Human | P5_GeometryHard | 0.098281 | -0.116420 | 0.000000 | 0.078848 | 6187070.015883 | 0.748552 |
| Campbell | P0_Random | 0.023504 | 0.000000 | 0.071849 | 0.018587 | 173937.886369 | 0.771761 |
| Campbell | P1_SupportPreserve | 0.053393 | 0.029888 | 0.000000 | 0.090435 | 262141.994009 | 0.753937 |
| Campbell | P2_SupportTarget | 0.170387 | 0.146883 | 0.090435 | 0.000000 | 403414.967774 | 0.781154 |
| Campbell | P3_FrequencyAware | 0.069621 | 0.046116 | 0.000000 | 0.090435 | 61318.238750 | 0.745523 |
| Campbell | P4_ResidualHard | 0.106125 | 0.082621 | 0.000000 | 0.090435 | 150221.831533 | 0.753203 |
| Campbell | P5_GeometryHard | 0.037891 | 0.014387 | 0.000000 | 0.090435 | 367325.986413 | 0.757323 |

## Decision boundary

- Material descriptive margin: `0.03` ARI.
- Simple-principle candidates (material on at least two development datasets): `P2_SupportTarget, P3_FrequencyAware, P4_ResidualHard`.
- Dataset best arms: `{"Baron Human": {"delta_ARI_vs_P0": 0.12606882071727035, "material": true, "principle": "P2_SupportTarget"}, "Campbell": {"delta_ARI_vs_P0": 0.14688252130143517, "material": true, "principle": "P2_SupportTarget"}, "Mouse_retina": {"delta_ARI_vs_P0": 0.39489814411354435, "material": true, "principle": "P2_SupportTarget"}}`.
- Score representation caveat: P4 uses standardized clean H0 residuals frozen per dataset×seed; P5 uses raw clean H0 geometry scores frozen per dataset.
- Adaptive policy, GAN, learned generator and C3 holdout runs remain locked; any future unlock requires a new explicit protocol.

Raw score arrays, H0, labels, embeddings, predictions, checkpoints and logs remain local and are not publication artifacts.
