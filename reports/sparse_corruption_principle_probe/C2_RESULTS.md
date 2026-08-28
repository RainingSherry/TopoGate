# C2 Static Corruption Principle Matrix

Status: `simple_static_principle_sufficient`; completed-valid runs: `54/54`.

> Support in C2 denotes the frozen threshold-defined support of dense H0, not raw-X zero/nonzero
> support; raw sparse-support claims require a separate validation.

Primary endpoint: `Delta_P = ARI(P) - ARI(P0_Random)`. Seeds `[42, 123, 7]` are paired repeats;
the dataset is the analysis unit. The three datasets are the predeclared development/mechanism
panel, not a generalization denominator.

## Dataset-level primary result

| Dataset | P1 | P2 | P3 | P4 | P5 | Best |
|---|---:|---:|---:|---:|---:|---|
| Mouse_retina | +0.319044 | **+0.394898** | +0.388034 | +0.384727 | +0.116976 | P2_SupportTarget |
| Baron Human | −0.126288 | **+0.126069** | −0.108278 | −0.102735 | −0.116420 | P2_SupportTarget |
| Campbell | +0.029888 | **+0.146883** | +0.046116 | +0.082621 | +0.014387 | P2_SupportTarget |

Material descriptive margin: `0.03 ARI`.

| Principle | Material datasets (`Delta_ARI >= 0.03`) |
|---|---:|
| P1_SupportPreserve | 1/3 |
| P2_SupportTarget | **3/3** |
| P3_FrequencyAware | 2/3 |
| P4_ResidualHard | 2/3 |
| P5_GeometryHard | 1/3 |

P2 is the only material winner on all three predeclared datasets. The frozen descriptive terminal
label is therefore `simple_static_principle_sufficient`; this does not authorize an adaptive policy
or imply that P2 is an oracle upper bound.

## Secondary corruption diagnostics

Each run also records support-change rate, value-change rate, total absolute change and reconstruction
loss (`L_rec`). These are diagnostics for interpreting ARI and are not additional promotion criteria.
The compact values are in `C2_DATASET_SUMMARY.csv` and `C2_PRINCIPLE_SUMMARY.csv`.

P4 uses standardized clean-H0 residual scores frozen per dataset×seed. P5 uses raw clean-H0 local
geometry scores frozen per dataset. GeometrySafe remains a secondary fixture and was not promoted to a
seventh primary arm.

## Locked follow-up and claim boundary

- C3 holdout runs: locked.
- Adaptive policy, GAN and learned generator: locked.
- Raw X, labels, H0, corruption arrays, score arrays, embeddings, predictions, checkpoints and logs:
  local/external only; not publication artifacts.
- This result supports only a tested static-library development-panel finding. It does not establish
  raw sparse-support semantics, universal transfer, or a learned adaptive method.
