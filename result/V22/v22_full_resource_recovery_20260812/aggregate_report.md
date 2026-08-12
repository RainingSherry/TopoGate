# V22 Full Single-Seed Audit

- Manifest: `v22_full_resource_recovery_20260812_v1`
- Queue status: `completed`
- Jobs: `12`
- Status counts: `{'completed': 12}`
- Artifact audits passed: `12/12`
- Boundary: single-seed full-component evidence only; no efficacy claim or configuration selection.

| Dataset | Stratum | Status | Artifact | ARI | NMI | D steps | Gate updates | Gate nonzero |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sms_spam_collection | original8_shared_text | completed | True | -0.0400770576889054 | 0.007300336839417565 | 80 | 80 | 1.0 |
| cnae9 | original8_shared_text | completed | True | 0.29012931734840497 | 0.3709908697870251 | 80 | 80 | 1.0 |
| PBMC3k | new_scRNA_unlabelled | completed | True | None | None | 80 | 80 | 1.0 |
| sentiment_labeld_sentences | original8_shared_text | completed | True | 0.002628889371973654 | 0.005015747151415455 | 80 | 80 | 1.0 |
| hate_speech | original8_shared_text | completed | True | 0.04107525437748214 | 0.03653363856768208 | 80 | 80 | 1.0 |
| imdb | original8_shared_text | completed | True | 0.0011578750257372555 | 0.0010965956064424049 | 80 | 80 | 1.0 |
| sector | new_sparse_highdim | completed | True | 0.037159554403322276 | 0.31218507525015965 | 160 | 160 | 1.0 |
| Mouse_retina | original8_clubench_bridge | completed | True | 0.29468396185778056 | 0.3518757136858703 | 240 | 240 | 1.0 |
| Baron Human | original8_clubench_bridge | completed | True | 0.379623680421238 | 0.5836202371287048 | 240 | 240 | 1.0 |
| Campbell | original8_clubench_bridge | completed | True | 0.15762441068131408 | 0.322392689024346 | 240 | 240 | 1.0 |
| real-sim | new_sparse_highdim | completed | True | -0.004230323567852784 | 0.00013774986540045266 | 1440 | 1440 | 1.0 |
| covtype | new_dense_control | completed | True | 0.05582919893049488 | 0.14167066701184272 | 11360 | 11360 | 1.0 |

## Strata

- `new_dense_control`: {'completed_with_ari': 1, 'mean_ari': 0.05582919893049488}
- `new_sparse_highdim`: {'completed_with_ari': 2, 'mean_ari': 0.016464615417734745}
- `original8_clubench_bridge`: {'completed_with_ari': 3, 'mean_ari': 0.2773106843201109}
- `original8_shared_text`: {'completed_with_ari': 5, 'mean_ari': 0.05898285568693853}
