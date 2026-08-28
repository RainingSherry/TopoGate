# V22 Full Single-Seed Audit

- Manifest: `v22_full_cooperative_keep_single_seed_20260812_v1`
- Queue status: `interrupted`
- Jobs: `16`
- Status counts: `{'completed': 14, 'incomplete_compute': 2}`
- Artifact audits passed: `14/16`
- Boundary: single-seed full-component evidence only; no efficacy claim or configuration selection.

| Dataset | Stratum | Status | Artifact | ARI | NMI | D steps | Gate updates | Gate nonzero |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sms_spam_collection | original8_shared_text | completed | True | -0.050867540837288595 | 0.013330032632676401 | 80 | 80 | 1.0 |
| cnae9 | original8_shared_text | completed | True | 0.20422926076666564 | 0.36557152036037455 | 80 | 80 | 1.0 |
| pbmc_1k_v3 | scRNA_count_unlabelled | completed | True | None | None | 80 | 80 | 1.0 |
| PBMC3k | new_scRNA_unlabelled | completed | True | None | None | 80 | 80 | 1.0 |
| sentiment_labeld_sentences | original8_shared_text | completed | True | 0.002705169715497749 | 0.005093978827066298 | 80 | 80 | 1.0 |
| hate_speech | original8_shared_text | completed | True | 0.05196627904057691 | 0.03445128354796639 | 80 | 80 | 1.0 |
| imdb | original8_shared_text | completed | True | -0.00016063445278214353 | 0.0001033760014482029 | 80 | 80 | 1.0 |
| sector | new_sparse_highdim | completed | True | 0.023828056115724767 | 0.25993928871692007 | 160 | 160 | 1.0 |
| Mouse_retina | original8_clubench_bridge | completed | True | 0.3952257074503743 | 0.5474015102883393 | 240 | 240 | 1.0 |
| Baron Human | original8_clubench_bridge | completed | True | 0.3161405000263956 | 0.5129813418661514 | 240 | 240 | 1.0 |
| Campbell | original8_clubench_bridge | completed | True | 0.13552117099423747 | 0.30764891286128765 | 240 | 240 | 1.0 |
| news20 | sparse_highdim_text | completed | True | 0.018747243165114162 | 0.055259831759399324 | 320 | 320 | 1.0 |
| rcv1_train | sparse_highdim_text | completed | True | 0.0012516907472253744 | 0.0007115587692821434 | 400 | 400 | 1.0 |
| mnist | dense_image_control | completed | True | 0.3147581680431644 | 0.41002828909204914 | 1200 | 1200 | 1.0 |
| real_sim__libsvm_sparse_highdim | new_sparse_highdim | incomplete_compute | False | None | None | None | None | None |
| covtype__libsvm_dense_control | new_dense_control | incomplete_compute | False | None | None | None | None | None |

## Strata

- `dense_image_control`: {'completed_with_ari': 1, 'mean_ari': 0.3147581680431644}
- `new_sparse_highdim`: {'completed_with_ari': 1, 'mean_ari': 0.023828056115724767}
- `original8_clubench_bridge`: {'completed_with_ari': 3, 'mean_ari': 0.2822957928236691}
- `original8_shared_text`: {'completed_with_ari': 5, 'mean_ari': 0.04157450684653391}
- `sparse_highdim_text`: {'completed_with_ari': 2, 'mean_ari': 0.009999466956169769}
