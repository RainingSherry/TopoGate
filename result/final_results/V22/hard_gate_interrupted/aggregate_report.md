# V22 Full Single-Seed Audit

- Manifest: `v22_full_single_seed_20260812_v1`
- Queue status: `interrupted`
- Jobs: `12`
- Status counts: `{'completed': 10, 'incomplete_compute': 2}`
- Artifact audits passed: `10/12`
- Boundary: single-seed full-component evidence only; no efficacy claim or configuration selection.

| Dataset | Stratum | Status | Artifact | ARI | NMI | D steps | Gate updates | Gate nonzero |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sms_spam_collection | original8_shared_text | completed | True | 0.38626186164732873 | 0.2867946497740293 | 560 | 560 | 1.0 |
| cnae9 | original8_shared_text | completed | True | 0.6347143930877188 | 0.6923878206303827 | 720 | 720 | 1.0 |
| PBMC3k | new_scRNA_unlabelled | completed | True | None | None | 1760 | 1760 | 1.0 |
| sentiment_labeld_sentences | original8_shared_text | completed | True | 0.0024775931507336657 | 0.004765131429461572 | 1760 | 1760 | 1.0 |
| hate_speech | original8_shared_text | completed | True | 0.04350075961886216 | 0.02472955300633038 | 2080 | 2080 | 1.0 |
| imdb | original8_shared_text | completed | True | -0.0002466721946324173 | 3.2543967966800175e-06 | 2080 | 2080 | 1.0 |
| sector | new_sparse_highdim | completed | True | 0.06637041783719777 | 0.3891328318928151 | 4080 | 4080 | 1.0 |
| Mouse_retina | original8_clubench_bridge | completed | True | 0.2901214038069284 | 0.5030165532113205 | 5280 | 5280 | 1.0 |
| Baron Human | original8_clubench_bridge | completed | True | 0.23055338936520237 | 0.4209614382090735 | 5360 | 5360 | 1.0 |
| Campbell | original8_clubench_bridge | completed | True | 0.17294485156816053 | 0.3194005429024996 | 6320 | 6320 | 1.0 |
| real_sim__libsvm_sparse_highdim | new_sparse_highdim | incomplete_compute | False | None | None | None | None | None |
| covtype__libsvm_dense_control | new_dense_control | incomplete_compute | False | None | None | None | None | None |

## Strata

- `new_sparse_highdim`: {'completed_with_ari': 1, 'mean_ari': 0.06637041783719777}
- `original8_clubench_bridge`: {'completed_with_ari': 3, 'mean_ari': 0.2312065482467638}
- `original8_shared_text`: {'completed_with_ari': 5, 'mean_ari': 0.2133415870620022}
