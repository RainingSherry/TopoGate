# V19 sparse/high-dimensional extension

- Matrix audit: `audit_ok` (78/78 runs).
- Dataset selection was fixed before extension outcomes; labels were not used during fitting or input selection.
- `rg_win_by_mean_ari` is an evaluation summary. It does not authorize changing the candidate panel.

| Dataset | RG ARI | scMAE ARI | Delta ARI | Positive seeds | All seeds positive |
|---|---:|---:|---:|---:|---|
| internet_advertisements__uci_sparse | -0.0603 | -0.0725 | +0.0123 | 2/3 | no |
| gina_prior2__local_sparse_highdim | 0.3620 | 0.3547 | +0.0073 | 1/3 | no |
| tr45_wc__local_sparse_text | 0.0092 | 0.0045 | +0.0047 | 3/3 | yes |
| dexter__uci_sparse_highdim | 0.0050 | 0.0009 | +0.0041 | 2/3 | no |
| madelon__uci_highdim_control | 0.0283 | 0.0258 | +0.0025 | 3/3 | yes |
| dorothea__uci_sparse_highdim | -0.0835 | -0.0844 | +0.0009 | 2/3 | no |
| arcene__uci_highdim | 0.0977 | 0.0977 | +0.0000 | 0/3 | no |
| micro_mass__local_sparse_highdim | 0.4944 | 0.5017 | -0.0073 | 0/3 | no |
| quake_smartseq2_lung__local_sparse_expression | 0.1696 | 0.1791 | -0.0095 | 1/3 | no |
| sms_spam_full__uci_sparse_text | 0.8614 | 0.8757 | -0.0142 | 0/3 | no |
| fbis_wc__local_sparse_text | 0.2906 | 0.3093 | -0.0187 | 0/3 | no |
| gisette__uci_highdim_dense | 0.0723 | 0.0968 | -0.0245 | 1/3 | no |
| fabert__local_sparse_text | 0.0328 | 0.0788 | -0.0460 | 0/3 | no |
