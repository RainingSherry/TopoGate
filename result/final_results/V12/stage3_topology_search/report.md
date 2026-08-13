# V12 stage-3 topology-signal grid report
This report summarizes the V12 stage-3 grid search. The grid amplifies
the topology signal by sweeping `lambda_topology`, `rank_margin`, and
`self_init_weight` (self_null only). Edge entropy is the headline metric:
the goal is to push conditional edge entropy below 1.0 (effective
neighbors < 3) and below `log(5) ≈ 1.6094` (effective neighbors < 5).

- Expected runs: 144
- Completed runs: 144
- Failed/incomplete records: 0
- Configs present: edge_only_lam0.3_rm0.5, edge_only_lam0.3_rm1.0, edge_only_lam0.5_rm0.5, edge_only_lam0.5_rm1.0, self_null_lam0.3_rm0.5_si0.3, self_null_lam0.3_rm0.5_si0.5, self_null_lam0.3_rm1.0_si0.3, self_null_lam0.3_rm1.0_si0.5, self_null_lam0.5_rm0.5_si0.3, self_null_lam0.5_rm0.5_si0.5, self_null_lam0.5_rm1.0_si0.3, self_null_lam0.5_rm1.0_si0.5
- Datasets present: balance_scale, flame, spect_heart, vehicle
- (dataset, config) cells with edge entropy < log(5): 48
- (dataset, config) cells with edge entropy < 1.0: 0

## Configuration count

- self_null: 2 (lambda) × 2 (rank_margin) × 2 (self_init) = 8 configs
- edge_only: 2 (lambda) × 2 (rank_margin) = 4 configs
- Total configs: 12

## Edge-entropy diagnostic (headline metric)

| dataset | config | edge_entropy | effective_neighbors | rank_loss | < log(5) | < 1.0 |
|---|---|---:|---:|---:|---:|---:|
| balance_scale | edge_only_lam0.3_rm0.5 | 1.4806 | 4.4173 | 0.1667 | yes | no |
| balance_scale | edge_only_lam0.3_rm1.0 | 1.3983 | 4.1011 | 0.3938 | yes | no |
| balance_scale | edge_only_lam0.5_rm0.5 | 1.4805 | 4.4169 | 0.1667 | yes | no |
| balance_scale | edge_only_lam0.5_rm1.0 | 1.3984 | 4.1015 | 0.3938 | yes | no |
| balance_scale | self_null_lam0.3_rm0.5_si0.3 | 1.4807 | 4.4176 | 0.1667 | yes | no |
| balance_scale | self_null_lam0.3_rm0.5_si0.5 | 1.4807 | 4.4177 | 0.1667 | yes | no |
| balance_scale | self_null_lam0.3_rm1.0_si0.3 | 1.3981 | 4.1003 | 0.3938 | yes | no |
| balance_scale | self_null_lam0.3_rm1.0_si0.5 | 1.3980 | 4.1001 | 0.3938 | yes | no |
| balance_scale | self_null_lam0.5_rm0.5_si0.3 | 1.4805 | 4.4169 | 0.1667 | yes | no |
| balance_scale | self_null_lam0.5_rm0.5_si0.5 | 1.4806 | 4.4172 | 0.1667 | yes | no |
| balance_scale | self_null_lam0.5_rm1.0_si0.3 | 1.3981 | 4.1003 | 0.3938 | yes | no |
| balance_scale | self_null_lam0.5_rm1.0_si0.5 | 1.3980 | 4.1001 | 0.3938 | yes | no |
| flame | edge_only_lam0.3_rm0.5 | 1.5914 | 4.9113 | 0.3136 | yes | no |
| flame | edge_only_lam0.3_rm1.0 | 1.5866 | 4.8886 | 0.6843 | yes | no |
| flame | edge_only_lam0.5_rm0.5 | 1.5914 | 4.9113 | 0.3136 | yes | no |
| flame | edge_only_lam0.5_rm1.0 | 1.5866 | 4.8886 | 0.6843 | yes | no |
| flame | self_null_lam0.3_rm0.5_si0.3 | 1.5914 | 4.9113 | 0.3136 | yes | no |
| flame | self_null_lam0.3_rm0.5_si0.5 | 1.5914 | 4.9113 | 0.3136 | yes | no |
| flame | self_null_lam0.3_rm1.0_si0.3 | 1.5866 | 4.8886 | 0.6843 | yes | no |
| flame | self_null_lam0.3_rm1.0_si0.5 | 1.5866 | 4.8885 | 0.6843 | yes | no |
| flame | self_null_lam0.5_rm0.5_si0.3 | 1.5914 | 4.9113 | 0.3136 | yes | no |
| flame | self_null_lam0.5_rm0.5_si0.5 | 1.5914 | 4.9113 | 0.3136 | yes | no |
| flame | self_null_lam0.5_rm1.0_si0.3 | 1.5866 | 4.8885 | 0.6843 | yes | no |
| flame | self_null_lam0.5_rm1.0_si0.5 | 1.5866 | 4.8886 | 0.6843 | yes | no |
| spect_heart | edge_only_lam0.3_rm0.5 | 1.5308 | 4.6338 | 0.1918 | yes | no |
| spect_heart | edge_only_lam0.3_rm1.0 | 1.4666 | 4.3691 | 0.4389 | yes | no |
| spect_heart | edge_only_lam0.5_rm0.5 | 1.5303 | 4.6316 | 0.1920 | yes | no |
| spect_heart | edge_only_lam0.5_rm1.0 | 1.4696 | 4.3807 | 0.4393 | yes | no |
| spect_heart | self_null_lam0.3_rm0.5_si0.3 | 1.5306 | 4.6330 | 0.1914 | yes | no |
| spect_heart | self_null_lam0.3_rm0.5_si0.5 | 1.5306 | 4.6331 | 0.1915 | yes | no |
| spect_heart | self_null_lam0.3_rm1.0_si0.3 | 1.4592 | 4.3409 | 0.4383 | yes | no |
| spect_heart | self_null_lam0.3_rm1.0_si0.5 | 1.4588 | 4.3394 | 0.4384 | yes | no |
| spect_heart | self_null_lam0.5_rm0.5_si0.3 | 1.5304 | 4.6320 | 0.1916 | yes | no |
| spect_heart | self_null_lam0.5_rm0.5_si0.5 | 1.5308 | 4.6339 | 0.1916 | yes | no |
| spect_heart | self_null_lam0.5_rm1.0_si0.3 | 1.4603 | 4.3451 | 0.4384 | yes | no |
| spect_heart | self_null_lam0.5_rm1.0_si0.5 | 1.4595 | 4.3421 | 0.4384 | yes | no |
| vehicle | edge_only_lam0.3_rm0.5 | 1.2522 | 3.5912 | 0.1783 | yes | no |
| vehicle | edge_only_lam0.3_rm1.0 | 1.2329 | 3.5259 | 0.4336 | yes | no |
| vehicle | edge_only_lam0.5_rm0.5 | 1.1970 | 3.4326 | 0.1779 | yes | no |
| vehicle | edge_only_lam0.5_rm1.0 | 1.1956 | 3.4165 | 0.4327 | yes | no |
| vehicle | self_null_lam0.3_rm0.5_si0.3 | 1.3236 | 3.8208 | 0.1793 | yes | no |
| vehicle | self_null_lam0.3_rm0.5_si0.5 | 1.3233 | 3.8197 | 0.1793 | yes | no |
| vehicle | self_null_lam0.3_rm1.0_si0.3 | 1.2577 | 3.6024 | 0.4340 | yes | no |
| vehicle | self_null_lam0.3_rm1.0_si0.5 | 1.2572 | 3.6007 | 0.4339 | yes | no |
| vehicle | self_null_lam0.5_rm0.5_si0.3 | 1.3228 | 3.8181 | 0.1794 | yes | no |
| vehicle | self_null_lam0.5_rm0.5_si0.5 | 1.3212 | 3.8127 | 0.1792 | yes | no |
| vehicle | self_null_lam0.5_rm1.0_si0.3 | 1.2572 | 3.6007 | 0.4341 | yes | no |
| vehicle | self_null_lam0.5_rm1.0_si0.5 | 1.2563 | 3.5979 | 0.4339 | yes | no |

## Per-config mean ARI (across all 4 datasets × 3 seeds)

| config | ARI mean ± std | edge_entropy mean | effective_neighbors mean | rank_loss mean |
|---|---:|---:|---:|---:|
| edge_only_lam0.3_rm0.5 | 0.1849 ± 0.1990 | 1.4637 | 4.3884 | 0.2126 |
| edge_only_lam0.3_rm1.0 | 0.1848 ± 0.1990 | 1.4211 | 4.2212 | 0.4876 |
| edge_only_lam0.5_rm0.5 | 0.1833 ± 0.1985 | 1.4498 | 4.3481 | 0.2125 |
| edge_only_lam0.5_rm1.0 | 0.1847 ± 0.1967 | 1.4126 | 4.1968 | 0.4875 |
| self_null_lam0.3_rm0.5_si0.3 | 0.1870 ± 0.1986 | 1.4816 | 4.4457 | 0.2128 |
| self_null_lam0.3_rm0.5_si0.5 | 0.1872 ± 0.1983 | 1.4815 | 4.4454 | 0.2128 |
| self_null_lam0.3_rm1.0_si0.3 | 0.1869 ± 0.1987 | 1.4254 | 4.2330 | 0.4876 |
| self_null_lam0.3_rm1.0_si0.5 | 0.1870 ± 0.1985 | 1.4252 | 4.2322 | 0.4876 |
| self_null_lam0.5_rm0.5_si0.3 | 0.1868 ± 0.1988 | 1.4813 | 4.4446 | 0.2128 |
| self_null_lam0.5_rm0.5_si0.5 | 0.1870 ± 0.1991 | 1.4810 | 4.4438 | 0.2128 |
| self_null_lam0.5_rm1.0_si0.3 | 0.1869 ± 0.1987 | 1.4255 | 4.2337 | 0.4876 |
| self_null_lam0.5_rm1.0_si0.5 | 0.1885 ± 0.1983 | 1.4251 | 4.2322 | 0.4876 |

## Paired interpretation

Use `paired_deltas_vs_stage2.csv` for seed-matched ARI comparisons against the
stage-2 self_null_lambda01 baseline (lambda=0.1, rank_margin=0.1, self_init=0.8).
A positive delta > 0.03 ARI is evidence for a real improvement; values in
[-0.03, 0.03] are within the documented noise band.
