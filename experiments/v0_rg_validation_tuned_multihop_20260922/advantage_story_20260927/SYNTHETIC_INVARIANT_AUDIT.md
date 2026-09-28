# Independent CPU and frozen-output checks (2026-09-27)

On `yazhouwan`, the existing synthetic tests were invoked with `/data/luolie/conda/envs/scssl_bench_py310/bin/python`, with the historical experiment base, its two experiment-script directories, and `/data/luolie/ToPoGate/worktrees/topogate_unified_missing5_20260916` on `PYTHONPATH` (the last supplies `methods.NeighborMix_scMAE`). The environment has no `pytest`, so test functions were called directly. No real dataset was trained and no remote source was edited.

| Historical synthetic test function | Result |
|---|---|
| `tests_factorial.test_traced_estimators_are_numerically_identical_and_rng_preserving` | Passed |
| `tests_factorial.test_reference_identity_of_wrappers_h1_and_flat` | Passed |
| `tests_factorial.test_frozen_graph_matches_trainer_diagnostics` | Passed |
| `tests_interventions.test_kwide_and_shuffle_match_count_without_self_or_source_mutation` | Passed |
| `tests_interventions.test_traced_reference_sampler_is_exact_and_rng_preserving` | Passed |

The actual command returned exit code 0 and `five CPU invariant tests passed`. The checks use tiny synthetic matrices. They do not validate the new H4 Kwide task guard or the 21 future fits.

After the E3/E4 patch was corrected, I compared its three-case outputs independently with each case's archived `all_readout_outcomes.csv` and `asset_manifest.json`. Each case has 60 one-to-one paired readout rows; the maximum absolute difference in ARI, split rate and merge rate is exactly 0 for Baron, Quake_Smart-seq2_Lung and student_grade. Off-diagonal community-pair coassignment, weighted by eligible pair count within each seed and arm, matches the archived per-readout summary to within `1.11e-16` in every case. The three input SHA-256 values in each new metadata file match its frozen asset manifest. Full E4 community-pair row counts are 336, 184 and 72, respectively, each with three seeds and ten readouts per seed.

These checks verify the three-case posthoc outputs. The E3 all-59 cohort placement remains a separate required deliverable.
