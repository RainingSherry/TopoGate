# M1 Results / Preflight Decision

M1 did not enter the GPU performance matrix. The full no-training 30-epoch
preflight completed all 9 dataset×seed rows and passed the structural contract,
but `P2_MM_SupportPreserve` was not magnitude-estimable for Baron Human:

| Dataset | Seed 42 | Seed 123 | Seed 7 |
|---|---:|---:|---:|
| Mouse_retina | 0.001582 | 0.001647 | 0.001686 |
| Baron Human | **0.094640** | **0.095877** | **0.094646** |
| Campbell | 0.005726 | 0.006001 | 0.005828 |

Values are dataset-total relative L1 mismatch between the matched control and
the replayed P2 action. The frozen limit is `0.05`; median row mismatch also
remained below `0.10` for all rows. The Baron rows therefore have status
`magnitude_match_not_estimable`, not a negative clustering result.

`gpu_runs_started=0`, `model_training_started=false`, and no ARI/NMI/ACC result
exists for M1. The launcher now refuses to construct a model unless the
preflight decision explicitly authorizes the complete matrix. M2 raw-X bridge,
M3 holdout, M4 full-backbone transfer, adaptive policy and GAN remain locked.

> This preflight does not establish or refute a support-crossing effect. It
> establishes that this active-active magnitude-matched control cannot provide
> the pre-registered three-dataset estimand under the frozen tolerance.
