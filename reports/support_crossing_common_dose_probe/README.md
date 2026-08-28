# support_crossing_common_dose_probe

This is an independent, non-V study that follows the terminal
`support_target_validation_probe` M1 preflight. It does not rerun C2 or M1 and
does not relax either study's magnitude tolerance.

The question is narrower than a new model claim:

> Can an active/inactive value-swap (`Cross`) and an unequal active/active
> value-swap (`Preserve`) be constructed at a common row-wise L1 dose while
> keeping the C2 changed-coordinate budget and row value multiset?

D0 and D1 are CPU-only. D1 uses constructive minimum/maximum matching
witnesses and a deterministic nearest-per-pair target matching; it does not
claim to enumerate every attainable matching. The three seeds are deterministic
tie-break reproductions, not independent statistical samples.

The frozen D1 result is `common_dose_not_estimable`: Mouse_retina passes its
dataset×seed contract, while Baron Human fails the common-row and tolerance
gates and Campbell fails the dataset-total tolerance. D2 was not authorized;
there are no ARI/NMI/ACC results and no GPU runs.

> Support means the fixed threshold-defined support of dense H0, not raw-X
> zero/nonzero support. This project does not establish a causal support effect.
