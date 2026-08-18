# Pre-registration: D0/D1 only

The only authorized work in this project is the D0 inheritance freeze followed
by the CPU-only D1 constructive feasibility map. The D1 thresholds are frozen
before reading D1 outcomes: 95% common positive-budget rows, 5% dataset-total
dose mismatch, 10% median row mismatch, exact changed count, Cross support
change positive, Preserve support change zero, and row-multiset preservation.

Any dataset×seed failure yields `common_dose_not_estimable` and stops before
D2. Changing the 5% tolerance, selecting a different target after inspection,
or replacing the constructive matcher after seeing the results is not allowed
within this project.

D2, raw-X validation, holdout, adaptive policy and GAN are separate locked
routes. If a future D2 is proposed, it must be a new frozen authorization that
consumes a positive `d1_gate_pass` artifact and uses the estimand
`ARI(Cross)-ARI(Preserve)` rather than the C2 P2 effect.
