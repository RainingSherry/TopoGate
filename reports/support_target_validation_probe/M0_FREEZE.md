# M0 New-project Freeze

M0 is a non-training stage. It freezes C2 evidence, source hashes, the dormant
holdout membership, the exact C2 reconstruction-probe contract and the M1
magnitude tolerances. It also proves that the missing C2 pair identity can be
deterministically reconstructed rather than approximated.

The executable audit is written to
`result/support_target_validation_probe/M0_freeze/`. Formal M1 jobs are not
authorized unless `audit.json` reports `audit_ok=true` and every replay row is
exact.

> Support in C2/M1 denotes threshold-defined support of dense H0, not raw-X
> zero/nonzero support; raw sparse-support claims require a separate validation.
