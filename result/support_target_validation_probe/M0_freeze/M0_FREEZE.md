# M0 Freeze Audit

Status: `passed`.

M0 freezes C2 P0/P2 evidence, H0/budget hashes, the matched reconstruction probe, and the dormant holdout membership.
The compact C2 artifacts did not store pair identities; the independent replay reproduced every P2 epoch and scalar audit before M1 authorization.

- C2 P2 records: `9/9`.
- Exact replay rows: `9/9`.
- Holdout dormant: `True`.
- M1 adds only `P2_MM_SupportPreserve`; M2/M3/M4, adaptive policy and GAN remain locked.

> Support in C2/M1 denotes threshold-defined support of dense H0, not raw-X zero/nonzero support; raw sparse-support claims require a separate validation.
