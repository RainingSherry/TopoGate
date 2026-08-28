# Decision — sparse_corruption_principle_probe

## Current decision

`C2_TERMINAL_SIMPLE_STATIC_PRINCIPLE`

The project passed C0/C1 contract checks and completed the frozen `3 x 6 x 3 = 54` static-library
GPU matrix under `sparse_corruption_principle_probe_c2_v1`. No adaptive policy, learned selector,
GAN, generator or C3 performance run is authorized.

Support interpretation firewall: support in C2 is threshold-defined support of dense H0, not raw-X
zero/nonzero support; raw sparse-support claims require a separate validation.

## Decision tree after the completed C2 matrix

| observed pattern | decision |
|---|---|
| no valid arm changes clustering relative to clean/random | stop and downgrade corruption direction |
| one fixed principle is material on at least two development datasets | stop before adaptation; pursue the simple principle only |
| distinct principles win on predeclared roles, but holdout does not reproduce it | downgrade to development-only heterogeneity; no adaptive model |
| distinct principles win and frozen holdout reproduces the pattern | adaptive policy may be proposed in a new protocol, starting with a decision tree/logistic baseline |
| B1 support/difficulty effects disappear under exact static matching | classify B1 as arm-specific/fragile and do not rescue with a learner |

## C2 observed result

The frozen C2 overlay completed `54/54` runs with `audit_ok=true` and no failures. Dataset-level
primary `Delta_ARI = ARI(P)-ARI(P0_Random)` was:

| dataset | best arm | Delta ARI |
|---|---|---:|
| Mouse_retina | P2_SupportTarget | +0.394898 |
| Baron Human | P2_SupportTarget | +0.126069 |
| Campbell | P2_SupportTarget | +0.146883 |

P2 is material on all three development datasets; P3 and P4 are also material on two datasets each.
This authorizes only the descriptive terminal label `simple_static_principle_sufficient`. It does
not unlock C3, adaptive policy, GAN or a learned generator.

Support firewall: support in C2 is the frozen threshold-defined support of dense H0, not raw-X
zero/nonzero support. Raw sparse-support claims require a separate validation.

The term `best-of-library` is descriptive only: `tested_static_library_opportunity`. It is never an
oracle upper bound.
