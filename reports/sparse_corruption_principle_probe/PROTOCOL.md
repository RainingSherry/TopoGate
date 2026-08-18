# Protocol — sparse_corruption_principle_probe

## 1. Scope and frozen starting point

| field | frozen value |
|---|---|
| project | `sparse_corruption_principle_probe` |
| protocol | `sparse_corruption_principle_probe_c0_v1` |
| old projects | `adaptive_corruption_probe`, `learned_relation_rule_probe`, `representation_consumer_probe`, `relation_selection_probe` (read-only) |
| development panel | `Mouse_retina`, `Baron Human`, `Campbell` |
| primary seeds | `[42,123,7]` |
| material descriptive margin | `delta_ARI=0.03` |
| legal GPU pool | `[1,2,3,4,5,6]` |
| forbidden GPUs | `[0,7]` |
| first authorized stages | `C0`, `C1`, `C2_library`, `C2_54_run_matrix` |

The repository currently has no usable Git object database. C0 therefore binds source paths and
SHA256 values in artifacts rather than inventing a commit identifier. The closed B1 compact result is
context, not a new training target.

## 2. Input and label boundary

C1 reuses the audited S0 `H0.npy` stem under
`result/representation_consumer_probe/S0_freeze/datasets/`. Its support is a fixed clean-row
threshold:

```text
support_ij = |H0_ij| >= max(1e-6, 0.05 * max_j |H0_ij|).
```

This is an H0 support proxy. It is not claimed to be the raw count-matrix support.

Corruption and future fit code accept only `X/H0`, frozen principle, seed, and an explicitly frozen
residual/geometry score where required. `y`, ARI, NMI, ACC, class purity and any class-derived feature
are forbidden in fit or principle selection. Labels may be loaded after fit for benchmark-known-K
readout and post-fit diagnostics only.

## 3. Stage order and locks

```text
C0 Freeze / inventory / toy contract
        |
        v
C1 structure replay (no model training)
        |
        v
C2 finite static library implementation + exact-budget tests
        |
        +--> only after contract pass: 3 x 6 x 3 GPU matrix
        |
        +--> only after C2 results are frozen: no adaptive model yet
```

C3 holdout membership is frozen before C2 results, but C3 runs stay locked. Adaptive policy, MLP,
GAN and learned generator stay locked until a later decision explicitly authorizes them.

The explicit C2 authorization uses protocol overlay `sparse_corruption_principle_probe_c2_v1`.
It does not unlock C3 or any adaptive/learned generator stage.

## 4. Development roles

- `Mouse_retina`: corruption-presence case.
- `Baron Human`: support-sensitive case.
- `Campbell`: difficulty-sensitive case.

These roles were inherited from the already burned B1 panel, not chosen after looking at new C2
outcomes.

## 5. Static principles

The primary library contains exactly six arms:

| arm | definition | support behavior |
|---|---|---|
| `P0_Random` | uniform coordinate selection, donor replacement | may change support |
| `P1_SupportPreserve` | value replacement on active coordinates only | support fixed |
| `P2_SupportTarget` | swap active and threshold-inactive positions, preserving the row nonzero multiset | support targeted |
| `P3_FrequencyAware` | replace values at rare active features first (one frozen rule, no frequency sweep) | support fixed |
| `P4_ResidualHard` | replace values at high frozen residual-score active coordinates | support fixed |
| `P5_GeometryHard` | replace values at high local-geometry sensitivity active coordinates | support fixed |

`P5_GeometrySafe` is a paired negative-control fixture only; it is not an unregistered seventh arm.

### Support interpretation firewall

Support in C2 denotes the frozen threshold-defined support of dense `H0`, not raw-X zero/nonzero
support; raw sparse-support claims require a separate validation.

## 6. Exact budget

For each row, let `a_i` be the number of active coordinates and `u_i` the inactive count. Freeze:

```text
m_i = min(ceil(0.25*a_i), floor(a_i/2), u_i)
q_i = 2*m_i
```

Every primary arm must change exactly `q_i` coordinates. If this cannot be satisfied, the run is
`protocol_mismatch` or `incomplete_compute`, never a negative performance result. Both changed
coordinate count and total absolute change are reported; a nominal rate alone is insufficient.

## 7. Primary estimand and claims

After a valid C2 fit, the primary comparison is:

```text
Delta_P(d) = ARI(P,d) - ARI(P0_Random,d).
```

The maximum over the tested six-arm library is named `tested_static_library_opportunity`; it is not
an oracle upper bound. `ARI(P)-ARI(clean)`, NMI, ACC and reconstruction loss are secondary or
diagnostic quantities.

The C2 reconstruction probe is fixed at `d_eff -> 64 -> 32 -> 64 -> d_eff`, ReLU, Adam with
learning rate `0.001`, batch size `512`, 30 epochs, and a five-epoch clean warm-up only for the
frozen P4 residual score. P4 residuals are computed on standardized clean H0 and frozen per
dataset×seed; P5 geometry scores are computed on raw clean H0 and frozen once per dataset before
performance runs. These score artifacts remain local and are excluded from publication.

## 8. Publication boundary

Only compact protocol/manifest/decision/audit files and important aggregate tables/figures are
publishable. Raw input matrices, labels, corruption arrays, embeddings, predictions, graphs,
weights, checkpoints and logs remain local/external.
