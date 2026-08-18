# Pre-registration — sparse_corruption_principle_probe

## Hypotheses

- **H1 (support role):** in the support-sensitive development case, preserving or targeting the
  H0 support proxy changes downstream utility beyond matched random corruption.
- **H2 (difficulty/geometry role):** in the difficulty-sensitive case, frozen residual or local
  geometry targeting explains more of the structured-vs-random gap than a generic harder task.
- **H3 (simple transfer):** one fixed static principle can produce material paired gains on at least
  two development datasets. If not, a heterogeneous pattern must be reproduced on an outcome-frozen
  holdout before any adaptive policy is considered.

## Anti-claims

The project will not claim that: (i) B1 effects were universal, (ii) a tested library maximum is an
oracle, (iii) H0 threshold support is biological zero semantics, (iv) harder reconstruction alone
improves clustering, or (v) an adaptive/GAN model is necessary without its prerequisite gates.

## Frozen matrix

```yaml
development_panel: [Mouse_retina, Baron Human, Campbell]
principles: [P0_Random, P1_SupportPreserve, P2_SupportTarget, P3_FrequencyAware, P4_ResidualHard, P5_GeometryHard]
negative_control_fixture: P5_GeometrySafe
seeds: [42, 123, 7]
formal_matrix_runs: 54
material_delta_ari: 0.03
labels_used_during_fit: false
legal_gpu_pool: [1, 2, 3, 4, 5, 6]
forbidden_gpu_ids: [0, 7]
```

## Stage gates

1. **C0 contract gate:** protocol validation, toy S/V/M sensitivity, exact-budget tests and
   outcome-independent holdout membership must pass. This gate passed before the explicit C2
   authorization below.
2. **C1 localization:** report support, value and geometry diagnostics. This stage has zero model
   fits and cannot produce an ARI claim.
3. **C2 static matrix:** compare all six arms under exact matched budgets. A structured principle is
   material only when its paired `Delta_P` reaches `+0.03` descriptively; seeds are repeats, not
   independent datasets.
4. **Simple-principle stop:** if one fixed principle is material on at least two development
   datasets, stop before adaptive model design and call it a `tested static principle`.
5. **Heterogeneity qualification:** only if different principles win on predeclared roles and the
   pattern survives a frozen holdout does adaptive policy become eligible for a new protocol.
6. **Fragility stop:** if the B1 support/difficulty clues do not survive the stricter static library,
   downgrade the direction and do not rescue it with a learner.

## Holdout freeze

C3 membership is generated before C2 results using only `log(n)`, `log(d)`, sampled sparsity,
sampled SVD-90 intrinsic-dimension proxy and source-family coverage. ARI, B1 effect, historical
gain, label values and best-arm identity are forbidden selection inputs. A shortfall below eight
valid, non-overlapping datasets is recorded as a shortfall and blocks C3 generalization claims.

## Resource and integrity rules

Formal GPU launchers must set `CUDA_VISIBLE_DEVICES` to one of `[1,2,3,4,5,6]` and reject `0`/`7`.
Every run records resolved protocol, source hashes, seed, device, metrics, label boundary and audit.
Failed/OOM/timeout jobs remain `incomplete_compute`; they never satisfy a gate.

### C2 interpretation firewall

`Support in C2 denotes the frozen threshold-defined support of dense H0, not raw-X zero/nonzero
support; raw sparse-support claims require a separate validation.`
