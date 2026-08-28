# corruption_objective_compatibility_probe

This is an independent, non-V-series mechanism study. It asks whether the
closed C2 `P2_SupportTarget` observation survives a cross-domain sentinel panel
and whether any advantage is compatible with the reconstruction objective.

The execution is a gated pipeline:

```text
E0 integrity closure -> E1/E1b opportunity + no-fit control
                     -> (G1 and G2 only) E2 objective matrix
                     -> E3 raw-X descriptive audit -> E4 decision
```

The study consumes the frozen dense SVD/H0 snapshot from
`representation_consumer_probe/S0_freeze`. In E1/E2, “support” means the
threshold-defined dense H0 proxy. E3 reports raw-X zero/nonzero structure for
future work only; it never enters fitting, gating, or the decision. The closed
`sparse_corruption_principle_probe`, `support_target_validation_probe`, and
`support_crossing_common_dose_probe` projects are read-only.

No adaptive policy, matcher optimization, GAN, learned generator, new Gate,
architecture search, corruption-rate sweep, or automatic new model is allowed.
GPU execution is restricted to physical GPUs 1–6; 0 and 7 are forbidden.

See [PROTOCOL.md](PROTOCOL.md), [PRE_REGISTRATION.md](PRE_REGISTRATION.md),
and [OVERNIGHT_DECISION.md](OVERNIGHT_DECISION.md) after the run.
