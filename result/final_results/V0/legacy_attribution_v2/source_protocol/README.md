# Attribution v2

This frozen design answers two questions only:

1. Does a local neighbor view add information beyond a second independently
   masked clean view?
2. Does the topology gate act through perturbation magnitude, auxiliary-loss
   weighting, or their combination?

The matrix contains seven variants on the six protocol-v1 datasets and seeds
42, 2024, and 3407 (126 runs). All backbone and optimization parameters are
inherited unchanged. ARI is the primary endpoint. Seeds are averaged within
dataset before paired inference.

Planned attribution contrasts are F--CleanPair and T--CleanPair. The crossed
gate contrasts are AdaptiveMix--F, AdaptiveWeight--F, and GateCoupled--F.
T--GateCoupled isolates topology-informed affinity after the gate analysis.
No parameter may be changed after formal results are inspected.

Formal execution is pending a GPU runtime and the project remote credential.
