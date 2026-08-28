# raw_sparse_mask_schedule_probe v2 — shared GPU amendment

`raw_sparse_mask_schedule_probe_v2_shared` is a user-authorized execution
amendment to v1. The v1 documents and artifacts remain immutable. The model,
data panel, mask definitions, seeds, readout, label firewall, and estimands are
unchanged; only the selected execution gates are removed.

## Cleared gates

The following user-selected gates are cleared for v2:

- E1: a GPU need not be idle;
- E2: a foreign process does not block launch;
- E3: shared runs are not automatically engineering-only;
- E4: the one-worker-per-GPU restriction and its occupancy-based launch gate;
- E8: the T0+11h launch cutoff and finalization reserve;
- E9: the one-retry cap;
- E10: the prohibition on operator-selected retry configuration changes;
- E11: dispatch-entry and per-bind occupancy rechecks;
- E12: the requirement to rerun v1 P0/P1 tests before v2 launch.

The v2 dispatcher has no hard wall or occupancy wait. `--attempts 0` means
retry a retryable failed cell until it succeeds or the operator stops the run.
Per-run subprocess timeout remains 1800 seconds. Timeout/OOM failures remain
`incomplete_compute` while retrying; NaN, shape/hash, label-leakage, assertion,
and protocol-integrity failures are terminal for that cell and are never
promoted by retry.

## Retained hard boundaries

- Physical GPU 0 and GPU 7 remain forbidden.
- No foreign process is killed or preempted.
- Training still receives no `y`, `K`, ARI, NMI, or ACC.
- Labels are still loaded only after fit.
- Source, adapter, zero-pattern, mask-budget, paired-seed, and finite-loss
  audits remain recorded.
- Failed cells are not silently counted as successful cells.
- The six-dataset, five-arm, three-seed matrix remains `90` cells.
- Existing v1 SVD32 rows are reused through an explicit `SVD_REUSE_MANIFEST`;
  no duplicate CPU SVD is launched.

## Resource mode

The v2 run records `execution_mode=shared_resource_allowed`, the physical GPU,
worker slot, retry attempts, and the v2 launcher hash in every manifest. Shared
resource use is now part of the declared v2 environment rather than a hidden
exception.

## Scope

This amendment does not authorize new gates, new model families, holdout
claims, post-hoc dataset removal, or automatic model promotion. Those remain
separate research decisions.

E10 is cleared at the scheduler boundary: a later explicit invocation may
choose different retry/resource parameters. Such a rerun is provenance-bearing
and is not silently treated as an identical scientific cell if its resolved
configuration or hashes differ.
