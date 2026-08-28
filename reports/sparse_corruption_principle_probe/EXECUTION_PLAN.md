# Execution plan — sparse_corruption_principle_probe

## Primary claim map

| claim | minimum evidence | stage |
|---|---|---|
| C1: B1 effects correspond to distinguishable support/value/geometry changes | zero-fit structural replay with finite diagnostics and honest H0-proxy limitation | C1 |
| C2: a simple static principle beats matched random corruption | toy contract, exact budget, 3×6×3 paired matrix, dataset-level `Delta_P` | C2 |

Adaptive policy is not a current claim. It earns eligibility only after distinct static winners survive
an outcome-independent holdout in a later protocol.

## Run order

| milestone | action | compute | gate |
|---|---|---:|---|
| M0 | C0 docs, protocol validation, inventory | CPU / I/O | no outcome-dependent membership |
| M1 | toy S/V/M fixtures and exact-budget tests | CPU, seconds | must pass before formal matrix |
| M2 | C1 B1 structural replay on S0 H0 | CPU, no fit | report support/value/geometry quadrants |
| M3 | C2 static library contract audit | CPU | all six arms exact-budget on fixtures |
| M4 | only if M0–M3 pass: 54 paired GPU runs | GPUs 1–6 | no adaptive unlock from a single result |
| M5 | only after C2 results are frozen: holdout run decision | GPUs as permitted | membership already frozen; config cannot change |

## GPU policy

When M4 is authorized, use explicit workers on physical GPUs `[1,2,3,4,5,6]`; never use `0` or `7`.
The common small reconstruction probe may use multiple GPUs for throughput, but each run records its
visible device and the launcher rejects an illegal pool. Resource abundance does not bypass the
cheap contract gates.

## Failure handling

OOM, timeout, missing source, shape mismatch, label leakage or budget mismatch is
`incomplete_compute`/`protocol_mismatch`; it is not a negative result and cannot satisfy a gate.

