# Execution plan

1. **P0 freeze (≤45 min).** Validate the frozen Python contract, resolve and
   hash all six E3 raw sources, build zero-preserving adapter manifests, run
   focused tests and `compileall`, and record the legal/forbidden GPU snapshot.
2. **Claude review.** Send the protocol, pre-registration, implementation,
   tests, and freeze manifest to the cross-family `claude-review` route. Store
   the full response in `review-stage/raw_sparse_mask_schedule_probe/`.
   Review unavailability is never a scientific pass.
3. **P1 (≤60 min).** Run SVD32 for all six datasets and seeds and the
   label-free sparse/dense first-projection benchmark.
4. **P2 (target ≤7 h 30 min from formal T0).** Launch only the frozen 90-cell
   MAIN matrix, one worker per idle legal physical GPU. Round-robin queue order
   is fixed before outcomes are visible. Never preempt foreign processes.
5. **P3.** Aggregate G0–G3 only after MAIN completion. If G0 fails, report
   `INCOMPLETE_COMPUTE`; if both G1/G2 fail, stop. Conditional probes are
   launched only by their predeclared gates and remaining wall-clock.
6. **P4/P5 (conditional).** Run the fixed-ratio seed-42 ceiling diagnostic or
   representation-space localization exactly as frozen; never promote a
   per-dataset winner.
7. **P6 (last 30 min).** Stop new launches at T0+11 h, mark unfinished cells
   `incomplete_compute` at T0+11 h 30 min, write compact reports and a
   publication-boundary manifest.

GPU 0 and GPU 7 are permanently forbidden. A legal GPU with an external
process is not available; unused GPU time is an acceptable outcome.
