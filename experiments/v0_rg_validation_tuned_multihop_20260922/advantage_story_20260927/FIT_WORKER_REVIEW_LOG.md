# Independent fit worker synthetic CPU review (2026-09-27)

Ran `test_fit_worker_review.py` on `yazhouwan` with `/data/luolie/conda/envs/scssl_bench_py310/bin/python`, the historical experiment base and the worktree supplying `methods.NeighborMix_scMAE` on `PYTHONPATH`. It ran only synthetic arrays and temporary outputs on CPU; no real dataset was trained. The script executes the actual `fit_worker.main` for H4/H8 path and Kwide variants with a one-epoch injected synthetic matrix, tests pinned task validation against `REUSE_MANIFEST.json`, and separately forces graph-unreachable direct candidates to exercise hop code 9.

```text
PASS pinned task validation and drift rejection
PASS synthetic full worker H4_B1 path
PASS synthetic full worker H4_B1 kwide
PASS synthetic original graph/gate and per-anchor count H4_B1
PASS synthetic full worker H8_B1 path
PASS synthetic full worker H8_B1 kwide
PASS synthetic original graph/gate and per-anchor count H8_B1
PASS H4 Kwide >8/unreachable code9 and traced-sampler identity
ALL SYNTHETIC WORKER REVIEW CHECKS PASSED
```

The process exited 0. For each full worker run, the test checked required artifacts, one training occurrence per anchor, four draws per occurrence, trace byte count and displacement count. Kwide and path graph indices, edge weights and node gates were bitwise identical within each synthetic variant; Kwide reference and intervention plan offsets and draw counts matched. The H4 disconnected-graph test confirmed positive trace mass in code 9 and unchanged original versus traced sampler output and RNG state. This is a runtime invariant check, not evidence of completed formal experiment cells.
