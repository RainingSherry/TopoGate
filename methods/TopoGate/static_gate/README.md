# StaticGate (Phase 1 — ablations before LearnableGate)

This directory contains the **v1-phase** TopoGate implementation, frozen for historical reproduction.

## What's in here

- `run.py` — original CSV-based runner (predates the .npz path)
- `model.py`, `neighbor_graph.py`, `mixing.py`, `diagnostics.py` — frozen modules
  (functionally identical to the V9-period implementation, duplicated to avoid coupling)
- `configs/static_gate_*.yaml` — the 8 ablation configs of Phase 1

## StaticGate vs LearnableGate — what differs

| Aspect             | static_gate (frozen V1)     | learnable_gate (V9 legacy mainline)                                |
|--------------------|------------------------------|--------------------------------------------------------------------|
| Topological gate   | 4 fixed `numpy`-implemented  | 4 `torch.nn.Parameter` (LearnableGate), enabled by `--gate_mode=learned` |
|                    | `β` (argparse defaults)      | + schedule (warmup + ramp) for stability                          |
| Config name        | `static_gate_<variant>`      | `learnable_gate_sched`                                                      |
| Config location    | `static_gate/configs/`      | `learnable_gate/configs/`                                          |
| `gate_mode` in cfg | `topology` (static path)     | `learned` (V9 legacy path)                                         |

The static 8-variant ablation table in CHANGELOG.md (`scripts/static_gate/run_topogate_ablation.py`)
runs the **static path** (`gate_mode=topology`) on top of the V9-period `run_npz.py` —
the algorithm logic for `gate_mode=topology` did not change.

## How to run a v1 ablation

```bash
# From the repo root
python scripts/static_gate/run_topogate_ablation.py --worker_id 0 --layer core \
    --epochs 150 --mask_ratio 0.3 --neighbor_k 5 --gpu_ids 1 4 5
```

The script calls `CLUBench.TopoGate` (wrapper in
`baseline/CLUBench/CLUBench/algorithms/ToPoGate.py`), which delegates to
`methods.TopoGate.learnable_gate.run_npz.run_topogate` for each
`static_gate_*` variant. The CLI flag `variant_name=static_gate_full` selects
the frozen static config; the underlying trainer is still the V9-period
`learnable_gate/run_npz.py`.

## Modifications log

- 2026-07-25: created this directory as part of the static_gate/learnable_gate split.  Source files
  are byte-identical to the pre-split originals (`/tmp/topogate_pre_v1v2_split_*.tar.gz`).
- 2026-07-25: configs renamed from `topogate_*.yaml` to `static_gate_*.yaml`
  to make the v1 origin explicit.
