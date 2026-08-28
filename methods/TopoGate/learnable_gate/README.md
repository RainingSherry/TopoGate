# LearnableGate — canonical mainline

This directory contains the **current mainline** TopoGate implementation.

It is the V9 legacy path. The independently implemented V10, V11 and V12
variants live in `../v10_reliable_graph/`, `../V11/` and
`../V12_latent_topology/`; similarly named YAML files in this directory still
use this legacy runner and must not be relabelled as those versions.

## What v2 adds over v1

The single key change is that the four topological-gate coefficients
(`β_mutual`, `β_snn`, `β_perturb`, `β_uncertainty`) became learnable
`torch.nn.Parameter` instead of fixed argparse defaults.  They participate
in the MAE loss gradient through a `LearnableGate` module and a per-epoch
schedule that interpolates from the v1 static gate to the learned gate
during warmup.

### New components

- `learnable_gate.py` — `LearnableGate` (4β → per-node gate via sigmoid) and
  `build_gate_stats_tensor` (graph features → 4-D stats tensor)
- `configs/learnable_gate_sched.yaml` — v2 config: `gate_mode=learned`, plus
  `warmup_epochs=20`, `ramp_epochs=10`, `learned_gate_init_mode=zero`

### New CLI flags (in `run_npz.py`)

- `--gate_mode learned`     (else behaves like v1)
- `--warmup_epochs N`       (default 20)
- `--ramp_epochs N`         (default 10)
- `--learned_gate_init_mode {zero, v1_default}`
- `--init_beta_mutual / _snn / _perturb / _uncertainty`
- `--freeze_mae_after_epoch N`   (default 1e9 = disabled; set to e.g. 30 to freeze
                                 MAE after the ramp so β can settle on a stable target)

### Diagnostics

- `summary.json` now contains `learned_gate_final_beta` and
  `learned_gate_beta_history` (per-epoch β values, including `mae_frozen`
  flag) for post-hoc β-curve analysis.

See `../CORE_CODE_INDEX.md` for the complete version map and output contracts.

## How to run

```bash
# Smoke test (3-way compare: v1, v2 schedule=0, v2 schedule=20/10)
python scripts/learnable_gate/run_learnable_gate_sched_smoke.py --gpu 4 --datasets har enron

# Direct Python API
from methods.TopoGate.learnable_gate.run_npz import run_topogate
labels = run_topogate(X, n_clusters=K, gpu=4, seed=42, variant="learnable_gate_sched",
                      epochs=150)
```

## Modifications log

- 2026-07-25: created this directory during the static_gate/learnable_gate split.  Source files
  byte-identical to the pre-split `methods/TopoGate/*.py` (verified by
  diff against `/tmp/topogate_pre_v1v2_split_*.tar.gz`).
- 2026-07-25: config renamed `learnable_gate.yaml` → `learnable_gate_sched.yaml`.
- 2026-07-25: added `--freeze_mae_after_epoch` and per-epoch
  `learned_gate_beta_history` logging for ablation studies.
