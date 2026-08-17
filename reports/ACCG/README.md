# ACCG Public Research Route

ACCG (Action-Conditional Compatibility-Constrained Topology Gate) is an
independent post-V25 research route. It keeps the V21 donor, exact effective
budget, matched schedule, warmup branchpoint, and clean-embedding KMeans
readout, and changes the adversarial selection policy by constraining the
actual joint donor action with a cross-fitted feature-conditional structural
energy.

## Public evidence status

- The final synthetic v3 action contract passed its frozen shortcut,
  exact-selector, and held-out-family incremental-information checks.
- The locked real panel completed `30/30` main runs and `48/48` ablation arms.
  The confirmatory scope is 27 labeled panels across 9 datasets plus 48
  ablation arms; 3 unlabeled PBMC3k runs are operational-only.
- The labeled primary effect `ARI(T_c) - ARI(T_s)` is mean `+0.007492`, median
  `+0.000363`, with dataset-bootstrap 95% CI
  `[-0.000879, +0.018889]`. Only `4/9` datasets are positive for all three
  seeds.
- Joint selection does not beat the coordinate control on the development
  subset: `+0.010751` versus `+0.015689`, with joint winning `1/12` paired
  seed rows.

The current real clustering-improvement claim is therefore **No-Go**. This is
an empirical promotion decision, not a claim that the code or structural audit
failed. No external-baseline or outcome-driven rescue result is included.

## Contents

- `../../methods/TopoGate/ACCG_action_constrained_gate/`: model, selector,
  feature energy, synthetic worlds, configs, protocol, and tests.
- `../../scripts/ACCG/`: dry-by-default manifest builders, audits, runners,
  and summary tools.
- `synthetic_v3_audit/`: small JSON-only synthetic contract evidence.
- `real_panel_v2_audit/`: JSON/Markdown summaries and sanitized manifests for
  the completed real panel.
- `SYNTHETIC_CONTRACT_V3.md`: frozen synthetic promotion contract.

## Reproduction boundary

The public snapshot contains no dataset binaries, checkpoints, prediction or
embedding arrays, memmaps, caches, worker logs, or local machine paths. Supply
an input dataset and an output directory explicitly. Runners are dry by
default; model execution requires an explicit `--execute` flag. Labeled
benchmarks use labels only for outer evaluation and known-K protocol; an
unlabeled run requires explicit `--n-clusters`.

Focused verification used for this snapshot: `29 passed` across ACCG method and
runner tests, plus Python `compileall` for `methods/TopoGate/ACCG_action_constrained_gate`
and `scripts/ACCG`.
