# V0_RG frozen-noise mechanism study

This directory publishes the executable code for the completed four-case
V0_RG study (Mouse_retina, dermatology, har, and micro-mass). The [final
report](../../reports/V0_RG_noise_four_step_20260923/FINAL_REPORT.md),
[per-run metric table](../../reports/V0_RG_noise_four_step_20260923/tables/metrics.csv),
seed summaries, paired ablations, provenance inventory, and figure sources are
under `reports/V0_RG_noise_four_step_20260923/`.

The study reused three existing checkpoint seeds per effective variant. It
evaluated four raw-input corruptions with five noise seeds each (840 audited
observations), plus matched-norm model-input direction probes (378 audited
observations). There were 870 validated full–control pairs and zero file-audit
issues. No weights were retrained or tuned on corrupted inputs. Labels were
used only for post-hoc ARI and neighbourhood-purity diagnostics. The noisy
graph is a diagnostic; frozen V0_RG inference does not rebuild it.

The public tables and figures are a curated snapshot. They omit dataset
binaries, checkpoints, embeddings, predictions, worker logs, and other
per-run arrays, following this repository's metadata-only publication policy.
The private full-results archive is identified by SHA-256 in
[`PUBLISH_MANIFEST.json`](../../reports/V0_RG_noise_four_step_20260923/PUBLISH_MANIFEST.json).
The separate 18-dataset baseline expansion was still in progress when this
four-case study closed, so this report makes no cross-model noisy-Top1 claim.

## Reproduction interface

1. Supply the exact ordered NPZ inputs and V0_RG checkpoint directories.
   Checkpoint and input hashes are recorded in the [public inventory](../../reports/V0_RG_noise_four_step_20260923/inventories/checkpoints.json).
   The current repository `methods/TopoGate/V0_RG` is not byte-identical to
   the experiment version. In a **separate disposable checkout**, overlay the
   files from `source_snapshot/methods/TopoGate/V0_RG/` onto that path before
   replay. The snapshot includes the exact audited Python files; do not alter
   the active model branch to reproduce this study.
2. Copy `config.example.json` and `cohort.example.json`, replace every
   `/ABS/PATH/TO/` placeholder with absolute local paths, and set the output
   directory. Each checkpoint root must end in `full/current`; the runner
   locates sibling ablation variants from that root.
3. From the repository root, run the inventory before any probe. The commands
   below use one CPU worker for clarity; the historical run used physical GPU
   4 with bounded parallel workers.

```bash
python experiments/V0_RG_noise_four_step_20260923/scripts/run_stages.py \
  --config /absolute/path/to/config.json --stage inventory
python experiments/V0_RG_noise_four_step_20260923/scripts/run_stages.py \
  --config /absolute/path/to/config.json --stage observe --device cpu --workers 1
python experiments/V0_RG_noise_four_step_20260923/scripts/run_stages.py \
  --config /absolute/path/to/config.json --stage directions --device cpu --workers 1
python experiments/V0_RG_noise_four_step_20260923/scripts/run_stages.py \
  --config /absolute/path/to/config.json --stage summarize
```

`run_full_four_steps.py` is the historical Linux/GPU-4 dispatcher and expects
`config.json` beside this README; `audit_full_results.py` checks the complete
per-run output tree. The public `run_frozen_v0_rg_raw_corruption.py` differs
from the audited server source only in resolving the repository import path
relative to its own file and using a portable default output path. The original source hash is retained in the
publication manifest. Consequently the public code is suitable for a new
replay, while the archived 2026-09-23 hashes identify the exact original run.

The study concerns deliberately selected cases and frozen readouts. A stable
ARI under corruption is not, by itself, evidence that a particular gate caused
robustness. Read the report's ablation and direction findings before making a
mechanism claim.
