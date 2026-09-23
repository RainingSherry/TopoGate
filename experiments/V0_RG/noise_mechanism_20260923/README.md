# V0_RG noise-mechanism probes

This directory contains the frozen-checkpoint analysis code for the four-case ToPoGate V0_RG noise-mechanism study. It measures input and neighborhood changes, frozen representation and clustering responses, effective ablations, and matched-norm direction probes. The scripts do not train weights.

## Source version

The audited experiment used V0_RG source commit `3be12512b571207d7b287df4f2a5836bd9354622`. The repository's newer main branch has later changes in three V0_RG files, so use this source snapshot for exact replay. From a ToPoGate checkout, create a separate worktree:

```bash
git worktree add ../topogate-v0rg-mechanism-source 3be12512b571207d7b287df4f2a5836bd9354622
```

Set `remote_repo` in `config.json` to the absolute path of that worktree. The analysis scripts and this source checkout can stay in separate worktrees.

## Privacy and data setup

Dataset files, checkpoint files, and the cohort manifest are intentionally excluded. Create a private cohort JSON and set its path in `config.json`. Each of the four entries (Mouse_retina, dermatology, har, micro-mass) must provide the fields consumed by `scripts/common.py`, including `dataset`, `input_kind`, `data_path`, and `checkpoint_root`. Paths must resolve on your machine. Keep this manifest and any machine-specific configuration out of Git.

The reference frozen-corruption evaluator used by these probes is included at `scripts/run_frozen_v0_rg_raw_corruption.py`. No weights, predictions, embeddings, labels, or result tables are included in this source release.

## Configure and run

Run on Linux from this directory:

```bash
cp config.example.json config.json
# Edit cohort, remote_repo, remote_python, reference_runner, and output_root
# to match your private data and host.
python scripts/test_numerics.py
python scripts/run_stages.py --config config.json --stage inventory
python scripts/run_full_four_steps.py
python scripts/audit_full_results.py
```

The example uses relative paths except for fields that must point to your private manifest/source checkout. The full-run launcher targets physical GPU 4 with four bounded workers; use the staged runner if that device is unavailable. The GPU allowlist is restricted to physical devices 1, 4, and 5. No CUDA device is needed for the numerical tests.

## Protocol and outputs

The run uses existing model seeds 42, 123, and 7; noise seeds 3001-3005; and four registered Gaussian/Replacement conditions. Labels are used only for post-fit ARI and neighborhood purity. A successful run writes each condition's `result.json` and `samples.npz` under the configured output root, then writes summaries, figures, dispatch logs, and `FILE_AUDIT.json`. Existing completed results are protected by task signatures.

The four cases were selected from earlier benchmark results. Treat this as a selected-case mechanism study, not a universal robustness estimate. The noise seeds reuse existing benchmark realizations and are not a previously unseen holdout. Read `config.json` and each result's provenance before comparing outputs.

## Main scripts

- `common.py`: private manifest and checkpoint loading, frozen preprocessing, clean replay, hashes.
- `probe.py`: input/representation probes and matched-norm direction probes.
- `metrics.py`: distribution, neighborhood, representation, and decision metrics.
- `summarize.py`: provenance checks, paired ablations, seed-aware summaries.
- `plot_results.py`: figures from validated summaries and actual paired embeddings.
- `run_stages.py`: bounded resumable stages.
- `run_full_four_steps.py`: registered end-to-end queue.
- `audit_full_results.py`: file-level coverage, hashes, and pair audit.
- `audit_checkpoints.py`: checkpoint completeness and no-op control audit.
- `test_numerics.py`: standalone metric and protocol invariants.
