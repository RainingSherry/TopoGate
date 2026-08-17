# ACCG Action-Constrained Gate

This directory implements the frozen Post-V25 ACCG route. The implementation
is complete enough to build manifests, run synthetic and real matched panels,
reuse canonical controls for ablations, and audit outputs. No ACCG experiment
is started by importing the package or invoking a matrix runner without
`--execute`.

## Layout

- `config.py`: frozen method and V21 compatibility configs.
- `feature_model.py`: robust transform and cross-fitted feature graph.
- `calibration.py`: label-free random joint-action epsilon calibration.
- `selector.py`: joint/coordinate selectors, pair lookahead, STE, exact oracle.
- `torch_energy.py`: differentiable joint structural barrier.
- `protocol.py`: shared-branchpoint `N/R/T_s/T_c` protocol and artifact writer.
- `synthetic.py`: support/marginal-matched W0-W5 generators.
- `synthetic_audit.py`: shortcut and outer-only oracle audits.
- `synthetic_probe.py`: grouped incremental AUC/PR action probes.
- `run.py`: one-panel CLI.
- `configs/`: primary and control variants.
- `tests/`: method, W5, branchpoint, and label-boundary regression tests.

Experiment orchestration lives in `scripts/ACCG/`.

## Execution boundary

The matrix runners are dry by default:

```bash
python scripts/ACCG/run_matrix.py --manifest /path/to/real_manifest.json
python scripts/ACCG/run_synthetic_matrix.py \
  --manifest /path/to/synthetic/manifest.json \
  --output-root /path/to/output
```

They print the planned jobs and do not create job output directories. Training
requires explicit `--execute` plus either `--cpu` or an allowed physical GPU in
`1..6`. GPUs 0 and 7 are rejected.

Main panels and ablations are separate phases. An ablation is blocked until
the canonical main panel has complete `N/R/T_s/T_c` metrics, branchpoint,
resolved config, runner profile, and matching source identity.

## Synthetic workflow

Input generation, shortcut audit, action probes, and exact-selector audit are
separate from end-to-end training:

```bash
python scripts/ACCG/build_synthetic_manifest.py \
  --output-dir /path/to/synthetic_inputs
python scripts/ACCG/audit_synthetic_contract.py \
  --manifest /path/to/synthetic_inputs/manifest.json \
  --output /path/to/synthetic_contract_audit.json
python scripts/ACCG/build_action_probes.py \
  --manifest /path/to/synthetic_inputs/manifest.json \
  --output-dir /path/to/action_probes
python scripts/ACCG/audit_exact_selector.py \
  --output /path/to/exact_selector_audit.json
```

These commands perform generator or selector analysis but do not train the
clustering model. End-to-end synthetic training remains behind the matrix
runner's `--execute` gate.

## Real manifest schema

`scripts/ACCG/build_real_manifest.py` expects a YAML file with 8-12 frozen
dataset rows. Each row contains:

```yaml
dataset_id: stable_id
name: display_name
source_path: /absolute/or/spec-relative/data.npz
domain: scRNA_or_text_or_table
source_family: independent_source_family
input_protocol: clubench_bridge_or_shared_text_or_scRNA_count
license: provenance_string
n_clusters: 5  # optional when benchmark labels are stored in the NPZ
```

The builder rejects outcome fields, duplicate execution keys, more than 2000
input features, invalid label lengths, K disagreement, missing K for unlabelled
data, fewer than two domains, or a panel outside 8-12 datasets. Formal primary
ARI/NMI aggregation additionally requires saved outer labels.

## Verification

```bash
python -m compileall -q methods/TopoGate/ACCG_action_constrained_gate scripts/ACCG
pytest -q methods/TopoGate/ACCG_action_constrained_gate/tests scripts/ACCG/tests
python -m methods.TopoGate.ACCG_action_constrained_gate.run --help
python scripts/ACCG/run_matrix.py --help
python scripts/ACCG/run_synthetic_matrix.py --help
```

Passing these checks establishes implementation and artifact contracts only.
It does not establish the synthetic identifiability gate or a real-data effect.

