#!/usr/bin/env python
"""Run the V12 stage-3 topology-signal hyperparameter sweep.

The launcher owns only orchestration. The V12 runner remains the source of
truth for preprocessing, graph construction, K protocol, losses, and output
contracts. Jobs are assigned explicitly to the allowed physical GPU pool.

Stage 3 amplifies the topology signal: ``lambda_topology`` lifts the share of
the topology loss, ``rank_margin`` sharpens the hinge force, and
``self_init_weight`` (self_null only) lowers the initial self mass so the
edge branch receives a larger share of the gradient early in training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "methods" / "TopoGate" / "V12_latent_topology" / "run_npz.py"
GPU_POOL = (1, 4, 5)
SEEDS = (42, 123, 7)

DATASETS = {
    "flame": REPO / "datasets" / "AHDPC" / "processed" / "flame.npz",
    "balance_scale": REPO / "datasets" / "AHDPC" / "processed" / "balance_scale.npz",
    "spect_heart": REPO / "datasets" / "AHDPC" / "processed" / "spect_heart.npz",
    "vehicle": REPO / "datasets" / "AHDPC" / "processed" / "vehicle.npz",
}

# Pre-registered search space. Keep aligned with the Stage-3 plan.
LAMBDA_VALUES = (0.3, 0.5)
RANK_MARGIN_VALUES = (0.5, 1.0)
SELF_INIT_VALUES = (0.3, 0.5)


@dataclass(frozen=True)
class Config:
    name: str
    topology_mode: str
    lambda_topology: float
    rank_margin: float
    self_init_weight: float = 0.8  # edge_only default


def _build_configs() -> dict[str, Config]:
    configs: dict[str, Config] = {}
    # self_null: full 3D grid.
    for lam in LAMBDA_VALUES:
        for margin in RANK_MARGIN_VALUES:
            for s_init in SELF_INIT_VALUES:
                name = f"self_null_lam{lam}_rm{margin}_si{s_init}"
                configs[name] = Config(
                    name=name,
                    topology_mode="self_null",
                    lambda_topology=lam,
                    rank_margin=margin,
                    self_init_weight=s_init,
                )
    # edge_only: 2D grid (self_init_weight ignored).
    for lam in LAMBDA_VALUES:
        for margin in RANK_MARGIN_VALUES:
            name = f"edge_only_lam{lam}_rm{margin}"
            configs[name] = Config(
                name=name,
                topology_mode="edge_only",
                lambda_topology=lam,
                rank_margin=margin,
            )
    return configs


CONFIGS = _build_configs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO
            / "result"
            / "V12"
            / "v12_topology_search_stage3_2026-08-04"
        ),
    )
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-loss-weight", type=float, default=0.1)
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--rank-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--datasets",
        default="flame,balance_scale,spect_heart,vehicle",
    )
    parser.add_argument(
        "--configs",
        default="",
        help="Comma-separated subset of config names; default = full grid.",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _split_csv(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def _job_command(
    dataset_name: str,
    data_path: Path,
    config: Config,
    seed: int,
    gpu: int,
    save_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--data_path",
        str(data_path),
        "--save_dir",
        str(save_dir),
        "--dataset_name",
        dataset_name,
        "--method_name",
        "TopoGate",
        "--variant_name",
        f"topogate_v12_{config.name}",
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--epochs",
        str(args.epochs),
        "--hidden_size",
        str(args.hidden_size),
        "--batch_size",
        str(args.batch_size),
        "--mask_ratio",
        str(args.mask_ratio),
        "--neighbor_k",
        str(args.neighbor_k),
        "--scale_input",
        "true",
        "--decoder_mode",
        "legacy_mask_conditioned",
        "--mask_loss_mode",
        "additive",
        "--mask_loss_weight",
        str(args.mask_loss_weight),
        "--topology_enabled",
        "true",
        "--topology_mode",
        config.topology_mode,
        "--lambda_topology",
        str(config.lambda_topology),
        "--rank_loss_weight",
        str(args.rank_loss_weight),
        "--rank_margin",
        str(config.rank_margin),
        "--topology_warmup_epochs",
        "20",
        "--topology_ramp_epochs",
        "10",
        "--self_init_weight",
        str(config.self_init_weight),
    ]
    if args.no_cuda:
        command.append("--no_cuda")
    return command


def _run_one(
    dataset_name: str,
    data_path: Path,
    config: Config,
    seed: int,
    gpu: int,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = output_root / dataset_name / config.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = _job_command(dataset_name, data_path, config, seed, gpu, run_dir, args)
    (run_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "dataset": dataset_name,
                "config": config.name,
                "seed": seed,
                "gpu": gpu,
                "topology_mode": config.topology_mode,
                "lambda_topology": config.lambda_topology,
                "rank_margin": config.rank_margin,
                "self_init_weight": config.self_init_weight,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return {
            "dataset": dataset_name,
            "config": config.name,
            "seed": seed,
            "gpu": gpu,
            "status": "dry_run",
            "save_dir": str(run_dir),
            "topology_mode": config.topology_mode,
            "lambda_topology": config.lambda_topology,
            "rank_margin": config.rank_margin,
            "self_init_weight": config.self_init_weight,
        }
    env = os.environ.copy()
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/mpl",
        }
    )
    log_path = run_dir / "run.log"
    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    summary_path = run_dir / "summary.json"
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "config": config.name,
        "seed": seed,
        "gpu": gpu,
        "returncode": int(process.returncode),
        "status": "completed" if process.returncode == 0 and summary_path.exists() else "failed",
        "save_dir": str(run_dir),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "topology_mode": config.topology_mode,
        "lambda_topology": config.lambda_topology,
        "rank_margin": config.rank_margin,
        "self_init_weight": config.self_init_weight,
    }
    if summary_path.exists():
        try:
            result["summary"] = json.loads(summary_path.read_text())
        except json.JSONDecodeError as exc:
            result["status"] = "failed_invalid_summary"
            result["summary_error"] = str(exc)
    return result


def _write_runs_csv(records: list[dict[str, Any]], output_root: Path) -> None:
    fields = [
        "dataset",
        "config",
        "topology_mode",
        "lambda_topology",
        "rank_margin",
        "self_init_weight",
        "seed",
        "gpu",
        "status",
        "returncode",
        "save_dir",
        "summary_path",
        "log_path",
        "ari",
        "nmi",
        "acc",
        "fmi",
        "self_mass",
        "edge_entropy",
        "effective_neighbor_count",
        "topology_loss",
        "rank_loss",
        "rank_active_fraction",
        "reconstruction_loss",
        "mask_loss",
        "source_path",
        "source_sha256",
        "runner_source_sha256",
        "model_source_sha256",
        "gate_source_sha256",
        "k_source",
        "labels_used_during_fit",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            summary = record.get("summary") or {}
            metrics = summary.get("metrics") or {}
            row = {key: record.get(key, "") for key in fields}
            for key in (
                "self_mass",
                "edge_entropy",
                "effective_neighbor_count",
                "topology_loss",
                "rank_loss",
                "rank_active_fraction",
                "reconstruction_loss",
                "mask_loss",
                "source_path",
                "source_sha256",
                "runner_source_sha256",
                "model_source_sha256",
                "gate_source_sha256",
                "k_source",
                "labels_used_during_fit",
            ):
                row[key] = summary.get(key, "")
            for key in ("ari", "nmi", "acc", "fmi"):
                row[key] = metrics.get(key, "")
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.max_parallel <= 0 or args.max_parallel > len(GPU_POOL):
        raise SystemExit(f"--max-parallel must be in [1, {len(GPU_POOL)}]")

    dataset_names = _split_csv(args.datasets)
    missing = [name for name in dataset_names if name not in DATASETS]
    if missing:
        raise SystemExit(f"unknown dataset(s): {missing}")

    if args.configs:
        config_names = _split_csv(args.configs)
        missing_configs = [name for name in config_names if name not in CONFIGS]
        if missing_configs:
            raise SystemExit(f"unknown config(s): {missing_configs}")
    else:
        config_names = list(CONFIGS.keys())

    jobs = [
        (
            dataset_name,
            DATASETS[dataset_name],
            CONFIGS[config_name],
            seed,
            GPU_POOL[index % len(GPU_POOL)],
        )
        for index, (dataset_name, config_name, seed) in enumerate(
            (dataset_name, config_name, seed)
            for dataset_name in dataset_names
            for config_name in config_names
            for seed in SEEDS
        )
    ]
    output_root = Path(args.output_dir).resolve()
    records: list[dict[str, Any]] = []
    if args.dry_run:
        for dataset_name, data_path, config, seed, gpu in jobs:
            records.append(_run_one(dataset_name, data_path, config, seed, gpu, output_root, args))
    else:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
            futures = [
                executor.submit(
                    _run_one,
                    dataset_name,
                    data_path,
                    config,
                    seed,
                    gpu,
                    output_root,
                    args,
                )
                for dataset_name, data_path, config, seed, gpu in jobs
            ]
            for future in as_completed(futures):
                records.append(future.result())
                latest = records[-1]
                print(
                    f"[{len(records)}/{len(jobs)}] {latest['dataset']} "
                    f"{latest['config']} seed={latest['seed']} status={latest['status']}",
                    flush=True,
                )
    records.sort(key=lambda row: (row["dataset"], row["config"], int(row["seed"])))
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": {name: str(DATASETS[name].resolve()) for name in dataset_names},
                "configs": {
                    name: {
                        "topology_mode": CONFIGS[name].topology_mode,
                        "lambda_topology": CONFIGS[name].lambda_topology,
                        "rank_margin": CONFIGS[name].rank_margin,
                        "self_init_weight": CONFIGS[name].self_init_weight,
                    }
                    for name in config_names
                },
                "seeds": list(SEEDS),
                "gpu_pool": list(GPU_POOL),
                "mask_loss_weight": float(args.mask_loss_weight),
                "rank_loss_weight": float(args.rank_loss_weight),
                "records": records,
            },
            indent=2,
            default=str,
        )
    )
    _write_runs_csv(records, output_root)
    failed = [row for row in records if row["status"] not in {"completed", "dry_run"}]
    print(f"wrote {len(records)} records to {output_root}; failed={len(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
