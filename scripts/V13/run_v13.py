#!/usr/bin/env python
"""Run the V13 Gumbel-Top-k experiments.

Stage 1 smoke: 2 datasets (flame, enron) × 2 variants × 3 seeds = 12 runs.
Stage 2 formal: 5 datasets × 2 variants × 3 seeds = 30 runs.

V13 replaces the V12 softmax + rank_loss gate with Gumbel-Top-k hard
selection.  The only two variants are:
- nomix: topology disabled (pure AE baseline).
- topk2: topology enabled with top_k=2 (hard top-k selection).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "methods" / "TopoGate" / "V13_hard_gate" / "run_npz.py"
GPU_POOL = (1, 4, 5)
SEEDS = (42, 123, 7)

DATASETS = {
    "flame": REPO / "datasets" / "AHDPC" / "processed" / "flame.npz",
    "balance_scale": REPO / "datasets" / "AHDPC" / "processed" / "balance_scale.npz",
    "spect_heart": REPO / "datasets" / "AHDPC" / "processed" / "spect_heart.npz",
    "vehicle": REPO / "datasets" / "AHDPC" / "processed" / "vehicle.npz",
    "enron": REPO / "datasets" / "enron.npz",
}


@dataclass(frozen=True)
class Variant:
    name: str
    topology_enabled: bool
    top_k_neighbors: int = 2


VARIANTS = {
    "nomix": Variant("nomix", False),
    "topk2": Variant("topk2", True, top_k_neighbors=2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "result" / "V13" / "v13_hard_gate_2026-08-04"),
    )
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-loss-weight", type=float, default=0.1)
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--lambda-topology", type=float, default=0.1)
    parser.add_argument("--topology-warmup-epochs", type=int, default=20)
    parser.add_argument("--topology-ramp-epochs", type=int, default=10)
    parser.add_argument("--gumbel-tau", type=float, default=1.0)
    parser.add_argument("--gumbel-tau-min", type=float, default=0.1)
    parser.add_argument("--gumbel-tau-anneal-epochs", type=int, default=50)
    parser.add_argument(
        "--datasets",
        default="flame,balance_scale,spect_heart,vehicle,enron",
    )
    parser.add_argument(
        "--variants",
        default="nomix,topk2",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", choices=["smoke", "formal"], default="formal")
    return parser.parse_args()


def _split(value: str) -> list[str]:
    return [t.strip() for t in value.split(",") if t.strip()]


def _job_command(
    dataset_name: str,
    data_path: Path,
    variant: Variant,
    seed: int,
    gpu: int,
    save_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--data_path", str(data_path),
        "--save_dir", str(save_dir),
        "--dataset_name", dataset_name,
        "--method_name", "TopoGate",
        "--variant_name", f"topogate_v13_{variant.name}",
        "--seed", str(seed),
        "--gpu", str(gpu),
        "--epochs", str(args.epochs),
        "--hidden_size", str(args.hidden_size),
        "--batch_size", str(args.batch_size),
        "--mask_ratio", str(args.mask_ratio),
        "--neighbor_k", str(args.neighbor_k),
        "--scale_input", "true",
        "--decoder_mode", "legacy_mask_conditioned",
        "--mask_loss_mode", "additive",
        "--mask_loss_weight", str(args.mask_loss_weight),
        "--topology_enabled", str(variant.topology_enabled).lower(),
        "--lambda_topology", str(args.lambda_topology),
        "--topology_warmup_epochs", str(args.topology_warmup_epochs),
        "--topology_ramp_epochs", str(args.topology_ramp_epochs),
        "--gumbel_tau", str(args.gumbel_tau),
        "--gumbel_tau_min", str(args.gumbel_tau_min),
        "--gumbel_tau_anneal_epochs", str(args.gumbel_tau_anneal_epochs),
        "--top_k_neighbors", str(variant.top_k_neighbors),
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")
    return cmd


def _run_one(
    dataset_name: str,
    data_path: Path,
    variant: Variant,
    seed: int,
    gpu: int,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = output_root / dataset_name / variant.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = _job_command(dataset_name, data_path, variant, seed, gpu, run_dir, args)
    (run_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "dataset": dataset_name,
                "variant": variant.name,
                "seed": seed,
                "gpu": gpu,
                "topology_enabled": variant.topology_enabled,
                "top_k_neighbors": variant.top_k_neighbors,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return {
            "dataset": dataset_name,
            "variant": variant.name,
            "seed": seed,
            "gpu": gpu,
            "status": "dry_run",
            "save_dir": str(run_dir),
            "topology_enabled": variant.topology_enabled,
            "top_k_neighbors": variant.top_k_neighbors,
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
        proc = subprocess.run(
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
        "variant": variant.name,
        "seed": seed,
        "gpu": gpu,
        "returncode": int(proc.returncode),
        "status": "completed" if proc.returncode == 0 and summary_path.exists() else "failed",
        "save_dir": str(run_dir),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "topology_enabled": variant.topology_enabled,
        "top_k_neighbors": variant.top_k_neighbors,
    }
    if summary_path.exists():
        try:
            result["summary"] = json.loads(summary_path.read_text())
        except json.JSONDecodeError as exc:
            result["status"] = "failed_invalid_summary"
            result["summary_error"] = str(exc)
    return result


def _write_csv(records: list[dict[str, Any]], output_root: Path) -> None:
    fields = [
        "dataset", "variant", "topology_enabled", "top_k_neighbors",
        "seed", "gpu", "status", "returncode", "save_dir",
        "summary_path", "log_path",
        "ari", "nmi", "acc", "fmi",
        "selected_neighbor_count", "effective_neighbor_count",
        "topology_loss", "reconstruction_loss", "mask_loss",
        "source_sha256", "runner_source_sha256",
        "model_source_sha256", "gate_source_sha256",
        "n_clusters", "k_source", "labels_used_during_fit",
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
                "selected_neighbor_count", "effective_neighbor_count",
                "topology_loss", "reconstruction_loss", "mask_loss",
                "source_sha256", "runner_source_sha256",
                "model_source_sha256", "gate_source_sha256",
                "n_clusters", "k_source", "labels_used_during_fit",
            ):
                row[key] = summary.get(key, "")
            for key in ("ari", "nmi", "acc", "fmi"):
                row[key] = metrics.get(key, "")
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    dataset_names = _split(args.datasets)
    variant_names = _split(args.variants)
    missing_ds = [n for n in dataset_names if n not in DATASETS]
    if missing_ds:
        raise SystemExit(f"unknown dataset(s): {missing_ds}")
    missing_var = [n for n in variant_names if n not in VARIANTS]
    if missing_var:
        raise SystemExit(f"unknown variant(s): {missing_var}")

    if args.stage == "smoke":
        dataset_names = [n for n in dataset_names if n in ("flame", "enron")]

    jobs = [
        (
            dataset_name,
            DATASETS[dataset_name],
            VARIANTS[variant_name],
            seed,
            GPU_POOL[index % len(GPU_POOL)],
        )
        for index, (dataset_name, variant_name, seed) in enumerate(
            (dataset_name, variant_name, seed)
            for dataset_name in dataset_names
            for variant_name in variant_names
            for seed in SEEDS
        )
    ]
    output_root = Path(args.output_dir).resolve()
    records: list[dict[str, Any]] = []
    if args.dry_run:
        for dataset_name, data_path, variant, seed, gpu in jobs:
            records.append(_run_one(dataset_name, data_path, variant, seed, gpu, output_root, args))
    else:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
            futures = [
                executor.submit(
                    _run_one,
                    dataset_name, data_path, variant, seed, gpu, output_root, args,
                )
                for dataset_name, data_path, variant, seed, gpu in jobs
            ]
            for future in as_completed(futures):
                records.append(future.result())
                latest = records[-1]
                print(
                    f"[{len(records)}/{len(jobs)}] {latest['dataset']} "
                    f"{latest['variant']} seed={latest['seed']} status={latest['status']}",
                    flush=True,
                )
    records.sort(key=lambda row: (row["dataset"], row["variant"], int(row["seed"])))
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": {name: str(DATASETS[name].resolve()) for name in dataset_names},
                "variants": {name: vars(VARIANTS[name]) for name in variant_names},
                "seeds": list(SEEDS),
                "gpu_pool": list(GPU_POOL),
                "stage": args.stage,
                "records": records,
            },
            indent=2,
            default=str,
        )
    )
    _write_csv(records, output_root)
    failed = [row for row in records if row["status"] not in {"completed", "dry_run"}]
    print(f"wrote {len(records)} records to {output_root}; failed={len(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
