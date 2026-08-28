#!/usr/bin/env python
"""Run the registered V12 stage-2 edge-rank paired benchmark.

The launcher owns only orchestration.  The V12 runner remains the source of
truth for preprocessing, graph construction, K protocol, losses, and output
contracts.  Jobs are assigned explicitly to the allowed physical GPU pool.

Stage 2 extends stage 1 with the per-edge rank signal registered in
`rank_alignment_loss`, expanding the dataset pool to five AHDPC datasets and
allowing a subset of variants to be selected for partial smoke runs.
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
    "enron": REPO / "datasets" / "enron.npz",
    "balance_scale": REPO / "datasets" / "AHDPC" / "processed" / "balance_scale.npz",
    "spect_heart": REPO / "datasets" / "AHDPC" / "processed" / "spect_heart.npz",
    "vehicle": REPO / "datasets" / "AHDPC" / "processed" / "vehicle.npz",
}


@dataclass(frozen=True)
class Variant:
    name: str
    topology_enabled: bool
    topology_mode: str
    lambda_topology: float
    rank_loss_weight: float = 0.1
    rank_margin: float = 0.1


VARIANTS = {
    "nomix": Variant("nomix", False, "self_null", 0.0, rank_loss_weight=0.0),
    "edge_only": Variant("edge_only", True, "edge_only", 0.1),
    "self_null_lambda001": Variant("self_null_lambda001", True, "self_null", 0.01),
    "self_null_lambda003": Variant("self_null_lambda003", True, "self_null", 0.03),
    "self_null_lambda01": Variant("self_null_lambda01", True, "self_null", 0.1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO
            / "result"
            / "V12"
            / "v12_edge_rank_stage2_2026-08-04"
        ),
    )
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-loss-weight", type=float, default=0.1)
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--datasets", default="flame,enron,balance_scale,spect_heart,vehicle")
    parser.add_argument(
        "--variants",
        default="nomix,edge_only,self_null_lambda01",
        help="Comma-separated subset of variant names; defaults to the three "
        "core variants (NoMix/edge-only/self_null@0.1). To reproduce the full "
        "stage-1 matrix use 'nomix,edge_only,self_null_lambda001,"
        "self_null_lambda003,self_null_lambda01'.",
    )
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _split_csv(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def _job_command(
    dataset_name: str,
    data_path: Path,
    variant: Variant,
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
        f"topogate_v12_{variant.name}",
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
        str(variant.topology_enabled).lower(),
        "--topology_mode",
        variant.topology_mode,
        "--lambda_topology",
        str(variant.lambda_topology),
        "--rank_loss_weight",
        str(variant.rank_loss_weight),
        "--rank_margin",
        str(variant.rank_margin),
        "--topology_warmup_epochs",
        "20",
        "--topology_ramp_epochs",
        "10",
        "--self_init_weight",
        "0.8",
    ]
    if args.no_cuda:
        command.append("--no_cuda")
    return command


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
        "variant": variant.name,
        "seed": seed,
        "gpu": gpu,
        "returncode": int(process.returncode),
        "status": "completed" if process.returncode == 0 and summary_path.exists() else "failed",
        "save_dir": str(run_dir),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
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
        "variant",
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
    variant_names = _split_csv(args.variants)
    missing_variants = [name for name in variant_names if name not in VARIANTS]
    if missing_variants:
        raise SystemExit(f"unknown variant(s): {missing_variants}")

    jobs = [
        (dataset_name, DATASETS[dataset_name], VARIANTS[variant_name], seed, GPU_POOL[index % len(GPU_POOL)])
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
                    dataset_name,
                    data_path,
                    variant,
                    seed,
                    gpu,
                    output_root,
                    args,
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
                "datasets": {name: str(path.resolve()) for name in dataset_names for path in [DATASETS[name]]},
                "variants": [VARIANTS[name].__dict__ for name in variant_names],
                "seeds": list(SEEDS),
                "gpu_pool": list(GPU_POOL),
                "mask_loss_weight": float(args.mask_loss_weight),
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
