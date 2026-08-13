#!/usr/bin/env python3
"""Run the manifest-frozen V21 readout-fix extension matrix."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "result/V21/v21_extended13_readoutfix_manifest_20260811.json"
DEFAULT_OUTPUT = ROOT / "result/V21/v21_extended13_readoutfix_v1"
EXPERIMENT_PROTOCOL_ID = "v21_assignment_adversarial_extended13_readoutfix_v1"
MODEL_PROTOCOL_ID = "v21_assignment_adversarial_v3_readoutfix_v1"
ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})
VARIANT_CONFIGS = {
    "topology_assignment_adversarial": ROOT
    / "methods/TopoGate/V21_assignment_adversarial_gate/configs/v21_topology_assignment_adversarial_readoutfix.yaml",
    "scmae_only": ROOT
    / "methods/TopoGate/V21_assignment_adversarial_gate/configs/v21_scmae_only_readoutfix.yaml",
}
DEFAULT_SEEDS = (42, 123, 7)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != EXPERIMENT_PROTOCOL_ID:
        raise ValueError("V21 extended runner received an incompatible manifest protocol")
    if payload.get("model_protocol_id") != MODEL_PROTOCOL_ID:
        raise ValueError("V21 extended manifest has an incompatible model protocol")
    if payload.get("selection_policy", {}).get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("V21 extension selection must be outcome-independent")
    records = [row for row in payload.get("datasets", []) if row.get("status") == "eligible"]
    if len(records) != int(payload.get("expected_dataset_count", -1)):
        raise ValueError("V21 extension eligible dataset count does not match the manifest")
    for row in records:
        for key in ("dataset_id", "name", "source_path", "input_protocol"):
            if not row.get(key):
                raise ValueError(f"V21 extension record is missing {key}")
        if row["input_protocol"] not in {"clubench_bridge", "shared_text"}:
            raise ValueError(f"unsupported V21 extension input protocol: {row['input_protocol']}")
        if not Path(str(row["source_path"])).is_file():
            raise FileNotFoundError(f"V21 extension source is missing: {row['source_path']}")
    return {**payload, "datasets": records}


def build_jobs(
    manifest: dict[str, Any],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    requested: set[str],
    output_root: Path,
) -> list[dict[str, Any]]:
    records = [row for row in manifest["datasets"] if not requested or str(row["dataset_id"]) in requested]
    if requested and requested != {str(row["dataset_id"]) for row in records}:
        missing = sorted(requested - {str(row["dataset_id"]) for row in records})
        raise ValueError(f"unknown V21 extension dataset ids: {missing}")
    records.sort(
        key=lambda row: (
            int(row.get("profile", {}).get("n_samples", 0)) * int(row.get("profile", {}).get("n_features", 0)),
            str(row["dataset_id"]),
        )
    )
    jobs = []
    for record in records:
        for variant in variants:
            for seed in seeds:
                dataset_id = str(record["dataset_id"])
                run_key = f"{manifest['manifest_id']}::{dataset_id}::{variant}::seed{seed}"
                output = output_root / dataset_id / variant / f"seed{seed}"
                jobs.append(
                    {
                        "run_key": run_key,
                        "record": record,
                        "variant": variant,
                        "seed": int(seed),
                        "config": VARIANT_CONFIGS[variant],
                        "output": output,
                    }
                )
    return jobs


def _required_outputs(variant: str) -> tuple[str, ...]:
    required = (
        "summary.json",
        "metrics.json",
        "resolved_config.json",
        "training_history.json",
        "readout_profile.json",
        "preprocess_profile.json",
        "predictions.npy",
        "labels_true.npy",
        "embedding_final.npy",
        "checkpoint.pt",
        "run_record.json",
    )
    if variant == "topology_assignment_adversarial":
        required += ("cluster_probabilities.npy", "student_t_predictions.npy", "graph_profile.json", "stats_profile.json")
    return required


def _completed(job: dict[str, Any]) -> bool:
    output = Path(job["output"])
    if any(not (output / name).is_file() for name in _required_outputs(str(job["variant"]))):
        return False
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
        record = json.loads((output / "run_record.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "completed"
        and summary.get("protocol_id") == MODEL_PROTOCOL_ID
        and summary.get("variant") == job["variant"]
        and summary.get("dataset") == job["record"]["name"]
        and int(summary.get("seed", -1)) == int(job["seed"])
        and summary.get("labels_used_during_fit") is False
        and summary.get("prediction_semantics") == "kmeans_embedding_known_k"
        and config.get("readout_mode") == "kmeans_embedding"
        and record.get("run_key") == job["run_key"]
        and record.get("status") == "completed"
    )


def _run_job(
    job: dict[str, Any],
    *,
    manifest: dict[str, Any],
    gpu: int | None,
    epochs: int | None,
    warmup_epochs: int | None,
    force: bool,
) -> dict[str, Any]:
    output = Path(job["output"])
    if not force and _completed(job):
        return {"run_key": job["run_key"], "status": "completed", "skipped": True}
    output.mkdir(parents=True, exist_ok=True)
    record = job["record"]
    run_record = {
        "run_key": job["run_key"],
        "status": "running",
        "manifest_id": manifest["manifest_id"],
        "experiment_protocol_id": manifest["protocol_id"],
        "model_protocol_id": MODEL_PROTOCOL_ID,
        "dataset_id": record["dataset_id"],
        "dataset": record["name"],
        "source_path": record["source_path"],
        "source_hash": record.get("processed_sha256", record.get("source_hash", "unavailable")),
        "source_provenance_status": record.get("source_provenance_status", "recorded"),
        "input_protocol": record["input_protocol"],
        "variant": job["variant"],
        "seed": int(job["seed"]),
        "physical_gpu": gpu,
        "labels_used_during_fit": False,
        "labels_used_for_preprocessing": False,
        "labels_used_for_graph": False,
        "labels_used_for_gate": False,
        "labels_used_for_loss": False,
        "extension_labels_used_for_selection": False,
        "K_source": "benchmark_oracle_from_y",
        "engineering_epoch_override": epochs,
    }
    _write_json(output / "run_record.json", run_record)
    command = [
        sys.executable,
        "-m",
        "methods.TopoGate.V21_assignment_adversarial_gate.run",
        "--data",
        str(record["source_path"]),
        "--dataset-name",
        str(record["name"]),
        "--input-protocol",
        str(record["input_protocol"]),
        "--config",
        str(job["config"]),
        "--output-dir",
        str(output),
        "--seed",
        str(job["seed"]),
        "--device",
        "cpu" if gpu is None else "cuda",
    ]
    if gpu is not None:
        command.extend(["--gpu", str(gpu)])
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if warmup_epochs is not None:
        command.extend(["--warmup-epochs", str(warmup_epochs)])
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": str(output / "mpl"),
        }
    )
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_path = output / "launcher.log"
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    run_record.update(
        {
            "status": "completed" if completed.returncode == 0 else "incomplete_compute",
            "return_code": int(completed.returncode),
            "wall_seconds": float(time.time() - started),
            "log": str(log_path),
        }
    )
    if completed.returncode == 0:
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "manifest_id": manifest["manifest_id"],
                "experiment_protocol_id": manifest["protocol_id"],
                "dataset_id": record["dataset_id"],
                "run_key": job["run_key"],
                "extension_labels_used_for_selection": False,
                "selection_evidence_type": "held-out extension transfer",
            }
        )
        _write_json(summary_path, summary)
    _write_json(output / "run_record.json", run_record)
    if completed.returncode == 0 and not _completed(job):
        run_record["status"] = "incomplete_compute"
        run_record["contract_error"] = "process exited successfully but required V21 artifacts failed validation"
        _write_json(output / "run_record.json", run_record)
    return {"run_key": job["run_key"], "status": run_record["status"], "return_code": completed.returncode}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None, help="engineering smoke override; omit for the formal 80 epochs")
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    variants = tuple(str(value) for value in args.variants)
    if not variants or set(variants) - set(VARIANT_CONFIGS):
        raise ValueError(f"variants must be a subset of {sorted(VARIANT_CONFIGS)}")
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or any(seed <= 0 for seed in seeds):
        raise ValueError("seeds must be positive")
    if args.cpu and args.gpu is not None:
        raise ValueError("--cpu and --gpu are mutually exclusive")
    gpu = None if args.cpu else args.gpu
    if gpu is None and not args.cpu and not args.dry_run:
        raise ValueError("formal execution requires either --cpu or an explicit --gpu in 1..6")
    if gpu is not None and int(gpu) not in ALLOWED_GPUS:
        raise ValueError(f"physical GPU {gpu} is forbidden; allowed={sorted(ALLOWED_GPUS)}")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.warmup_epochs is not None and args.epochs is None:
        raise ValueError("--warmup-epochs requires --epochs")
    worker_count = int(args.num_workers)
    worker_id = int(args.worker_id)
    if worker_count <= 0 or not 0 <= worker_id < worker_count:
        raise ValueError("worker-id must be in [0, num-workers)")

    all_jobs = build_jobs(manifest, variants, seeds, set(args.datasets or []), args.output_dir)
    jobs = [job for index, job in enumerate(all_jobs) if index % worker_count == worker_id]
    header = {
        "manifest_id": manifest["manifest_id"],
        "experiment_protocol_id": manifest["protocol_id"],
        "model_protocol_id": MODEL_PROTOCOL_ID,
        "jobs_total": len(all_jobs),
        "jobs_for_worker": len(jobs),
        "worker_id": worker_id,
        "num_workers": worker_count,
        "physical_gpu": gpu,
        "variants": list(variants),
        "seeds": list(seeds),
        "labels_used_during_fit": False,
        "extension_labels_used_for_selection": False,
        "environment": {"python": platform.python_version()},
    }
    print(json.dumps(header, ensure_ascii=True), flush=True)
    if args.dry_run:
        for job in jobs:
            print(job["run_key"])
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job['run_key']}", flush=True)
        row = _run_job(
            job,
            manifest=manifest,
            gpu=None if args.cpu else int(gpu),
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            force=bool(args.force),
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    worker_summary = {
        **header,
        "completed": sum(row["status"] == "completed" for row in rows),
        "incomplete_compute": sum(row["status"] != "completed" for row in rows),
        "runs": rows,
    }
    _write_json(args.output_dir / f"matrix_worker{worker_id}.json", worker_summary)
    return 0 if worker_summary["incomplete_compute"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
