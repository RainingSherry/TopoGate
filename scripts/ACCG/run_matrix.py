#!/usr/bin/env python3
"""Run a frozen ACCG manifest; training requires an explicit --execute flag."""
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
ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})
SUPPORTED_MANIFEST_IDS = frozenset({"accg_locked_real_panel_v1", "accg_locked_real_panel_v2"})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_id") not in SUPPORTED_MANIFEST_IDS:
        raise ValueError("runner received an incompatible ACCG manifest")
    if payload.get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("ACCG panel selection must be outcome-independent")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("ACCG manifest has no jobs")
    keys = [str(job.get("run_key")) for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("ACCG manifest contains duplicate run keys")
    return payload


def _completed(job: dict[str, Any]) -> bool:
    output = Path(str(job["output_dir"]))
    required = [output / "summary.json", output / "runner_profile.json", output / "T_c/metrics.json"]
    if job["role"] == "main":
        required.extend(output / f"{arm}/metrics.json" for arm in ("N", "R", "T_s"))
        required.append(output / "branchpoint.pt")
    if any(not path.is_file() for path in required):
        return False
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        runner = json.loads((output / "runner_profile.json").read_text(encoding="utf-8"))
        resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "completed"
        and int(summary.get("seed", -1)) == int(job["seed"])
        and summary.get("variant") == resolved.get("variant")
        and runner.get("dataset") == job["record"]["name"]
        and runner.get("dataset_sha256") == job["record"]["source_sha256"]
        and runner.get("config_sha256") == job.get("config_sha256")
        and runner.get("labels_used_during_fit") is False
        and bool(runner.get("branchpoint_reused")) == (job["role"] == "ablation")
    )


def _canonical_control_ready(job: dict[str, Any]) -> bool:
    if job.get("role") != "ablation" or not job.get("reused_from"):
        return False
    output = Path(str(job["reused_from"]))
    required = [
        output / "summary.json",
        output / "runner_profile.json",
        output / "resolved_config.json",
        output / "branchpoint.pt",
    ]
    required.extend(output / arm / "metrics.json" for arm in ("N", "R", "T_s", "T_c"))
    if any(not path.is_file() for path in required):
        return False
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        runner = json.loads((output / "runner_profile.json").read_text(encoding="utf-8"))
        resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        summary.get("status") == "completed"
        and summary.get("variant") == "accg_joint"
        and int(summary.get("seed", -1)) == int(job["seed"])
        and resolved.get("variant") == "accg_joint"
        and runner.get("dataset") == job["record"]["name"]
        and runner.get("dataset_sha256") == job["record"]["source_sha256"]
        and runner.get("labels_used_during_fit") is False
        and runner.get("branchpoint_reused") is False
    )


def _command(job: dict[str, Any], gpu: int | None, epochs: int | None, warmup_epochs: int | None) -> list[str]:
    record = job["record"]
    command = [
        sys.executable,
        "-m",
        "methods.TopoGate.ACCG_action_constrained_gate.run",
        "--data",
        str(record["source_path"]),
        "--dataset-name",
        str(record["name"]),
        "--input-protocol",
        str(record["input_protocol"]),
        "--config",
        str(job["config"]),
        "--output-dir",
        str(job["output_dir"]),
        "--seed",
        str(job["seed"]),
        "--device",
        "cpu" if gpu is None else "cuda",
    ]
    if gpu is not None:
        command.extend(["--gpu", str(gpu)])
    if record.get("n_clusters") is not None:
        command.extend(["--n-clusters", str(record["n_clusters"])])
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if warmup_epochs is not None:
        command.extend(["--warmup-epochs", str(warmup_epochs)])
    if job["role"] == "ablation":
        command.extend(["--branchpoint-from", str(job["reused_from"])])
    return command


def _run_job(
    job: dict[str, Any],
    *,
    gpu: int | None,
    epochs: int | None,
    warmup_epochs: int | None,
    force: bool,
) -> dict[str, Any]:
    if not force and _completed(job):
        return {"run_key": job["run_key"], "status": "completed", "skipped": True}
    if job["role"] == "ablation" and not _canonical_control_ready(job):
        return {
            "run_key": job["run_key"],
            "status": "blocked_missing_canonical_control",
            "reused_from": job["reused_from"],
        }
    output = Path(str(job["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "run_key": job["run_key"],
        "status": "running",
        "role": job["role"],
        "dataset_id": job["dataset_id"],
        "seed": int(job["seed"]),
        "physical_gpu": gpu,
        "source_path": job["record"]["source_path"],
        "source_sha256": job["record"]["source_sha256"],
        "config": job["config"],
        "config_sha256": job.get("config_sha256"),
        "reused_from": job.get("reused_from"),
        "labels_used_during_fit": False,
        "started_at": time.time(),
    }
    _write_json(output / "run_record.json", record)
    command = _command(job, gpu, epochs, warmup_epochs)
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
            "NUMBA_CACHE_DIR": str(output / "numba_cache"),
        }
    )
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    log = output / "launcher.log"
    started = time.time()
    with log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    record.update(
        {
            "status": "completed" if completed.returncode == 0 and _completed(job) else "incomplete_compute",
            "return_code": int(completed.returncode),
            "wall_seconds": float(time.time() - started),
            "log": str(log),
        }
    )
    _write_json(output / "run_record.json", record)
    return {"run_key": job["run_key"], "status": record["status"], "return_code": completed.returncode}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="required to launch any training")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=("main", "ablation"),
        default=["main"],
        help="run main panels first; launch ablations in a second phase after canonical branchpoints exist",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="engineering-only override")
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--queue-state", type=Path, default=None)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.cpu and args.gpu is not None:
        raise ValueError("--cpu and --gpu are mutually exclusive")
    gpu = None if args.cpu else args.gpu
    if args.execute and gpu is None and not args.cpu:
        raise ValueError("execution requires --cpu or an explicit --gpu in 1..6")
    if gpu is not None and gpu not in ALLOWED_GPUS:
        raise ValueError(f"physical GPU {gpu} is forbidden; allowed={sorted(ALLOWED_GPUS)}")
    if args.warmup_epochs is not None and args.epochs is None:
        raise ValueError("--warmup-epochs requires --epochs")
    requested = set(args.datasets or [])
    jobs = [
        job
        for job in manifest["jobs"]
        if job["role"] in set(args.roles) and (not requested or job["dataset_id"] in requested)
    ]
    if requested - {job["dataset_id"] for job in jobs}:
        raise ValueError(f"unknown requested datasets: {sorted(requested - {job['dataset_id'] for job in jobs})}")
    if args.num_workers <= 0 or not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    jobs = [job for index, job in enumerate(jobs) if index % args.num_workers == args.worker_id]
    header = {
        "manifest_id": manifest["manifest_id"],
        "execute": bool(args.execute),
        "jobs_for_worker": len(jobs),
        "worker_id": int(args.worker_id),
        "num_workers": int(args.num_workers),
        "physical_gpu": gpu,
        "environment": {"python": platform.python_version()},
    }
    print(json.dumps(header, ensure_ascii=True), flush=True)
    if not args.execute:
        for job in jobs:
            print(json.dumps({"run_key": job["run_key"], "role": job["role"], "reused_from": job.get("reused_from")}, ensure_ascii=True))
        return 0
    rows = []
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job['run_key']}", flush=True)
        rows.append(
            _run_job(
                job,
                gpu=gpu,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                force=args.force,
            )
        )
        if args.queue_state is not None:
            _write_json(args.queue_state, {**header, "rows": rows, "updated_at": time.time()})
    return 0 if all(row["status"] == "completed" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
