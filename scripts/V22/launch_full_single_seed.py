#!/usr/bin/env python3
"""Run the V22 full-component panel on the frozen single-seed manifest.

This is a local, resource-aware queue.  It never stops unrelated processes and
never uses physical GPU 0 or 7.  Jobs are ordered from small to large inputs,
so the first wave provides evidence while large sparse/dense controls wait for
an allowed GPU with enough free memory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "datasets" / "external" / "v22_full_single_seed_20260812" / "manifest.json"
DEFAULT_CONFIG = ROOT / "methods" / "TopoGate" / "V22_topology_discriminator_hard_mask" / "configs" / "v22_topology_discriminator_hard_gate.yaml"
DEFAULT_VARIANT = "v22_topology_discriminator_hard_gate"
MODULE = "methods.TopoGate.V22_topology_discriminator_hard_mask.run"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_manifest(path: Path, expected_variant: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != 1 or not str(variants[0]):
        raise ValueError("the single-seed launcher requires exactly one V22 Full variant")
    if expected_variant is not None and str(variants[0]) != str(expected_variant):
        raise ValueError(f"manifest variant {variants[0]!r} does not match --variant {expected_variant!r}")
    if payload.get("seeds") != [42]:
        raise ValueError("the single-seed launcher is frozen to seed 42")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest has no records")
    seen: set[str] = set()
    for record in records:
        for key in ("dataset_id", "name", "input_protocol", "source_path", "source_sha256", "n_samples"):
            if key not in record:
                raise ValueError(f"manifest record missing {key}: {record}")
        dataset_id = str(record["dataset_id"])
        if dataset_id in seen:
            raise ValueError(f"duplicate dataset_id: {dataset_id}")
        seen.add(dataset_id)
        source = Path(str(record["source_path"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        if not str(record["source_sha256"]):
            raise ValueError(f"empty source_sha256 for {dataset_id}")
        if int(record["n_samples"]) <= 0:
            raise ValueError(f"invalid n_samples for {dataset_id}")
        if record.get("labels_available_outer_only") is False and record.get("n_clusters") is None:
            raise ValueError(f"unlabelled record requires explicit n_clusters: {dataset_id}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _query_free_memory() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    free: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index_text, memory_text = [part.strip() for part in line.split(",", 1)]
        free[int(index_text)] = int(float(memory_text))
    return free


def _job_key(record: dict[str, Any], variant: str) -> str:
    return f"{record['dataset_id']}::{variant}::seed42"


def _summary_is_complete(job: dict[str, Any]) -> bool:
    summary_path = Path(job["output_dir"]) / "summary.json"
    config_path = Path(job["output_dir"]) / "resolved_config.json"
    if not summary_path.is_file() or not config_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        summary.get("status") == "completed"
        and summary.get("protocol_id") == job["protocol_id"]
        and summary.get("variant") == job["variant"]
        and int(summary.get("seed", -1)) == 42
        and summary.get("dataset") == job["record"]["name"]
        and summary.get("source_sha256") == job["record"]["source_sha256"]
        and config.get("variant") == job["variant"]
        and int(config.get("epochs", -1)) == int(job["epochs"])
        and config.get("source_sha256") == job["record"]["source_sha256"]
        and (
            job.get("batch_size") is None
            or int(config.get("batch_size", -1)) == int(job["batch_size"])
        )
    )


def _build_jobs(
    manifest: dict[str, Any],
    output_root: Path,
    config_path: Path,
    *,
    variant: str,
    batch_size: int | None = None,
    topology_cache_root: Path | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for record in manifest["records"]:
        dataset_id = str(record["dataset_id"])
        output_dir = output_root / dataset_id / variant / "seed42"
        jobs.append(
            {
                "run_key": _job_key(record, variant),
                "dataset_id": dataset_id,
                "record": record,
                "variant": variant,
                "protocol_id": str(manifest["protocol_id"]),
                "seed": 42,
                "epochs": int(manifest.get("epochs", 80)),
                "config": str(config_path),
                "output_dir": str(output_dir),
                "batch_size": None if batch_size is None else int(batch_size),
                "topology_cache_dir": (
                    str(topology_cache_root / dataset_id / variant / "seed42" / "cache")
                    if topology_cache_root is not None
                    else None
                ),
                "status": "reused" if _summary_is_complete(
                    {
                        "output_dir": str(output_dir),
                        "protocol_id": manifest["protocol_id"],
                        "variant": variant,
                        "record": record,
                        "epochs": int(manifest.get("epochs", 80)),
                        "batch_size": batch_size,
                    }
                ) else "queued",
                "attempts": 0,
                "gpu": None,
                "pid": None,
                "started_at": None,
                "finished_at": None,
                "returncode": None,
                "error": None,
            }
        )
    return sorted(jobs, key=lambda job: (int(job["record"]["n_samples"]), job["dataset_id"]))


def _command(job: dict[str, Any], gpu: int) -> list[str]:
    record = job["record"]
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--data",
        str(record["source_path"]),
        "--dataset-name",
        str(record["name"]),
        "--input-protocol",
        str(record["input_protocol"]),
        "--config",
        str(job["config"]),
        "--variant",
        str(job["variant"]),
        "--output-dir",
        str(job["output_dir"]),
        "--seed",
        "42",
        "--device",
        "cuda",
        "--gpu",
        str(gpu),
        "--epochs",
        str(int(job["epochs"])),
        "--source-sha256",
        str(record["source_sha256"]),
    ]
    if record.get("n_clusters") is not None:
        command.extend(["--n-clusters", str(int(record["n_clusters"]))])
    if job.get("batch_size") is not None:
        command.extend(["--batch-size", str(int(job["batch_size"]))])
    if job.get("topology_cache_dir") is not None:
        command.extend(["--reuse-topology-cache", "--topology-cache-dir", str(job["topology_cache_dir"])])
    return command


def _is_oom(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return "CUDA out of memory" in text or "torch.OutOfMemoryError" in text


def _save_state(path: Path, manifest: dict[str, Any], jobs: list[dict[str, Any]], started_at: str, status: str) -> None:
    _write_json(
        path,
        {
            "manifest_id": manifest["manifest_id"],
            "protocol_id": manifest["protocol_id"],
            "phase": manifest["phase"],
            "started_at": started_at,
            "updated_at": _now(),
            "status": status,
            "jobs": jobs,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Queue V22 full single-seed jobs on local GPUs")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", default=None, help="single V22 Full variant; must match manifest")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result" / "V22" / "v22_full_single_seed_20260812")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--min-free-mib", type=int, default=12000)
    parser.add_argument("--max-parallel-per-gpu", type=int, default=1)
    parser.add_argument("--gpu-pool", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--reuse-topology-cache-root",
        type=Path,
        default=None,
        help="root of a prior completed topology cache tree, keyed by dataset_id",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_parallel_per_gpu <= 0 or args.min_free_mib <= 0:
        raise ValueError("max-parallel-per-gpu and min-free-mib must be positive")
    if any(gpu in {0, 7} or gpu < 0 for gpu in args.gpu_pool):
        raise ValueError("GPU 0 and GPU 7 are forbidden")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if args.batch_size is not None and int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    if args.reuse_topology_cache_root is not None and not args.reuse_topology_cache_root.is_dir():
        raise FileNotFoundError(args.reuse_topology_cache_root)
    manifest = _load_manifest(args.manifest, expected_variant=args.variant)
    variant = str(manifest["variants"][0])
    if args.variant is not None and str(args.variant) != variant:
        raise ValueError(f"--variant {args.variant!r} does not match manifest variant {variant!r}")
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "manifest.json", manifest)
    state_path = output_root / "queue_state.json"
    jobs = _build_jobs(
        manifest,
        output_root,
        args.config,
        variant=variant,
        batch_size=args.batch_size,
        topology_cache_root=args.reuse_topology_cache_root,
    )
    if args.reuse_topology_cache_root is not None:
        missing_cache = [
            job["dataset_id"]
            for job in jobs
            if not (Path(str(job["topology_cache_dir"])) / "topology_statistics.dat").is_file()
        ]
        if missing_cache:
            raise FileNotFoundError(f"topology cache missing for datasets: {missing_cache}")
    started_at = _now()
    if args.dry_run:
        _save_state(state_path, manifest, jobs, started_at, "dry_run")
        print(json.dumps({"manifest_id": manifest["manifest_id"], "jobs": jobs}, ensure_ascii=True, indent=2, default=str))
        return 0

    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    processes: dict[str, subprocess.Popen[Any]] = {}
    log_handles: dict[str, Any] = {}
    interrupted = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        for process in processes.values():
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while True:
        for job in jobs:
            key = job["run_key"]
            process = processes.get(key)
            if process is None:
                continue
            returncode = process.poll()
            if returncode is None:
                continue
            process_log = logs_root / f"{job['dataset_id']}.seed42.log"
            handle = log_handles.pop(key, None)
            if handle is not None:
                handle.close()
            processes.pop(key, None)
            job["pid"] = None
            job["returncode"] = int(returncode)
            job["finished_at"] = _now()
            if returncode == 0 and _summary_is_complete(job):
                job["status"] = "completed"
                job["error"] = None
            elif _is_oom(process_log) and int(job["attempts"]) < 3 and not interrupted:
                job["status"] = "queued"
                job["error"] = "CUDA out of memory; requeued"
            else:
                job["status"] = "incomplete_compute"
                job["error"] = f"returncode={returncode}; log={process_log}"

        if interrupted:
            for job in jobs:
                if job["run_key"] in processes:
                    job["status"] = "incomplete_compute"
                    job["error"] = "launcher interrupted"
            _save_state(state_path, manifest, jobs, started_at, "interrupted")
            return 130

        active_by_gpu: dict[int, int] = {}
        for job in jobs:
            if job["status"] == "running" and job["gpu"] is not None:
                active_by_gpu[int(job["gpu"])] = active_by_gpu.get(int(job["gpu"]), 0) + 1
        free_memory = _query_free_memory()
        for job in jobs:
            if job["status"] != "queued":
                continue
            candidates = [
                gpu
                for gpu in args.gpu_pool
                if active_by_gpu.get(gpu, 0) < args.max_parallel_per_gpu
                and free_memory.get(gpu, 0) >= args.min_free_mib
            ]
            if not candidates:
                break
            gpu = max(candidates, key=lambda item: free_memory.get(item, 0))
            output_dir = Path(job["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                output_dir / "manifest_record.json",
                job["record"]
                | {
                    "run_key": job["run_key"],
                    "batch_size": job.get("batch_size"),
                    "topology_cache_source": job.get("topology_cache_dir"),
                },
            )
            log_path = logs_root / f"{job['dataset_id']}.seed42.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            env.setdefault("MPLCONFIGDIR", str(output_root / "mplconfig"))
            command = _command(job, gpu)
            _write_json(
                output_dir / "launch_record.json",
                {
                    "run_key": job["run_key"],
                    "gpu": gpu,
                    "started_at": _now(),
                    "command": command,
                    "batch_size": job.get("batch_size"),
                    "topology_cache_source": job.get("topology_cache_dir"),
                },
            )
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            processes[job["run_key"]] = process
            log_handles[job["run_key"]] = log_handle
            job["status"] = "running"
            job["attempts"] = int(job["attempts"]) + 1
            job["gpu"] = gpu
            job["pid"] = process.pid
            job["started_at"] = _now()
            job["error"] = None
            active_by_gpu[gpu] = active_by_gpu.get(gpu, 0) + 1
            free_memory[gpu] = 0

        terminal = all(job["status"] in {"completed", "reused", "incomplete_compute"} for job in jobs)
        status = "completed" if terminal and all(job["status"] in {"completed", "reused"} for job in jobs) else "running"
        _save_state(state_path, manifest, jobs, started_at, status)
        if terminal:
            for handle in log_handles.values():
                handle.close()
            print(json.dumps({"manifest_id": manifest["manifest_id"], "status": status, "jobs": jobs}, ensure_ascii=True, indent=2, default=str))
            return 0 if status == "completed" else 2
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
