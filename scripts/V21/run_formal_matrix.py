#!/usr/bin/env python3
"""Launch the frozen V21 six-dataset Full/scMAE-only matrix.

The launcher never stops an existing process. It waits for an explicitly
allowed physical GPU to be idle, isolates that GPU for one child, and records
each stable run key and terminal state under a unique result root.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "result/V21/v21_formal6_full_20260811_graphfix"
# Keep the matrix protocol separate from the model/config protocol recorded by
# each run.  The former freezes the benchmark job set; the latter identifies
# the V21 implementation and is emitted by the model runner.
MATRIX_PROTOCOL_ID = "v21_assignment_adversarial_full6_graphfix_v1"
MODEL_PROTOCOL_ID = "v21_assignment_adversarial_v2_graphfix_v1"
DEFAULT_GPU_POOL = (1, 2, 3, 4, 5, 6)
FORBIDDEN_GPUS = frozenset({0, 7})
SEEDS = (42, 123, 7)
CPU_FALLBACK_DATASETS = frozenset({"cnae9", "sms_spam_collection", "hate_speech"})
EXTERNAL_CPU_FALLBACK_DATASETS = frozenset({"Mouse_retina", "Baron Human", "Campbell"})
DEFAULT_CPU_WORKERS = 6
DATASETS = (
    ("cnae9", "shared_text", ROOT / "datasets/cnae9.npz"),
    ("Mouse_retina", "clubench_bridge", ROOT / "datasets/Mouse_retina.npz"),
    ("Baron Human", "clubench_bridge", ROOT / "datasets/Baron Human.npz"),
    ("Campbell", "clubench_bridge", ROOT / "datasets/Campbell.npz"),
    ("sms_spam_collection", "shared_text", ROOT / "datasets/sms_spam_collection.npz"),
    ("hate_speech", "shared_text", ROOT / "datasets/hate_speech.npz"),
)
VARIANTS = (
    (
        "topology_assignment_adversarial",
        ROOT
        / "methods/TopoGate/V21_assignment_adversarial_gate/configs/v21_topology_assignment_adversarial.yaml",
    ),
    (
        "scmae_only",
        ROOT / "methods/TopoGate/V21_assignment_adversarial_gate/configs/v21_scmae_only.yaml",
    ),
)
POLL_SECONDS = 30
IDLE_MEMORY_MIB = 3000


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _safe_name(value: str) -> str:
    return value.casefold().replace(" ", "_")


def _run_key(dataset: str, protocol: str, variant: str, seed: int) -> str:
    return f"{_safe_name(dataset)}__{protocol}__{variant}__seed{seed}"


def _jobs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for dataset, protocol, data_path in DATASETS:
        for variant, config_path in VARIANTS:
            for seed in SEEDS:
                key = _run_key(dataset, protocol, variant, seed)
                output = OUTPUT_ROOT / f"{_safe_name(dataset)}__{protocol}" / variant / f"seed{seed}"
                items.append(
                    {
                        "key": key,
                        "dataset": dataset,
                        "input_protocol": protocol,
                        "data": str(data_path.resolve()),
                        "variant": variant,
                        "config": str(config_path.resolve()),
                        "seed": int(seed),
                        "output": str(output),
                        "log": str(OUTPUT_ROOT / "logs" / f"{key}.log"),
                        "status": "queued",
                    }
                )
    return items


def _stage_spec(gpu_pool: tuple[int, ...], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "protocol_id": MATRIX_PROTOCOL_ID,
        "stage": "formal_six_dataset_matrix",
        "datasets": [
            {
                "dataset": dataset,
                "input_protocol": protocol,
                "data": str(path.resolve()),
            }
            for dataset, protocol, path in DATASETS
        ],
        "variants": [variant for variant, _config in VARIANTS],
        "seeds": list(SEEDS),
        "gpu_pool": list(gpu_pool),
        "forbidden_gpus": sorted(FORBIDDEN_GPUS),
        "expected_jobs": len(jobs),
        "label_isolation": {
            "fit_receives_y": False,
            "graph_receives_y": False,
            "gate_receives_y": False,
            "loss_receives_y": False,
            "K_source": "benchmark_oracle_from_y_for_cluster_head_variants",
            "metrics_use_y_after_fit": True,
        },
        "selection": "fixed_v21_config; no ARI-based variant or hyperparameter selection",
        "cpu_fallback": {
            "enabled": True,
            "datasets": sorted(CPU_FALLBACK_DATASETS),
            "max_workers": DEFAULT_CPU_WORKERS,
        },
        "created_at": _timestamp(),
    }


def _gpu_snapshot(gpu: int) -> tuple[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory, utilization = (int(part.strip()) for part in result.stdout.strip().split(",", 1))
    return memory, utilization


def _is_idle(gpu: int, idle_memory_mib: int = IDLE_MEMORY_MIB) -> bool:
    try:
        memory, utilization = _gpu_snapshot(gpu)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False
    return memory < idle_memory_mib and utilization == 0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _required_outputs(job: dict[str, Any]) -> tuple[str, ...]:
    required = (
        "summary.json",
        "metrics.json",
        "resolved_config.json",
        "training_history.json",
        "preprocess_profile.json",
        "graph_profile.json",
        "stats_profile.json",
        "selected_feature_indices.npy",
        "embedding_final.npy",
        "predictions.npy",
        "labels_true.npy",
        "checkpoint.pt",
    )
    if job["variant"] != "scmae_only":
        required += ("cluster_probabilities.npy",)
    return required


def _valid_completed(job: dict[str, Any]) -> bool:
    output = Path(job["output"])
    if any(not (output / name).is_file() for name in _required_outputs(job)):
        return False
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "completed"
        and summary.get("protocol_id") == MODEL_PROTOCOL_ID
        and summary.get("variant") == job["variant"]
        and summary.get("dataset") == job["dataset"]
        and int(summary.get("seed", -1)) == int(job["seed"])
        and summary.get("labels_used_during_fit") is False
    )


def _external_cpu_fallback_active(dataset: str) -> bool:
    """Reserve keys while a separately launched CPU batch owns their outputs."""
    if dataset not in EXTERNAL_CPU_FALLBACK_DATASETS:
        return False
    state_path = OUTPUT_ROOT / f"cpu_fallback_{_safe_name(dataset)}.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    jobs = payload.get("jobs", [])
    return bool(jobs) and any(job.get("status") not in {"completed", "incomplete_compute"} for job in jobs)


def _validate_paths() -> None:
    missing = [str(path) for _name, _protocol, path in DATASETS if not path.is_file()]
    missing += [str(path) for _variant, path in VARIANTS if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing V21 matrix inputs: " + ", ".join(missing))


def _load_or_write_spec(spec_path: Path, expected: dict[str, Any]) -> None:
    if spec_path.exists():
        actual = json.loads(spec_path.read_text(encoding="utf-8"))
        comparable_actual = dict(actual)
        comparable_expected = dict(expected)
        comparable_actual.pop("created_at", None)
        comparable_expected.pop("created_at", None)
        if comparable_actual != comparable_expected:
            raise RuntimeError(f"existing stage_spec.json does not match frozen protocol: {spec_path}")
    else:
        _write_json(spec_path, expected)


def _load_previous_state(state_path: Path) -> list[dict[str, Any]] | None:
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != MATRIX_PROTOCOL_ID:
        raise RuntimeError(f"existing launcher state has another protocol: {state_path}")
    return payload.get("jobs")


def _prepare_jobs(state_path: Path, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous = _load_previous_state(state_path)
    previous_by_key = {str(item.get("key")): item for item in previous or []}
    for job in jobs:
        old = previous_by_key.get(job["key"], {})
        if old.get("status") == "running" and old.get("pid") and _pid_alive(int(old["pid"])):
            raise RuntimeError(f"run key is active in another launcher: {job['key']} pid={old['pid']}")
        if _valid_completed(job):
            job.update({"status": "completed", "return_code": 0, "resumed": True})
        elif old.get("status") == "incomplete_compute":
            job.update({"status": "queued", "retry_of": old.get("return_code")})
    return jobs


def _write_state(state_path: Path, jobs: list[dict[str, Any]], active_gpus: list[int], status: str) -> None:
    _write_json(
        state_path,
        {
            "protocol_id": MATRIX_PROTOCOL_ID,
            "updated_at": _timestamp(),
            "status": status,
            "expected_jobs": len(jobs),
            "completed_jobs": sum(item.get("status") == "completed" for item in jobs),
            "incomplete_jobs": sum(item.get("status") == "incomplete_compute" for item in jobs),
            "queued_jobs": sum(item.get("status") == "queued" for item in jobs),
            "active_gpus": sorted(active_gpus),
            "jobs": jobs,
        },
    )


def _spawn(job: dict[str, Any], gpu: int | None, cpu_slot: int | None = None) -> tuple[subprocess.Popen[str], Any]:
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "methods.TopoGate.V21_assignment_adversarial_gate.run",
        "--data",
        job["data"],
        "--dataset-name",
        job["dataset"],
        "--input-protocol",
        job["input_protocol"],
        "--config",
        job["config"],
        "--output-dir",
        job["output"],
        "--seed",
        str(job["seed"]),
    ]
    env = os.environ.copy()
    worker_name: str
    if gpu is not None:
        command.extend(["--device", "cuda", "--gpu", str(gpu)])
        worker_name = f"gpu{gpu}"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        mpl_dir = OUTPUT_ROOT / "mpl" / f"gpu{gpu}"
    else:
        if job["dataset"] not in CPU_FALLBACK_DATASETS:
            raise ValueError(f"CPU fallback is not enabled for {job['dataset']}")
        command.extend(["--device", "cpu"])
        worker_name = f"cpu{cpu_slot}"
        env["CUDA_VISIBLE_DEVICES"] = ""
        mpl_dir = OUTPUT_ROOT / "mpl" / worker_name
    env.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": str(mpl_dir),
        }
    )
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(f"\n[{_timestamp()}] launch worker={worker_name} key={job['key']}\n")
    log_handle.flush()
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    job.update(
        {
            "status": "running",
            "device": "cuda" if gpu is not None else "cpu",
            "gpu": None if gpu is None else int(gpu),
            "worker": worker_name,
            "pid": int(child.pid),
            "started_at": _timestamp(),
        }
    )
    return child, log_handle


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen V21 six-dataset Full/scMAE-only matrix")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(DEFAULT_GPU_POOL))
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument(
        "--idle-memory-mib",
        type=int,
        default=IDLE_MEMORY_MIB,
        help="maximum observed GPU memory (MiB) for a zero-utilization GPU; use a higher value only for explicit shared-GPU scheduling",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=DEFAULT_CPU_WORKERS,
        help="optional concurrent CPU fallback workers for cnae9/sms_spam_collection/hate_speech",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpu_pool = tuple(int(gpu) for gpu in args.gpus)
    if not gpu_pool or any(gpu in FORBIDDEN_GPUS or gpu < 1 or gpu > 6 for gpu in gpu_pool):
        raise ValueError(f"GPU pool must contain only physical GPUs 1..6, got {gpu_pool}")
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if args.idle_memory_mib <= 0:
        raise ValueError("idle-memory-mib must be positive")
    if args.cpu_workers < 0:
        raise ValueError("cpu-workers must be non-negative")
    _validate_paths()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT_ROOT / "launcher.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = _jobs()
        spec = _stage_spec(gpu_pool, jobs)
        _load_or_write_spec(OUTPUT_ROOT / "stage_spec.json", spec)
        jobs = _prepare_jobs(OUTPUT_ROOT / "launcher_state.json", jobs)
        _write_state(OUTPUT_ROOT / "launcher_state.json", jobs, [], "dry_run" if args.dry_run else "queued")
        if args.dry_run:
            return 0

        children: dict[str, tuple[subprocess.Popen[str], Any, dict[str, Any]]] = {}
        while True:
            for item in jobs:
                if item.get("status") == "queued" and _valid_completed(item):
                    item.update({"status": "completed", "return_code": 0, "resumed": True})
            for gpu in gpu_pool:
                worker_name = f"gpu{gpu}"
                if worker_name in children:
                    continue
                queued = next(
                    (
                        item
                        for item in jobs
                        if item.get("status") == "queued"
                        and not _external_cpu_fallback_active(str(item.get("dataset", "")))
                    ),
                    None,
                )
                if queued is None or not _is_idle(gpu, args.idle_memory_mib):
                    continue
                child, log_handle = _spawn(queued, gpu)
                children[worker_name] = (child, log_handle, queued)

            for cpu_slot in range(args.cpu_workers):
                worker_name = f"cpu{cpu_slot}"
                if worker_name in children:
                    continue
                queued = next(
                    (
                        item
                        for item in jobs
                        if item.get("status") == "queued" and item.get("dataset") in CPU_FALLBACK_DATASETS
                    ),
                    None,
                )
                if queued is None:
                    continue
                child, log_handle = _spawn(queued, None, cpu_slot=cpu_slot)
                children[worker_name] = (child, log_handle, queued)

            for worker_name, (child, log_handle, job) in list(children.items()):
                return_code = child.poll()
                if return_code is None:
                    continue
                log_handle.close()
                job.update(
                    {
                        "status": "completed" if return_code == 0 and _valid_completed(job) else "incomplete_compute",
                        "return_code": int(return_code),
                        "finished_at": _timestamp(),
                    }
                )
                children.pop(worker_name)

            _write_state(
                OUTPUT_ROOT / "launcher_state.json",
                jobs,
                [int(item[2]["gpu"]) for item in children.values() if item[2].get("gpu") is not None],
                "running" if children or any(item.get("status") == "queued" for item in jobs) else "finished",
            )
            if not children and not any(item.get("status") == "queued" for item in jobs):
                return 0 if all(item.get("status") == "completed" for item in jobs) else 1
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
