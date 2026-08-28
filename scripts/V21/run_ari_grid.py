#!/usr/bin/env python3
"""Run the explicitly label-selected V21 hyperparameter development grid.

This launcher is intentionally separate from the formal matrix.  ARI is used
only by the post-fit selector; ``fit_v21`` still receives no labels.
"""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "result/V21/v21_ari_grid_seed42_20260811"
BASE_CONFIG = ROOT / "methods/TopoGate/V21_assignment_adversarial_gate/configs/v21_topology_assignment_adversarial.yaml"
DATASETS = (
    ("cnae9", "shared_text", ROOT / "datasets/cnae9.npz"),
    ("Mouse_retina", "clubench_bridge", ROOT / "datasets/Mouse_retina.npz"),
    ("Baron Human", "clubench_bridge", ROOT / "datasets/Baron Human.npz"),
    ("Campbell", "clubench_bridge", ROOT / "datasets/Campbell.npz"),
    ("sms_spam_collection", "shared_text", ROOT / "datasets/sms_spam_collection.npz"),
    ("hate_speech", "shared_text", ROOT / "datasets/hate_speech.npz"),
)
SEED = 42
GRID_PROTOCOL_ID = "v21_ari_grid_seed42_v1"
MODEL_PROTOCOL_ID = "v21_assignment_adversarial_v2_graphfix_v1"
GPU_SLOTS = {1: 1, 5: 2}
IDLE_MEMORY_MIB = 45000
SHARED_MEMORY_MIB = 70000


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _safe_name(value: str) -> str:
    return value.casefold().replace(" ", "_")


def _candidate_id(assignment_weight: float, gate_lr: float, epochs: int) -> str:
    return f"aw{assignment_weight:g}_glr{gate_lr:g}_ep{epochs}"


def _candidates() -> list[dict[str, Any]]:
    values = itertools.product((0.1, 0.5, 1.0), (2.5e-4, 5e-4), (40, 80))
    candidates = []
    for assignment_weight, gate_lr, epochs in values:
        candidates.append(
            {
                "candidate_id": _candidate_id(assignment_weight, gate_lr, epochs),
                "assignment_weight": float(assignment_weight),
                "gate_lr": float(gate_lr),
                "epochs": int(epochs),
                "warmup_epochs": int(epochs // 2),
                "infomax_weight": 0.05,
            }
        )
    return candidates


def _jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for candidate in _candidates():
        for dataset, protocol, data_path in DATASETS:
            candidate_id = str(candidate["candidate_id"])
            key = f"{candidate_id}__{_safe_name(dataset)}__seed{SEED}"
            output = OUTPUT_ROOT / candidate_id / f"{_safe_name(dataset)}__{protocol}" / f"seed{SEED}"
            jobs.append(
                {
                    "key": key,
                    "candidate_id": candidate_id,
                    "candidate": candidate,
                    "dataset": dataset,
                    "input_protocol": protocol,
                    "data": str(data_path.resolve()),
                    "config": str(BASE_CONFIG.resolve()),
                    "variant": "topology_assignment_adversarial",
                    "seed": SEED,
                    "output": str(output),
                    "log": str(OUTPUT_ROOT / "logs" / f"{key}.log"),
                    "status": "queued",
                }
            )
    return jobs


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


def _gpu_ready(gpu: int, *, has_local_job: bool = False) -> bool:
    try:
        memory, utilization = _gpu_snapshot(gpu)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False
    if has_local_job:
        # A second slot is allowed only for memory left by this launcher.  The
        # first slot is admitted by the zero-utilization check below, so an
        # external compute process cannot silently turn into a shared slot.
        return memory < SHARED_MEMORY_MIB
    return memory < IDLE_MEMORY_MIB and utilization == 0


def _required_outputs(job: dict[str, Any]) -> tuple[str, ...]:
    return (
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
        "cluster_probabilities.npy",
        "checkpoint.pt",
    )


def _valid_completed(job: dict[str, Any]) -> bool:
    output = Path(job["output"])
    if any(not (output / name).is_file() for name in _required_outputs(job)):
        return False
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    candidate = job["candidate"]
    return (
        summary.get("status") == "completed"
        and summary.get("protocol_id") == MODEL_PROTOCOL_ID
        and summary.get("variant") == job["variant"]
        and summary.get("dataset") == job["dataset"]
        and int(summary.get("seed", -1)) == SEED
        and summary.get("labels_used_during_fit") is False
        and int(resolved.get("epochs", -1)) == int(candidate["epochs"])
        and int(resolved.get("warmup_epochs", -1)) == int(candidate["warmup_epochs"])
        and abs(float(resolved.get("assignment_weight", -1.0)) - float(candidate["assignment_weight"])) < 1e-12
        and abs(float(resolved.get("gate_lr", -1.0)) - float(candidate["gate_lr"])) < 1e-12
    )


def _load_or_write_spec(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        comparable_actual = dict(actual)
        comparable_expected = dict(expected)
        comparable_actual.pop("created_at", None)
        comparable_expected.pop("created_at", None)
        if comparable_actual != comparable_expected:
            raise RuntimeError(f"existing grid_spec.json does not match frozen grid: {path}")
    else:
        _write_json(path, expected)


def _stage_spec(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "protocol_id": GRID_PROTOCOL_ID,
        "model_protocol_id": MODEL_PROTOCOL_ID,
        "stage": "ari_selected_development_grid",
        "datasets": [
            {"dataset": dataset, "input_protocol": protocol, "data": str(path.resolve())}
            for dataset, protocol, path in DATASETS
        ],
        "variant": "topology_assignment_adversarial",
        "seed": SEED,
        "candidate_count": len(_candidates()),
        "expected_jobs": len(jobs),
        "candidate_grid": {
            "assignment_weight": [0.1, 0.5, 1.0],
            "gate_lr": [2.5e-4, 5e-4],
            "epochs": [40, 80],
            "warmup_epochs": "epochs // 2",
            "infomax_weight": 0.05,
        },
        "selection_metric": "macro_mean_over_six_dataset_ari",
        "selection_uses_labels": True,
        "fit_receives_y": False,
        "gpu_slots": {str(gpu): slots for gpu, slots in GPU_SLOTS.items()},
        "created_at": _timestamp(),
    }


def _spawn(job: dict[str, Any], gpu: int, slot: str) -> tuple[subprocess.Popen[str], Any]:
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "grid_job.json",
        {
            "grid_protocol_id": GRID_PROTOCOL_ID,
            "candidate_id": job["candidate_id"],
            "candidate": job["candidate"],
            "dataset": job["dataset"],
            "seed": SEED,
            "selection_uses_labels": True,
            "fit_receives_y": False,
        },
    )
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
        str(SEED),
        "--device",
        "cuda",
        "--gpu",
        str(gpu),
        "--epochs",
        str(job["candidate"]["epochs"]),
        "--warmup-epochs",
        str(job["candidate"]["warmup_epochs"]),
        "--gate-lr",
        str(job["candidate"]["gate_lr"]),
        "--assignment-weight",
        str(job["candidate"]["assignment_weight"]),
        "--infomax-weight",
        str(job["candidate"]["infomax_weight"]),
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": str(OUTPUT_ROOT / "mpl" / slot),
        }
    )
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"\n[{_timestamp()}] launch slot={slot} key={job['key']}\n")
    handle.flush()
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    job.update({"status": "running", "pid": int(child.pid), "gpu": gpu, "slot": slot, "started_at": _timestamp()})
    return child, handle


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V21 ARI-selected development grid")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if not BASE_CONFIG.is_file() or any(not data.is_file() for _name, _protocol, data in DATASETS):
        raise FileNotFoundError("V21 ARI grid input/config is missing")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT_ROOT / "grid.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = _jobs()
        _load_or_write_spec(OUTPUT_ROOT / "grid_spec.json", _stage_spec(jobs))
        state_path = OUTPUT_ROOT / "grid_state.json"
        previous: dict[str, Any] = {}
        if state_path.exists():
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if payload.get("protocol_id") != GRID_PROTOCOL_ID:
                raise RuntimeError("existing grid_state has another protocol")
            previous = {item["key"]: item for item in payload.get("jobs", [])}
        for job in jobs:
            old = previous.get(job["key"], {})
            if old.get("status") == "incomplete_compute":
                job["retry_of"] = old.get("return_code")
            if _valid_completed(job):
                job.update({"status": "completed", "return_code": 0, "resumed": True})
        if args.dry_run:
            _write_json(OUTPUT_ROOT / "grid_state.json", {"protocol_id": GRID_PROTOCOL_ID, "status": "dry_run", "jobs": jobs})
            return 0

        children: dict[str, tuple[subprocess.Popen[str], Any, dict[str, Any]]] = {}
        while True:
            for gpu, slots in GPU_SLOTS.items():
                slot_names = [f"gpu{gpu}_{index}" for index in range(slots)]
                free = [slot for slot in slot_names if slot not in children]
                active_on_gpu = sum(1 for item in children.values() if int(item[2].get("gpu", -1)) == gpu)
                if not free or not _gpu_ready(gpu, has_local_job=active_on_gpu > 0):
                    continue
                for slot in free:
                    queued = next((item for item in jobs if item.get("status") == "queued"), None)
                    if queued is None:
                        break
                    child, handle = _spawn(queued, gpu, slot)
                    children[slot] = (child, handle, queued)

            for slot, (child, handle, job) in list(children.items()):
                return_code = child.poll()
                if return_code is None:
                    continue
                handle.close()
                job.update(
                    {
                        "status": "completed" if return_code == 0 and _valid_completed(job) else "incomplete_compute",
                        "return_code": int(return_code),
                        "finished_at": _timestamp(),
                    }
                )
                children.pop(slot)

            _write_json(
                state_path,
                {
                    "protocol_id": GRID_PROTOCOL_ID,
                    "updated_at": _timestamp(),
                    "status": "running" if children or any(item.get("status") == "queued" for item in jobs) else "finished",
                    "expected_jobs": len(jobs),
                    "completed_jobs": sum(item.get("status") == "completed" for item in jobs),
                    "incomplete_jobs": sum(item.get("status") == "incomplete_compute" for item in jobs),
                    "queued_jobs": sum(item.get("status") == "queued" for item in jobs),
                    "jobs": jobs,
                },
            )
            if not children and not any(item.get("status") == "queued" for item in jobs):
                return 0 if all(item.get("status") == "completed" for item in jobs) else 1
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
