#!/usr/bin/env python3
"""Run the three-seed confirmation for the ARI-selected V21 configuration."""

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
OUTPUT_ROOT = ROOT / "result/V21/v21_ari_confirm_aw0.1_glr0.00025_ep80_20260811"
CONFIG = ROOT / "methods/TopoGate/V21_assignment_adversarial_gate/configs/v21_topology_assignment_adversarial.yaml"
DATASETS = (
    ("cnae9", "shared_text", ROOT / "datasets/cnae9.npz"),
    ("Mouse_retina", "clubench_bridge", ROOT / "datasets/Mouse_retina.npz"),
    ("Baron Human", "clubench_bridge", ROOT / "datasets/Baron Human.npz"),
    ("Campbell", "clubench_bridge", ROOT / "datasets/Campbell.npz"),
    ("sms_spam_collection", "shared_text", ROOT / "datasets/sms_spam_collection.npz"),
    ("hate_speech", "shared_text", ROOT / "datasets/hate_speech.npz"),
)
SEEDS = (42, 123, 7)
PROTOCOL_ID = "v21_ari_confirm_v1"
MODEL_PROTOCOL_ID = "v21_assignment_adversarial_v2_graphfix_v1"
PARAMS = {
    "assignment_weight": 0.1,
    "gate_lr": 2.5e-4,
    "infomax_weight": 0.05,
    "epochs": 80,
    "warmup_epochs": 40,
}
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


def _jobs() -> list[dict[str, Any]]:
    jobs = []
    for dataset, protocol, data_path in DATASETS:
        for seed in SEEDS:
            key = f"{_safe_name(dataset)}__{protocol}__topology_assignment_adversarial__seed{seed}"
            output = OUTPUT_ROOT / f"{_safe_name(dataset)}__{protocol}" / "topology_assignment_adversarial" / f"seed{seed}"
            jobs.append(
                {
                    "key": key,
                    "dataset": dataset,
                    "input_protocol": protocol,
                    "data": str(data_path.resolve()),
                    "config": str(CONFIG.resolve()),
                    "variant": "topology_assignment_adversarial",
                    "seed": seed,
                    "output": str(output),
                    "log": str(OUTPUT_ROOT / "logs" / f"{key}.log"),
                    "status": "queued",
                }
            )
    return jobs


def _gpu_snapshot(gpu: int) -> tuple[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    memory, utilization = (int(part.strip()) for part in result.stdout.strip().split(",", 1))
    return memory, utilization


def _gpu_ready(gpu: int, *, has_local_job: bool) -> bool:
    try:
        memory, utilization = _gpu_snapshot(gpu)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False
    if has_local_job:
        return memory < SHARED_MEMORY_MIB
    return memory < IDLE_MEMORY_MIB and utilization == 0


def _valid_completed(job: dict[str, Any]) -> bool:
    output = Path(job["output"])
    required = ("summary.json", "metrics.json", "resolved_config.json", "training_history.json", "checkpoint.pt")
    if any(not (output / name).is_file() for name in required):
        return False
    try:
        summary = json.loads((output / "summary.json").read_text())
        resolved = json.loads((output / "resolved_config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "completed"
        and summary.get("protocol_id") == MODEL_PROTOCOL_ID
        and summary.get("dataset") == job["dataset"]
        and summary.get("variant") == job["variant"]
        and int(summary.get("seed", -1)) == int(job["seed"])
        and summary.get("labels_used_during_fit") is False
        and all(abs(float(resolved.get(key, float("nan"))) - float(value)) < 1e-12 for key, value in PARAMS.items())
    )


def _spawn(job: dict[str, Any], gpu: int, slot: str) -> tuple[subprocess.Popen[str], Any]:
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "confirmation_job.json", {"protocol_id": PROTOCOL_ID, "params": PARAMS, "fit_receives_y": False, "selection_uses_labels": True})
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
        "--device",
        "cuda",
        "--gpu",
        str(gpu),
        "--epochs",
        str(PARAMS["epochs"]),
        "--warmup-epochs",
        str(PARAMS["warmup_epochs"]),
        "--gate-lr",
        str(PARAMS["gate_lr"]),
        "--assignment-weight",
        str(PARAMS["assignment_weight"]),
        "--infomax-weight",
        str(PARAMS["infomax_weight"]),
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
    job.update({"status": "running", "pid": child.pid, "gpu": gpu, "slot": slot, "started_at": _timestamp()})
    return child, handle


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the three-seed ARI-selected V21 confirmation")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = _jobs()
    spec = {
        "protocol_id": PROTOCOL_ID,
        "model_protocol_id": MODEL_PROTOCOL_ID,
        "expected_jobs": len(jobs),
        "datasets": [dataset for dataset, _protocol, _path in DATASETS],
        "seeds": list(SEEDS),
        "params": PARAMS,
        "selection_uses_labels": True,
        "fit_receives_y": False,
        "created_at": _timestamp(),
    }
    spec_path = OUTPUT_ROOT / "confirm_spec.json"
    if spec_path.exists():
        actual = json.loads(spec_path.read_text())
        comparable_actual = dict(actual); comparable_actual.pop("created_at", None)
        comparable_spec = dict(spec); comparable_spec.pop("created_at", None)
        if comparable_actual != comparable_spec:
            raise RuntimeError("existing confirmation spec mismatch")
    else:
        _write_json(spec_path, spec)
    state_path = OUTPUT_ROOT / "confirm_state.json"
    previous = {}
    if state_path.exists():
        payload = json.loads(state_path.read_text())
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError("existing confirmation state mismatch")
        previous = {item["key"]: item for item in payload.get("jobs", [])}
    for job in jobs:
        if _valid_completed(job):
            job.update({"status": "completed", "return_code": 0, "resumed": True})
        elif previous.get(job["key"], {}).get("status") == "incomplete_compute":
            job["retry_of"] = previous[job["key"]].get("return_code")
    if args.dry_run:
        _write_json(state_path, {"protocol_id": PROTOCOL_ID, "status": "dry_run", "jobs": jobs})
        return 0

    with (OUTPUT_ROOT / "confirm.lock").open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        children: dict[str, tuple[subprocess.Popen[str], Any, dict[str, Any]]] = {}
        while True:
            for gpu, slots in GPU_SLOTS.items():
                free = [f"gpu{gpu}_{index}" for index in range(slots) if f"gpu{gpu}_{index}" not in children]
                active = sum(1 for item in children.values() if int(item[2].get("gpu", -1)) == gpu)
                if not free or not _gpu_ready(gpu, has_local_job=active > 0):
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
                job.update({"status": "completed" if return_code == 0 and _valid_completed(job) else "incomplete_compute", "return_code": int(return_code), "finished_at": _timestamp()})
                children.pop(slot)
            _write_json(
                state_path,
                {
                    "protocol_id": PROTOCOL_ID,
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
