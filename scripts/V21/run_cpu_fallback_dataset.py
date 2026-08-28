#!/usr/bin/env python3
"""Run one explicitly approved V21 dataset batch on CPU.

This helper is used only when the formal GPU queue is blocked by external
processes. It preserves the formal run keys, configs, seeds, output paths, and
80-epoch budget; device differences remain visible in each resolved config.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "result/V21/v21_formal6_full_20260811_graphfix"
SEEDS = (42, 123, 7)
DATASETS = {
    "Mouse_retina": ("clubench_bridge", ROOT / "datasets/Mouse_retina.npz"),
    "Baron Human": ("clubench_bridge", ROOT / "datasets/Baron Human.npz"),
    "Campbell": ("clubench_bridge", ROOT / "datasets/Campbell.npz"),
}
VARIANTS = (
    (
        "topology_assignment_adversarial",
        ROOT / "methods/TopoGate/V21_assignment_adversarial_gate/configs/v21_topology_assignment_adversarial.yaml",
    ),
    (
        "scmae_only",
        ROOT / "methods/TopoGate/V21_assignment_adversarial_gate/configs/v21_scmae_only.yaml",
    ),
)


def _safe_name(value: str) -> str:
    return value.casefold().replace(" ", "_")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one V21 formal dataset batch on CPU")
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    protocol, data_path = DATASETS[args.dataset]
    jobs: list[dict[str, Any]] = []
    for variant, config in VARIANTS:
        for seed in SEEDS:
            key = f"{_safe_name(args.dataset)}__{protocol}__{variant}__seed{seed}"
            output = OUTPUT_ROOT / f"{_safe_name(args.dataset)}__{protocol}" / variant / f"seed{seed}"
            jobs.append(
                {
                    "key": key,
                    "dataset": args.dataset,
                    "input_protocol": protocol,
                    "variant": variant,
                    "seed": seed,
                    "data": str(data_path.resolve()),
                    "config": str(config.resolve()),
                    "output": str(output),
                    "log": str(OUTPUT_ROOT / "logs" / f"{key}.log"),
                    "device": "cpu",
                    "status": "queued",
                }
            )
    state_path = OUTPUT_ROOT / f"cpu_fallback_{_safe_name(args.dataset)}.json"
    _write_json(
        state_path,
        {
            "protocol_id": "v21_assignment_adversarial_full6_graphfix_v1",
            "dataset": args.dataset,
            "device": "cpu",
            "created_at": _timestamp(),
            "jobs": jobs,
        },
    )
    children: dict[int, tuple[subprocess.Popen[str], Any, dict[str, Any]]] = {}
    next_job = 0
    while next_job < len(jobs) or children:
        while next_job < len(jobs) and len(children) < args.workers:
            job = jobs[next_job]
            next_job += 1
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
                "--device",
                "cpu",
            ]
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "TOPOGATE_CPU_THREADS": "32",
                    "PYTHONUNBUFFERED": "1",
                    "MPLCONFIGDIR": str(OUTPUT_ROOT / "mpl" / f"cpu_{_safe_name(args.dataset)}"),
                }
            )
            Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            handle.write(f"\n[{_timestamp()}] CPU fallback launch key={job['key']}\n")
            handle.flush()
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            job.update({"status": "running", "pid": child.pid, "started_at": _timestamp()})
            children[child.pid] = (child, handle, job)
        for pid, (child, handle, job) in list(children.items()):
            code = child.poll()
            if code is None:
                continue
            handle.close()
            job.update({"status": "completed" if code == 0 else "incomplete_compute", "return_code": code, "finished_at": _timestamp()})
            children.pop(pid)
        _write_json(state_path, {"protocol_id": "v21_assignment_adversarial_full6_graphfix_v1", "dataset": args.dataset, "device": "cpu", "updated_at": _timestamp(), "jobs": jobs})
        if children:
            next_job = next_job
            import time

            time.sleep(5)
    return 0 if all(job["status"] == "completed" for job in jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
