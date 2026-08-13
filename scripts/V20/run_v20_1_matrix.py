#!/usr/bin/env python3
"""Queue the V20.1 Full/scMAE-only direction check on explicitly allowed GPUs.

The launcher never terminates or reuses existing processes.  It starts at most
one child per GPU and waits until the requested GPU is genuinely idle.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "result/V20/v20_1_effective_mask_short_adv_v1"
GPU_POOL = (5, 6)
DATASETS = (
    ("cnae9", "shared_text", ROOT / "datasets/cnae9.npz"),
    ("Mouse_retina", "clubench_bridge", ROOT / "datasets/Mouse_retina.npz"),
    ("Campbell", "clubench_bridge", ROOT / "datasets/Campbell.npz"),
)
VARIANTS = (
    ("topology_adversarial_full", ROOT / "methods/TopoGate/V20_topology_conditioned_adv_mask/configs/v20_1_full.yaml"),
    ("scmae_only", ROOT / "methods/TopoGate/V20_topology_conditioned_adv_mask/configs/v20_1_scmae_only.yaml"),
)
SEED = 42
PROTOCOL_ID = "v20_1_effective_mask_short_adv_v1"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


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


def _is_idle(gpu: int) -> bool:
    memory, utilization = _gpu_snapshot(gpu)
    # Driver/context overhead is below 1 GiB here.  A 3 GiB threshold avoids
    # colliding with an existing training process while allowing our jobs.
    return memory < 3000 and utilization == 0


def _jobs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for dataset, protocol, data in DATASETS:
        for variant, config in VARIANTS:
            output = OUTPUT_ROOT / f"{dataset.lower()}__{protocol}" / variant / "seed42"
            log = OUTPUT_ROOT / "logs" / f"{dataset.lower()}_{variant}.log"
            items.append(
                {
                    "dataset": dataset,
                    "input_protocol": protocol,
                    "data": str(data),
                    "variant": variant,
                    "config": str(config),
                    "output": str(output),
                    "log": str(log),
                    "status": "queued",
                }
            )
    return items


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT_ROOT / "launcher_state.json"
    jobs = _jobs()
    _write(
        OUTPUT_ROOT / "stage_spec.json",
        {
            "protocol_id": PROTOCOL_ID,
            "stage": "direction_check",
            "datasets": [item[0] for item in DATASETS],
            "variants": [item[0] for item in VARIANTS],
            "seed": SEED,
            "gpu_pool": list(GPU_POOL),
            "forbidden_gpus": [0, 7],
            "label_isolation": "fit_and_topology_do_not_read_y; labels only outer K and metrics",
            "jobs": len(jobs),
        },
    )
    children: dict[int, tuple[subprocess.Popen[str], dict[str, Any]]] = {}
    remaining = list(jobs)
    while remaining or children:
        for gpu in GPU_POOL:
            if gpu in children or not remaining or not _is_idle(gpu):
                continue
            item = remaining.pop(0)
            output = Path(item["output"])
            log_path = Path(item["log"])
            output.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "methods.TopoGate.V20_topology_conditioned_adv_mask.run",
                "--data",
                item["data"],
                "--dataset-name",
                item["dataset"],
                "--input-protocol",
                item["input_protocol"],
                "--config",
                item["config"],
                "--output-dir",
                item["output"],
                "--seed",
                str(SEED),
                "--gpu",
                str(gpu),
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
                }
            )
            log_handle = log_path.open("w", encoding="utf-8")
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            item.update({"status": "running", "gpu": gpu, "pid": child.pid})
            children[gpu] = (child, item)
        for gpu, (child, item) in list(children.items()):
            return_code = child.poll()
            if return_code is None:
                continue
            item.update({"status": "completed" if return_code == 0 else "incomplete_compute", "return_code": return_code})
            children.pop(gpu)
        _write(state_path, {"protocol_id": PROTOCOL_ID, "jobs": jobs, "active_gpus": sorted(children)})
        if remaining or children:
            time.sleep(30)
    _write(state_path, {"protocol_id": PROTOCOL_ID, "jobs": jobs, "active_gpus": [], "status": "finished"})
    return 0 if all(item["status"] == "completed" for item in jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
