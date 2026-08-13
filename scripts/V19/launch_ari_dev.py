#!/usr/bin/env python
"""Parallel small-first launcher for the V19 ARI development protocol."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.V19.tune_ari_dev import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    PROTOCOL_ID,
    TARGET_DATASET_IDS,
    _read_json,
    _write_json,
    load_manifest,
)


ALLOWED_GPUS = (1, 2, 3, 4, 5, 6)


def _gpu_snapshot() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required before launching GPU workers")
    command = [
        executable,
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3:
            continue
        rows.append({"index": int(fields[0]), "memory_used_mib": int(fields[1]), "memory_total_mib": int(fields[2])})
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")
    forbidden = {int(row["index"]) for row in rows}.intersection({0, 7})
    if forbidden:
        # Seeing physical cards 0/7 is expected on this host; the launcher
        # records them but never assigns a worker to either card.
        pass
    return rows


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def _stage_command(args: argparse.Namespace, worker_id: int, gpu: int) -> list[str]:
    if args.stage in {"screen", "refine"}:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "V19" / "tune_ari_dev.py"),
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(args.output_dir),
            "--config",
            str(args.config),
            "--stage",
            args.stage,
            "--worker-id",
            str(worker_id),
            "--num-workers",
            str(len(args.gpus)),
            "--gpu",
            str(gpu),
        ]
        if args.stage == "refine":
            selected = _read_json(args.output_dir / "screen" / "top12_config.json")
            ids = selected.get("top_candidate_ids", [])
            if len(ids) != 12:
                raise ValueError("screen top12_config.json does not contain 12 candidate ids")
            command.extend(["--candidate-ids", *[str(value) for value in ids]])
        return command
    if args.stage == "reference":
        return [
            sys.executable,
            str(ROOT / "scripts" / "V19" / "run_scmae_reference_ari_dev.py"),
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(args.output_dir),
            "--config",
            str(args.config),
            "--worker-id",
            str(worker_id),
            "--num-workers",
            str(len(args.gpus)),
            "--gpu",
            str(gpu),
        ]
    if args.stage == "final":
        selected = args.output_dir / "refine" / "selected_config.json"
        if not selected.is_file():
            raise FileNotFoundError(f"missing selected ARI config: {selected}")
        return [
            sys.executable,
            str(ROOT / "scripts" / "V19" / "run_ari_final.py"),
            "--manifest",
            str(args.manifest),
            "--selected-config",
            str(selected),
            "--output-dir",
            str(args.output_dir / "final"),
            "--config",
            str(args.config),
            "--worker-id",
            str(worker_id),
            "--num-workers",
            str(len(args.gpus)),
            "--gpu",
            str(gpu),
        ]
    raise ValueError(args.stage)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch V19 ARI development workers")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("screen", "reference", "refine", "final"), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(ALLOWED_GPUS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.gpus = tuple(int(value) for value in args.gpus)
    if not args.gpus or any(gpu not in ALLOWED_GPUS for gpu in args.gpus):
        raise ValueError(f"GPU pool must be a non-empty subset of {ALLOWED_GPUS}; GPU0/7 are forbidden")
    manifest = load_manifest(args.manifest)
    if set(TARGET_DATASET_IDS) != {
        str(row["dataset_id"])
        for row in manifest["datasets"]
        if row.get("status") == "eligible" and row.get("input_protocol") in {"clubench_bridge", "shared_text"}
    }:
        raise ValueError("manifest does not expose the exact 8 target layers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _gpu_snapshot() if not args.dry_run else []
    launcher_status = args.output_dir / f"launcher_{args.stage}_status.json"
    payload = {
        "status": "running",
        "protocol_id": PROTOCOL_ID,
        "stage": args.stage,
        "output_dir": str(args.output_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "config": str(args.config.resolve()),
        "gpus": list(args.gpus),
        "gpu_snapshot_before_launch": snapshot,
        "small_first": True,
        "labels_used_for_selection": True,
        "selection_evidence_type": "ARI-selected development evidence",
        "started_at": time.time(),
    }
    _write_status(launcher_status, payload)
    commands = [_stage_command(args, index, gpu) for index, gpu in enumerate(args.gpus)]
    if args.dry_run:
        for index, command in enumerate(commands):
            print(json.dumps({"worker_id": index, "gpu": args.gpus[index], "command": command}, ensure_ascii=True))
        payload.update({"status": "dry_run", "commands": commands})
        _write_status(launcher_status, payload)
        return 0
    processes: list[tuple[int, int, subprocess.Popen[str], Any]] = []
    logs = args.output_dir / "launcher_logs" / args.stage
    logs.mkdir(parents=True, exist_ok=True)
    for worker_id, gpu in enumerate(args.gpus):
        environment = dict(os.environ)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_handle = (logs / f"worker{worker_id}_gpu{gpu}.log").open("a", encoding="utf-8")
        command = commands[worker_id]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_id, gpu, process, log_handle))
    payload["workers"] = [{"worker_id": worker_id, "gpu": gpu, "pid": process.pid} for worker_id, gpu, process, _ in processes]
    _write_status(launcher_status, payload)
    exit_codes: dict[str, int] = {}
    for worker_id, gpu, process, log_handle in processes:
        code = process.wait()
        log_handle.close()
        exit_codes[str(worker_id)] = int(code)
    completed = all(code == 0 for code in exit_codes.values())
    payload.update({
        "status": "completed" if completed else "incomplete_compute",
        "exit_codes": exit_codes,
        "finished_at": time.time(),
    })
    _write_status(launcher_status, payload)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
