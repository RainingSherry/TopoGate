#!/usr/bin/env python
"""Continue the V19 ARI development protocol after the screen stage.

The watcher only sequences already audited commands.  It does not select
parameters itself and never reads labels; selection remains in the dedicated
ARI summarizer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.V19.tune_ari_dev import DEFAULT_CONFIG, DEFAULT_OUTPUT, _read_json, _write_json


def _status(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with code {completed.returncode}: {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Continue V19 ARI development stages")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if not args.gpus or any(int(gpu) not in {1, 2, 3, 4, 5, 6} for gpu in args.gpus):
        raise ValueError("pipeline GPUs must be a subset of physical GPUs 1-6")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "pipeline_status.json"
    log_path = args.output_dir / "pipeline.log"
    payload: dict[str, Any] = {
        "status": "waiting_for_screen",
        "protocol_id": "v19_rg_ari_dev_tuning_v1",
        "output_dir": str(args.output_dir.resolve()),
        "gpus": [int(gpu) for gpu in args.gpus],
        "started_at": time.time(),
    }
    _status(status_path, payload)
    screen_status_path = args.output_dir / "launcher_screen_status.json"
    while True:
        if screen_status_path.is_file():
            screen_status = _read_json(screen_status_path)
            if screen_status.get("status") in {"completed", "incomplete_compute"}:
                break
        time.sleep(max(5, int(args.poll_seconds)))
    screen_status = _read_json(screen_status_path)
    if screen_status.get("status") != "completed":
        payload.update({"status": "incomplete_compute", "failed_stage": "screen", "screen_status": screen_status})
        _status(status_path, payload)
        return 1
    try:
        payload["status"] = "summarizing_screen"
        _status(status_path, payload)
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "V19" / "summarize_ari_dev.py"),
                "--manifest",
                str(args.manifest),
                "--output-dir",
                str(args.output_dir / "screen"),
                "--stage",
                "screen",
                "--top-k",
                "12",
            ],
            log_path,
        )
        payload["status"] = "running_reference"
        _status(status_path, payload)
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "V19" / "launch_ari_dev.py"),
                "--manifest",
                str(args.manifest),
                "--output-dir",
                str(args.output_dir),
                "--config",
                str(args.config),
                "--stage",
                "reference",
                "--gpus",
                *[str(gpu) for gpu in args.gpus],
            ],
            log_path,
        )
        payload["status"] = "running_refine"
        _status(status_path, payload)
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "V19" / "launch_ari_dev.py"),
                "--manifest",
                str(args.manifest),
                "--output-dir",
                str(args.output_dir),
                "--config",
                str(args.config),
                "--stage",
                "refine",
                "--gpus",
                *[str(gpu) for gpu in args.gpus],
            ],
            log_path,
        )
        payload["status"] = "summarizing_refine"
        _status(status_path, payload)
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "V19" / "summarize_ari_dev.py"),
                "--manifest",
                str(args.manifest),
                "--output-dir",
                str(args.output_dir / "refine"),
                "--stage",
                "refine",
                "--reference-dir",
                str(args.output_dir / "reference"),
            ],
            log_path,
        )
        payload["status"] = "running_final"
        _status(status_path, payload)
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "V19" / "launch_ari_dev.py"),
                "--manifest",
                str(args.manifest),
                "--output-dir",
                str(args.output_dir),
                "--config",
                str(args.config),
                "--stage",
                "final",
                "--gpus",
                *[str(gpu) for gpu in args.gpus],
            ],
            log_path,
        )
        payload["status"] = "summarizing_final"
        _status(status_path, payload)
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "V19" / "summarize_ari_final.py"),
                "--final-dir",
                str(args.output_dir / "final"),
                "--output-dir",
                str(args.output_dir / "final_summary"),
            ],
            log_path,
        )
        payload.update({"status": "completed", "finished_at": time.time()})
        _status(status_path, payload)
        return 0
    except Exception as exc:
        payload.update({"status": "incomplete_compute", "error_type": type(exc).__name__, "error": str(exc), "finished_at": time.time()})
        _status(status_path, payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
