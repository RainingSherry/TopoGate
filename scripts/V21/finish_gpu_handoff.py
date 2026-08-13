#!/usr/bin/env python3
"""Finish the explicit Campbell CPU-to-GPU handoff and resume the matrix."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "result/V21/v21_formal6_full_20260811_graphfix"
CAMPBELL_STATE = OUTPUT_ROOT / "cpu_fallback_campbell.json"
CPU_PARENT_PID = 245442
WAIT_PIDS = (462292, 483119, 491305, 561382, 561383)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _summary_ok(job: dict[str, object]) -> bool:
    output = Path(str(job["output"]))
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "completed"
        and summary.get("protocol_id") == "v21_assignment_adversarial_v2_graphfix_v1"
        and summary.get("dataset") == job.get("dataset")
        and summary.get("variant") == job.get("variant")
        and int(summary.get("seed", -1)) == int(job.get("seed", -2))
        and summary.get("labels_used_during_fit") is False
    )


def _write_campbell_state() -> None:
    payload = json.loads(CAMPBELL_STATE.read_text(encoding="utf-8"))
    for job in payload.get("jobs", []):
        if job.get("status") == "running":
            if _summary_ok(job):
                job.update({"status": "completed", "return_code": 0, "finished_at": _timestamp()})
            else:
                job.update({"status": "incomplete_compute", "return_code": -1, "finished_at": _timestamp()})
        elif job.get("status") == "incomplete_compute" and _summary_ok(job):
            job.update({"status": "completed", "return_code": 0, "finished_at": _timestamp()})
    payload["updated_at"] = _timestamp()
    payload["handoff_finished"] = True
    CAMPBELL_STATE.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    while any(_alive(pid) for pid in WAIT_PIDS):
        time.sleep(30)

    # The stopped CPU parent has no live children now and cannot launch queued jobs.
    if _alive(CPU_PARENT_PID):
        try:
            os.kill(CPU_PARENT_PID, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _write_campbell_state()

    command = [
        sys.executable,
        str(ROOT / "scripts/V21/run_formal_matrix.py"),
        "--gpus",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "--idle-memory-mib",
        "45000",
        "--cpu-workers",
        "0",
        "--poll-seconds",
        "30",
    ]
    log = (OUTPUT_ROOT / "gpu_handoff_resume.log").open("a", encoding="utf-8")
    log.write(f"\n[{_timestamp()}] launching formal matrix after GPU handoff\n")
    log.flush()
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    log.close()
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
