#!/usr/bin/env python3
"""Resume the formal V21 launcher after an explicit CPU fallback batch."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "result/V21/v21_formal6_full_20260811_graphfix"
CPU_STATES = (
    OUTPUT_ROOT / "cpu_fallback_mouse_retina.json",
    OUTPUT_ROOT / "cpu_fallback_baron_human.json",
    OUTPUT_ROOT / "cpu_fallback_campbell.json",
)


def _all_cpu_batches_terminal() -> bool:
    for state_path in CPU_STATES:
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        jobs = payload.get("jobs", [])
        if not jobs or not all(job.get("status") in {"completed", "incomplete_compute"} for job in jobs):
            return False
    return True


def main() -> int:
    while True:
        if _all_cpu_batches_terminal():
            os.execv(
                sys.executable,
                [
                    sys.executable,
                    str(ROOT / "scripts/V21/run_formal_matrix.py"),
                    "--gpus",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "--cpu-workers",
                    "6",
                    "--poll-seconds",
                    "30",
                ],
            )
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
