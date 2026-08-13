#!/usr/bin/env python
"""Materialize the final sparse RG goal audit after the required runs finish."""

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


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _wait(path: Path, seconds: int) -> dict[str, Any]:
    while not path.is_file():
        time.sleep(max(5, seconds))
    return _read(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--batch2-summary", type=Path, required=True)
    parser.add_argument("--primary-baseline", type=Path, required=True)
    parser.add_argument("--batch2-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=30)
    args = parser.parse_args()
    status_path = args.output_dir / "goal_audit_launcher_status.json"
    primary = _wait(args.primary_summary, args.wait_seconds)
    if primary.get("audit_ok") is not True:
        _write(status_path, {"status": "blocked_incomplete_primary"})
        return 2
    primary_wins = sum(bool(row.get("promotion_rg_win_by_mean_ari") is True) for row in primary.get("datasets", []))
    extension_args = ["--extension-summary", str(args.primary_summary)]
    baseline_args: list[str] = []
    if primary_wins >= 5:
        while not args.primary_baseline.is_file():
            time.sleep(max(5, args.wait_seconds))
        baseline_args.extend(["--baseline-summary", str(args.primary_baseline)])
    else:
        batch2 = _wait(args.batch2_summary, args.wait_seconds)
        if batch2.get("audit_ok") is not True:
            _write(status_path, {"status": "blocked_incomplete_batch2", "primary_winner_count": primary_wins})
            return 2
        extension_args.extend(["--extension-summary", str(args.batch2_summary)])
        # The cross-panel runner writes a summary only for a panel with at least
        # one winner; absent files are valid when that panel has no winners.
        if args.primary_baseline.is_file():
            baseline_args.extend(["--baseline-summary", str(args.primary_baseline)])
        if args.batch2_baseline.is_file():
            baseline_args.extend(["--baseline-summary", str(args.batch2_baseline)])
        if not baseline_args:
            _write(status_path, {"status": "incomplete_compute", "reason": "no baseline summary for any winner"})
            return 1
    command = [
        sys.executable, str(ROOT / "scripts" / "V19" / "audit_rg_sparse_goal.py"),
        *extension_args, *baseline_args, "--output-dir", str(args.output_dir),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "audit_launcher.log").open("a", encoding="utf-8") as handle:
        code = int(subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode)
    _write(status_path, {"status": "completed" if code == 0 else "audit_incomplete_or_goal_not_met", "exit_code": code, "primary_winner_count": primary_wins, "finished_at": time.time()})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
