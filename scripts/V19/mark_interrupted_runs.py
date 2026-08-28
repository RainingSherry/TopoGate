#!/usr/bin/env python
"""Mark verified interrupted V19 runs without fabricating summaries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    root = args.root
    running = []
    for path in sorted(root.rglob("run_record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "running":
            continue
        running.append(path)
        record.update(
            {
                "status": "incomplete_compute",
                "error_type": "ExternalInterruption",
                "error": str(args.reason),
            }
        )
        _write(path, record)
        status_path = path.parent / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        status.update(
            {
                "status": "incomplete_compute",
                "run_key": record.get("run_key"),
                "protocol_id": record.get("protocol_id"),
                "error_type": "ExternalInterruption",
                "error": str(args.reason),
            }
        )
        _write(status_path, status)

    if int(args.expected_count) and len(running) != int(args.expected_count):
        raise SystemExit(
            f"expected {int(args.expected_count)} running records, found {len(running)}; no launcher status changed"
        )

    launcher_path = root / "launcher_status.json"
    launcher = json.loads(launcher_path.read_text(encoding="utf-8")) if launcher_path.is_file() else {}
    launcher.update(
        {
            "status": "incomplete_compute",
            "audit_ok": False,
            "audit_message": str(args.reason),
            "termination_reason": str(args.reason),
            "terminated_unix_time": float(time.time()),
            "interrupted_run_count": len(running),
        }
    )
    _write(launcher_path, launcher)
    for path in running:
        print(path)
    print(f"marked_incomplete_compute={len(running)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
