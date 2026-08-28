#!/usr/bin/env python
"""Run fixed baselines for all RG-positive datasets across the two panels."""

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

PRIMARY_SUMMARY = ROOT / "result" / "V19" / "v19_rg_extended_sparse_ari_v1" / "summary" / "extension_summary.json"
PRIMARY_MANIFEST = ROOT / "result" / "V19" / "v19_rg_extended_sparse_manifest_20260811.json"
PRIMARY_BASELINE_ROOT = ROOT / "result" / "V19" / "v19_rg_extended_winner_baselines_primary_v1"
BATCH2_SUMMARY = ROOT / "result" / "V19" / "v19_rg_extended_sparse_batch2_ari_v1" / "summary" / "extension_summary.json"
BATCH2_MANIFEST = ROOT / "result" / "V19" / "v19_rg_extended_sparse_batch2_manifest_20260811.json"
BATCH2_BASELINE_ROOT = ROOT / "result" / "V19" / "v19_rg_extended_winner_baselines_batch2_v1"
METHODS = ("AHDPC", "DPC_GFNN", "GCC")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _wait_for(path: Path, seconds: int) -> dict[str, Any]:
    while not path.is_file():
        time.sleep(max(5, seconds))
    return _read(path)


def _run_panel(summary: Path, manifest: Path, output: Path, workers: int) -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "V19" / "run_extended_winner_baselines.py"),
        "--extension-summary", str(summary),
        "--manifest", str(manifest),
        "--output-dir", str(output),
        "--methods", *METHODS,
        "--min-wins", "1",
        "--workers", str(max(1, int(workers))),
    ]
    log_path = output / "launcher.log"
    output.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        return int(subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-summary", type=Path, default=PRIMARY_SUMMARY)
    parser.add_argument("--primary-manifest", type=Path, default=PRIMARY_MANIFEST)
    parser.add_argument("--primary-output", type=Path, default=PRIMARY_BASELINE_ROOT)
    parser.add_argument("--batch2-summary", type=Path, default=BATCH2_SUMMARY)
    parser.add_argument("--batch2-manifest", type=Path, default=BATCH2_MANIFEST)
    parser.add_argument("--batch2-output", type=Path, default=BATCH2_BASELINE_ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--wait-seconds", type=int, default=30)
    args = parser.parse_args()
    primary = _wait_for(args.primary_summary, int(args.wait_seconds))
    status = args.primary_output.parent / "winner_baselines_after_panels_status.json"
    if primary.get("audit_ok") is not True:
        _write(status, {"status": "blocked_incomplete_primary"})
        return 2
    primary_winners = [row for row in primary.get("datasets", []) if row.get("promotion_rg_win_by_mean_ari") is True]
    # The original single-panel launcher owns the >=5 path. This helper only
    # handles the conditional multi-panel path, so it cannot duplicate those runs.
    if len(primary_winners) >= 5:
        _write(status, {"status": "delegated_primary_met", "primary_winner_count": len(primary_winners)})
        return 0
    _write(status, {"status": "running_primary_panel", "primary_winner_count": len(primary_winners)})
    primary_code = 0
    if primary_winners:
        primary_code = _run_panel(args.primary_summary, args.primary_manifest, args.primary_output, args.workers)
    batch2 = _wait_for(args.batch2_summary, int(args.wait_seconds))
    batch2_winners = [row for row in batch2.get("datasets", []) if row.get("promotion_rg_win_by_mean_ari") is True] if batch2.get("audit_ok") is True else []
    batch2_code = 0
    if batch2.get("audit_ok") is True and batch2_winners:
        batch2_code = _run_panel(args.batch2_summary, args.batch2_manifest, args.batch2_output, args.workers)
    total_winners = len(primary_winners) + len(batch2_winners)
    result = {
        "status": "completed" if primary_code == 0 and batch2_code == 0 else "incomplete_compute",
        "primary_winner_count": len(primary_winners),
        "batch2_winner_count": len(batch2_winners),
        "total_winner_count": total_winners,
        "minimum_required": 5,
        "primary_baseline_exit_code": primary_code,
        "batch2_baseline_exit_code": batch2_code,
        "methods": list(METHODS),
        "primary_baseline_summary": str((args.primary_output / "baseline_summary.json").resolve()) if primary_winners else None,
        "batch2_baseline_summary": str((args.batch2_output / "baseline_summary.json").resolve()) if batch2_winners else None,
    }
    _write(status, result)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
