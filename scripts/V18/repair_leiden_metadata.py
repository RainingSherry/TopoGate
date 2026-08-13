#!/usr/bin/env python
"""Repair only legacy V18 Leiden K metadata after the matrix is terminal.

The old v2.2 runner derived K from benchmark labels even though Leiden did not
use it.  This utility changes only JSON metadata; predictions, matrices,
configs, and metrics are never recomputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "v18_scmae_mainline_v2_2"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")


def repair(root: Path) -> dict[str, int]:
    changed_summaries = 0
    changed_records = 0
    skipped = 0
    for summary_path in sorted(root.glob("*/*/seed*/summary.json")):
        summary = _read(summary_path)
        if summary is None or summary.get("protocol_id") != PROTOCOL_ID \
                or summary.get("variant") != "v18_leiden" \
                or summary.get("status") != "completed":
            continue
        summary["n_clusters"] = None
        summary["K_source"] = "not_applicable_leiden"
        summary["benchmark_oracle_from_y"] = False
        summary["K_used_only_in_readout"] = False
        readout = summary.get("readout")
        if isinstance(readout, dict):
            readout["K_used_only_in_readout"] = False
        _write(summary_path, summary)
        changed_summaries += 1

        record_path = summary_path.parent / "run_record.json"
        record = _read(record_path)
        if record is None:
            skipped += 1
            continue
        record["k_source"] = "not_applicable_leiden"
        _write(record_path, record)
        changed_records += 1
    return {"changed_summaries": changed_summaries, "changed_records": changed_records,
            "missing_run_records": skipped, "hashes_recomputed": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair V18 Leiden metadata without changing model artifacts")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = repair(args.root)
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
