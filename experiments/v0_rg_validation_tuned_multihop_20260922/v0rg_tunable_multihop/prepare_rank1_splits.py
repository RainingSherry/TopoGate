"""Trusted one-time split preparation for every archived rank-1 data set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .validation_protocol import prepare_isolated_split


def main() -> None:
    parser = argparse.ArgumentParser(description="prepare feature-only rank-1 split sidecars")
    parser.add_argument("--ranking-manifest", required=True)
    parser.add_argument("--historical-tasks", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    if args.worker_count < 1 or args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")
    ranking = json.loads(Path(args.ranking_manifest).read_text(encoding="utf-8"))
    historical = json.loads(Path(args.historical_tasks).read_text(encoding="utf-8"))
    tasks = historical["tasks"] if isinstance(historical, dict) else historical
    split_root = Path(args.split_root)
    for entry in ranking["entries"]:
        if int(entry["rank"]) != 1:
            continue
        bucket = int(hashlib.sha256(entry["entry_id"].encode("utf-8")).hexdigest()[:16], 16) % args.worker_count
        if bucket != args.worker_index:
            continue
        task = next(item for item in tasks if item.get("entry_id") == entry["entry_id"])
        output = split_root / task["dataset"]
        if (output / "tuning_manifest.json").exists():
            print(json.dumps({"dataset": task["dataset"], "status": "skipped_existing"}), flush=True)
            continue
        try:
            manifest = prepare_isolated_split(task["data_path"], output)
            print(json.dumps({"dataset": task["dataset"], "status": "completed",
                              "sizes": manifest["split_sizes"],
                              "strategy": manifest["split_strategy"]}), flush=True)
        except Exception as exc:
            print(json.dumps({"dataset": task["dataset"], "status": "failed", "error": repr(exc)}), flush=True)
            raise


if __name__ == "__main__":
    main()
