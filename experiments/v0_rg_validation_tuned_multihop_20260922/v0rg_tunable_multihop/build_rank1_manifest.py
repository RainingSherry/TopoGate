"""Build the full archived-rank-1 multi-hop validation manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_pilot_manifest import HOP_VARIANTS, SEEDS, _write_json


def build_rank1_manifest(ranking_manifest_path: str | Path, historical_tasks_path: str | Path,
                         split_root: str | Path, output_path: str | Path) -> dict:
    ranking = json.loads(Path(ranking_manifest_path).read_text(encoding="utf-8"))
    historical = json.loads(Path(historical_tasks_path).read_text(encoding="utf-8"))
    tasks = historical["tasks"] if isinstance(historical, dict) else historical
    entries = [entry for entry in ranking["entries"] if int(entry["rank"]) == 1]
    split_root = Path(split_root)
    rows = []
    for entry in entries:
        candidates = [task for task in tasks if task.get("entry_id") == entry["entry_id"]]
        historical_task = next(
            task for task in candidates if task.get("hop_variant") == "H1" and int(task.get("seed", -1)) == 42
        )
        selected = json.loads(Path(historical_task["selected_path"]).read_text(encoding="utf-8"))
        base_config = dict(selected["winner"]["config"])
        tuning_path = split_root / historical_task["dataset"] / "tuning_manifest.json"
        if not tuning_path.exists():
            raise FileNotFoundError(tuning_path)
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        if tuning.get("test_label_path_exposed", True):
            raise ValueError(f"{entry['entry_id']} tuning manifest exposes test labels")
        for hop_variant in HOP_VARIANTS:
            config = dict(base_config)
            config["hop_variant"] = hop_variant
            for seed in SEEDS:
                rows.append({
                    "entry_id": entry["entry_id"],
                    "dataset": historical_task["dataset"],
                    "panel": entry["panel"],
                    "archived_rank": 1,
                    "archived_stratum": entry["stratum"],
                    "seed": int(seed),
                    "hop_variant": hop_variant,
                    "config": config,
                    "tuning_manifest_path": str(tuning_path.resolve()),
                    "input_kind": historical_task["input_kind"],
                    "feature_limit": 2000,
                    "K_provenance": tuning["K_provenance"],
                    "test_labels_loaded": False,
                })
    payload = {
        "protocol": "v0_rg_expanded_multihop_validation_only_v1",
        "scope": "all archived rank-1 datasets",
        "rank1_entries": len(entries),
        "seeds": list(SEEDS),
        "hop_variants": list(HOP_VARIANTS),
        "tasks": rows,
    }
    _write_json(Path(output_path), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="build full archived-rank-1 V0_RG task manifest")
    parser.add_argument("--ranking-manifest", required=True)
    parser.add_argument("--historical-tasks", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_rank1_manifest(args.ranking_manifest, args.historical_tasks, args.split_root, args.output)


if __name__ == "__main__":
    main()
