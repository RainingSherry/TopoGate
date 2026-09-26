"""Build a no-label-access task manifest from frozen historical configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PILOT_DATASETS = ("cnae9", "student_grade", "imdb", "Mouse_retina")
SEEDS = (42, 123, 7)
HOP_VARIANTS = ("H1", "H2_B1", "H2_B2", "H2_B4", "H3_B1", "H3_B2", "H3_B4",
                "H4_B1", "H4_B2", "H4_B4", "H5_B1", "H5_B2", "H5_B4")


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def build_manifest(historical_tasks_path: str | Path, split_root: str | Path, output_path: str | Path) -> dict:
    historical = json.loads(Path(historical_tasks_path).read_text(encoding="utf-8"))
    tasks = historical["tasks"] if isinstance(historical, dict) else historical
    split_root = Path(split_root)
    selected: dict[str, dict] = {}
    for task in tasks:
        name = task.get("dataset")
        if name in PILOT_DATASETS and task.get("hop_variant") == "H1" and int(task.get("seed", -1)) == 42:
            selected.setdefault(name, task)
    if set(selected) != set(PILOT_DATASETS):
        raise ValueError(f"missing frozen historical task(s): {sorted(set(PILOT_DATASETS) - set(selected))}")
    rows = []
    for name in PILOT_DATASETS:
        historical_task = selected[name]
        selected_payload = json.loads(Path(historical_task["selected_path"]).read_text(encoding="utf-8"))
        base_config = dict(selected_payload["winner"]["config"])
        split_dir = split_root / name
        tuning_manifest = json.loads((split_dir / "tuning_manifest.json").read_text(encoding="utf-8"))
        if tuning_manifest.get("test_label_path_exposed", True):
            raise ValueError(f"{name} tuning manifest exposes test labels")
        for hop_variant in HOP_VARIANTS:
            config = dict(base_config)
            config["hop_variant"] = hop_variant
            for seed in SEEDS:
                rows.append({
                    "dataset": name,
                    "seed": int(seed),
                    "hop_variant": hop_variant,
                    "config": config,
                    "tuning_manifest_path": str((split_dir / "tuning_manifest.json").resolve()),
                    "input_kind": historical_task["input_kind"],
                    "feature_limit": 2000,
                    "K_provenance": tuning_manifest["K_provenance"],
                    "test_labels_loaded": False,
                })
    payload = {
        "protocol": "v0_rg_expanded_multihop_validation_only_v1",
        "source": str(Path(historical_tasks_path).resolve()),
        "selection": "frozen historical selected configuration; no labels read by manifest builder",
        "datasets": list(PILOT_DATASETS),
        "seeds": list(SEEDS),
        "hop_variants": list(HOP_VARIANTS),
        "tasks": rows,
    }
    _write_json(Path(output_path), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="build V0_RG multihop pilot task manifest")
    parser.add_argument("--historical-tasks", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_manifest(args.historical_tasks, args.split_root, args.output)


if __name__ == "__main__":
    main()
