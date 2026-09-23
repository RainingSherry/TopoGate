"""Select one expanded path setting from validation records only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SELECTION_SEEDS = (42, 123, 7)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _rank_key(variant: str) -> tuple[int, int]:
    if variant == "H1":
        return (1, 1)
    if variant == "H12":
        return (2, 1)
    if variant == "H123":
        return (3, 1)
    head, breadth = variant.split("_B")
    return (int(head[1:]), int(breadth))


def select_validation_records(record_paths: list[str | Path]) -> dict:
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in record_paths]
    if not records:
        raise ValueError("no validation records")
    grouped: dict[str, list[dict]] = {}
    for record in records:
        if record.get("test_labels_loaded") is not False:
            raise ValueError("selection refuses records that accessed final-test labels")
        if record.get("selection_label_scope") != "validation_only":
            raise ValueError("selection requires validation-only records")
        grouped.setdefault(record["config"]["hop_variant"], []).append(record)
    scored: dict[str, float] = {}
    for variant, group in grouped.items():
        seeds = tuple(sorted(int(item["seed"]) for item in group))
        if seeds != tuple(sorted(SELECTION_SEEDS)):
            raise ValueError(f"{variant} has seeds {seeds}, expected {SELECTION_SEEDS}")
        scored[variant] = float(np.mean([item["validation_metrics"]["ari"] for item in group]))
    if "H1" not in scored:
        raise ValueError("H1 validation baseline is required")
    candidates = {name: value for name, value in scored.items() if name != "H1"}
    if not candidates:
        raise ValueError("no multihop candidates")
    best_score = max(candidates.values())
    tied = [name for name, value in candidates.items() if best_score - value <= 0.001]
    chosen = min(tied, key=_rank_key)
    representative = next(item for item in grouped[chosen] if int(item["seed"]) == 42)
    return {
        "selection_locked": True,
        "selection_metric": "validation_ari_mean",
        "selection_seeds": list(SELECTION_SEEDS),
        "test_labels_loaded": False,
        "baseline": "H1",
        "baseline_validation_ari_mean": scored["H1"],
        "chosen_hop_variant": chosen,
        "chosen_validation_ari_mean": scored[chosen],
        "chosen_validation_delta_ari": scored[chosen] - scored["H1"],
        "all_validation_ari_means": scored,
        "chosen_config": representative["config"],
        "K_provenance": representative["K_provenance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="select V0_RG multihop config with validation records only")
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _write_json(Path(args.output), select_validation_records(args.records))


if __name__ == "__main__":
    main()
