#!/usr/bin/env python3
from __future__ import annotations

"""Lock the 70/30 discovery/confirmation split from X-only features."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.v9_regime.protocol import SPLIT_SEED, PROTOCOL_ID, read_manifest, write_json


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _bin(value: float, edges: tuple[float, ...]) -> str:
    return str(int(np.digitize([value], edges, right=False)[0]))


def make_stratum(row: dict[str, str]) -> str:
    n = max(1.0, _number(row, "n"))
    d = max(1.0, _number(row, "d"))
    return "|".join(
        [
            row.get("family", "unknown"),
            f"nd={_bin(math.log10(n), (2.5, 3.5, 4.5))}",
            f"dd={_bin(math.log10(d), (1.0, 2.0, 3.0, 4.0))}",
            f"zero={_bin(_number(row, 'zero_fraction'), (0.2, 0.5, 0.8, 0.95))}",
            f"dist={_bin(_number(row, 'cv_knn_distance'), (0.1, 0.25, 0.5, 1.0))}",
            f"graph={_bin(_number(row, 'graph_largest_component_fraction'), (0.5, 0.8, 0.95, 0.995))}",
        ]
    )


def lock_split(manifest_path: Path, feature_csv: Path, output: Path, seed: int = SPLIT_SEED, confirmation_fraction: float = 0.30) -> dict:
    manifest = read_manifest(manifest_path)
    feature_rows = list(csv.DictReader(feature_csv.open(newline="", encoding="utf-8")))
    by_id = {row["dataset_id"]: row for row in feature_rows}
    strata: dict[str, list[str]] = defaultdict(list)
    assignments: dict[str, dict] = {}
    for record in manifest["datasets"]:
        dataset_id = str(record["dataset_id"])
        feature = by_id.get(dataset_id, {})
        if record.get("status") != "eligible" or feature.get("feature_error"):
            assignments[dataset_id] = {
                "dataset_id": dataset_id,
                "split": "excluded",
                "stratum": None,
                "reason": record.get("status_reason") or feature.get("feature_error") or "missing_feature_row",
            }
            continue
        stratum = make_stratum(feature)
        strata[stratum].append(dataset_id)

    rng = np.random.default_rng(int(seed))
    for stratum, dataset_ids in sorted(strata.items()):
        ordered = list(dataset_ids)
        rng.shuffle(ordered)
        if len(ordered) == 1:
            confirmation_count = 0
        else:
            confirmation_count = max(1, int(round(len(ordered) * confirmation_fraction)))
            confirmation_count = min(confirmation_count, len(ordered) - 1)
        confirmation = set(ordered[:confirmation_count])
        for dataset_id in ordered:
            assignments[dataset_id] = {
                "dataset_id": dataset_id,
                "split": "confirmation" if dataset_id in confirmation else "discovery",
                "stratum": stratum,
                "reason": "x_only_stratified_split",
            }

    # Preserve an external-family holdout even when every member of a small
    # family falls into a singleton geometry stratum. This still uses only X
    # metadata; it prevents a whole modality (notably scRNA) from disappearing
    # from confirmation by construction.
    family_groups: dict[str, list[str]] = defaultdict(list)
    for record in manifest["datasets"]:
        dataset_id = str(record["dataset_id"])
        if assignments.get(dataset_id, {}).get("split") != "discovery":
            continue
        feature = by_id.get(dataset_id, {})
        family_groups[feature.get("family", "unknown")].append(dataset_id)
    for family, dataset_ids in sorted(family_groups.items()):
        all_family = [
            str(record["dataset_id"])
            for record in manifest["datasets"]
            if str(record["dataset_id"]) in by_id and by_id[str(record["dataset_id"])].get("family", "unknown") == family
            and assignments.get(str(record["dataset_id"]), {}).get("split") in {"discovery", "confirmation"}
        ]
        if len(all_family) < 2:
            continue
        if not any(assignments.get(dataset_id, {}).get("split") == "confirmation" for dataset_id in all_family):
            chosen = sorted(all_family)[0]
            assignments[chosen]["split"] = "confirmation"
            assignments[chosen]["reason"] = "x_only_family_holdout"

    payload = {
        "protocol_id": PROTOCOL_ID,
        "manifest": str(manifest_path),
        "features": str(feature_csv),
        "split_seed": int(seed),
        "confirmation_fraction_target": float(confirmation_fraction),
        "split_uses_labels_or_outcomes": False,
        "assignments": [assignments[key] for key in sorted(assignments)],
    }
    write_json(output, payload)
    counts = defaultdict(int)
    for assignment in payload["assignments"]:
        counts[assignment["split"]] += 1
    print(json.dumps({"output": str(output), "counts": dict(counts), "strata": len(strata)}))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--confirmation-fraction", type=float, default=0.30)
    args = parser.parse_args()
    lock_split(args.manifest, args.features, args.output, args.seed, args.confirmation_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
