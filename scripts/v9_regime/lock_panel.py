#!/usr/bin/env python3
from __future__ import annotations

"""Lock a discovery-only mechanism panel from X-only structure features."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.v9_regime.lock_split import make_stratum
from scripts.v9_regime.protocol import PROTOCOL_ID, read_manifest, write_json


ROLE_NAMES = (
    "normal_local_structure",
    "high_distance_concentration",
    "low_mutual_snn",
    "graph_fragmented",
    "high_sparsity",
    "dense_matched_control",
)


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _quantile(rows: list[dict[str, str]], key: str, q: float) -> float:
    values = np.asarray([_number(row, key) for row in rows], dtype=float)
    return float(np.quantile(values, q)) if values.size else 0.0


def _take(rows: list[dict[str, str]], score, count: int, reverse: bool = False) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (score(row), row["dataset_id"]), reverse=reverse)
    return ordered[: max(1, int(count))]


def lock_panel(
    manifest_path: Path,
    features_path: Path,
    split_path: Path,
    output: Path,
    per_role: int = 2,
) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    feature_rows = {
        row["dataset_id"]: row
        for row in csv.DictReader(features_path.open(newline="", encoding="utf-8"))
    }
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    split_by_id = {row["dataset_id"]: row.get("split") for row in split_payload.get("assignments", [])}
    eligible: list[dict[str, str]] = []
    for record in manifest["datasets"]:
        dataset_id = str(record["dataset_id"])
        feature = feature_rows.get(dataset_id)
        if record.get("status") != "eligible" or split_by_id.get(dataset_id) != "discovery" or not feature:
            continue
        if feature.get("feature_error"):
            continue
        eligible.append(feature)
    if len(eligible) < 6:
        raise ValueError(f"at least six discovery feature rows are required, got {len(eligible)}")

    q25 = {key: _quantile(eligible, key, 0.25) for key in ("zero_fraction", "cv_knn_distance", "mean_mutual_ratio", "mean_snn", "graph_largest_component_fraction")}
    q50 = {key: _quantile(eligible, key, 0.50) for key in q25}
    q75 = {key: _quantile(eligible, key, 0.75) for key in q25}
    selected: dict[str, list[dict[str, str]]] = {role: [] for role in ROLE_NAMES}
    used: set[str] = set()

    def add(role: str, rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["dataset_id"] in used:
                continue
            selected[role].append(row)
            used.add(row["dataset_id"])
            if len(selected[role]) >= per_role:
                break

    # All scores are computed from the frozen Stage-0 feature table.  Labels,
    # metrics and run artifacts are intentionally unavailable in this function.
    add(
        "normal_local_structure",
        _take(
            eligible,
            lambda row: sum(abs(_number(row, key) - q50[key]) for key in q50),
            per_role,
        ),
    )
    add("high_distance_concentration", _take(eligible, lambda row: _number(row, "cv_knn_distance"), per_role, reverse=True))
    add(
        "low_mutual_snn",
        _take(
            eligible,
            lambda row: _number(row, "mean_mutual_ratio") + _number(row, "mean_snn"),
            per_role,
        ),
    )
    add(
        "graph_fragmented",
        _take(
            eligible,
            lambda row: _number(row, "graph_largest_component_fraction") - 0.01 * _number(row, "graph_components"),
            per_role,
        ),
    )
    add("high_sparsity", _take(eligible, lambda row: _number(row, "zero_fraction"), per_role, reverse=True))

    sparse_refs = selected["high_sparsity"] or _take(eligible, lambda row: _number(row, "zero_fraction"), 1, reverse=True)
    dense_candidates = [row for row in eligible if _number(row, "zero_fraction") <= q25["zero_fraction"]]
    for reference in sparse_refs:
        if not dense_candidates:
            break
        chosen = min(
            dense_candidates,
            key=lambda row: (
                abs(math.log10(max(1.0, _number(row, "n"))) - math.log10(max(1.0, _number(reference, "n"))))
                + abs(math.log10(max(1.0, _number(row, "d"))) - math.log10(max(1.0, _number(reference, "d")))),
                row["dataset_id"],
            ),
        )
        add("dense_matched_control", [chosen])
        dense_candidates = [row for row in dense_candidates if row["dataset_id"] != chosen["dataset_id"]]

    assignments = []
    for role in ROLE_NAMES:
        for row in selected[role]:
            assignments.append(
                {
                    "dataset_id": row["dataset_id"],
                    "role": role,
                    "stratum": make_stratum(row),
                    "family": row.get("family"),
                    "n": int(float(row.get("n", 0))),
                    "d": int(float(row.get("d", 0))),
                    "zero_fraction": _number(row, "zero_fraction"),
                    "cv_knn_distance": _number(row, "cv_knn_distance"),
                    "mean_mutual_ratio": _number(row, "mean_mutual_ratio"),
                    "mean_snn": _number(row, "mean_snn"),
                    "graph_largest_component_fraction": _number(row, "graph_largest_component_fraction"),
                }
            )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "manifest": str(manifest_path),
        "features": str(features_path),
        "split": str(split_path),
        "selection_uses_labels_or_outcomes": False,
        "selection_rule": "fixed X-only quantiles and deterministic dataset_id tie-breaks",
        "per_role_target": int(per_role),
        "roles": list(ROLE_NAMES),
        "panel_ids": sorted({row["dataset_id"] for row in assignments}),
        "assignments": sorted(assignments, key=lambda row: (row["role"], row["dataset_id"])),
    }
    write_json(output, payload)
    print(json.dumps({"output": str(output), "panel_size": len(payload["panel_ids"]), "roles": {role: len(selected[role]) for role in ROLE_NAMES}}))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-role", type=int, default=2)
    args = parser.parse_args()
    lock_panel(args.manifest, args.features, args.split, args.output, args.per_role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
