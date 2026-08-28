#!/usr/bin/env python
"""Aggregate V19 RG tuning runs using X-only diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.V19.tune_unsupervised import (  # noqa: E402
    CANDIDATE_BY_ID,
    PROTOCOL_ID,
    _load_manifest,
    _write_json,
)


DEFAULT_OUTPUT = ROOT / "result" / "V19" / "v19_rg_unsup_tuning_v1"


def _normalised_ranks(values: list[float], *, higher_is_better: bool) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    if higher_is_better:
        order.reverse()
    scores = [0.0] * len(values)
    denominator = max(1, len(values) - 1)
    for rank, index in enumerate(order):
        scores[index] = 1.0 - float(rank) / float(denominator)
    return scores


def _read_completed(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("protocol_id") != PROTOCOL_ID or payload.get("status") != "completed":
            continue
        if payload.get("labels_accessed") is not False:
            raise ValueError(f"label access audit failed: {path}")
        if payload.get("y_key_read") is not False:
            raise ValueError(f"NPZ y-key audit failed: {path}")
        if payload.get("n_clusters_used") is not None or payload.get("readout_enabled") is not False:
            raise ValueError(f"readout audit failed: {path}")
        if "metrics" in payload or "labels_true" in payload:
            raise ValueError(f"label-derived output found: {path}")
        diag = payload.get("unsupervised_diagnostics")
        if not isinstance(diag, dict):
            raise ValueError(f"missing unsupervised diagnostics: {path}")
        rows.append(payload)
    return rows


def _complete_units(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
) -> set[tuple[str, int]]:
    expected = {(str(record["dataset_id"]), int(seed)) for record in manifest["datasets"] for seed in seeds}
    return {
        (str(row["dataset_id"]), int(row["seed"]))
        for row in rows
        if (str(row["dataset_id"]), int(row["seed"])) in expected
    }


def summarize(root: Path, manifest_path: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    rows = _read_completed(root)
    expected_units = {
        (str(record["dataset_id"]), int(seed))
        for record in manifest["datasets"]
        for seed in seeds
    }
    completed_units = _complete_units(rows, manifest, seeds)
    if completed_units != expected_units:
        missing = sorted(expected_units - completed_units)
        raise RuntimeError(
            f"refusing to select from incomplete tuning matrix: "
            f"completed_units={len(completed_units)}/{len(expected_units)}, missing={missing[:8]}"
        )

    by_unit: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        unit = (str(row["dataset_id"]), int(row["seed"]))
        by_unit[unit][str(row["candidate_id"])] = row
    candidate_units = defaultdict(set)
    for unit, candidates in by_unit.items():
        for candidate_id in candidates:
            candidate_units[candidate_id].add(unit)
    complete_candidates = sorted(
        candidate_id
        for candidate_id, units in candidate_units.items()
        if units == expected_units
    )
    if not complete_candidates:
        raise RuntimeError("no candidate has a complete set of X-only tuning units")

    unit_scores: list[dict[str, Any]] = []
    for unit in sorted(expected_units):
        candidates = [candidate_id for candidate_id in complete_candidates if candidate_id in by_unit[unit]]
        losses = [float(by_unit[unit][candidate_id]["unsupervised_diagnostics"]["eval_mask_loss"]) for candidate_id in candidates]
        stabilities = [float(by_unit[unit][candidate_id]["unsupervised_diagnostics"]["latent_view_cosine_mean"]) for candidate_id in candidates]
        overlaps = [float(by_unit[unit][candidate_id]["unsupervised_diagnostics"]["input_neighbor_overlap"]) for candidate_id in candidates]
        loss_scores = _normalised_ranks(losses, higher_is_better=False)
        stability_scores = _normalised_ranks(stabilities, higher_is_better=True)
        overlap_scores = _normalised_ranks(overlaps, higher_is_better=True)
        for index, candidate_id in enumerate(candidates):
            row = by_unit[unit][candidate_id]
            diag = row["unsupervised_diagnostics"]
            unit_scores.append(
                {
                    "dataset_id": unit[0],
                    "seed": unit[1],
                    "input_protocol": row["input_protocol"],
                    "candidate_id": candidate_id,
                    "eval_mask_loss": losses[index],
                    "latent_view_cosine_mean": stabilities[index],
                    "input_neighbor_overlap": overlaps[index],
                    "latent_mean_feature_std": float(diag["latent_mean_feature_std"]),
                    "mean_node_gate": float(row["gate_summary"]["mean_node_gate"]),
                    "x_only_score": float(
                        (loss_scores[index] + stability_scores[index] + overlap_scores[index]) / 3.0
                    ),
                }
            )

    candidate_rows: list[dict[str, Any]] = []
    for candidate_id in complete_candidates:
        selected = [row for row in unit_scores if row["candidate_id"] == candidate_id]
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "x_only_score": float(sum(row["x_only_score"] for row in selected) / len(selected)),
                "mean_eval_mask_loss": float(sum(row["eval_mask_loss"] for row in selected) / len(selected)),
                "mean_latent_view_cosine": float(sum(row["latent_view_cosine_mean"] for row in selected) / len(selected)),
                "mean_input_neighbor_overlap": float(sum(row["input_neighbor_overlap"] for row in selected) / len(selected)),
                "mean_latent_feature_std": float(sum(row["latent_mean_feature_std"] for row in selected) / len(selected)),
                "mean_node_gate": float(sum(row["mean_node_gate"] for row in selected) / len(selected)),
                "n_units": len(selected),
                "protocol_scores": {
                    protocol: float(
                        sum(row["x_only_score"] for row in selected if row["input_protocol"] == protocol)
                        / max(1, sum(row["input_protocol"] == protocol for row in selected))
                    )
                    for protocol in sorted({row["input_protocol"] for row in selected})
                },
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            -row["x_only_score"],
            row["mean_eval_mask_loss"],
            -row["mean_latent_view_cosine"],
            row["candidate_id"],
        )
    )
    best = candidate_rows[0]
    selected_config = {
        "protocol_id": PROTOCOL_ID,
        "selection_method": "X-only equal-rank mean of masked recovery, latent view stability, and input-neighbor overlap",
        "candidate_id": best["candidate_id"],
        "overrides": CANDIDATE_BY_ID[best["candidate_id"]]["overrides"],
        "base_config": "methods/TopoGate/V19_rg_adapter/configs/v19_rg.yaml",
        "labels_accessed": False,
        "y_key_read": False,
        "n_clusters_used": None,
        "readout_enabled": False,
        "n_datasets": len(manifest["datasets"]),
        "seeds": list(seeds),
        "best_summary": best,
    }
    summary = {
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "manifest_id": manifest.get("manifest_id"),
        "selection_method": selected_config["selection_method"],
        "labels_accessed": False,
        "y_key_read": False,
        "n_clusters_used": None,
        "readout_enabled": False,
        "expected_units": len(expected_units),
        "completed_units": len(completed_units),
        "completed_candidates": complete_candidates,
        "selected_config": selected_config,
        "candidate_scores": candidate_rows,
    }
    _write_json(root / "unsupervised_selection.json", summary)
    _write_json(root / "selected_config.json", selected_config)
    with (root / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "candidate_id",
            "x_only_score",
            "mean_eval_mask_loss",
            "mean_latent_view_cosine",
            "mean_input_neighbor_overlap",
            "mean_latent_feature_std",
            "mean_node_gate",
            "n_units",
            "protocol_scores",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow(row)
    with (root / "unit_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(unit_scores[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(unit_scores)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize V19 X-only RG tuning")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    args = parser.parse_args()
    summary = summarize(args.output_dir, args.manifest, tuple(int(seed) for seed in args.seeds))
    print(json.dumps(summary["selected_config"], ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
