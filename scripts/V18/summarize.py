#!/usr/bin/env python
"""Aggregate V18 completed/incomplete run artifacts without label-based selection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(value, dict) or "variant" not in value or "dataset" not in value:
        return None
    return value


def _values(rows: list[dict[str, Any]], *path: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                values.append(number)
    return values


def _stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def _paired_delta_summary(
    records: list[dict[str, Any]],
    left_variant: str,
    right_variant: str,
) -> dict[str, Any]:
    """Compute pre-registered same-dataset/seed deltas without selecting runs."""
    keyed = {
        (str(row.get("dataset_id", row.get("dataset"))), int(row["seed"]), str(row["variant"])): row
        for row in records
        if row.get("status") == "completed" and "seed" in row
    }
    metrics = ("ari_active", "nmi_active", "ari_all_with_abstention_label", "nmi_all_with_abstention_label")
    deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
    by_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: {metric: [] for metric in metrics})
    pair_count = 0
    datasets: set[str] = set()
    for (dataset_id, seed, variant), left in keyed.items():
        if variant != left_variant:
            continue
        right = keyed.get((dataset_id, seed, right_variant))
        if right is None:
            continue
        pair_count += 1
        datasets.add(dataset_id)
        left_metrics = left.get("metrics", {})
        right_metrics = right.get("metrics", {})
        for metric in metrics:
            left_value = left_metrics.get(metric)
            right_value = right_metrics.get(metric)
            if left_value is None or right_value is None:
                continue
            delta = float(right_value) - float(left_value)
            deltas[metric].append(delta)
            by_dataset[dataset_id][metric].append(delta)

    aggregate: dict[str, Any] = {}
    for metric, values in deltas.items():
        mean, std = _stats(values)
        aggregate[metric] = {
            "n": len(values),
            "mean_right_minus_left": mean,
            "std": std,
            "positive_fraction": float(np.mean(np.asarray(values) > 0.0)) if values else None,
        }
    dataset_summary: dict[str, Any] = {}
    for dataset_id, metric_values in sorted(by_dataset.items()):
        dataset_summary[dataset_id] = {}
        for metric, values in metric_values.items():
            mean, std = _stats(values)
            dataset_summary[dataset_id][metric] = {
                "n": len(values), "mean_right_minus_left": mean, "std": std,
            }
    return {
        "left_variant": left_variant,
        "right_variant": right_variant,
        "pair_count": pair_count,
        "datasets_with_pairs": len(datasets),
        "aggregate": aggregate,
        "by_dataset": dataset_summary,
    }


def summarize(root: Path, manifest_id: str | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/summary.json")):
        record = _read(path)
        if record is None:
            continue
        if record.get("protocol_id") != "v18_scmae_mainline_v2_2":
            continue
        if manifest_id is not None and record.get("manifest_id") != manifest_id:
            continue
        record["_path"] = str(path)
        records.append(record)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("dataset_id", record["dataset"])), str(record["variant"]))].append(record)
    aggregate: list[dict[str, Any]] = []
    for (dataset_id, variant), rows in sorted(groups.items()):
        completed = [row for row in rows if row.get("status") == "completed"]
        ari_mean, ari_std = _stats(_values(completed, "metrics", "ari_active"))
        nmi_mean, nmi_std = _stats(_values(completed, "metrics", "nmi_active"))
        ami_mean, ami_std = _stats(_values(completed, "metrics", "ami_active"))
        ari_all_mean, ari_all_std = _stats(_values(completed, "metrics", "ari_all_with_abstention_label"))
        nmi_all_mean, nmi_all_std = _stats(_values(completed, "metrics", "nmi_all_with_abstention_label"))
        gate_mean, gate_std = _stats(_values(completed, "gate", "hard_open_rate"))
        expected_gate_mean, expected_gate_std = _stats(_values(completed, "gate", "expected_open_rate"))
        retention_mean, retention_std = _stats(_values(completed, "relation", "edge_retention_rate"))
        zero_row_mean, zero_row_std = _stats(_values(completed, "relation", "zero_outgoing_row_fraction"))
        abstain_mean, abstain_std = _stats(_values(completed, "readout", "abstention_rate"))
        components_mean, components_std = _stats(_values(completed, "readout", "connected_components"))
        candidate_mean, candidate_std = _stats(_values(completed, "relation", "candidate_edges"))
        nnz_mean, nnz_std = _stats(_values(completed, "relation", "coefficient_nnz"))
        readout_status: dict[str, int] = defaultdict(int)
        for row in completed:
            readout_status[str(row.get("readout", {}).get("status", "unknown"))] += 1
        aggregate.append({
            "dataset_id": dataset_id,
            "dataset": str(rows[0].get("dataset", dataset_id)), "variant": variant, "runs_seen": len(rows),
            "completed": len(completed), "incomplete": len(rows) - len(completed),
            "seeds_completed": sorted(int(row["seed"]) for row in completed if "seed" in row),
            "ari_active_mean": ari_mean, "ari_active_std": ari_std,
            "nmi_active_mean": nmi_mean, "nmi_active_std": nmi_std,
            "ami_active_mean": ami_mean, "ami_active_std": ami_std,
            "ari_all_mean": ari_all_mean, "ari_all_std": ari_all_std,
            "nmi_all_mean": nmi_all_mean, "nmi_all_std": nmi_all_std,
            "hard_open_rate_mean": gate_mean, "hard_open_rate_std": gate_std,
            "expected_open_rate_mean": expected_gate_mean, "expected_open_rate_std": expected_gate_std,
            "edge_retention_rate_mean": retention_mean, "edge_retention_rate_std": retention_std,
            "zero_outgoing_row_fraction_mean": zero_row_mean, "zero_outgoing_row_std": zero_row_std,
            "abstention_rate_mean": abstain_mean, "abstention_rate_std": abstain_std,
            "connected_components_mean": components_mean, "connected_components_std": components_std,
            "candidate_edges_mean": candidate_mean, "candidate_edges_std": candidate_std,
            "coefficient_nnz_mean": nnz_mean, "coefficient_nnz_std": nnz_std,
            "readout_status_counts": dict(sorted(readout_status.items())),
        })
    status_counts: dict[str, int] = defaultdict(int)
    for path in sorted(root.glob("**/run_record.json")):
        record = _read(path) or {}
        status_counts[str(record.get("status", "unknown"))] += 1
    comparisons = [
        ("latent_candidate_spectral", "latent_GW_frozen"),
        ("latent_GW_frozen", "v18_full"),
        ("latent_GW_frozen", "latent_C_exactzero"),
    ]
    return {"root": str(root), "manifest_id": manifest_id, "run_summaries": len(records),
            "status_counts": dict(sorted(status_counts.items())), "groups": aggregate,
            "paired_deltas": [_paired_delta_summary(records, left, right) for left, right in comparisons],
            "selection_or_tuning_uses_labels": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize V18 run summaries")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-id", default=None)
    args = parser.parse_args()
    payload = summarize(args.root, args.manifest_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = sorted({key for row in payload["groups"] for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["groups"])
    print(json.dumps({"run_summaries": payload["run_summaries"], "status_counts": payload["status_counts"],
                      "json": str(args.output), "csv": str(csv_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
