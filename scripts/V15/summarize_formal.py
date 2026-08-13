#!/usr/bin/env python3
"""Summarize V15 formal runs with paired deltas against self_only."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for summary_path in sorted(args.run_root.rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary.get("metrics", {})
            rows.append(
                {
                    "dataset": summary.get("dataset"),
                    "condition": summary.get("run_metadata", {}).get("condition", "clean"),
                    "variant": summary.get("run_metadata", {}).get("variant")
                    or summary.get("config", {}).get("gate_mode", "counterfactual"),
                    "seed": summary.get("seed"),
                    "ari": metrics.get("ari"),
                    "nmi": metrics.get("nmi"),
                    "ami": metrics.get("ami"),
                    "null_mass": metrics.get("gate", {}).get("final_null_mass"),
                    "edge_mass": metrics.get("gate", {}).get("final_edge_mass"),
                    "effective_neighbors": metrics.get("gate", {}).get("final_effective_neighbors"),
                    "labels_used_during_fit": summary.get("labels_used_during_fit"),
                }
            )
        except Exception as exc:
            errors.append({"path": str(summary_path), "error": f"{type(exc).__name__}: {exc}"})
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["condition"]), str(row["variant"]))].append(row)
    self_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row["variant"] == "self_only":
            self_rows[(str(row["dataset"]), str(row["condition"]), int(row["seed"]))] = row
    aggregate: list[dict[str, Any]] = []
    for (dataset, condition, variant), members in sorted(grouped.items()):
        numeric = {key: np.asarray([row[key] for row in members if row[key] is not None], dtype=float) for key in ("ari", "nmi", "ami", "null_mass", "edge_mass", "effective_neighbors")}
        deltas = []
        for row in members:
            reference = self_rows.get((dataset, condition, int(row["seed"])))
            if reference is not None and row["ari"] is not None and reference["ari"] is not None:
                deltas.append(float(row["ari"] - reference["ari"]))
        aggregate.append(
            {
                "dataset": dataset,
                "condition": condition,
                "variant": variant,
                "n": len(members),
                "ari_mean": float(np.mean(numeric["ari"])) if numeric["ari"].size else None,
                "ari_std": float(np.std(numeric["ari"])) if numeric["ari"].size else None,
                "nmi_mean": float(np.mean(numeric["nmi"])) if numeric["nmi"].size else None,
                "ami_mean": float(np.mean(numeric["ami"])) if numeric["ami"].size else None,
                "delta_ari_mean_vs_self": float(np.mean(deltas)) if deltas else None,
                "delta_ari_median_vs_self": float(np.median(deltas)) if deltas else None,
                "positive_delta_fraction": float(np.mean(np.asarray(deltas) > 0.0)) if deltas else None,
                "null_mass_mean": float(np.mean(numeric["null_mass"])) if numeric["null_mass"].size else None,
                "edge_mass_mean": float(np.mean(numeric["edge_mass"])) if numeric["edge_mass"].size else None,
                "effective_neighbors_mean": float(np.mean(numeric["effective_neighbors"])) if numeric["effective_neighbors"].size else None,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "V15-formal-summary-1",
        "runs": rows,
        "aggregate": aggregate,
        "errors": errors,
        "labels_used_during_fit_audit": all(row["labels_used_during_fit"] is False for row in rows),
    }
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    if aggregate:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
            writer.writeheader()
            writer.writerows(aggregate)
    print(json.dumps({"runs": len(rows), "groups": len(aggregate), "errors": len(errors), "output": str(args.output)}))


if __name__ == "__main__":
    main()
