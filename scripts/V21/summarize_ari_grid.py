#!/usr/bin/env python3
"""Audit and select the V21 ARI-selected development grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


MODEL_PROTOCOL_ID = "v21_assignment_adversarial_v2_graphfix_v1"
REQUIRED = (
    "summary.json",
    "metrics.json",
    "resolved_config.json",
    "training_history.json",
    "preprocess_profile.json",
    "graph_profile.json",
    "stats_profile.json",
    "selected_feature_indices.npy",
    "embedding_final.npy",
    "predictions.npy",
    "labels_true.npy",
    "cluster_probabilities.npy",
    "checkpoint.pt",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(mean(values)), float(stdev(values)) if len(values) > 1 else 0.0


def _audit_job(job: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    output = Path(job["output"])
    missing = [name for name in REQUIRED if not (output / name).is_file()]
    if missing:
        return None, [f"{job['key']}: missing {','.join(missing)}"]
    try:
        summary = _read(output / "summary.json")
        metrics = _read(output / "metrics.json")
        resolved = _read(output / "resolved_config.json")
        preprocess = _read(output / "preprocess_profile.json")
        graph = _read(output / "graph_profile.json")
        history = _read(output / "training_history.json")
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{job['key']}: invalid JSON {exc}"]
    candidate = job["candidate"]
    errors: list[str] = []
    checks = {
        "status": summary.get("status") == "completed",
        "protocol": summary.get("protocol_id") == MODEL_PROTOCOL_ID,
        "variant": summary.get("variant") == "topology_assignment_adversarial",
        "dataset": summary.get("dataset") == job["dataset"],
        "seed": int(summary.get("seed", -1)) == 42,
        "labels_used_during_fit": summary.get("labels_used_during_fit") is False,
        "metrics_labels_used_during_fit": metrics.get("labels_used_during_fit") is False,
        "resolved_epochs": int(resolved.get("epochs", -1)) == int(candidate["epochs"]),
        "resolved_warmup": int(resolved.get("warmup_epochs", -1)) == int(candidate["warmup_epochs"]),
        "resolved_assignment_weight": abs(float(resolved.get("assignment_weight", -1.0)) - float(candidate["assignment_weight"])) < 1e-12,
        "resolved_gate_lr": abs(float(resolved.get("gate_lr", -1.0)) - float(candidate["gate_lr"])) < 1e-12,
        "preprocess_labels": preprocess.get("labels_used") is False and preprocess.get("labels_used_during_fit") is False,
        "preprocess_K": preprocess.get("K_used") is False,
        "graph_no_label_leakage": graph.get("label_leakage_diagnostic", False) is False,
        "graph_no_self_edges": int(graph.get("self_edges", -1)) == 0,
    }
    errors.extend(f"{job['key']}: failed {name}" for name, passed in checks.items() if not passed)
    history_values = [value for record in history if isinstance(record, dict) for value in record.values() if isinstance(value, (int, float))]
    if not isinstance(history, list) or len(history) != int(candidate["epochs"]):
        errors.append(f"{job['key']}: history length mismatch")
    if not history_values or not all(math.isfinite(float(value)) for value in history_values):
        errors.append(f"{job['key']}: non-finite history")
    metric_values = [metrics.get("ari"), metrics.get("nmi"), metrics.get("acc")]
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in metric_values):
        errors.append(f"{job['key']}: non-finite metrics")
    if errors:
        return None, errors
    try:
        embedding = np.load(output / "embedding_final.npy", mmap_mode="r")
        predictions = np.load(output / "predictions.npy", mmap_mode="r")
        labels_true = np.load(output / "labels_true.npy", mmap_mode="r")
        probabilities = np.load(output / "cluster_probabilities.npy", mmap_mode="r")
    except (OSError, ValueError) as exc:
        return None, [f"{job['key']}: invalid numpy artifact {exc}"]
    n_samples = int(summary.get("n_samples", -1))
    hidden_size = int(resolved.get("hidden_size", -1))
    n_clusters = int(metrics.get("n_clusters", -1))
    shape_checks = {
        "embedding": embedding.shape == (n_samples, hidden_size),
        "predictions": predictions.shape == (n_samples,),
        "labels_true": labels_true.shape == (n_samples,),
        "probabilities": probabilities.shape == (n_samples, n_clusters),
        "finite_arrays": bool(np.isfinite(embedding).all() and np.isfinite(predictions).all() and np.isfinite(labels_true).all() and np.isfinite(probabilities).all()),
    }
    errors.extend(f"{job['key']}: failed shape/{name}" for name, passed in shape_checks.items() if not passed)
    if errors:
        return None, errors
    return {
        "key": job["key"],
        "candidate_id": job["candidate_id"],
        "dataset": job["dataset"],
        "seed": 42,
        "assignment_weight": float(candidate["assignment_weight"]),
        "gate_lr": float(candidate["gate_lr"]),
        "epochs": int(candidate["epochs"]),
        "warmup_epochs": int(candidate["warmup_epochs"]),
        "infomax_weight": float(candidate["infomax_weight"]),
        "ari": float(metrics["ari"]),
        "nmi": float(metrics["nmi"]),
        "acc": float(metrics["acc"]),
    }, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the V21 ARI-selected development grid")
    parser.add_argument("--output-dir", type=Path, default=Path("result/V21/v21_ari_grid_seed42_20260811"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.output_dir
    spec = _read(root / "grid_spec.json")
    state = _read(root / "grid_state.json")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for job in state.get("jobs", []):
        row, job_errors = _audit_job(job)
        errors.extend(job_errors)
        if row is not None:
            rows.append(row)
    expected = int(spec["expected_jobs"])
    audit = {
        "grid_protocol_id": spec["protocol_id"],
        "model_protocol_id": MODEL_PROTOCOL_ID,
        "expected_jobs": expected,
        "audited_jobs": len(state.get("jobs", [])),
        "completed_valid_jobs": len(rows),
        "selection_uses_labels": True,
        "fit_receives_y": False,
        "errors": errors,
        "audit_ok": not errors and len(rows) == expected,
    }
    (root / "grid_audit.json").write_text(json.dumps(audit, ensure_ascii=True, indent=2), encoding="utf-8")
    if errors and not args.allow_incomplete:
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        return 2
    with (root / "grid_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["candidate_id", "dataset", "ari"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_candidate_dataset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate_dataset[(row["candidate_id"], row["dataset"])].append(row)
    candidate_rows: list[dict[str, Any]] = []
    for candidate in spec.get("candidate_grid", {}).get("assignment_weight", []):
        _ = candidate
    candidates = {job["candidate_id"]: job["candidate"] for job in state.get("jobs", [])}
    for candidate_id, candidate in sorted(candidates.items()):
        dataset_means = []
        dataset_rows = []
        for dataset, _protocol, _path in (
            ("cnae9", "shared_text", None),
            ("Mouse_retina", "clubench_bridge", None),
            ("Baron Human", "clubench_bridge", None),
            ("Campbell", "clubench_bridge", None),
            ("sms_spam_collection", "shared_text", None),
            ("hate_speech", "shared_text", None),
        ):
            values = [row["ari"] for row in by_candidate_dataset.get((candidate_id, dataset), [])]
            if values:
                dataset_mean, dataset_std = _mean_std(values)
                dataset_means.append(dataset_mean)
                dataset_rows.append({"dataset": dataset, "ari_mean": dataset_mean, "ari_std": dataset_std})
        if len(dataset_means) == 6:
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "assignment_weight": float(candidate["assignment_weight"]),
                    "gate_lr": float(candidate["gate_lr"]),
                    "epochs": int(candidate["epochs"]),
                    "warmup_epochs": int(candidate["warmup_epochs"]),
                    "infomax_weight": float(candidate["infomax_weight"]),
                    "macro_ari": float(mean(dataset_means)),
                    "worst_dataset_ari": float(min(dataset_means)),
                    "best_dataset_ari": float(max(dataset_means)),
                    "dataset_rows": dataset_rows,
                }
            )
    candidate_rows.sort(key=lambda row: (-row["macro_ari"], row["candidate_id"]))
    selected = candidate_rows[0] if candidate_rows else None
    payload = audit | {
        "candidate_ranked": candidate_rows,
        "selected_candidate": selected,
        "selection_metric": "macro_mean_over_six_dataset_ari",
        "evidence_level": "ari_selected_development",
    }
    (root / "grid_summary.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if selected is not None:
        (root / "selected_config.json").write_text(json.dumps(selected, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0 if audit["audit_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
