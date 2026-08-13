#!/usr/bin/env python3
"""Audit and summarize the three-seed ARI-selected V21 confirmation."""

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
    errors: list[str] = []
    checks = {
        "status": summary.get("status") == "completed",
        "protocol": summary.get("protocol_id") == MODEL_PROTOCOL_ID,
        "variant": summary.get("variant") == "topology_assignment_adversarial",
        "dataset": summary.get("dataset") == job["dataset"],
        "seed": int(summary.get("seed", -1)) == int(job["seed"]),
        "labels_used_during_fit": summary.get("labels_used_during_fit") is False,
        "metrics_labels_used_during_fit": metrics.get("labels_used_during_fit") is False,
        "resolved_epochs": int(resolved.get("epochs", -1)) == 80,
        "resolved_warmup": int(resolved.get("warmup_epochs", -1)) == 40,
        "resolved_assignment_weight": abs(float(resolved.get("assignment_weight", -1.0)) - 0.1) < 1e-12,
        "resolved_gate_lr": abs(float(resolved.get("gate_lr", -1.0)) - 2.5e-4) < 1e-12,
        "preprocess_labels": preprocess.get("labels_used") is False and preprocess.get("labels_used_during_fit") is False,
        "preprocess_K": preprocess.get("K_used") is False,
        "graph_no_label_leakage": graph.get("label_leakage_diagnostic", False) is False,
        "graph_no_self_edges": int(graph.get("self_edges", -1)) == 0,
    }
    errors.extend(f"{job['key']}: failed {name}" for name, passed in checks.items() if not passed)
    history_values = [value for record in history if isinstance(record, dict) for value in record.values() if isinstance(value, (int, float))]
    if not isinstance(history, list) or len(history) != 80:
        errors.append(f"{job['key']}: history length is not 80")
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
    array_checks = {
        "embedding_shape": embedding.shape == (n_samples, hidden_size),
        "predictions_shape": predictions.shape == (n_samples,),
        "labels_true_shape": labels_true.shape == (n_samples,),
        "probabilities_shape": probabilities.shape == (n_samples, n_clusters),
        "arrays_finite": bool(np.isfinite(embedding).all() and np.isfinite(predictions).all() and np.isfinite(labels_true).all() and np.isfinite(probabilities).all()),
    }
    errors.extend(f"{job['key']}: failed {name}" for name, passed in array_checks.items() if not passed)
    if errors:
        return None, errors
    return {
        "key": job["key"],
        "dataset": job["dataset"],
        "seed": int(job["seed"]),
        "ari": float(metrics["ari"]),
        "nmi": float(metrics["nmi"]),
        "acc": float(metrics["acc"]),
    }, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the ARI-selected V21 confirmation")
    parser.add_argument("--output-dir", type=Path, default=Path("result/V21/v21_ari_confirm_aw0.1_glr0.00025_ep80_20260811"))
    args = parser.parse_args()
    root = args.output_dir
    spec = _read(root / "confirm_spec.json")
    state = _read(root / "confirm_state.json")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for job in state.get("jobs", []):
        row, job_errors = _audit_job(job)
        errors.extend(job_errors)
        if row is not None:
            rows.append(row)
    audit = {
        "protocol_id": spec["protocol_id"],
        "model_protocol_id": MODEL_PROTOCOL_ID,
        "expected_jobs": int(spec["expected_jobs"]),
        "audited_jobs": len(state.get("jobs", [])),
        "completed_valid_jobs": len(rows),
        "selection_uses_labels": True,
        "fit_receives_y": False,
        "errors": errors,
        "audit_ok": not errors and len(rows) == int(spec["expected_jobs"]),
    }
    (root / "confirm_audit.json").write_text(json.dumps(audit, ensure_ascii=True, indent=2), encoding="utf-8")
    with (root / "confirm_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["dataset", "seed", "ari"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    formal = _read(Path("result/V21/v21_formal6_full_20260811_graphfix/aggregate_summary.json"))
    formal_full = {row["dataset"]: float(row["ari_mean"]) for row in formal["per_dataset_variant"] if row["variant"] == "topology_assignment_adversarial"}
    formal_scmae = {row["dataset"]: float(row["ari_mean"]) for row in formal["per_dataset_variant"] if row["variant"] == "scmae_only"}
    per_dataset = []
    for dataset in sorted(by_dataset):
        values = by_dataset[dataset]
        ari_mean, ari_std = _mean_std([row["ari"] for row in values])
        per_dataset.append(
            {
                "dataset": dataset,
                "n_seeds": len(values),
                "ari_mean": ari_mean,
                "ari_std": ari_std,
                "nmi_mean": _mean_std([row["nmi"] for row in values])[0],
                "acc_mean": _mean_std([row["acc"] for row in values])[0],
                "formal_full_ari": formal_full.get(dataset),
                "scmae_only_ari": formal_scmae.get(dataset),
                "delta_vs_formal_full": ari_mean - formal_full[dataset] if dataset in formal_full else None,
                "delta_vs_scmae_only": ari_mean - formal_scmae[dataset] if dataset in formal_scmae else None,
            }
        )
    macro_confirm = float(mean([row["ari_mean"] for row in per_dataset])) if per_dataset else None
    macro_formal_full = float(mean(formal_full.values())) if formal_full else None
    macro_scmae = float(mean(formal_scmae.values())) if formal_scmae else None
    payload = audit | {
        "evidence_level": "ari_selected_development_confirmation",
        "selection_metric": "macro_mean_over_six_dataset_ari",
        "params": spec["params"],
        "per_dataset": per_dataset,
        "macro_confirm_ari": macro_confirm,
        "macro_formal_full_ari": macro_formal_full,
        "macro_scmae_only_ari": macro_scmae,
        "macro_delta_vs_formal_full": macro_confirm - macro_formal_full if macro_confirm is not None and macro_formal_full is not None else None,
        "macro_delta_vs_scmae_only": macro_confirm - macro_scmae if macro_confirm is not None and macro_scmae is not None else None,
    }
    (root / "confirm_summary.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0 if audit["audit_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
