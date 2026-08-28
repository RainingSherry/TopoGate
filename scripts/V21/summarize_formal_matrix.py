#!/usr/bin/env python3
"""Audit and summarize completed V21 formal matrix runs."""

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
    "checkpoint.pt",
)
MODEL_PROTOCOL_ID = "v21_assignment_adversarial_v2_graphfix_v1"
EXPECTED_EPOCHS = 80


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_job(job: dict[str, Any], model_protocol_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    output = Path(job["output"])
    missing = [name for name in REQUIRED if not (output / name).is_file()]
    if job["variant"] != "scmae_only" and not (output / "cluster_probabilities.npy").is_file():
        missing.append("cluster_probabilities.npy")
    if missing:
        return None, [f"{job['key']}: missing {','.join(missing)}"]
    try:
        summary = _read(output / "summary.json")
        metrics = _read(output / "metrics.json")
        resolved = _read(output / "resolved_config.json")
        preprocess = _read(output / "preprocess_profile.json")
        graph = _read(output / "graph_profile.json")
        stats = _read(output / "stats_profile.json")
        history = _read(output / "training_history.json")
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{job['key']}: invalid JSON {exc}"]
    errors: list[str] = []
    uses_head = job["variant"] != "scmae_only"
    uses_gate = job["variant"] == "topology_assignment_adversarial"
    checks = {
        "status": summary.get("status") == "completed",
        "model_protocol_id": summary.get("protocol_id") == model_protocol_id,
        "variant": summary.get("variant") == job["variant"],
        "dataset": summary.get("dataset") == job["dataset"],
        "seed": int(summary.get("seed", -1)) == int(job["seed"]),
        "labels_used_during_fit": summary.get("labels_used_during_fit") is False,
        "metrics_labels_used_during_fit": metrics.get("labels_used_during_fit") is False,
        "K_source": metrics.get("K_source") in {"benchmark_oracle_from_y", "explicit_n_clusters"},
        "resolved_protocol_id": resolved.get("protocol_id") == model_protocol_id,
        "resolved_variant": resolved.get("variant") == job["variant"],
        "resolved_dataset": resolved.get("dataset") == job["dataset"],
        "resolved_seed": int(resolved.get("seed", -1)) == int(job["seed"]),
        "resolved_epochs": int(resolved.get("epochs", -1)) == EXPECTED_EPOCHS,
        "preprocess_labels_used": preprocess.get("labels_used") is False
        and preprocess.get("labels_used_during_fit") is False,
        "preprocess_K_not_used": preprocess.get("K_used") is False,
        "preprocess_K_fit_contract": preprocess.get("K_used_during_fit") is uses_head,
        "summary_K_fit_contract": summary.get("K_used_during_fit") is uses_head,
        "metrics_K_fit_contract": metrics.get("K_used_during_fit") is uses_head,
        "graph_contract": graph.get("enabled") is uses_gate,
        "graph_no_label_leakage": graph.get("label_leakage_diagnostic", False) is False,
        "graph_no_self_edges": (not uses_gate) or int(graph.get("self_edges", -1)) == 0,
        "labels_available": metrics.get("labels_available") is True,
    }
    errors.extend(f"{job['key']}: failed {name}" for name, passed in checks.items() if not passed)
    if errors:
        return None, errors
    if not isinstance(history, list) or len(history) != EXPECTED_EPOCHS:
        return None, [f"{job['key']}: history length is not {EXPECTED_EPOCHS}"]
    numeric_history = [
        value
        for record in history
        if isinstance(record, dict)
        for value in record.values()
        if isinstance(value, (int, float))
    ]
    if not numeric_history or not all(math.isfinite(float(value)) for value in numeric_history):
        return None, [f"{job['key']}: non-finite value in training history"]
    last = history[-1]
    finite_values = [metrics.get("ari"), metrics.get("nmi"), metrics.get("acc")]
    finite_values.extend(last.get(name) for name in ("loss", "assignment_divergence", "gate_divergence"))
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in finite_values):
        return None, [f"{job['key']}: non-finite metric or final history value"]
    try:
        embedding = np.load(output / "embedding_final.npy", mmap_mode="r")
        predictions = np.load(output / "predictions.npy", mmap_mode="r")
        labels_true = np.load(output / "labels_true.npy", mmap_mode="r")
        selected_features = np.load(output / "selected_feature_indices.npy", mmap_mode="r")
        probabilities = (
            np.load(output / "cluster_probabilities.npy", mmap_mode="r") if uses_head else None
        )
    except (OSError, ValueError) as exc:
        return None, [f"{job['key']}: invalid numpy artifact {exc}"]
    n_samples = int(summary.get("n_samples", -1))
    n_features = int(summary.get("n_features", -1))
    hidden_size = int(resolved.get("hidden_size", -1))
    array_checks = {
        "embedding_shape": embedding.shape == (n_samples, hidden_size),
        "predictions_shape": predictions.shape == (n_samples,),
        "labels_true_shape": labels_true.shape == (n_samples,),
        "selected_feature_shape": selected_features.shape == (n_features,),
        "embedding_finite": bool(np.isfinite(embedding).all()),
        "predictions_finite": bool(np.isfinite(predictions).all()),
        "labels_true_finite": bool(np.isfinite(labels_true).all()),
        "selected_feature_finite": bool(np.isfinite(selected_features).all()),
    }
    if uses_head:
        expected_clusters = int(metrics.get("n_clusters", -1))
        array_checks.update(
            {
                "cluster_probabilities_shape": probabilities is not None
                and probabilities.shape == (n_samples, expected_clusters),
                "cluster_probabilities_finite": probabilities is not None
                and bool(np.isfinite(probabilities).all()),
            }
        )
    errors.extend(f"{job['key']}: failed {name}" for name, passed in array_checks.items() if not passed)
    if errors:
        return None, errors
    row = {
        "key": job["key"],
        "dataset": job["dataset"],
        "input_protocol": job["input_protocol"],
        "variant": job["variant"],
        "seed": int(job["seed"]),
        "ari": float(metrics["ari"]),
        "nmi": float(metrics["nmi"]),
        "acc": float(metrics["acc"]),
        "gate_updates": int(summary.get("diagnostics", {}).get("gate_updates", 0)),
        "final_loss": float(last.get("loss", float("nan"))),
        "final_assignment_divergence": float(last.get("assignment_divergence", 0.0)),
        "final_gate_divergence": float(last.get("gate_divergence", 0.0)),
        "final_assignment_eligible_rate": float(last.get("assignment_eligible_rate", 0.0)),
        "final_assignment_effective_rate": float(last.get("assignment_effective_rate", 0.0)),
    }
    return row, []


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(mean(values)), float(stdev(values)) if len(values) > 1 else 0.0


def _provenance_errors(root: Path, matrix_protocol_id: str) -> list[str]:
    path = root / "provenance.json"
    if not path.is_file():
        return ["missing provenance.json"]
    try:
        provenance = _read(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid provenance.json: {exc}"]
    errors: list[str] = []
    if provenance.get("matrix_protocol_id") != matrix_protocol_id:
        errors.append("provenance matrix protocol mismatch")
    if provenance.get("model_protocol_id") != MODEL_PROTOCOL_ID:
        errors.append("provenance model protocol mismatch")
    for category, expected_count in (("source_files", 8), ("config_files", 2), ("data_files", 6)):
        records = provenance.get(category)
        if not isinstance(records, list) or len(records) != expected_count:
            errors.append(f"provenance {category} count mismatch")
            continue
        for record in records:
            candidate = Path(str(record.get("path", "")))
            digest = str(record.get("sha256", ""))
            if not candidate.is_file() or len(digest) != 64:
                errors.append(f"invalid provenance record in {category}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and summarize a V21 formal matrix")
    parser.add_argument("--output-dir", type=Path, default=Path("result/V21/v21_formal6_full_20260811_graphfix"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.output_dir
    spec = _read(root / "stage_spec.json")
    jobs = _read(root / "launcher_state.json")["jobs"] if (root / "launcher_state.json").is_file() else []
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    errors.extend(_provenance_errors(root, spec["protocol_id"]))
    for job in jobs:
        row, job_errors = _audit_job(job, MODEL_PROTOCOL_ID)
        errors.extend(job_errors)
        if row is not None:
            rows.append(row)
    audit = {
        "matrix_protocol_id": spec["protocol_id"],
        "model_protocol_id": MODEL_PROTOCOL_ID,
        "expected_jobs": int(spec["expected_jobs"]),
        "audited_jobs": len(jobs),
        "completed_valid_jobs": len(rows),
        "provenance_ok": not any(error.startswith(("missing provenance", "invalid provenance", "provenance ")) for error in errors),
        "errors": errors,
        "audit_ok": not errors and len(rows) == int(spec["expected_jobs"]),
    }
    (root / "matrix_audit.json").write_text(json.dumps(audit, ensure_ascii=True, indent=2), encoding="utf-8")
    if errors and not args.allow_incomplete:
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        return 2

    with (root / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["key", "dataset", "variant", "seed", "ari", "nmi", "acc"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_dataset_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset_variant[(row["dataset"], row["variant"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for (dataset, variant), values in sorted(by_dataset_variant.items()):
        ari_mean, ari_std = _mean_std([v["ari"] for v in values])
        nmi_mean, nmi_std = _mean_std([v["nmi"] for v in values])
        aggregate.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n_runs": len(values),
                "ari_mean": ari_mean,
                "ari_std": ari_std,
                "nmi_mean": nmi_mean,
                "nmi_std": nmi_std,
            }
        )

    paired: list[dict[str, Any]] = []
    by_dataset_seed: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_dataset_seed[(row["dataset"], row["seed"])][row["variant"]] = row
    for (dataset, seed), variants in sorted(by_dataset_seed.items()):
        full = variants.get("topology_assignment_adversarial")
        control = variants.get("scmae_only")
        if full is not None and control is not None:
            paired.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "full_ari": full["ari"],
                    "scmae_only_ari": control["ari"],
                    "delta_ari": full["ari"] - control["ari"],
                    "full_nmi": full["nmi"],
                    "scmae_only_nmi": control["nmi"],
                    "delta_nmi": full["nmi"] - control["nmi"],
                }
            )
    with (root / "paired_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(paired[0]) if paired else ["dataset", "seed", "delta_ari"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)

    dataset_delta: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in paired}):
        values = [row["delta_ari"] for row in paired if row["dataset"] == dataset]
        delta_mean, delta_std = _mean_std(values)
        dataset_delta.append({"dataset": dataset, "delta_ari_mean": delta_mean, "delta_ari_std": delta_std, "n_seeds": len(values)})
    aggregate_payload = audit | {
        "per_dataset_variant": aggregate,
        "paired_seed_rows": paired,
        "per_dataset_delta": dataset_delta,
        "macro_dataset_delta_ari": float(mean([row["delta_ari_mean"] for row in dataset_delta])) if dataset_delta else None,
    }
    (root / "aggregate_summary.json").write_text(
        json.dumps(aggregate_payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate_payload, ensure_ascii=True, indent=2))
    return 0 if audit["audit_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
