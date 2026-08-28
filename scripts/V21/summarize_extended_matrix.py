#!/usr/bin/env python3
"""Strictly audit and summarize the V21 readout-fix extension matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from run_extended_matrix import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    MODEL_PROTOCOL_ID,
    VARIANT_CONFIGS,
    build_jobs,
    load_manifest,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(mean(values)), float(pstdev(values)) if len(values) > 1 else 0.0


def _audit_job(job: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    output = Path(job["output"])
    errors: list[str] = []
    required = [
        "summary.json",
        "metrics.json",
        "resolved_config.json",
        "preprocess_profile.json",
        "readout_profile.json",
        "training_history.json",
        "predictions.npy",
        "labels_true.npy",
        "embedding_final.npy",
        "run_record.json",
    ]
    if job["variant"] == "topology_assignment_adversarial":
        required += ["cluster_probabilities.npy", "student_t_predictions.npy", "graph_profile.json", "stats_profile.json"]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        return None, [f"{job['run_key']}: missing {missing}"]
    try:
        summary = _read(output / "summary.json")
        metrics = _read(output / "metrics.json")
        config = _read(output / "resolved_config.json")
        preprocess = _read(output / "preprocess_profile.json")
        readout = _read(output / "readout_profile.json")
        run_record = _read(output / "run_record.json")
        predictions = np.load(output / "predictions.npy", allow_pickle=False)
        labels = np.load(output / "labels_true.npy", allow_pickle=False)
        embedding = np.load(output / "embedding_final.npy", allow_pickle=False)
    except Exception as exc:
        return None, [f"{job['run_key']}: unreadable artifact: {type(exc).__name__}: {exc}"]

    expected_profile = job["record"].get("profile", {})
    checks = {
        "summary status": summary.get("status") == "completed",
        "model protocol": summary.get("protocol_id") == MODEL_PROTOCOL_ID,
        "experiment protocol": summary.get("experiment_protocol_id") == run_record.get("experiment_protocol_id"),
        "run key": summary.get("run_key") == job["run_key"] == run_record.get("run_key"),
        "dataset": summary.get("dataset") == job["record"]["name"],
        "dataset id": summary.get("dataset_id") == job["record"]["dataset_id"],
        "variant": summary.get("variant") == job["variant"],
        "seed": int(summary.get("seed", -1)) == int(job["seed"]),
        "labels isolated": summary.get("labels_used_during_fit") is False,
        "preprocess labels isolated": preprocess.get("labels_used_during_fit") is False,
        "extension selection isolated": summary.get("extension_labels_used_for_selection") is False,
        "benchmark K": summary.get("K_source") == "benchmark_oracle_from_y",
        "readout config": config.get("readout_mode") == "kmeans_embedding",
        "readout semantics": summary.get("prediction_semantics") == "kmeans_embedding_known_k",
        "readout labels isolated": readout.get("labels_used_for_readout") is False,
        "readout effective": readout.get("effective_mode") == "kmeans_embedding",
        "no empty primary clusters": int(readout.get("primary", {}).get("empty_clusters", -1)) == 0,
        "prediction shape": predictions.shape == labels.shape == (embedding.shape[0],),
        "embedding finite": embedding.ndim == 2 and bool(np.isfinite(embedding).all()),
        "predictions finite": bool(np.isfinite(predictions).all()),
        "labels finite": bool(np.isfinite(labels).all()),
        "sample count": not expected_profile or embedding.shape[0] == int(expected_profile.get("n_samples", embedding.shape[0])),
        "feature count": not expected_profile or int(summary.get("n_features", -1)) == int(expected_profile.get("n_features", summary.get("n_features", -1))),
        "run record completed": run_record.get("status") == "completed",
    }
    for label, passed in checks.items():
        if not passed:
            errors.append(f"{job['run_key']}: {label} failed")
    for key in ("ari", "nmi", "acc"):
        try:
            if not np.isfinite(float(metrics[key])):
                errors.append(f"{job['run_key']}: non-finite {key}")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{job['run_key']}: missing {key}")
    if errors:
        return None, errors
    head_metrics = metrics.get("student_t_training_head", {})
    return (
        {
            "run_key": job["run_key"],
            "dataset_id": job["record"]["dataset_id"],
            "dataset": job["record"]["name"],
            "family": job["record"].get("family"),
            "input_protocol": job["record"]["input_protocol"],
            "variant": job["variant"],
            "seed": int(job["seed"]),
            "ari": float(metrics["ari"]),
            "nmi": float(metrics["nmi"]),
            "acc": float(metrics["acc"]),
            "primary_empty_clusters": int(readout["primary"]["empty_clusters"]),
            "student_t_ari": float(head_metrics["ari"]) if "ari" in head_metrics else None,
            "student_t_empty_clusters": readout.get("student_t_training_head", {}).get("empty_clusters"),
        },
        [],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    variants = tuple(str(value) for value in manifest["variants"])
    seeds = tuple(int(value) for value in (args.seeds if args.seeds else manifest["seeds"]))
    selected_datasets = set(args.datasets or [])
    jobs = build_jobs(manifest, variants, seeds, selected_datasets, args.output_dir)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for job in jobs:
        row, job_errors = _audit_job(job)
        errors.extend(job_errors)
        if row is not None:
            rows.append(row)
    audit = {
        "manifest_id": manifest["manifest_id"],
        "experiment_protocol_id": manifest["protocol_id"],
        "model_protocol_id": MODEL_PROTOCOL_ID,
        "expected_jobs": len(jobs),
        "dataset_scope": sorted(selected_datasets) if selected_datasets else "all_manifest_datasets",
        "seed_scope": list(seeds),
        "completed_valid_jobs": len(rows),
        "incomplete_jobs": len(jobs) - len(rows),
        "errors": errors,
        "labels_used_during_fit": False,
        "extension_labels_used_for_selection": False,
        "audit_ok": not errors and len(rows) == len(jobs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "extended_audit.json", audit)
    if errors and not args.allow_incomplete:
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        return 2

    with (args.output_dir / "extended_run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["run_key", "dataset_id", "variant", "seed", "ari"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset_id"]), str(row["variant"]))].append(row)
    per_variant = []
    for (dataset_id, variant), values in sorted(grouped.items()):
        ari_mean, ari_std = _mean_std([float(row["ari"]) for row in values])
        nmi_mean, nmi_std = _mean_std([float(row["nmi"]) for row in values])
        per_variant.append(
            {
                "dataset_id": dataset_id,
                "dataset": values[0]["dataset"],
                "variant": variant,
                "n_runs": len(values),
                "ari_mean": ari_mean,
                "ari_std": ari_std,
                "nmi_mean": nmi_mean,
                "nmi_std": nmi_std,
            }
        )
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[(str(row["dataset_id"]), int(row["seed"]))][str(row["variant"])] = row
    paired = []
    for (dataset_id, seed), pair in sorted(by_pair.items()):
        full = pair.get("topology_assignment_adversarial")
        control = pair.get("scmae_only")
        if full is not None and control is not None:
            paired.append(
                {
                    "dataset_id": dataset_id,
                    "dataset": full["dataset"],
                    "seed": seed,
                    "full_ari": full["ari"],
                    "scmae_only_ari": control["ari"],
                    "delta_ari": float(full["ari"]) - float(control["ari"]),
                }
            )
    dataset_deltas = []
    for dataset_id in sorted({str(row["dataset_id"]) for row in paired}):
        values = [float(row["delta_ari"]) for row in paired if row["dataset_id"] == dataset_id]
        delta_mean, delta_std = _mean_std(values)
        dataset_deltas.append(
            {"dataset_id": dataset_id, "n_seeds": len(values), "delta_ari_mean": delta_mean, "delta_ari_std": delta_std}
        )
    aggregate = audit | {
        "per_dataset_variant": per_variant,
        "paired_seed_rows": paired,
        "per_dataset_delta": dataset_deltas,
        "macro_dataset_delta_ari": float(mean(row["delta_ari_mean"] for row in dataset_deltas)) if dataset_deltas else None,
    }
    _write_json(args.output_dir / "extended_summary.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=True, indent=2))
    return 0 if audit["audit_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
