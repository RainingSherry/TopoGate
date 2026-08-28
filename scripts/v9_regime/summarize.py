#!/usr/bin/env python3
from __future__ import annotations

"""Aggregate V9 paired runs and evaluate the predeclared regime test."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.v9_regime.protocol import write_csv, write_json

FEATURE_COLUMNS = (
    "n", "d", "zero_fraction", "analysis_pca_dim_ratio", "cv_knn_distance",
    "mean_mutual_ratio", "mean_snn", "graph_components",
    "graph_largest_component_fraction", "reliability_entropy",
    "effective_neighbor_count", "neighbor_perturbation_norm",
)


def _records(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_root.rglob("run_record.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row["run_record_path"] = str(path)
        metrics = row.get("metrics") or {}
        for key, value in metrics.items():
            row[f"metric_{key}"] = value
        rows.append(row)
    return rows


def _bootstrap(values: np.ndarray, seed: int = 20260806, rounds: int = 10000) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(rounds, values.size), replace=True)
    means = sample.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _wilcoxon(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    result: dict[str, Any] = {"statistic": None, "p_value": None}
    if values.size < 2 or not np.any(values != 0):
        return result
    try:
        statistic, p_value = wilcoxon(values)
    except ValueError:
        return result
    return {"statistic": float(statistic), "p_value": float(p_value)}


def _paired(
    rows: list[dict[str, Any]],
    reference_variant: str = "nomix",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(row.get("dataset_id"), row.get("seed"), row.get("variant")): row for row in rows if row.get("status") == "completed"}
    deltas = []
    for (dataset_id, seed, variant), full in sorted(by_key.items(), key=str):
        if variant != "full":
            continue
        reference = by_key.get((dataset_id, seed, reference_variant))
        if reference is None:
            continue
        delta = {
            "dataset_id": dataset_id,
            "seed": int(seed),
            "reference_variant": reference_variant,
            "delta_ari": float(full.get("metric_ari", float("nan")) - reference.get("metric_ari", float("nan"))),
            "delta_nmi": float(full.get("metric_nmi", float("nan")) - reference.get("metric_nmi", float("nan"))),
            "full_ari": float(full.get("metric_ari", float("nan"))),
            f"{reference_variant}_ari": float(reference.get("metric_ari", float("nan"))),
        }
        deltas.append(delta)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deltas:
        grouped[str(row["dataset_id"])].append(row)
    dataset_rows = []
    for dataset_id, group in sorted(grouped.items()):
        values = np.asarray([row["delta_ari"] for row in group], dtype=float)
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "reference_variant": reference_variant,
                "n_seeds": int(values.size),
                "mean_delta_ari": float(np.mean(values)),
                "median_delta_ari": float(np.median(values)),
                "std_delta_ari": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "mean_delta_nmi": float(np.mean([row["delta_nmi"] for row in group])),
                "all_seeds_positive": bool(np.all(values > 0.0)),
                "positive_seed_count": int(np.sum(values > 0.0)),
                "seed_sign_consistency": float(np.mean(values > 0.0)),
                "confirmed_positive_candidate": bool(values.size >= 5 and np.all(values > 0.0) and np.mean(values) >= 0.03),
            }
        )
    return deltas, dataset_rows


def _control_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair Full with mechanism controls without selecting a winner."""
    by_key = {
        (row.get("dataset_id"), row.get("seed"), row.get("variant")): row
        for row in rows
        if row.get("status") == "completed"
    }
    controls = []
    for (dataset_id, seed, variant), full in sorted(by_key.items(), key=str):
        if variant != "full":
            continue
        for control in ("static", "random", "far"):
            other = by_key.get((dataset_id, seed, control))
            if other is None:
                continue
            controls.append(
                {
                    "dataset_id": dataset_id,
                    "seed": int(seed),
                    "control": control,
                    "full_ari": float(full.get("metric_ari", float("nan"))),
                    "control_ari": float(other.get("metric_ari", float("nan"))),
                    "control_minus_full_ari": float(other.get("metric_ari", float("nan")) - full.get("metric_ari", float("nan"))),
                    "full_nmi": float(full.get("metric_nmi", float("nan"))),
                    "control_nmi": float(other.get("metric_nmi", float("nan"))),
                }
            )
    return controls


def _split_statistics(dataset_rows: list[dict[str, Any]], split_json: Path | None) -> dict[str, Any]:
    if split_json is None:
        return {}
    assignments = json.loads(split_json.read_text(encoding="utf-8")).get("assignments", [])
    split_by_id = {row.get("dataset_id"): row.get("split") for row in assignments}
    result: dict[str, Any] = {}
    for split_name in ("discovery", "confirmation"):
        values = np.asarray(
            [row["mean_delta_ari"] for row in dataset_rows if split_by_id.get(row["dataset_id"]) == split_name],
            dtype=float,
        )
        ci_low, ci_high = _bootstrap(values)
        result[split_name] = {
            "n_datasets": int(values.size),
            "mean_delta_ari": float(np.mean(values)) if values.size else None,
            "median_delta_ari": float(np.median(values)) if values.size else None,
            "dataset_bootstrap_95ci": [ci_low, ci_high],
            "wilcoxon": _wilcoxon(values),
            "no_go_if_ci_crosses_zero": bool(ci_low is not None and ci_high is not None and ci_low <= 0.0 <= ci_high) if values.size else True,
        }
    return result


def _regime_predictor(features_csv: Path | None, split_json: Path | None, dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if features_csv is None or split_json is None:
        return {"status": "not_run", "reason": "features_and_split_required"}
    feature_rows = {row["dataset_id"]: row for row in csv.DictReader(features_csv.open(newline="", encoding="utf-8"))}
    assignments = json.loads(split_json.read_text(encoding="utf-8")).get("assignments", [])
    split_by_id = {row["dataset_id"]: row.get("split") for row in assignments}
    joined = [
        (row, feature_rows.get(row["dataset_id"], {}), split_by_id.get(row["dataset_id"]))
        for row in dataset_rows
        if row["dataset_id"] in feature_rows and split_by_id.get(row["dataset_id"]) in {"discovery", "confirmation"}
    ]
    if len(joined) < 8:
        return {"status": "not_run", "reason": "too_few_joined_datasets", "n": len(joined)}
    discovery = [item for item in joined if item[2] == "discovery"]
    confirmation = [item for item in joined if item[2] == "confirmation"]
    if len(discovery) < 6 or len(confirmation) < 2:
        return {"status": "not_run", "reason": "insufficient_discovery_or_confirmation", "n_discovery": len(discovery), "n_confirmation": len(confirmation)}

    def matrix(items: list[tuple[dict, dict, str]]) -> np.ndarray:
        return np.asarray([[float(item[1].get(key, "nan") or "nan") for key in FEATURE_COLUMNS] for item in items], dtype=float)

    x_train = matrix(discovery)
    x_test = matrix(confirmation)
    y_train_delta = np.asarray([item[0]["mean_delta_ari"] for item in discovery], dtype=float)
    y_test_delta = np.asarray([item[0]["mean_delta_ari"] for item in confirmation], dtype=float)
    y_train = (y_train_delta >= 0.03).astype(int)
    y_test = (y_test_delta >= 0.03).astype(int)
    if np.unique(y_train).size < 2:
        return {"status": "not_run", "reason": "discovery_has_one_class", "n_discovery": len(discovery), "n_confirmation": len(confirmation)}
    classifier = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=20260806))
    classifier.fit(x_train, y_train)
    proba = classifier.predict_proba(x_test)[:, 1]
    auc = None if np.unique(y_test).size < 2 else float(roc_auc_score(y_test, proba))
    regressor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    regressor.fit(x_train, y_train_delta)
    predictions = regressor.predict(x_test)
    return {
        "status": "completed",
        "n_discovery": len(discovery),
        "n_confirmation": len(confirmation),
        "feature_columns": list(FEATURE_COLUMNS),
        "confirmation_auc": auc,
        "confirmation_predicted_delta": predictions.tolist(),
        "confirmation_observed_delta": y_test_delta.tolist(),
        "auc_passes_0p65": bool(auc is not None and auc > 0.65),
    }


def summarize(
    run_root: Path,
    output_dir: Path,
    features_csv: Path | None = None,
    split_json: Path | None = None,
    reference_variant: str = "nomix",
) -> dict[str, Any]:
    rows = _records(run_root)
    deltas, dataset_rows = _paired(rows, reference_variant=reference_variant)
    controls = _control_deltas(rows)
    write_csv(output_dir / "runs.csv", rows)
    write_csv(output_dir / "paired_deltas.csv", deltas)
    write_csv(output_dir / "dataset_summary.csv", dataset_rows)
    write_csv(output_dir / "control_deltas.csv", controls)
    values = np.asarray([row["delta_ari"] for row in deltas], dtype=float)
    dataset_values = np.asarray([row["mean_delta_ari"] for row in dataset_rows], dtype=float)
    wilcoxon_result = _wilcoxon(dataset_values)
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[str(row.get("status", "unknown"))] += 1
    variant_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("status") == "completed":
            variant_counts[str(row.get("variant", "unknown"))] += 1
    ci_low, ci_high = _bootstrap(dataset_values)
    summary = {
        "run_root": str(run_root),
        "total_run_records": len(rows),
        "completed_runs": int(status_counts.get("completed", 0)),
        "error_runs": int(status_counts.get("error", 0)),
        "running_runs": int(status_counts.get("running", 0)),
        "status_counts": dict(status_counts),
        "completed_variant_counts": dict(variant_counts),
        "paired_seed_rows": len(deltas),
        "paired_dataset_rows": len(dataset_rows),
        "paired_reference_variant": reference_variant,
        "mean_delta_ari_over_seed_pairs": float(np.mean(values)) if values.size else None,
        "mean_delta_ari_over_datasets": float(np.mean(dataset_values)) if dataset_values.size else None,
        "median_delta_ari_over_datasets": float(np.median(dataset_values)) if dataset_values.size else None,
        "dataset_bootstrap_95ci": [ci_low, ci_high],
        "positive_dataset_count": int(np.sum(dataset_values > 0)) if dataset_values.size else 0,
        "confirmed_positive_candidates": [row["dataset_id"] for row in dataset_rows if row["confirmed_positive_candidate"]],
        "wilcoxon": wilcoxon_result,
        "no_go_if_ci_crosses_zero": bool(ci_low is not None and ci_high is not None and ci_low <= 0.0 <= ci_high) if dataset_values.size else True,
        "split_statistics": _split_statistics(dataset_rows, split_json),
        "control_pair_rows": len(controls),
        "regime_predictor": _regime_predictor(features_csv, split_json, dataset_rows),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "regime_predictor.json", summary["regime_predictor"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--reference-variant", default="nomix")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(
        args.runs,
        args.output_dir,
        args.features,
        args.split,
        reference_variant=args.reference_variant,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
