#!/usr/bin/env python3
"""Audit V15 Stage-1 mechanism diagnostics from saved run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_metric(fn, *args) -> float | None:
    try:
        value = float(fn(*args))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _run_record(path: Path) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    with np.load(path / "gate_diagnostics.npz") as diagnostics:
        target = diagnostics["utility_target"].astype(np.float64)
        predicted = diagnostics["utility_hat"].astype(np.float64)
        valid = diagnostics["utility_valid"].astype(bool)
        pi = diagnostics["predicted_pi"].astype(np.float64)
    target_flat = target[valid]
    predicted_flat = predicted[valid]
    binary = target_flat > 0.0
    auc = _safe_metric(roc_auc_score, binary.astype(np.int8), predicted_flat) if np.unique(binary).size == 2 else None
    auprc = _safe_metric(average_precision_score, binary.astype(np.int8), predicted_flat) if np.unique(binary).size == 2 else None
    try:
        rank_value = spearmanr(target_flat, predicted_flat).statistic
        rank = float(rank_value) if np.isfinite(rank_value) else None
    except (TypeError, ValueError):
        rank = None
    probabilities = np.load(path / "cluster_probabilities.npy")
    predictions = np.load(path / "predictions.npy")
    counts = np.bincount(predictions, minlength=probabilities.shape[1]).astype(np.float64)
    counts /= max(1, counts.sum())
    entropy = float(-(counts[counts > 0] * np.log(counts[counts > 0])).sum())
    row_null = pi[:, 0]
    edge_mass = pi[:, 1:].sum(axis=1)
    effective = np.exp(-(pi[:, 1:] * np.log(np.clip(pi[:, 1:], 1e-8, None))).sum(axis=1))
    record: dict[str, Any] = {
        "run": str(path),
        "dataset": summary.get("dataset"),
        "gate_mode": summary.get("config", {}).get("gate_mode", "counterfactual"),
        "seed": summary.get("seed"),
        "utility_auroc": auc,
        "utility_auprc": auprc,
        "utility_spearman": rank,
        "utility_positive_rate": float(np.mean(binary)) if binary.size else None,
        "null_mass": float(np.mean(row_null)),
        "edge_mass": float(np.mean(edge_mass)),
        "effective_neighbors": float(np.mean(effective)),
        "predicted_cluster_count": int(np.unique(predictions).size),
        "max_cluster_fraction": float(np.max(counts)) if counts.size else 0.0,
        "cluster_entropy": entropy,
        "candidate_recall": summary.get("graph_profile", {}).get("posthoc_candidate_recall"),
        "edge_purity": summary.get("graph_profile", {}).get("posthoc_edge_purity"),
        "labels_used_during_fit": summary.get("labels_used_during_fit"),
    }
    masks_path = path / "mechanism_masks.npz"
    if masks_path.exists():
        with np.load(masks_path) as masks:
            for name in masks.files:
                mask = masks[name].astype(bool)
                if mask.shape == row_null.shape and np.unique(mask).size == 2:
                    record[f"null_auc_{name}"] = _safe_metric(roc_auc_score, mask.astype(np.int8), row_null)
    corruption_mask_path = path / "corruption_mask.npy"
    if corruption_mask_path.exists():
        mask = np.asarray(np.load(corruption_mask_path)).reshape(-1).astype(bool)
        if mask.shape == row_null.shape and np.unique(mask).size == 2:
            record["null_auc_corruption"] = _safe_metric(roc_auc_score, mask.astype(np.int8), row_null)
    return record


def _pollution_trend(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[float, list[float]] = {}
    for record in records:
        summary = json.loads((Path(record["run"]) / "summary.json").read_text(encoding="utf-8"))
        fraction = float(summary.get("config", {}).get("graph_replacement_fraction", 0.0))
        grouped.setdefault(fraction, []).append(float(record["null_mass"]))
    means = {str(key): float(np.mean(value)) for key, value in sorted(grouped.items())}
    fractions = sorted(grouped)
    endpoint_trend = None
    monotonic = None
    if len(fractions) >= 2:
        ordered = [means[str(value)] for value in fractions]
        endpoint_trend = bool(ordered[-1] >= ordered[0])
        monotonic = bool(all(current + 1e-8 >= previous for previous, current in zip(ordered, ordered[1:])))
    return {
        "null_mass_by_graph_replacement_fraction": means,
        "endpoint_non_decreasing": endpoint_trend,
        "monotonic_non_decreasing": monotonic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = sorted({path.parent for path in args.run_root.rglob("summary.json")})
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in run_dirs:
        try:
            records.append(_run_record(path))
        except Exception as exc:
            errors.append({"run": str(path), "error": f"{type(exc).__name__}: {exc}"})
    payload = {
        "schema_version": "V15-stage1-1",
        "records": records,
        "errors": errors,
        "pollution_audit": _pollution_trend(records),
        "utility_target_contract": {
            "detached_artifact": True,
            "gate_input_present_in_target": False,
            "source": "operator_aligned_utility(q_teacher_clean,q_self_masked,q_edge_masked,rec_self,rec_edge,valid)",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"runs": len(records), "errors": len(errors), "output": str(args.output)}))


if __name__ == "__main__":
    main()
