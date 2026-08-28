#!/usr/bin/env python3
"""Audit the four independent V15 Stage-1B bottleneck certificates.

This script is deliberately read-only with respect to training artifacts.  It
uses benchmark labels only for post-hoc graph diagnostics; labels are never
used to fit a scorer, select a graph, or choose a variant.  Missing artifacts
are reported as ``not_available`` instead of being inferred from a proxy.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _metric(fn: Any, y_true: np.ndarray, score: np.ndarray) -> float | None:
    try:
        return _finite(float(fn(y_true, score)))
    except (TypeError, ValueError):
        return None


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path, name: str) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive[name]) if name in archive.files else None


def _optional_array(path: Path, names: Iterable[str]) -> tuple[np.ndarray | None, str | None]:
    for name in names:
        candidate = path / name
        if candidate.exists():
            return np.asarray(np.load(candidate, allow_pickle=False)), name
    return None, None


def _cluster_usage(predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    if predictions.size == 0:
        return {"predicted_cluster_count": 0, "max_cluster_fraction": None, "entropy": None}
    counts = np.bincount(predictions.astype(np.int64), minlength=probabilities.shape[1]).astype(np.float64)
    fractions = counts / max(1.0, counts.sum())
    nonzero = fractions > 0
    entropy = float(-(fractions[nonzero] * np.log(fractions[nonzero])).sum())
    return {
        "predicted_cluster_count": int(np.count_nonzero(nonzero)),
        "max_cluster_fraction": _finite(float(fractions.max())) if fractions.size else None,
        "entropy": entropy,
    }


def _partition_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    return {
        "ari": float(adjusted_rand_score(labels, predictions)),
        "nmi": float(normalized_mutual_info_score(labels, predictions)),
        "ami": float(adjusted_mutual_info_score(labels, predictions)),
    }


def _hungarian_mapping(predictions: np.ndarray, labels: np.ndarray) -> tuple[dict[int, int], int]:
    predictions = np.asarray(predictions).reshape(-1).astype(np.int64)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    predicted_values = np.unique(predictions)
    true_values = np.unique(labels)
    contingency = np.zeros((predicted_values.size, true_values.size), dtype=np.int64)
    for row, predicted_value in enumerate(predicted_values):
        for col, true_value in enumerate(true_values):
            contingency[row, col] = int(np.sum((predictions == predicted_value) & (labels == true_value)))
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {int(predicted_values[row]): int(true_values[col]) for row, col in zip(row_ind, col_ind)}
    fallback = int(np.bincount(labels).argmax())
    return mapping, fallback


def _teacher_certificate(
    run: Path,
    summary: dict[str, Any],
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return only teacher checks supported by persisted artifacts."""
    q_teacher, q_name = _optional_array(
        run,
        (
            "teacher_cluster_probabilities.npy",
            "teacher_probabilities.npy",
            "teacher_probabilities_clean.npy",
            "assignments_teacher.npy",
        ),
    )
    z_teacher, z_name = _optional_array(
        run,
        ("embedding_teacher.npy", "teacher_embedding.npy", "teacher_embeddings.npy"),
    )
    certificate: dict[str, Any] = {
        "status": "partial" if q_teacher is not None or z_teacher is not None else "not_available",
        "assignment_artifact": q_name,
        "embedding_artifact": z_name,
        "checks": {},
        "missing": [],
        "label_isolation": summary.get("labels_used_during_fit") is False,
    }
    if q_teacher is not None:
        q = np.asarray(q_teacher, dtype=np.float64)
        if q.ndim == 2 and q.shape[0] > 0:
            q = np.clip(q, 1e-8, None)
            q /= q.sum(axis=1, keepdims=True)
            marginal = q.mean(axis=0)
            certificate["checks"]["cluster_usage_entropy"] = float(
                -(marginal * np.log(marginal)).sum()
            )
            certificate["checks"]["teacher_confidence_mean"] = _finite(float(q.max(axis=1).mean()))
            if labels is not None and labels.shape[0] == q.shape[0]:
                certificate["posthoc_partitions"] = {
                    "label_use": "posthoc_only",
                    "ema_clean": _partition_metrics(labels, q.argmax(axis=1)),
                }
                for key, artifact in (
                    ("raw_aligned", "teacher_probabilities_raw_aligned.npy"),
                    ("reference", "teacher_probabilities_reference.npy"),
                    ("augmented", "teacher_probabilities_augmented.npy"),
                ):
                    artifact_path = run / artifact
                    if artifact_path.exists():
                        values = np.asarray(np.load(artifact_path, allow_pickle=False))
                        if values.ndim == 2 and values.shape[0] == labels.shape[0]:
                            certificate["posthoc_partitions"][key] = _partition_metrics(
                                labels,
                                values.argmax(axis=1),
                            )
        else:
            certificate["checks"]["assignment_shape_valid"] = False
    else:
        certificate["missing"].append("teacher_cluster_probabilities")
    if z_teacher is not None:
        z = np.asarray(z_teacher, dtype=np.float64)
        if z.ndim == 2 and z.shape[0] > 0:
            std = z.std(axis=0)
            certificate["checks"]["embedding_per_dimension_std_min"] = _finite(float(std.min()))
            certificate["checks"]["embedding_per_dimension_std_median"] = _finite(float(np.median(std)))
            singular = np.linalg.svd(z - z.mean(axis=0, keepdims=True), compute_uv=False)
            if singular.size and singular[0] > 0:
                certificate["checks"]["embedding_effective_rank_99"] = int(
                    np.searchsorted(np.cumsum(singular**2) / np.sum(singular**2), 0.99) + 1
                )
        else:
            certificate["checks"]["embedding_shape_valid"] = False
    else:
        certificate["missing"].append("teacher_embedding")

    # These require paired clean/augmented or temporal teacher artifacts.
    paired_names = (
        ("cross_view_assignment_jsd_mean", "teacher_probabilities_clean.npy", "teacher_probabilities_augmented.npy"),
        ("temporal_assignment_jsd_mean", "teacher_probabilities_epoch0.npy", "teacher_probabilities_epoch_last.npy"),
        (
            "raw_reference_assignment_jsd_mean",
            "teacher_probabilities_clean.npy",
            "teacher_probabilities_raw_aligned.npy",
        ),
    )
    paired_available = False
    for metric_name, left_name, right_name in paired_names:
        left, right = run / left_name, run / right_name
        if left.exists() and right.exists():
            left_q = np.clip(np.asarray(np.load(left), dtype=np.float64), 1e-8, None)
            right_q = np.clip(np.asarray(np.load(right), dtype=np.float64), 1e-8, None)
            left_q /= left_q.sum(axis=1, keepdims=True)
            right_q /= right_q.sum(axis=1, keepdims=True)
            midpoint = 0.5 * (left_q + right_q)
            jsd = 0.5 * (left_q * np.log(left_q / midpoint)).sum(axis=1)
            jsd += 0.5 * (right_q * np.log(right_q / midpoint)).sum(axis=1)
            certificate["checks"][metric_name] = _finite(float(jsd.mean()))
            certificate["checks"][metric_name.replace("jsd_mean", "partition_ari")] = float(
                adjusted_rand_score(left_q.argmax(axis=1), right_q.argmax(axis=1))
            )
            paired_available = True
    if not paired_available:
        certificate["missing"].append("cross_view_or_temporal_teacher_pair")
    shuffled_path = run / "teacher_probabilities_shuffled.npy"
    clean_path = run / "teacher_probabilities_clean.npy"
    if shuffled_path.exists() and clean_path.exists():
        clean_q = np.clip(np.asarray(np.load(clean_path), dtype=np.float64), 1e-8, None)
        shuffled_q = np.clip(np.asarray(np.load(shuffled_path), dtype=np.float64), 1e-8, None)
        clean_q /= clean_q.sum(axis=1, keepdims=True)
        shuffled_q /= shuffled_q.sum(axis=1, keepdims=True)
        midpoint = 0.5 * (clean_q + shuffled_q)
        negative_jsd = 0.5 * (clean_q * np.log(clean_q / midpoint)).sum(axis=1)
        negative_jsd += 0.5 * (shuffled_q * np.log(shuffled_q / midpoint)).sum(axis=1)
        certificate["negative_controls"] = {
            "random_or_shuffled_teacher": "available",
            "assignment_jsd_mean": _finite(float(negative_jsd.mean())),
        }
    else:
        certificate["negative_controls"] = {
            "random_or_shuffled_teacher": "not_available",
            "reason": "no negative-control teacher artifacts are persisted",
        }
    if certificate["status"] == "partial" and q_teacher is not None and z_teacher is not None and paired_available:
        certificate["status"] = "available"
    if certificate["status"] == "not_available":
        certificate["reason"] = "current V15 output contract does not save teacher assignments or embeddings"
    elif certificate["status"] == "partial":
        certificate["reason"] = "teacher artifact exists, but paired cross-view/temporal correctness check is missing"
    return certificate


def _posthoc_graph_metrics(indices: np.ndarray, valid: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    if indices.shape != valid.shape or indices.ndim != 2 or labels.shape[0] != indices.shape[0]:
        raise ValueError("candidate arrays and labels have incompatible shapes")
    rows, cols = np.where(valid)
    donors = indices[rows, cols]
    in_bounds = (donors >= 0) & (donors < labels.shape[0])
    rows, donors = rows[in_bounds], donors[in_bounds]
    if rows.size:
        same = labels[rows] == labels[donors]
        edge_purity = float(np.mean(same))
    else:
        same = np.zeros(0, dtype=bool)
        edge_purity = 0.0
    per_anchor: list[float] = []
    coverage: list[bool] = []
    for i in range(indices.shape[0]):
        chosen = indices[i, valid[i]]
        chosen = chosen[(chosen >= 0) & (chosen < labels.shape[0]) & (chosen != i)]
        same_nodes = np.flatnonzero(labels == labels[i])
        same_nodes = same_nodes[same_nodes != i]
        denominator = min(chosen.size, same_nodes.size)
        per_anchor.append(float(np.intersect1d(chosen, same_nodes).size / denominator) if denominator else 0.0)
        coverage.append(bool(np.any(labels[chosen] == labels[i])) if chosen.size else False)
    row_ids = np.broadcast_to(np.arange(indices.shape[0])[:, None], indices.shape)
    return {
        "valid_edge_count": int(rows.size),
        "candidate_width": int(indices.shape[1]),
        "edge_purity": edge_purity,
        "candidate_recall_budget_normalized": _finite(float(np.mean(per_anchor))) if per_anchor else None,
        "candidate_coverage_any_same_label": _finite(float(np.mean(coverage))) if coverage else None,
        "self_edge_count": int(np.sum(indices[valid] == row_ids[valid])) if valid.any() else 0,
        "invalid_index_count": int(np.sum((indices[valid] < 0) | (indices[valid] >= labels.shape[0]))) if valid.any() else 0,
    }


def _graph_certificate(run: Path, summary: dict[str, Any], labels: np.ndarray) -> dict[str, Any]:
    diagnostics = run / "gate_diagnostics.npz"
    indices = _load_npz(diagnostics, "candidate_indices") if diagnostics.exists() else None
    valid = _load_npz(diagnostics, "candidate_valid") if diagnostics.exists() else None
    if indices is None or valid is None:
        return {
            "status": "not_available",
            "reason": "candidate_indices/candidate_valid are not persisted",
            "label_use": "posthoc_only",
        }
    metrics = _posthoc_graph_metrics(indices, valid, labels)
    for graph_name, key in (
        ("raw_only", "raw_candidate_indices"),
        ("latent_only", "latent_candidate_indices"),
    ):
        graph_indices = _load_npz(diagnostics, key)
        if graph_indices is not None:
            graph_valid = np.asarray(graph_indices) >= 0
            metrics[graph_name] = _posthoc_graph_metrics(graph_indices, graph_valid, labels)
    features = _load_npz(diagnostics, "candidate_features")
    if features is not None and features.ndim == 3 and features.shape[:2] == valid.shape:
        feature_metrics: dict[str, Any] = {}
        # Source indicator is raw-only=-1, both=0, latent-only=+1.
        source = features[:, :, 2]
        in_bounds = (indices >= 0) & (indices < labels.shape[0])
        for key, mask in {
            "both_views": valid & in_bounds & np.isclose(source, 0.0),
            "raw_only": valid & in_bounds & (source < 0.0),
            "latent_only": valid & in_bounds & (source > 0.0),
        }.items():
            flat = mask.reshape(-1)
            if np.any(flat):
                feature_metrics[key] = {
                    "edge_count": int(flat.sum()),
                    "edge_purity": float(
                        np.mean(
                            labels[np.repeat(np.arange(labels.size), valid.shape[1])[flat]]
                            == labels[indices.reshape(-1)[flat]]
                        )
                    ),
                }
        metrics["source_stratified"] = feature_metrics
        metrics["source_indicator_view_identity"] = "raw_only=-1, both=0, latent_only=+1"
    return {
        "status": "available",
        "label_use": "posthoc_only",
        "labels_used_during_fit": summary.get("labels_used_during_fit"),
        "metrics": metrics,
        "stored_summary_values": {
            "candidate_recall": summary.get("graph_profile", {}).get("posthoc_candidate_recall"),
            "edge_purity": summary.get("graph_profile", {}).get("posthoc_edge_purity"),
        },
    }


def _posthoc_assignment_gain(
    run: Path,
    labels: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    train_anchor: np.ndarray,
) -> dict[str, Any]:
    diagnostics = run / "gate_diagnostics.npz"
    self_prediction = _load_npz(diagnostics, "final_probe_self_prediction")
    edge_prediction = _load_npz(diagnostics, "final_probe_edge_prediction")
    if self_prediction is None or edge_prediction is None:
        self_prediction = _load_npz(diagnostics, "utility_probe_self_prediction")
        edge_prediction = _load_npz(diagnostics, "utility_probe_edge_prediction")
    if self_prediction is None or edge_prediction is None:
        return {"status": "not_available", "reason": "single-edge probe predictions are not persisted"}
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    self_prediction = np.asarray(self_prediction).reshape(-1).astype(np.int64)
    edge_prediction = np.asarray(edge_prediction).astype(np.int64)
    if self_prediction.shape != labels.shape or edge_prediction.shape != target.shape:
        return {"status": "invalid_artifact", "reason": "probe prediction shapes do not match anchors/edges"}
    mapping, fallback = _hungarian_mapping(self_prediction, labels)
    mapped_self = np.asarray([mapping.get(int(value), fallback) for value in self_prediction], dtype=np.int64)
    mapped_edge = np.vectorize(lambda value: mapping.get(int(value), fallback), otypes=[np.int64])(edge_prediction)
    self_correct = mapped_self == labels
    edge_correct = mapped_edge == labels[:, None]
    gain_matrix = edge_correct.astype(np.int8) - self_correct[:, None].astype(np.int8)
    heldout_valid = valid & ~train_anchor[:, None]
    gain = gain_matrix[heldout_valid]
    utility_hat = predicted[heldout_valid]
    utility_target = target[heldout_valid]
    positive = gain > 0
    predicted_pi = _load_npz(diagnostics, "final_predicted_pi")
    if predicted_pi is None:
        predicted_pi = _load_npz(diagnostics, "predicted_pi")
    selected_gain = None
    if predicted_pi is not None and np.asarray(predicted_pi).shape == (target.shape[0], target.shape[1] + 1):
        selected = heldout_valid & (np.asarray(predicted_pi)[:, 1:] > 0.0)
        selected_gain = _finite(float(gain_matrix[selected].mean())) if np.any(selected) else None
    return {
        "status": "available",
        "label_use": "posthoc_only",
        "definition": "change in Hungarian-mapped cluster-assignment correctness after one edge",
        "positive_rate": _finite(float(np.mean(gain > 0))) if gain.size else None,
        "negative_rate": _finite(float(np.mean(gain < 0))) if gain.size else None,
        "assignment_change_rate": _finite(float(np.mean(gain != 0))) if gain.size else None,
        "utility_hat_auroc": _metric(roc_auc_score, positive.astype(np.int8), utility_hat)
        if np.unique(positive).size == 2
        else None,
        "utility_hat_spearman": _finite(float(spearmanr(gain, utility_hat).statistic)) if gain.size > 1 else None,
        "utility_target_spearman": _finite(float(spearmanr(gain, utility_target).statistic)) if gain.size > 1 else None,
        "selected_edge_mean_gain": selected_gain,
        "edge_count": int(gain.size),
    }


def _posthoc_edge_alignment(
    run: Path,
    labels: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    train_anchor: np.ndarray,
) -> dict[str, Any]:
    """Audit whether utility ranks same-label candidate edges.

    This is a post-hoc label audit, not a fitting target. It is reported next
    to assignment-correction gain because one-edge argmax changes are often
    sparse even when a candidate is a useful same-cluster connection.
    """
    diagnostics = run / "gate_diagnostics.npz"
    indices = _load_npz(diagnostics, "candidate_indices") if diagnostics.exists() else None
    if indices is None or np.asarray(indices).shape != valid.shape:
        return {"status": "not_available", "reason": "candidate indices are missing"}
    indices = np.asarray(indices, dtype=np.int64)
    heldout = valid & ~np.asarray(train_anchor, dtype=bool)[:, None]
    rows, cols = np.where(heldout)
    in_bounds = (indices[rows, cols] >= 0) & (indices[rows, cols] < labels.shape[0])
    rows, cols = rows[in_bounds], cols[in_bounds]
    if rows.size == 0:
        return {"status": "available", "edge_count": 0, "label_use": "posthoc_only"}
    donors = indices[rows, cols]
    same = (labels[rows] == labels[donors]).astype(np.int8)
    scores = np.asarray(predicted)[heldout][in_bounds]
    predicted_pi = _load_npz(diagnostics, "final_predicted_pi")
    selected_purity = None
    if predicted_pi is not None and np.asarray(predicted_pi).shape[0] == valid.shape[0]:
        selected = heldout.copy()
        selected &= np.asarray(predicted_pi)[:, 1:] > 0.0
        selected_rows, selected_cols = np.where(selected)
        selected_in_bounds = (
            indices[selected_rows, selected_cols] >= 0
        ) & (indices[selected_rows, selected_cols] < labels.shape[0])
        if np.any(selected_in_bounds):
            selected_purity = _finite(float(np.mean(
                labels[selected_rows[selected_in_bounds]]
                == labels[indices[selected_rows[selected_in_bounds], selected_cols[selected_in_bounds]]]
            )))
    return {
        "status": "available",
        "label_use": "posthoc_only",
        "edge_count": int(same.size),
        "same_label_rate": _finite(float(same.mean())),
        "utility_hat_auroc": _metric(roc_auc_score, same, scores)
        if np.unique(same).size == 2
        else None,
        "selected_edge_same_label_rate": selected_purity,
    }


def _utility_certificate(
    run: Path,
    summary: dict[str, Any],
    labels: np.ndarray | None = None,
) -> dict[str, Any]:
    diagnostics = run / "gate_diagnostics.npz"
    target = _load_npz(diagnostics, "utility_target") if diagnostics.exists() else None
    predicted = _load_npz(diagnostics, "utility_hat") if diagnostics.exists() else None
    valid = _load_npz(diagnostics, "utility_valid") if diagnostics.exists() else None
    if target is None or predicted is None or valid is None:
        return {
            "status": "not_available",
            "reason": "utility target, prediction, or validity mask is missing",
        }
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if target.shape != predicted.shape or target.shape != valid.shape:
        return {"status": "invalid_artifact", "reason": "utility arrays have different shapes"}
    y = target[valid]
    score = predicted[valid]
    binary = y > 0.0
    row_has_valid = valid.any(axis=1)
    row_medians = np.median(
        np.where(valid[row_has_valid], target[row_has_valid], np.inf),
        axis=1,
    )
    row_medians = row_medians[np.isfinite(row_medians)]
    rowwise_centering = bool(row_medians.size and np.mean(np.abs(row_medians) < 1e-5) > 0.8)
    train_anchor = _load_npz(diagnostics, "utility_train_anchor")
    gate_mode = str(summary.get("config", {}).get("gate_mode", ""))
    direct_identity = gate_mode in {
        "direct_target",
        "direct_counterfactual",
        "union_uniform",
        "forced_topk",
        "shuffled_utility",
        "output_disabled",
    }
    heldout_metrics: dict[str, Any]
    if train_anchor is not None and np.asarray(train_anchor).shape == (target.shape[0],):
        train_anchor = np.asarray(train_anchor, dtype=bool)
        heldout_valid = valid & ~train_anchor[:, None]
        heldout_y = target[heldout_valid]
        heldout_score = predicted[heldout_valid]
        heldout_binary = heldout_y > 0.0
        heldout_metrics = (
            {
                "status": "not_applicable",
                "reason": "direct gate uses the detached utility target without an amortized scorer",
                "anchor_count": int((~train_anchor).sum()),
                "edge_count": int(heldout_y.size),
            }
            if direct_identity
            else {
                "status": "available",
                "auroc": _metric(roc_auc_score, heldout_binary.astype(np.int8), heldout_score)
                if np.unique(heldout_binary).size == 2
                else None,
                "auprc": _metric(average_precision_score, heldout_binary.astype(np.int8), heldout_score)
                if np.unique(heldout_binary).size == 2
                else None,
                "spearman": _finite(float(spearmanr(heldout_y, heldout_score).statistic))
                if heldout_y.size > 1
                else None,
                "anchor_count": int((~train_anchor).sum()),
                "edge_count": int(heldout_y.size),
            }
        )
    else:
        heldout_metrics = {
            "status": "not_available",
            "reason": "utility_train_anchor split is not persisted",
        }
    independent_gain = _load_npz(diagnostics, "utility_independent_cluster_gain")
    if (
        independent_gain is not None
        and np.asarray(independent_gain).shape == target.shape
        and train_anchor is not None
        and np.asarray(train_anchor).shape == (target.shape[0],)
    ):
        independent_gain = np.asarray(independent_gain, dtype=np.float64)
        independent_valid = valid & ~np.asarray(train_anchor, dtype=bool)[:, None]
        gain = independent_gain[independent_valid]
        gain_binary = gain > 0.0
        heldout_prediction = predicted[independent_valid]
        heldout_target = target[independent_valid]
        independent_metrics = {
            "status": "available",
            "definition": "held-out teacher-side cross-view counterfactual utility from the exact assignment operator",
            "gain_positive_rate": _finite(float(gain_binary.mean())) if gain.size else None,
            "utility_hat_auroc": _metric(roc_auc_score, gain_binary.astype(np.int8), heldout_prediction)
            if np.unique(gain_binary).size == 2
            else None,
            "utility_hat_spearman": _finite(float(spearmanr(gain, heldout_prediction).statistic))
            if gain.size > 1
            else None,
            "utility_target_spearman": _finite(float(spearmanr(gain, heldout_target).statistic))
            if gain.size > 1
            else None,
            "edge_count": int(gain.size),
        }
    else:
        independent_metrics = {
            "status": "not_available",
            "reason": "held-out independent cluster gain is not persisted",
        }
    posthoc_gain = (
        _posthoc_assignment_gain(run, labels, target, predicted, valid, np.asarray(train_anchor, dtype=bool))
        if labels is not None and train_anchor is not None and np.asarray(train_anchor).shape == (target.shape[0],)
        else {"status": "not_available", "reason": "labels or held-out anchor split are unavailable"}
    )
    posthoc_alignment = (
        _posthoc_edge_alignment(run, labels, predicted, valid, np.asarray(train_anchor, dtype=bool))
        if labels is not None and train_anchor is not None and np.asarray(train_anchor).shape == (target.shape[0],)
        else {"status": "not_available", "reason": "labels or held-out anchor split are unavailable"}
    )
    result: dict[str, Any] = {
        "status": "available",
        "in_sample_scorer_fit": (
            {
                "status": "not_applicable",
                "label": "direct detached utility; no amortized scorer",
                "positive_rate": _finite(float(binary.mean())) if binary.size else None,
                "edge_count": int(y.size),
            }
            if direct_identity
            else {
                "status": "available",
                "label": "in-sample diagnostic only",
                "auroc": _metric(roc_auc_score, binary.astype(np.int8), score)
                if np.unique(binary).size == 2
                else None,
                "auprc": _metric(average_precision_score, binary.astype(np.int8), score)
                if np.unique(binary).size == 2
                else None,
                "spearman": _finite(float(spearmanr(y, score).statistic)) if y.size > 1 else None,
                "positive_rate": _finite(float(binary.mean())) if binary.size else None,
                "edge_count": int(y.size),
            }
        ),
        "target_contract": {
            "invalid_edges_excluded": True,
            "target_detached": "not_verifiable_from_numpy_artifact",
            "gate_input_in_target": "not_verifiable_from_numpy_artifact",
            "rowwise_centering_detected": rowwise_centering,
            "row_median_abs_mean": _finite(float(np.mean(np.abs(row_medians)))) if row_medians.size else None,
            "clipped_target_fraction": _finite(float(np.mean(np.isclose(y, -4.0)))) if y.size else None,
        },
        "held_out_utility_prediction": heldout_metrics,
        "independent_view_counterfactual_gain": independent_metrics,
        "posthoc_assignment_correction_gain": posthoc_gain,
        "posthoc_edge_same_label_alignment": posthoc_alignment,
        "same_target_warning": direct_identity,
        "labels_used_during_fit": summary.get("labels_used_during_fit"),
    }
    return result


def _pollution(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        fraction = record.get("graph_replacement_fraction")
        graph = record.get("certificates", {}).get("graph", {}).get("metrics", {})
        if fraction is None or not graph:
            continue
        for metric in ("edge_purity", "candidate_recall_budget_normalized", "candidate_coverage_any_same_label"):
            value = graph.get(metric)
            if value is not None:
                grouped[float(fraction)][metric].append(float(value))
    means = {
        str(fraction): {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for fraction, metrics in sorted(grouped.items())
    }
    return {
        "by_replacement_fraction": means,
        "interpretation": "posthoc label audit; no monotonicity claim unless an explicit stress axis is present",
    }


def _posthoc_oracle_ceiling(run: Path, labels: np.ndarray) -> dict[str, Any]:
    """Measure whether the persisted candidate/operator space can beat self.

    This deliberately uses labels to choose the best single edge after fitting.
    It is a diagnostic upper bound, never a model result or fitting target.
    """
    diagnostics = run / "gate_diagnostics.npz"
    self_prediction = _load_npz(diagnostics, "final_probe_self_prediction") if diagnostics.exists() else None
    edge_prediction = _load_npz(diagnostics, "final_probe_edge_prediction") if diagnostics.exists() else None
    valid = _load_npz(diagnostics, "final_gate_valid") if diagnostics.exists() else None
    if valid is None and diagnostics.exists():
        valid = _load_npz(diagnostics, "candidate_valid")
    actual_path = run / "predictions.npy"
    if self_prediction is None or edge_prediction is None or valid is None or not actual_path.exists():
        return {
            "status": "not_available",
            "reason": "final self/edge predictions, validity, or actual predictions are missing",
            "label_use": "posthoc_only",
        }
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    self_prediction = np.asarray(self_prediction).reshape(-1).astype(np.int64)
    edge_prediction = np.asarray(edge_prediction).astype(np.int64)
    valid = np.asarray(valid, dtype=bool)
    actual_prediction = np.asarray(np.load(actual_path, allow_pickle=False)).reshape(-1).astype(np.int64)
    gate_readout_probabilities = _load_npz(diagnostics, "final_gate_readout_probabilities")
    student_probabilities = _load_npz(diagnostics, "final_student_probabilities")
    reference_path = run / "teacher_probabilities_reference.npy"
    if (
        self_prediction.shape != labels.shape
        or actual_prediction.shape != labels.shape
        or edge_prediction.ndim != 2
        or edge_prediction.shape != valid.shape
        or edge_prediction.shape[0] != labels.shape[0]
    ):
        return {
            "status": "invalid_artifact",
            "reason": "oracle ceiling arrays have incompatible shapes",
            "label_use": "posthoc_only",
        }
    mapping, fallback = _hungarian_mapping(self_prediction, labels)
    mapped_self = np.asarray([mapping.get(int(value), fallback) for value in self_prediction], dtype=np.int64)
    mapped_edge = np.vectorize(lambda value: mapping.get(int(value), fallback), otypes=[np.int64])(
        edge_prediction
    )
    self_correct = mapped_self == labels
    oracle_prediction = self_prediction.copy()
    recoverable = np.zeros(labels.shape[0], dtype=bool)
    for row in range(labels.shape[0]):
        correcting = np.flatnonzero(valid[row] & (mapped_edge[row] == labels[row]))
        if not self_correct[row] and correcting.size:
            oracle_prediction[row] = edge_prediction[row, correcting[0]]
            recoverable[row] = True
    self_metrics = _partition_metrics(labels, self_prediction)
    oracle_metrics = _partition_metrics(labels, oracle_prediction)
    actual_metrics = _partition_metrics(labels, actual_prediction)
    gate_readout_metrics = (
        _partition_metrics(labels, np.asarray(gate_readout_probabilities).argmax(axis=1))
        if gate_readout_probabilities is not None
        and np.asarray(gate_readout_probabilities).ndim == 2
        and np.asarray(gate_readout_probabilities).shape[0] == labels.shape[0]
        else None
    )
    student_metrics = (
        _partition_metrics(labels, np.asarray(student_probabilities).argmax(axis=1))
        if student_probabilities is not None
        and np.asarray(student_probabilities).ndim == 2
        and np.asarray(student_probabilities).shape[0] == labels.shape[0]
        else None
    )
    reference_gate_metrics = None
    reference_gate_prediction = None
    if reference_path.exists():
        reference = np.asarray(np.load(reference_path, allow_pickle=False), dtype=np.float64)
        if reference.ndim == 2 and reference.shape[0] == labels.shape[0]:
            reference = np.clip(reference, 1e-8, None)
            reference /= reference.sum(axis=1, keepdims=True)
            cluster_count = reference.shape[1]
            if (
                np.all((self_prediction >= 0) & (self_prediction < cluster_count))
                and np.all((edge_prediction >= 0) & (edge_prediction < cluster_count))
            ):
                rows = np.arange(labels.shape[0])
                self_support = np.log(reference[rows, self_prediction])
                edge_support = np.log(reference[rows[:, None], edge_prediction])
                reference_gain = edge_support - self_support[:, None]
                reference_gain = np.where(valid, reference_gain, -np.inf)
                best = np.argmax(reference_gain, axis=1)
                best_gain = reference_gain[rows, best]
                use_edge = np.isfinite(best_gain) & (best_gain > 0.0)
                reference_gate_prediction = self_prediction.copy()
                reference_gate_prediction[use_edge] = edge_prediction[rows[use_edge], best[use_edge]]
                reference_gate_metrics = _partition_metrics(labels, reference_gate_prediction)
    wrong = ~self_correct
    return {
        "status": "available",
        "label_use": "posthoc_only",
        "definition": "best one-edge assignment correction available after fitting",
        "self": self_metrics,
        "exported_prediction": actual_metrics,
        "student_clean": student_metrics,
        "gate_readout": gate_readout_metrics,
        "clean_reference_gate": reference_gate_metrics,
        "oracle_gate": oracle_metrics,
        "actual_delta_ari_vs_self": float(actual_metrics["ari"] - self_metrics["ari"]),
        "gate_readout_delta_ari_vs_student_clean": (
            float(gate_readout_metrics["ari"] - student_metrics["ari"])
            if gate_readout_metrics is not None and student_metrics is not None
            else None
        ),
        "clean_reference_gate_delta_ari_vs_student_clean": (
            float(reference_gate_metrics["ari"] - student_metrics["ari"])
            if reference_gate_metrics is not None and student_metrics is not None
            else None
        ),
        "oracle_delta_ari_vs_self": float(oracle_metrics["ari"] - self_metrics["ari"]),
        "oracle_delta_ari_vs_student_clean": (
            float(oracle_metrics["ari"] - student_metrics["ari"])
            if student_metrics is not None
            else None
        ),
        "self_error_count": int(wrong.sum()),
        "recoverable_error_count": int(recoverable.sum()),
        "recoverable_error_fraction": float(recoverable.sum() / max(1, wrong.sum())),
    }


def audit_run(run: Path) -> dict[str, Any]:
    summary_path = run / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {run}")
    summary = _json(summary_path)
    labels_path = run / "labels_true.npy"
    labels = np.load(labels_path, allow_pickle=False) if labels_path.exists() else None
    records: dict[str, Any] = {
        "run": str(run),
        "dataset": summary.get("dataset"),
        "seed": summary.get("seed"),
        "gate_mode": summary.get("config", {}).get("gate_mode"),
        "graph_replacement_fraction": summary.get("config", {}).get("graph_replacement_fraction", 0.0),
        "labels_used_during_fit": summary.get("labels_used_during_fit"),
        "certificates": {
            "teacher": _teacher_certificate(run, summary, labels),
            "graph": _graph_certificate(run, summary, labels) if labels is not None else {
                "status": "not_available",
                "reason": "labels_true.npy missing; graph certificate is posthoc only",
                "label_use": "posthoc_only",
            },
            "utility": _utility_certificate(run, summary, labels),
            "output_oracle": _posthoc_oracle_ceiling(run, labels) if labels is not None else {
                "status": "not_available",
                "reason": "labels_true.npy missing; output oracle is posthoc only",
                "label_use": "posthoc_only",
            },
        },
    }
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = sorted({path.parent for path in args.run_root.rglob("summary.json")})
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for run in run_dirs:
        try:
            records.append(audit_run(run))
        except Exception as exc:  # keep one malformed run from hiding the panel boundary
            errors.append({"run": str(run), "error": f"{type(exc).__name__}: {exc}"})
    utility_available = [
        record["certificates"]["utility"]
        for record in records
        if record["certificates"]["utility"].get("status") == "available"
    ]
    payload = {
        "schema_version": "V15-stage1b-certificates-2",
        "run_root": str(args.run_root.resolve()),
        "records": records,
        "errors": errors,
        "certificate_contract": {
            "teacher": "requires persisted teacher assignment/embedding plus cross-view or temporal checks",
            "graph": "posthoc labels only; labels_used_during_fit must remain false",
            "utility": "separate in-sample scorer fit, held-out prediction, and independent downstream gain",
            "output_oracle": "posthoc best-edge ceiling; labels never enter fitting",
        },
        "panel_summary": {
            "runs": len(records),
            "errors": len(errors),
            "teacher_available": sum(r["certificates"]["teacher"]["status"] == "available" for r in records),
            "graph_available": sum(r["certificates"]["graph"]["status"] == "available" for r in records),
            "utility_available": len(utility_available),
            "utility_held_out_available": sum(
                r["certificates"]["utility"].get("held_out_utility_prediction", {}).get("status") == "available"
                for r in records
            ),
            "independent_view_gain_available": sum(
                r["certificates"]["utility"].get("independent_view_counterfactual_gain", {}).get("status")
                == "available"
                for r in records
            ),
            "posthoc_assignment_gain_available": sum(
                r["certificates"]["utility"].get("posthoc_assignment_correction_gain", {}).get("status")
                == "available"
                for r in records
            ),
            "output_oracle_available": sum(
                r["certificates"]["output_oracle"].get("status") == "available"
                for r in records
            ),
        },
        "pollution_audit": _pollution(records),
        "interpretation": [
            "Do not call in-sample AUROC evidence that utility generalizes to clustering gain.",
            "Do not infer teacher correctness from EMA existence alone.",
            "Graph recall and purity use labels only after fitting and are not training signals.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"runs": len(records), "errors": len(errors), "output": str(args.output)}))


if __name__ == "__main__":
    main()
