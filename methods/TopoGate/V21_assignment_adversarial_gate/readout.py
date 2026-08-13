from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans


READOUT_MODES = frozenset({"student_t_head", "kmeans_embedding"})


def _occupancy_profile(predictions: np.ndarray, n_clusters: int) -> dict[str, Any]:
    counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=int(n_clusters))
    fractions = counts.astype(np.float64) / max(1, int(counts.sum()))
    return {
        "unique_clusters": int(np.count_nonzero(counts)),
        "empty_clusters": int(np.count_nonzero(counts == 0)),
        "cluster_counts": counts.tolist(),
        "min_cluster_fraction": float(fractions.min()),
        "max_cluster_fraction": float(fractions.max()),
    }


def select_readout(
    embedding: np.ndarray,
    probabilities: np.ndarray | None,
    *,
    n_clusters: int,
    mode: str,
    kmeans_n_init: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Select a label-free primary readout and retain the training-head audit."""

    values = np.asarray(embedding, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < int(n_clusters):
        raise ValueError("embedding must be 2D with at least n_clusters rows")
    if not np.isfinite(values).all():
        raise ValueError("embedding contains non-finite values")
    if mode not in READOUT_MODES:
        raise ValueError(f"unsupported readout mode: {mode!r}")

    head_predictions: np.ndarray | None = None
    head_profile: dict[str, Any] = {"available": False}
    if probabilities is not None:
        q = np.asarray(probabilities, dtype=np.float32)
        if q.shape != (values.shape[0], int(n_clusters)):
            raise ValueError("cluster probabilities do not match embedding and n_clusters")
        if not np.isfinite(q).all():
            raise ValueError("cluster probabilities contain non-finite values")
        head_predictions = q.argmax(axis=1).astype(np.int64)
        q_safe = np.clip(q, 1e-12, 1.0)
        entropy = -(q_safe * np.log(q_safe)).sum(axis=1)
        head_profile = {
            "available": True,
            "mean_max_probability": float(q.max(axis=1).mean()),
            "mean_normalized_entropy": float(entropy.mean() / np.log(int(n_clusters))),
            **_occupancy_profile(head_predictions, int(n_clusters)),
        }

    if mode == "student_t_head":
        if head_predictions is None:
            effective_mode = "kmeans_embedding"
        else:
            predictions = head_predictions.copy()
            effective_mode = "student_t_head"
    else:
        effective_mode = "kmeans_embedding"

    if effective_mode == "kmeans_embedding":
        predictions = KMeans(
            n_clusters=int(n_clusters),
            n_init=int(kmeans_n_init),
            random_state=int(seed),
        ).fit_predict(values).astype(np.int64)

    if effective_mode == "student_t_head":
        primary_method = "student_t_head_known_k"
    elif mode == "student_t_head" and probabilities is None:
        primary_method = "kmeans_known_k"
    else:
        primary_method = "kmeans_embedding_known_k"
    profile = {
        "requested_mode": mode,
        "effective_mode": effective_mode,
        "primary_method": primary_method,
        "labels_used_for_readout": False,
        "n_clusters": int(n_clusters),
        "kmeans_n_init": int(kmeans_n_init) if effective_mode == "kmeans_embedding" else None,
        "primary": _occupancy_profile(predictions, int(n_clusters)),
        "student_t_training_head": head_profile,
    }
    return predictions, head_predictions, profile
