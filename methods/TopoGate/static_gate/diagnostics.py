from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_fscore_support


def embedding_geometry(embedding: np.ndarray, labels: np.ndarray) -> dict:
    emb = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(labels)
    centroids = []
    within = []
    for lab in np.unique(labels):
        block = emb[labels == lab]
        if block.size == 0:
            continue
        c = block.mean(axis=0)
        centroids.append(c)
        within.append(np.linalg.norm(block - c, axis=1).mean())
    if len(centroids) <= 1:
        between = 0.0
    else:
        c = np.vstack(centroids)
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        between = float(d[np.triu_indices_from(d, k=1)].mean())
    within_mean = float(np.mean(within)) if within else 0.0
    return {
        "within_class_distance": within_mean,
        "between_class_distance": between,
        "between_within_ratio": float(between / max(within_mean, 1e-8)),
    }


def mapped_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    n = max(len(true_values), len(pred_values))
    counts = np.zeros((n, n), dtype=np.int64)
    for i, t in enumerate(true_values):
        for j, p in enumerate(pred_values):
            counts[i, j] = int(np.sum((y_true == t) & (y_pred == p)))
    rows, cols = linear_sum_assignment(-counts)
    mapped = np.zeros_like(y_pred, dtype=np.int64)
    for row, col in zip(rows, cols):
        if row < len(true_values) and col < len(pred_values):
            mapped[y_pred == pred_values[col]] = true_values[row]
    return mapped


def per_cell_type_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    mapped = mapped_predictions(y_true, y_pred)
    labels = np.unique(y_true)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, mapped, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "label": labels,
            "n_cells": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "is_rare_lt_50": support < 50,
        }
    )
