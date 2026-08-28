"""Post-fit readout and label-free representation diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np


def representation_diagnostics(embedding: np.ndarray) -> dict[str, float]:
    value = np.asarray(embedding, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("embedding must be two-dimensional")
    if value.shape[0] == 0:
        return {"effective_rank": 0.0, "variance_floor": 0.0, "variance_median": 0.0, "low_variance_dimension_ratio": 1.0, "mean_l2_norm": 0.0}
    centered = value - np.mean(value, axis=0, keepdims=True)
    variances = np.var(centered, axis=0)
    total = float(np.sum(variances))
    if total <= 0.0:
        effective_rank = 0.0
    else:
        probabilities = variances / total
        probabilities = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "effective_rank": effective_rank,
        "variance_floor": float(np.min(variances)) if variances.size else 0.0,
        "variance_median": float(np.median(variances)) if variances.size else 0.0,
        "low_variance_dimension_ratio": float(np.mean(variances < 1e-8)) if variances.size else 1.0,
        "mean_l2_norm": float(np.mean(np.linalg.norm(value, axis=1))),
    }


def clustering_metrics(embedding: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float | int]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from scipy.optimize import linear_sum_assignment

    z = np.asarray(embedding, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 2 or y.ndim != 1 or z.shape[0] != y.shape[0]:
        raise ValueError("embedding and labels shape mismatch")
    k = int(np.unique(y).size)
    if k < 1:
        raise ValueError("labels contain no clusters")
    predicted = KMeans(n_clusters=k, random_state=int(seed), n_init=10).fit_predict(z)
    ari = float(adjusted_rand_score(y, predicted))
    nmi = float(normalized_mutual_info_score(y, predicted))
    # ACC is the optimal one-to-one mapping between predicted and true IDs.
    true_ids = np.unique(y)
    pred_ids = np.unique(predicted)
    matrix = np.zeros((max(true_ids.size, pred_ids.size), max(true_ids.size, pred_ids.size)), dtype=np.int64)
    for i, true_id in enumerate(true_ids):
        for j, pred_id in enumerate(pred_ids):
            matrix[i, j] = int(np.count_nonzero((y == true_id) & (predicted == pred_id)))
    rows, cols = linear_sum_assignment(matrix.max() - matrix)
    acc = float(matrix[rows, cols].sum() / max(y.size, 1))
    return {"ARI": ari, "NMI": nmi, "ACC": acc, "labels_unique": k, "kmeans_seed": int(seed)}


def svd_embedding(x0: np.ndarray, seed: int, n_components: int = 32) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import TruncatedSVD
    from scipy import sparse

    value = np.asarray(x0, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("x0 must be two-dimensional")
    k = min(int(n_components), max(1, value.shape[0] - 1), max(1, value.shape[1] - 1))
    # The raw source is stored as a dense NPZ for provenance, but the frozen
    # SVD baseline uses an equivalent CSR view so high-dimensional sparse
    # matrices do not create a second dense work array.
    sparse_value = sparse.csr_matrix(value)
    transformer = TruncatedSVD(n_components=k, random_state=int(seed), n_iter=5)
    embedding = transformer.fit_transform(sparse_value).astype(np.float32, copy=False)
    return embedding, {"components": int(k), "explained_variance_sum": float(np.sum(transformer.explained_variance_ratio_)), "seed": int(seed), "input_representation": "csr_zero_preserving_view"}


def evaluate_after_fit(embedding: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, Any]:
    diagnostics = representation_diagnostics(embedding)
    metrics = clustering_metrics(embedding, labels, seed)
    return {**metrics, **diagnostics}
