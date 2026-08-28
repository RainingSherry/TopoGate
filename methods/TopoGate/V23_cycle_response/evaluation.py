from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


PRIMARY_FINGERPRINT = "cycle_repair_standardized"
SECONDARY_RECOVERABILITY = "recovery_gain_standardized"


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, object]
    predictions: dict[str, np.ndarray]
    benchmark_validity: dict[str, object]


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    _, encoded = np.unique(np.asarray(labels).reshape(-1), return_inverse=True)
    return encoded.astype(np.int64, copy=False)


def _mapped_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = _encode_labels(labels)
    predictions = _encode_labels(predictions)
    size = max(int(labels.max(initial=0)), int(predictions.max(initial=0))) + 1
    counts = np.zeros((size, size), dtype=np.int64)
    np.add.at(counts, (predictions, labels), 1)
    rows, cols = linear_sum_assignment(counts.max() - counts)
    return float(counts[rows, cols].sum() / max(1, labels.size))


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def _cluster_representation(values: np.ndarray, n_clusters: int, seed: int, fingerprint: bool) -> np.ndarray:
    data = _row_normalize(values) if fingerprint else np.asarray(values, dtype=np.float32)
    return KMeans(n_clusters=int(n_clusters), n_init=20, random_state=int(seed)).fit_predict(data)


def _is_degenerate(values: np.ndarray) -> bool:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        return True
    return bool(np.all(np.var(values, axis=0) <= 1e-12) or np.unique(values, axis=0).shape[0] < 2)


def _knn_purity(values: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    if values.shape[0] <= 1:
        return float("nan")
    neighbors = min(int(k), values.shape[0] - 1)
    indices = NearestNeighbors(n_neighbors=neighbors + 1, metric="cosine").fit(values).kneighbors(return_distance=False)
    indices = indices[:, 1:]
    return float(np.mean(labels[indices] == labels[:, None]))


def _sample_pair_scores(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    pairs_per_class: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    classes = np.unique(labels)
    class_rows = {int(label): np.flatnonzero(labels == label) for label in classes}
    eligible_same = [label for label, rows in class_rows.items() if rows.size >= 2]
    if len(classes) < 2 or not eligible_same:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    pair_labels: list[int] = []
    scores: list[float] = []

    def cosine_score(i: int, j: int) -> float:
        left = values[i]
        right = values[j]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)

    for _ in range(int(pairs_per_class)):
        label = int(rng.choice(eligible_same))
        i, j = rng.choice(class_rows[label], size=2, replace=False)
        pair_labels.append(1)
        scores.append(cosine_score(int(i), int(j)))
    for _ in range(int(pairs_per_class)):
        first, second = rng.choice(classes, size=2, replace=False)
        i = int(rng.choice(class_rows[int(first)]))
        j = int(rng.choice(class_rows[int(second)]))
        pair_labels.append(0)
        scores.append(cosine_score(i, j))
    return np.asarray(pair_labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def benchmark_validity_profile(
    clean_embedding: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    permutations: int = 32,
    sample_cap: int = 3000,
) -> dict[str, object]:
    """Evaluation-only CLM proxy; this is explicitly not Jeon et al.'s Adjusted IVM."""

    labels = _encode_labels(labels)
    rng = np.random.default_rng(int(seed) + 811)
    if clean_embedding.shape[0] > sample_cap:
        rows = np.sort(rng.choice(clean_embedding.shape[0], size=sample_cap, replace=False))
        embedding = clean_embedding[rows]
        sampled_labels = labels[rows]
    else:
        embedding = clean_embedding
        sampled_labels = labels
    unique = np.unique(sampled_labels)
    if unique.size <= 1 or unique.size >= sampled_labels.size:
        raw = float("nan")
        null_values = np.empty(0, dtype=np.float64)
    else:
        raw = float(silhouette_score(embedding, sampled_labels, metric="cosine"))
        null_values = np.asarray(
            [
                silhouette_score(embedding, rng.permutation(sampled_labels), metric="cosine")
                for _ in range(int(permutations))
            ],
            dtype=np.float64,
        )
    if null_values.size and np.isfinite(raw):
        null_mean = float(null_values.mean())
        null_std = float(null_values.std(ddof=1)) if null_values.size > 1 else 0.0
        z_score = float((raw - null_mean) / null_std) if null_std > 1e-12 else float("nan")
        percentile = float((1.0 + np.count_nonzero(null_values <= raw)) / (null_values.size + 1.0))
    else:
        null_mean = float("nan")
        null_std = float("nan")
        z_score = float("nan")
        percentile = float("nan")
    if np.isfinite(raw) and raw > 0.0 and np.isfinite(z_score) and z_score >= 2.0:
        interpretation = "high_confidence"
    elif np.isfinite(raw) and raw > 0.0 and np.isfinite(z_score) and z_score >= 0.0:
        interpretation = "qualified"
    else:
        interpretation = "low_confidence"
    return {
        "audit_scope": "evaluation_only",
        "enters_training": False,
        "enters_hyperparameter_selection": False,
        "proxy_name": "permutation_adjusted_label_silhouette_on_clean_embedding",
        "adjusted_ivma_implemented": False,
        "adjusted_ivma_note": "This proxy is not the Adjusted IVM protocol of Jeon et al.",
        "label_partition_intrinsic_validity": {
            "silhouette_cosine": raw,
            "null_mean": null_mean,
            "null_std": null_std,
            "null_z_score": z_score,
            "null_percentile": percentile,
            "permutations": int(permutations),
            "sample_count": int(embedding.shape[0]),
        },
        "external_metric_interpretation": interpretation,
        "n": int(labels.size),
        "d": int(clean_embedding.shape[1]),
        "K": int(np.unique(labels).size),
    }


def evaluate_fingerprints(
    arrays: dict[str, np.ndarray],
    *,
    labels: np.ndarray | None,
    external_k: int | None,
    seed: int,
) -> EvaluationResult:
    if "clean_embedding" not in arrays:
        raise ValueError("fingerprints artifact lacks clean_embedding")
    n_samples = int(arrays["clean_embedding"].shape[0])
    if labels is not None and np.asarray(labels).reshape(-1).size != n_samples:
        raise ValueError("outer label count differs from fingerprint rows")
    encoded = None if labels is None else _encode_labels(labels)
    if encoded is not None:
        n_clusters = int(np.unique(encoded).size)
        k_source = "benchmark_oracle_from_outer_labels"
        if external_k is not None and int(external_k) != n_clusters:
            raise ValueError("external K conflicts with outer label count")
    elif external_k is not None:
        n_clusters = int(external_k)
        k_source = "explicit_external_k"
    else:
        n_clusters = None
        k_source = "unavailable"

    representation_names = [
        "clean_embedding",
        "precycle_standardized",
        "cycle_repair_standardized",
        "recovery_gain_standardized",
        "support_standardized",
        "untrained_cycle_standardized",
        "linear_cycle_standardized",
        "lowrank_cycle_standardized",
        "full_cycle_adjusted_standardized",
    ]
    metrics: dict[str, object] = {
        "primary_fingerprint": PRIMARY_FINGERPRINT,
        "secondary_recoverability": SECONDARY_RECOVERABILITY,
        "primary_distance": "cosine",
        "labels_available_outer_only": encoded is not None,
        "K_source": k_source,
        "n_clusters": n_clusters,
        "pairwise_observations_treated_as_iid": False,
        "representations": {},
    }
    predictions: dict[str, np.ndarray] = {}
    if n_clusters is not None:
        if n_clusters <= 1 or n_clusters >= n_samples:
            raise ValueError("readout K must be in [2, n_samples-1]")
        for name in representation_names:
            if name not in arrays:
                continue
            fingerprint = name != "clean_embedding"
            if _is_degenerate(arrays[name]):
                metrics["representations"][name] = {
                    "status": "degenerate_representation",
                    "cluster_method": None,
                }
                continue
            prediction = _cluster_representation(arrays[name], n_clusters, seed, fingerprint)
            predictions[name] = prediction.astype(np.int64, copy=False)
            row: dict[str, object] = {"cluster_method": "spherical_kmeans" if fingerprint else "kmeans"}
            if encoded is not None:
                pair_labels, pair_scores = _sample_pair_scores(arrays[name], encoded, seed=seed)
                row.update(
                    {
                        "ari": float(adjusted_rand_score(encoded, prediction)),
                        "nmi": float(normalized_mutual_info_score(encoded, prediction)),
                        "acc": _mapped_accuracy(encoded, prediction),
                        "knn_purity_at_10": _knn_purity(arrays[name], encoded, k=10),
                        "pair_auc": float(roc_auc_score(pair_labels, pair_scores)) if pair_labels.size else float("nan"),
                        "pair_auprc_balanced": (
                            float(average_precision_score(pair_labels, pair_scores)) if pair_labels.size else float("nan")
                        ),
                        "pair_sampling_prevalence": 0.5 if pair_labels.size else float("nan"),
                    }
                )
            metrics["representations"][name] = row
    if encoded is None:
        benchmark_validity = {
            "audit_scope": "evaluation_only",
            "status": "unavailable_without_outer_labels",
            "enters_training": False,
            "enters_hyperparameter_selection": False,
            "n": n_samples,
            "d": int(arrays["clean_embedding"].shape[1]),
            "K": n_clusters,
        }
    else:
        benchmark_validity = benchmark_validity_profile(arrays["clean_embedding"], encoded, seed=seed)
    return EvaluationResult(metrics=metrics, predictions=predictions, benchmark_validity=benchmark_validity)
