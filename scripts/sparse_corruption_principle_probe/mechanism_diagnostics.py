"""Label-free structural diagnostics for C1/C2 corruption replays."""
from __future__ import annotations

from typing import Any

import numpy as np

from . import protocol
from .corruption_library import support_mask


def _quantiles(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        return {"q05": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q95": 0.0}
    q = np.quantile(flat, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {name: float(value) for name, value in zip(("q05", "q25", "q50", "q75", "q95"), q, strict=True)}


def _binary_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    terms = np.zeros_like(p)
    left = p > 0
    right = p < 1
    terms[left] -= p[left] * np.log(p[left])
    terms[right] -= (1.0 - p[right]) * np.log(1.0 - p[right])
    return float(np.mean(terms))


def support_diagnostics(clean: np.ndarray, corrupted: np.ndarray) -> dict[str, float]:
    """Measure support transitions under a fixed clean-reference threshold."""

    x = np.asarray(clean, dtype=np.float32)
    z = np.asarray(corrupted, dtype=np.float32)
    before = support_mask(x, reference=x)
    after = support_mask(z, reference=x)
    n, d = x.shape
    transition_01 = (~before & after)
    transition_10 = (before & ~after)
    union = before | after
    intersection = before & after
    row_union = np.sum(union, axis=1)
    row_jaccard = np.divide(
        np.sum(intersection, axis=1),
        row_union,
        out=np.ones(n, dtype=np.float64),
        where=row_union > 0,
    )
    clean_prev = np.mean(before, axis=0)
    after_prev = np.mean(after, axis=0)
    clean_co = (before.T @ before).astype(np.float64) / max(n, 1)
    after_co = (after.T @ after).astype(np.float64) / max(n, 1)
    co_norm = max(float(np.linalg.norm(clean_co)), 1e-12)
    return {
        "support_p_zero_to_nonzero": float(np.mean(transition_01)),
        "support_p_nonzero_to_zero": float(np.mean(transition_10)),
        "support_change_rate": float(np.mean(before != after)),
        "cell_nnz_before_mean": float(np.mean(np.sum(before, axis=1))),
        "cell_nnz_before_std": float(np.std(np.sum(before, axis=1))),
        "cell_nnz_after_mean": float(np.mean(np.sum(after, axis=1))),
        "cell_nnz_after_std": float(np.std(np.sum(after, axis=1))),
        "gene_prevalence_abs_change_mean": float(np.mean(np.abs(after_prev - clean_prev))),
        "gene_prevalence_abs_change_max": float(np.max(np.abs(after_prev - clean_prev))) if d else 0.0,
        "support_jaccard_mean": float(np.mean(row_jaccard)),
        "support_jaccard_q05": float(np.quantile(row_jaccard, 0.05)) if n else 0.0,
        "support_entropy_before": _binary_entropy(clean_prev),
        "support_entropy_after": _binary_entropy(after_prev),
        "support_cooccurrence_relative_fro_distortion": float(np.linalg.norm(after_co - clean_co) / co_norm),
    }


def value_diagnostics(clean: np.ndarray, corrupted: np.ndarray) -> dict[str, float]:
    """Measure value changes separately from support transitions."""

    x = np.asarray(clean, dtype=np.float32)
    z = np.asarray(corrupted, dtype=np.float32)
    before = support_mask(x, reference=x)
    after = support_mask(z, reference=x)
    overlap = before & after
    old = x[overlap].astype(np.float64)
    new = z[overlap].astype(np.float64)
    delta = np.abs(z.astype(np.float64) - x.astype(np.float64))
    high = np.abs(x) >= (np.quantile(np.abs(x), 0.9) if x.size else 0.0)
    result: dict[str, float] = {
        "value_overlap_count": float(old.size),
        "value_mean_before": float(np.mean(old)) if old.size else 0.0,
        "value_std_before": float(np.std(old)) if old.size else 0.0,
        "value_mean_after": float(np.mean(new)) if new.size else 0.0,
        "value_std_after": float(np.std(new)) if new.size else 0.0,
        "value_total_absolute_change": float(np.sum(delta, dtype=np.float64)),
        "value_mean_absolute_change": float(np.mean(delta)),
        "high_expression_coordinate_distortion": float(np.mean(delta[high])) if np.any(high) else 0.0,
    }
    result.update({f"value_before_{name}": value for name, value in _quantiles(old).items()})
    result.update({f"value_after_{name}": value for name, value in _quantiles(new).items()})
    if old.size >= 3 and np.std(old) > 1e-12 and np.std(new) > 1e-12:
        from scipy.stats import spearmanr

        result["value_rank_spearman"] = float(spearmanr(old, new).statistic)
    else:
        result["value_rank_spearman"] = 1.0 if old.size else 0.0
    return result


def _cosine_neighbors(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(matrix, dtype=np.float32)
    if x.shape[0] <= 1:
        return np.empty((x.shape[0], 0), dtype=np.int64), np.empty((x.shape[0], 0), dtype=np.float32)
    kk = min(max(1, int(k)), x.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1, metric="cosine")
    nn.fit(x)
    distances, indices = nn.kneighbors(x, return_distance=True)
    return indices[:, 1:], distances[:, 1:]


def geometry_diagnostics(
    clean: np.ndarray,
    corrupted: np.ndarray,
    *,
    k: int = protocol.GEOMETRY_K,
    sample_seed: int = 20260818,
) -> dict[str, float]:
    """Measure cell-cell geometry changes without labels."""

    x = np.asarray(clean, dtype=np.float32)
    z = np.asarray(corrupted, dtype=np.float32)
    idx_x, dist_x = _cosine_neighbors(x, k)
    idx_z, dist_z = _cosine_neighbors(z, k)
    if x.shape[0] == 0:
        return {"cosine_knn_jaccard_mean": 0.0, "neighbor_rank_stability_mean": 0.0, "local_density_relative_change": 0.0, "pairwise_distance_relative_distortion": 0.0}
    jaccards = []
    rank_stabilities = []
    for row in range(x.shape[0]):
        left = list(map(int, idx_x[row]))
        right = list(map(int, idx_z[row]))
        a, b = set(left), set(right)
        jaccards.append(len(a & b) / max(len(a | b), 1))
        if a & b:
            rank_x = {value: pos for pos, value in enumerate(left)}
            rank_z = {value: pos for pos, value in enumerate(right)}
            diffs = [abs(rank_x[value] - rank_z[value]) for value in a & b]
            rank_stabilities.append(1.0 - float(np.mean(diffs)) / max(len(left), 1))
        else:
            rank_stabilities.append(0.0)
    density_x = np.mean(dist_x, axis=1) if dist_x.size else np.zeros(x.shape[0])
    density_z = np.mean(dist_z, axis=1) if dist_z.size else np.zeros(x.shape[0])
    density_relative = np.abs(density_z - density_x) / np.maximum(np.abs(density_x), 1e-6)

    rng = np.random.default_rng(sample_seed)
    n = x.shape[0]
    pair_count = min(protocol.PAIRWISE_GEOMETRY_SAMPLE, n * max(n - 1, 0) // 2)
    if pair_count:
        left = rng.integers(0, n, size=pair_count)
        right = rng.integers(0, n, size=pair_count)
        keep = left != right
        left, right = left[keep], right[keep]
        dx = np.linalg.norm(x[left] - x[right], axis=1)
        dz = np.linalg.norm(z[left] - z[right], axis=1)
        pair_distortion = np.abs(dz - dx) / np.maximum(dx, 1e-6)
    else:
        pair_distortion = np.zeros(1, dtype=np.float32)
    return {
        "cosine_knn_jaccard_mean": float(np.mean(jaccards)),
        "cosine_knn_jaccard_q05": float(np.quantile(jaccards, 0.05)),
        "neighbor_rank_stability_mean": float(np.mean(rank_stabilities)),
        "local_density_relative_change": float(np.mean(density_relative)),
        "local_density_change_q95": float(np.quantile(density_relative, 0.95)),
        "pairwise_distance_relative_distortion": float(np.median(pair_distortion)),
    }


def representation_diagnostics(embedding: np.ndarray) -> dict[str, float]:
    """Label-free diagnostics for a post-fit clean embedding."""

    z = np.asarray(embedding, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("embedding must be a finite matrix")
    centered = z - np.mean(z, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(z.shape[0], 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(np.sum(eigenvalues))
    if total <= 1e-12:
        effective_rank = 0.0
    else:
        p = eigenvalues / total
        effective_rank = float(np.exp(-np.sum(np.where(p > 0, p * np.log(p), 0.0))))
    std = np.std(z, axis=0)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    normalized = z / np.maximum(norm, 1e-12)
    cosine = normalized @ normalized.T if z.shape[0] <= 2048 else None
    result = {
        "effective_rank": effective_rank,
        "variance_floor": float(np.min(std)) if std.size else 0.0,
        "variance_median": float(np.median(std)) if std.size else 0.0,
        "low_variance_dimension_ratio": float(np.mean(std < 1e-5)) if std.size else 1.0,
        "mean_pairwise_distance": float(np.mean(np.linalg.norm(z[1:] - z[:-1], axis=1))) if z.shape[0] > 1 else 0.0,
    }
    if cosine is not None and cosine.size:
        result["mean_off_diagonal_cosine"] = float(np.mean(cosine[~np.eye(z.shape[0], dtype=bool)]))
    else:
        result["mean_off_diagonal_cosine"] = 0.0
    return result


def combined_diagnostics(clean: np.ndarray, corrupted: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    result.update(support_diagnostics(clean, corrupted))
    result.update(value_diagnostics(clean, corrupted))
    result.update(geometry_diagnostics(clean, corrupted))
    return result
