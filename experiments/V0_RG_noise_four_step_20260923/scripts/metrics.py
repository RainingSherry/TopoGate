"""Descriptive frozen-probe metrics. Labels enter only post-hoc diagnostics."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors


def mean_or_none(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def rms_or_none(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if values.size else None


def applied_direction_family(clean, directions, multiplier, rtol=1e-4, atol=1e-8):
    """Audit actual float32 additions, then intersect validity across all families."""
    clean = np.asarray(clean, dtype=np.float32)
    applied = {}
    for name, (delta, nominal_valid, target) in directions.items():
        noisy = np.asarray(clean + float(multiplier) * delta, dtype=np.float32)
        norms = np.linalg.norm(noisy.astype(float) - clean, axis=1)
        target_norm = np.asarray(target, dtype=float) * float(multiplier)
        error = np.abs(norms - target_norm)
        valid = (np.asarray(nominal_valid, dtype=bool) & (norms > 0) & (target_norm > 0)
                 & np.isfinite(norms) & (error <= atol + rtol * np.abs(target_norm)))
        applied[name] = {"noisy": noisy, "norm": norms, "target": target_norm,
                         "error": error, "valid": valid, "nominal_valid": np.asarray(nominal_valid, dtype=bool)}
    if not applied:
        raise ValueError("At least one direction family is required")
    targets = [v["target"] for v in applied.values()]
    if not all(np.array_equal(targets[0], v) for v in targets[1:]):
        raise ValueError("Direction families must share identical target norms")
    common = np.logical_and.reduce([v["valid"] for v in applied.values()])
    return applied, common


def subset_prediction_metrics(labels, clean_pred, noisy_pred, valid):
    valid = np.asarray(valid, dtype=bool)
    y, p0, p1 = np.asarray(labels)[valid], np.asarray(clean_pred)[valid], np.asarray(noisy_pred)[valid]
    ari_defined = len(y) > 0 and len(np.unique(y)) > 1
    clean_ari = float(adjusted_rand_score(y, p0)) if ari_defined else None
    noisy_ari = float(adjusted_rand_score(y, p1)) if ari_defined else None
    return {"n": len(y), "clean_ari": clean_ari, "noisy_ari": noisy_ari,
            "ari_drop": clean_ari - noisy_ari if ari_defined else None,
            "ari_status": "ok" if ari_defined else ("empty_common_mask" if len(y) == 0 else "single_ground_truth_class"),
            "flip_rate": float(np.mean(p0 != p1)) if len(y) else None}


def displacement(clean, noisy):
    clean, noisy = np.asarray(clean, np.float32), np.asarray(noisy, np.float32)
    clean64 = clean.astype(float)
    delta = np.linalg.norm(noisy.astype(float) - clean64, axis=1)
    scale = float(np.sqrt(np.mean(np.sum((clean64 - clean64.mean(axis=0)) ** 2, axis=1))))
    normalized = delta / scale if scale > 0 else np.full(len(clean), np.nan)
    return {"changed_fraction": float(np.mean(clean != noisy)), "mean_l2_displacement": float(delta.mean()),
            "clean_rms_centered_scale": scale, "normalized_displacement": rms_or_none(normalized),
            "normalization": "Frobenius_displacement_over_clean_centered_Frobenius",
            "status": "ok" if scale > 0 else "zero_clean_variance"}, normalized


def distribution_metrics(clean, noisy, max_rows=512, max_features=128, seed=9102026):
    clean, noisy = np.asarray(clean), np.asarray(noisy)
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(clean), min(len(clean), int(max_rows)), replace=False)
    a, b = clean[ids].astype(float), noisy[ids].astype(float)
    if len(a) < 2:
        mmd = {"status": "insufficient_rows", "mmd2": None}
    else:
        d = pairwise_distances(a, metric="sqeuclidean")
        positive = d[np.triu_indices(len(a), 1)]
        positive = positive[positive > 0]
        bandwidth2 = float(np.median(positive)) if len(positive) else 0.0
        def effect(u, v):
            if not bandwidth2:
                return None
            ku = np.exp(-pairwise_distances(u, metric="sqeuclidean") / (2 * bandwidth2))
            kv = np.exp(-pairwise_distances(v, metric="sqeuclidean") / (2 * bandwidth2))
            kuv = np.exp(-pairwise_distances(u, v, metric="sqeuclidean") / (2 * bandwidth2))
            return float(ku.mean() + kv.mean() - 2 * kuv.mean())
        permutation = rng.permutation(len(clean))
        half = min(len(clean) // 2, int(max_rows))
        calibration = effect(clean[permutation[:half]], clean[permutation[half:2 * half]]) if half >= 2 else None
        contrasts = []
        if half >= 2 and bandwidth2 > 0:
            for _ in range(5):
                split = rng.permutation(len(clean))
                reference, comparison = split[:half], split[half:2 * half]
                baseline = effect(clean[reference], clean[comparison])
                perturbed = effect(clean[reference], noisy[comparison])
                contrasts.append({"clean_reference_vs_clean_comparison": baseline,
                                  "clean_reference_vs_noisy_comparison": perturbed,
                                  "difference": perturbed - baseline})
        mmd = {"status": "ok" if bandwidth2 > 0 else "zero_clean_bandwidth", "mmd2": effect(a, b),
               "estimator": "biased_empirical_RBF_MMD_squared_descriptive", "bandwidth_squared": bandwidth2,
               "rows_per_distribution": len(a), "clean_clean_calibration_mmd2": calibration,
               "calibration_rows_per_group": half, "calibration": "disjoint_clean_subsamples_descriptive_only_not_a_null_test",
               "paired_effect_note": "paired_clean_noisy_rows; do_not_ratio_against_different_size_clean_calibration",
               "reference_contrast": {"status": "ok" if contrasts else "undefined", "splits": contrasts,
                   "rows_per_group": half, "split_count": len(contrasts),
                   "mean_clean_clean_mmd2": mean_or_none([v["clean_reference_vs_clean_comparison"] for v in contrasts]),
                   "mean_clean_noisy_mmd2": mean_or_none([v["clean_reference_vs_noisy_comparison"] for v in contrasts]),
                   "mean_difference": mean_or_none([v["difference"] for v in contrasts]),
                   "interpretation": "equal_group_size_disjoint_reference_contrast; descriptive_not_p_value_or_threshold"},
               "p_value": None}
    variance = np.var(clean.astype(float), axis=0)
    selected = np.argsort(-variance, kind="stable")[:int(max_features)]
    selected = selected[variance[selected] > 0]
    # Noise can make formerly variable features constant; such correlations are undefined.
    selected = selected[np.var(noisy[:, selected].astype(float), axis=0) > 0]
    if len(selected) < 2 or len(clean) < 2:
        correlation = {"status": "insufficient_variable_features", "rms_correlation_change": None, "features": selected.tolist()}
    else:
        ca = np.corrcoef(clean[:, selected].astype(float), rowvar=False)
        cb = np.corrcoef(noisy[:, selected].astype(float), rowvar=False)
        off = np.triu_indices(len(selected), 1)
        diff = (cb - ca)[off]
        correlation = {"status": "ok", "rms_correlation_change": float(np.sqrt(np.mean(diff ** 2))),
                       "features": selected.tolist(), "feature_rule": "clean_variance_top_then_drop_undefined_noisy_correlations"}
    return {"mmd": mmd, "correlation": correlation}


def normalized_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def knn_self_excluded(embedding, k):
    n = len(embedding)
    k = min(int(k), n - 1)
    if k < 1:
        return np.empty((n, 0), dtype=np.int64)
    nn = NearestNeighbors(n_neighbors=min(n, k + 1), metric="cosine").fit(embedding)
    candidate = nn.kneighbors(embedding, return_distance=False)
    return np.asarray([row[row != i][:k] for i, row in enumerate(candidate)], dtype=np.int64)


def graph_comparison(clean_embedding, noisy_embedding, labels, test_indices, k):
    clean_nn = knn_self_excluded(clean_embedding, k)
    noisy_nn = knn_self_excluded(noisy_embedding, k)
    t, y = np.asarray(test_indices), np.asarray(labels)
    width = clean_nn.shape[1]
    if width == 0:
        return {"status": "insufficient_rows", "neighbor_retention": None}, {}
    retention = np.asarray([len(set(clean_nn[i]) & set(noisy_nn[i])) / width for i in t])
    clean_purity = np.mean(y[clean_nn[t]] == y[t, None], axis=1)
    noisy_purity = np.mean(y[noisy_nn[t]] == y[t, None], axis=1)
    valid = (np.linalg.norm(clean_embedding[t], axis=1) > 0) & (np.linalg.norm(noisy_embedding[t], axis=1) > 0)
    valid &= np.all(np.linalg.norm(clean_embedding[clean_nn[t]], axis=2) > 0, axis=1)
    valid &= np.all(np.linalg.norm(noisy_embedding[noisy_nn[t]], axis=2) > 0, axis=1)
    retention[~valid] = np.nan
    clean_purity[~valid] = np.nan
    noisy_purity[~valid] = np.nan
    classes = np.unique(y[t])
    def macro(v):
        values = [mean_or_none(v[y[t] == c]) for c in classes]
        return mean_or_none([v for v in values if v is not None])
    def topology(neighbors):
        sets = [set(row.tolist()) for row in neighbors]
        mutual, snn = [], []
        for i in t[valid]:
            mutual.append(np.mean([i in sets[j] for j in neighbors[i]]))
            snn.append(np.mean([len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for j in neighbors[i]]))
        return mean_or_none(mutual), mean_or_none(snn)
    cm, cs = topology(clean_nn)
    nm, ns = topology(noisy_nn)
    cp, npurity = mean_or_none(clean_purity), mean_or_none(noisy_purity)
    return {"status": "ok" if np.all(valid) else "undefined_cosine_rows_excluded", "k": width, "space": "clean_fitted_PCA_row_L2_cosine_self_excluded",
            "valid_query_count": int(valid.sum()), "undefined_query_count": int((~valid).sum()),
            "clean_zero_candidate_rows": int(np.sum(np.linalg.norm(clean_embedding, axis=1) == 0)),
            "noisy_zero_candidate_rows": int(np.sum(np.linalg.norm(noisy_embedding, axis=1) == 0)),
            "query_scope": "saved_test_rows", "candidate_scope": "all_clean_scope_row_identities",
            "neighbor_retention": mean_or_none(retention), "clean_purity_micro": cp,
            "noisy_purity_micro": npurity, "purity_micro_change": npurity - cp if cp is not None and npurity is not None else None,
            "clean_purity_macro": macro(clean_purity), "noisy_purity_macro": macro(noisy_purity),
            "clean_mutual": cm, "noisy_mutual": nm, "clean_snn": cs, "noisy_snn": ns,
            "label_role": "posthoc_purity_only"}, {"neighbor_retention": retention,
            "clean_neighbor_purity": clean_purity, "noisy_neighbor_purity": noisy_purity}


def representation_metrics(clean_z, noisy_z, centers, labels, test_indices, clean_pred=None):
    t = np.asarray(test_indices)
    z0, z1 = np.asarray(clean_z)[t], np.asarray(noisy_z)[t]
    centers = np.asarray(centers)
    # Match the audited runner's direct float32 squared-distance arithmetic,
    # including its first-index argmin tie convention.
    d0 = ((z0[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    d1 = ((z1[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    p0 = d0.argmin(axis=1) if clean_pred is None else np.asarray(clean_pred)[t]
    p1 = d1.argmin(axis=1)
    stats, drift = displacement(z0, z1)
    row = np.arange(len(t))
    margin0 = np.full(len(t), np.nan)
    margin1 = np.full(len(t), np.nan)
    if len(centers) > 1:
        distances = pairwise_distances(centers, metric="sqeuclidean")
        scale_values = distances[np.triu_indices(len(centers), 1)]
        scale = float(np.median(scale_values[scale_values > 0])) if np.any(scale_values > 0) else 0.0
        if scale > 0:
            other0, other1 = d0.copy(), d1.copy()
            other0[row, p0] = np.inf
            other1[row, p0] = np.inf
            margin0 = (other0.min(axis=1) - d0[row, p0]) / scale
            margin1 = (other1.min(axis=1) - d1[row, p0]) / scale
        margin_status = "ok" if scale > 0 else "coincident_centroids"
    else:
        scale, margin_status = None, "single_cluster"
    y = np.asarray(labels)[t]
    ari_defined = len(np.unique(y)) > 1
    prediction = {"flip_rate": float(np.mean(p0 != p1)),
                  "clean_ari": float(adjusted_rand_score(y, p0)) if ari_defined else None,
                  "noisy_ari": float(adjusted_rand_score(y, p1)) if ari_defined else None,
                  "ari_status": "ok" if ari_defined else "single_ground_truth_class",
                  "clean_margin": mean_or_none(margin0), "noisy_margin": mean_or_none(margin1),
                  "margin_change": mean_or_none(margin1 - margin0), "margin_status": margin_status,
                  "margin_scale": scale, "margin_definition": "(nearest_other_center_sqdist-clean_assigned_center_sqdist)/median_positive_clean_center_pair_sqdist",
                  "classes": [{"label": str(c), "n": int(np.sum(y == c)), "flip_rate": float(np.mean(p0[y == c] != p1[y == c])),
                               "normalized_drift": rms_or_none(drift[y == c])} for c in np.unique(y)]}
    prediction["ari_drop"] = prediction["clean_ari"] - prediction["noisy_ari"] if ari_defined else None
    rep = {"normalized_drift": stats["normalized_displacement"], "clean_rms_centered_scale": stats["clean_rms_centered_scale"], "status": stats["status"]}
    arrays = {"test_indices": t, "labels": y, "clean_predictions": p0, "noisy_predictions": p1,
              "clean_embedding": z0, "noisy_embedding": z1,
              "normalized_drift": drift, "flipped": p0 != p1, "clean_margin": margin0, "noisy_margin": margin1}
    # One clean projection maps both matrices; no independent noisy PCA fit.
    dim = min(2, len(clean_z) - 1, clean_z.shape[1])
    if dim > 0 and np.any(np.var(clean_z, axis=0) > 0):
        projection = PCA(n_components=dim, random_state=9102026).fit(clean_z)
        arrays["clean_projection_2d"] = np.pad(projection.transform(z0), ((0, 0), (0, 2 - dim)))
        arrays["noisy_projection_2d"] = np.pad(projection.transform(z1), ((0, 0), (0, 2 - dim)))
        rep["projection_status"] = "clean_fit_shared_axes"
    else:
        arrays["clean_projection_2d"] = np.full((len(t), 2), np.nan)
        arrays["noisy_projection_2d"] = np.full((len(t), 2), np.nan)
        rep["projection_status"] = "undefined_zero_variance_or_insufficient_rows"
    return rep, prediction, arrays
