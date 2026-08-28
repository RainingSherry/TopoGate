from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .config import FeatureConstraintConfig


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RobustTransform:
    center: np.ndarray
    scale: np.ndarray
    clip: float
    profile: dict[str, Any]

    def apply(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.center.size:
            raise ValueError("matrix shape does not match the robust transform")
        transformed = (values - self.center[None, :]) / self.scale[None, :]
        return np.clip(transformed, -self.clip, self.clip).astype(np.float32, copy=False)


@dataclass(frozen=True)
class FeatureFoldModel:
    fold: int
    neighbors: np.ndarray
    weights: np.ndarray
    prediction: sp.csr_matrix
    residual_operator_csc: sp.csc_matrix
    residual_scale: np.ndarray
    footprints: tuple[np.ndarray, ...]
    footprint_indptr: np.ndarray
    footprint_indices: np.ndarray
    profile: dict[str, Any]

    @property
    def n_features(self) -> int:
        return int(self.prediction.shape[0])

    def normalized_residual(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=np.float64).reshape(-1)
        if values.size != self.n_features:
            raise ValueError("z has the wrong feature count")
        predicted = self.prediction.dot(values)
        return ((values - predicted) / self.residual_scale).astype(np.float64, copy=False)

    def footprint(self, mask: np.ndarray) -> np.ndarray:
        selected = np.flatnonzero(np.asarray(mask, dtype=np.bool_).reshape(-1))
        if not selected.size:
            return np.empty(0, dtype=np.int64)
        result = np.zeros(self.n_features, dtype=np.bool_)
        for feature in selected:
            result[self.footprints[int(feature)]] = True
        return np.flatnonzero(result).astype(np.int64, copy=False)

    def joint_energy(self, z: np.ndarray, mask: np.ndarray | None = None) -> float:
        residual = self.normalized_residual(z)
        if mask is None:
            footprint = np.arange(self.n_features, dtype=np.int64)
        else:
            footprint = self.footprint(mask)
        if not footprint.size:
            return 0.0
        return float(np.mean(np.square(residual[footprint])))

    def action_delta(self, z: np.ndarray, donor_z: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
        hard = np.asarray(mask, dtype=np.bool_).reshape(-1)
        if hard.size != self.n_features:
            raise ValueError("mask has the wrong feature count")
        action = np.asarray(z, dtype=np.float64).reshape(-1).copy()
        donor = np.asarray(donor_z, dtype=np.float64).reshape(-1)
        action[hard] = donor[hard]
        clean_energy = self.joint_energy(z, hard)
        action_energy = self.joint_energy(action, hard)
        return float(action_energy - clean_energy), clean_energy, action_energy


@dataclass(frozen=True)
class CrossFittedFeatureModel:
    transform: RobustTransform
    fold_ids: np.ndarray
    folds: tuple[FeatureFoldModel, ...]
    profile: dict[str, Any]

    @property
    def n_features(self) -> int:
        return int(self.transform.center.size)

    def transform_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.transform.apply(X)

    def fold_for_row(self, row_index: int) -> FeatureFoldModel:
        index = int(row_index)
        if not 0 <= index < self.fold_ids.size:
            raise IndexError("row_index is outside the fitted sample range")
        return self.folds[int(self.fold_ids[index])]


def _fit_transform(X: np.ndarray, config: FeatureConstraintConfig) -> RobustTransform:
    values = np.asarray(X, dtype=np.float64)
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, float(config.robust_scale_floor))
    profile = {
        "name": config.transform,
        "clip": float(config.transform_clip),
        "scale_floor": float(config.robust_scale_floor),
        "floored_feature_count": int(np.count_nonzero(1.4826 * mad < config.robust_scale_floor)),
        "parameter_hash": _sha256_arrays(center.astype(np.float32), scale.astype(np.float32)),
        "labels_used": False,
    }
    return RobustTransform(
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
        clip=float(config.transform_clip),
        profile=profile,
    )


def _fold_assignments(n_samples: int, folds: int, seed: int) -> np.ndarray:
    effective = min(int(folds), int(n_samples))
    if effective < 2:
        raise ValueError("cross-fitted feature graph requires at least two samples")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_samples)
    assignments = np.empty(n_samples, dtype=np.int64)
    assignments[order] = np.arange(n_samples, dtype=np.int64) % effective
    return assignments


def _cosine_graph(
    train: np.ndarray,
    *,
    graph_k: int,
    weight_floor: float,
    control: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix, dict[str, Any]]:
    n_features = int(train.shape[1])
    if n_features <= 1:
        return (
            np.empty((n_features, 0), dtype=np.int64),
            np.empty((n_features, 0), dtype=np.float32),
            sp.csr_matrix((n_features, n_features), dtype=np.float32),
            {"graph_k_effective": 0, "zero_positive_rows": n_features},
        )
    if control == "marginal":
        return (
            np.empty((n_features, 0), dtype=np.int64),
            np.empty((n_features, 0), dtype=np.float32),
            sp.csr_matrix((n_features, n_features), dtype=np.float32),
            {"graph_k_effective": 0, "zero_positive_rows": n_features, "control": "marginal"},
        )
    k = min(int(graph_k), n_features - 1)
    values = np.asarray(train, dtype=np.float32)
    norms = np.linalg.norm(values, axis=0).astype(np.float64)
    gram = np.asarray(values.T @ values, dtype=np.float64)
    denominator = np.maximum(norms[:, None] * norms[None, :], 1e-12)
    similarity = gram / denominator
    np.fill_diagonal(similarity, -np.inf)
    candidate = np.argpartition(similarity, kth=n_features - k, axis=1)[:, -k:]
    candidate_values = np.take_along_axis(similarity, candidate, axis=1)
    order = np.argsort(candidate_values, axis=1)[:, ::-1]
    neighbors = np.take_along_axis(candidate, order, axis=1).astype(np.int64, copy=False)
    raw_weights = np.take_along_axis(candidate_values, order, axis=1)
    if control == "shuffled":
        rng = np.random.default_rng(int(seed))
        for feature in range(n_features):
            pool = np.concatenate((np.arange(feature), np.arange(feature + 1, n_features)))
            neighbors[feature] = rng.choice(pool, size=k, replace=False)
    weights = np.maximum(raw_weights, 0.0)
    if weight_floor:
        weights += float(weight_floor)
    sums = weights.sum(axis=1, keepdims=True)
    zero_rows = sums[:, 0] <= 1e-12
    if np.any(zero_rows):
        weights[zero_rows] = 1.0
        sums = weights.sum(axis=1, keepdims=True)
    weights = (weights / sums).astype(np.float32, copy=False)
    rows = np.repeat(np.arange(n_features, dtype=np.int64), k)
    prediction = sp.csr_matrix(
        (weights.reshape(-1), (rows, neighbors.reshape(-1))),
        shape=(n_features, n_features),
        dtype=np.float32,
    )
    prediction.sum_duplicates()
    prediction.sort_indices()
    profile = {
        "graph_k_effective": int(k),
        "zero_positive_rows": int(np.count_nonzero(zero_rows)),
        "mean_selected_cosine": float(np.mean(np.maximum(raw_weights, 0.0))),
        "self_edges": int(np.count_nonzero(prediction.diagonal())),
        "negative_weights": int(np.count_nonzero(prediction.data < 0.0)),
        "row_sum_max_error": float(np.max(np.abs(np.asarray(prediction.sum(axis=1)).reshape(-1) - 1.0))),
        "control": control,
    }
    return neighbors, weights, prediction, profile


def _fit_fold(
    z: np.ndarray,
    fold_ids: np.ndarray,
    fold: int,
    config: FeatureConstraintConfig,
    seed: int,
) -> FeatureFoldModel:
    train_indices = np.flatnonzero(fold_ids != int(fold))
    heldout_indices = np.flatnonzero(fold_ids == int(fold))
    train = np.asarray(z[train_indices], dtype=np.float32)
    neighbors, weights, prediction, graph_profile = _cosine_graph(
        train,
        graph_k=config.graph_k,
        weight_floor=config.graph_weight_floor,
        control=config.graph_control,
        seed=int(seed) + 1009 * int(fold),
    )
    residual = train - np.asarray(prediction.dot(train.T).T, dtype=np.float32)
    residual_center = np.median(residual, axis=0)
    residual_mad = np.median(np.abs(residual - residual_center[None, :]), axis=0)
    residual_scale = np.maximum(1.4826 * residual_mad, float(config.residual_scale_floor)).astype(np.float64)
    identity = sp.eye(prediction.shape[0], dtype=np.float64, format="csr")
    operator = sp.diags(1.0 / residual_scale, format="csr") @ (identity - prediction.astype(np.float64))
    operator_csc = operator.tocsc()
    footprints: list[np.ndarray] = []
    for feature in range(prediction.shape[0]):
        if neighbors.shape[1]:
            footprint = np.unique(np.concatenate((np.asarray([feature], dtype=np.int64), neighbors[feature])))
        else:
            footprint = np.asarray([feature], dtype=np.int64)
        footprints.append(footprint.astype(np.int64, copy=False))
    footprint_indptr = np.zeros(prediction.shape[0] + 1, dtype=np.int64)
    for feature, footprint in enumerate(footprints):
        footprint_indptr[feature + 1] = footprint_indptr[feature] + footprint.size
    footprint_indices = np.concatenate(footprints).astype(np.int64, copy=False)
    profile = {
        "fold": int(fold),
        "train_rows": int(train_indices.size),
        "heldout_rows": int(heldout_indices.size),
        "train_index_hash": _sha256_arrays(train_indices),
        "heldout_index_hash": _sha256_arrays(heldout_indices),
        "residual_scale_floor": float(config.residual_scale_floor),
        "residual_scale_floored": int(np.count_nonzero(1.4826 * residual_mad < config.residual_scale_floor)),
        "graph_hash": _sha256_arrays(neighbors, weights),
        "labels_used": False,
        **graph_profile,
    }
    return FeatureFoldModel(
        fold=int(fold),
        neighbors=neighbors,
        weights=weights,
        prediction=prediction,
        residual_operator_csc=operator_csc,
        residual_scale=residual_scale,
        footprints=tuple(footprints),
        footprint_indptr=footprint_indptr,
        footprint_indices=footprint_indices,
        profile=profile,
    )


def fit_cross_fitted_feature_model(
    X_model: np.ndarray,
    *,
    config: FeatureConstraintConfig,
    seed: int,
) -> CrossFittedFeatureModel:
    """Fit label-free fold-specific feature predictors on the capped interface."""

    config.validate()
    X = np.asarray(X_model, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X_model must be a non-empty 2D matrix with at least two rows")
    if not np.isfinite(X).all():
        raise ValueError("X_model contains non-finite values")
    if X.shape[1] > int(config.max_features):
        raise ValueError(f"ACCG capped interface supports at most {config.max_features} features")
    transform = _fit_transform(X, config)
    z = transform.apply(X)
    fold_ids = _fold_assignments(X.shape[0], config.graph_crossfit_folds, int(seed) + 701)
    folds = tuple(
        _fit_fold(z, fold_ids, fold, config, int(seed))
        for fold in range(int(fold_ids.max()) + 1)
    )
    profile = {
        "protocol": "cross_fitted_feature_conditional_model_v1",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "crossfit_folds_requested": int(config.graph_crossfit_folds),
        "crossfit_folds_effective": int(len(folds)),
        "fold_assignment_hash": _sha256_arrays(fold_ids),
        "transform": transform.profile,
        "graph_estimator": config.graph_estimator,
        "graph_k": int(config.graph_k),
        "graph_control": config.graph_control,
        "labels_used": False,
        "fold_profiles": [fold.profile for fold in folds],
    }
    return CrossFittedFeatureModel(transform=transform, fold_ids=fold_ids, folds=folds, profile=profile)
