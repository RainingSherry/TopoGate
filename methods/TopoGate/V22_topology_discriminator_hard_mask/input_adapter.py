from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LoadedNPZ:
    X: np.ndarray | sp.csr_matrix
    labels: np.ndarray | None
    profile: dict[str, Any]


@dataclass(frozen=True)
class PreparedDualInput:
    X_model: np.ndarray
    X_graph: sp.csr_matrix
    X_support: sp.csr_matrix
    profile: dict[str, Any]
    selected_feature_indices: np.ndarray


def _read_matrix(payload: Any) -> np.ndarray | sp.csr_matrix:
    sparse_keys = {"data", "indices", "indptr", "shape"}
    if sparse_keys.issubset(payload.files):
        shape = tuple(int(v) for v in np.asarray(payload["shape"]).reshape(-1))
        return sp.csr_matrix(
            (
                np.asarray(payload["data"]),
                np.asarray(payload["indices"], dtype=np.int64),
                np.asarray(payload["indptr"], dtype=np.int64),
            ),
            shape=shape,
            dtype=np.float32,
        )
    for key in ("X", "x", "features", "data"):
        if key in payload.files:
            return np.asarray(payload[key])
    raise ValueError("NPZ has no X/x/features/data matrix")


def load_npz(path: str | Path) -> LoadedNPZ:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        matrix = _read_matrix(payload)
        labels = None
        for key in ("y", "labels", "label"):
            if key in payload.files:
                labels = np.asarray(payload[key]).reshape(-1)
                break
        keys = list(payload.files)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"X must be a non-empty 2D matrix, got {matrix.shape}")
    if labels is not None and labels.shape[0] != matrix.shape[0]:
        raise ValueError("label length does not match sample count")
    return LoadedNPZ(
        X=matrix,
        labels=labels,
        profile={
            "path": str(source.resolve()),
            "format": "npz",
            "npz_keys": keys,
            "n_samples_original": int(matrix.shape[0]),
            "n_features_original": int(matrix.shape[1]),
            "sparse_storage": bool(sp.issparse(matrix)),
            "labels_loaded_by_outer_runner": labels is not None,
        },
    )


def _csr_finite(X: np.ndarray | sp.spmatrix) -> sp.csr_matrix:
    matrix = sp.csr_matrix(X, dtype=np.float32)
    matrix.data = np.nan_to_num(matrix.data, nan=0.0, posinf=0.0, neginf=0.0)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _variance_subset(X: sp.csr_matrix, feature_cap: int) -> tuple[sp.csr_matrix, np.ndarray]:
    n_features = int(X.shape[1])
    if n_features <= int(feature_cap):
        return X, np.arange(n_features, dtype=np.int64)
    mean = np.asarray(X.mean(axis=0)).reshape(-1).astype(np.float64)
    mean_square = np.asarray(X.multiply(X).mean(axis=0)).reshape(-1).astype(np.float64)
    variance = np.nan_to_num(mean_square - mean * mean, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    selected = np.argpartition(-variance, int(feature_cap) - 1)[: int(feature_cap)]
    selected = selected[np.argsort(-variance[selected], kind="stable")].astype(np.int64, copy=False)
    return X[:, selected].tocsr(), selected


def prepare_dual_input(
    X: np.ndarray | sp.spmatrix,
    *,
    dataset_name: str,
    input_protocol: str,
    feature_cap: int = 2000,
) -> PreparedDualInput:
    """Build sparse graph/support views and a bounded dense scMAE view.

    Feature capping is variance-only and fit on X; it never reads labels.  The
    selected feature indices are saved so high-dimensional runs remain auditable.
    """
    if input_protocol not in {"clubench_bridge", "shared_text", "scRNA_count"}:
        raise ValueError("unsupported V22 input protocol")
    source = _csr_finite(X)
    if source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("X must be non-empty")
    selected, selected_indices = _variance_subset(source, int(feature_cap))
    support = selected.copy()
    view_source = selected.copy()
    transform = "identity"
    if input_protocol == "scRNA_count":
        view_source.data = np.log1p(np.maximum(view_source.data, 0.0)).astype(np.float32, copy=False)
        transform = "log1p_counts"
    graph_view = StandardScaler(with_mean=False, copy=True).fit_transform(view_source).tocsr()
    dense = view_source.toarray().astype(np.float32, copy=False)
    model_view = StandardScaler(with_mean=True, copy=False).fit_transform(dense).astype(np.float32, copy=False)
    nnz = int(selected.nnz)
    profile = {
        "dataset_name": str(dataset_name),
        "input_protocol": input_protocol,
        "n_samples": int(selected.shape[0]),
        "n_features_original": int(source.shape[1]),
        "n_features_selected": int(selected.shape[1]),
        "feature_cap": int(feature_cap),
        "feature_selection": "top_variance_label_free" if source.shape[1] > feature_cap else "none",
        "selected_feature_indices_count": int(selected_indices.size),
        "source_zero_fraction": float(1.0 - source.nnz / float(source.shape[0] * source.shape[1])),
        "selected_zero_fraction": float(1.0 - nnz / float(selected.shape[0] * selected.shape[1])),
        "graph_view_storage": "csr",
        "graph_view_scaler": "StandardScaler_with_mean_false",
        "model_view_scaler": "StandardScaler_with_mean_true",
        "count_transform": transform,
        "labels_used": False,
        "K_used": False,
        "full_pairwise_matrix_materialized": False,
    }
    return PreparedDualInput(
        X_model=np.ascontiguousarray(model_view, dtype=np.float32),
        X_graph=graph_view,
        X_support=support,
        profile=profile,
        selected_feature_indices=selected_indices,
    )
