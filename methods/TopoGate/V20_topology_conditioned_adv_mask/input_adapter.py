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


def load_npz_matrix_only(path: str | Path) -> np.ndarray | sp.csr_matrix:
    """Load only X; the NPZ label keys are intentionally never accessed."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        matrix = _read_matrix(payload)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"X must be a non-empty 2D matrix, got {matrix.shape}")
    return matrix


def _csr_finite(X: np.ndarray | sp.spmatrix) -> sp.csr_matrix:
    matrix = sp.csr_matrix(X, dtype=np.float32)
    matrix.data = np.nan_to_num(matrix.data, nan=0.0, posinf=0.0, neginf=0.0)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def prepare_dual_input(
    X: np.ndarray | sp.spmatrix,
    *,
    dataset_name: str,
    input_protocol: str,
) -> PreparedDualInput:
    """Create dense scMAE and sparse graph views without labels."""
    if input_protocol not in {"clubench_bridge", "shared_text"}:
        raise ValueError("V20 first-round input protocol must be clubench_bridge or shared_text")
    source = _csr_finite(X)
    if source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("X must be non-empty")
    graph_view = StandardScaler(with_mean=False, copy=True).fit_transform(source).tocsr()
    dense = source.toarray().astype(np.float32, copy=False)
    model_view = StandardScaler(with_mean=True, copy=False).fit_transform(dense).astype(np.float32, copy=False)
    indices = np.arange(source.shape[1], dtype=np.int64)
    profile = {
        "dataset_name": str(dataset_name),
        "input_protocol": input_protocol,
        "n_samples": int(source.shape[0]),
        "n_features_original": int(source.shape[1]),
        "n_features_selected": int(source.shape[1]),
        "graph_view_storage": "csr",
        "graph_view_scaler": "StandardScaler_with_mean_false",
        "model_view_scaler": "StandardScaler_with_mean_true",
        "labels_used": False,
        "K_used": False,
        "full_pairwise_matrix_materialized": False,
    }
    return PreparedDualInput(
        X_model=np.ascontiguousarray(model_view, dtype=np.float32),
        X_graph=graph_view,
        profile=profile,
        selected_feature_indices=indices,
    )
