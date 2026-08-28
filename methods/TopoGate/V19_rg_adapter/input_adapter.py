from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import INPUT_PROTOCOLS


BIOLOGICAL_INPUT_KIND = {
    "baron human": "raw_count",
    "baron_human": "raw_count",
    "baron-human": "raw_count",
    "mouse_retina": "log1p_expression",
    "mouse retina": "log1p_expression",
    "campbell": "log1p_expression",
}


@dataclass(frozen=True)
class LoadedNPZ:
    X: np.ndarray | sp.csr_matrix
    labels: np.ndarray | None
    profile: dict[str, Any]


@dataclass(frozen=True)
class PreparedInput:
    X: np.ndarray
    profile: dict[str, Any]
    selected_feature_indices: np.ndarray


def _first(payload: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in payload.files:
            return np.asarray(payload[name])
    return None


def load_npz(path: str | Path) -> LoadedNPZ:
    """Load X and optional benchmark labels; callers keep labels outside preprocessing."""
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError(f"V19 selected-dataset adapter requires NPZ input: {source}")
    with np.load(source, allow_pickle=False) as payload:
        sparse_keys = {"data", "indices", "indptr", "shape"}
        if sparse_keys.issubset(payload.files):
            shape = tuple(int(value) for value in np.asarray(payload["shape"]).reshape(-1))
            X: np.ndarray | sp.csr_matrix = sp.csr_matrix(
                (
                    np.asarray(payload["data"]),
                    np.asarray(payload["indices"], dtype=np.int64),
                    np.asarray(payload["indptr"], dtype=np.int64),
                ),
                shape=shape,
            )
        else:
            X = _first(payload, ("X", "x", "features", "data"))
            if X is None:
                raise ValueError(f"NPZ has no X/x/features/data matrix: {source}")
        labels = _first(payload, ("y", "labels", "label"))
        keys = list(payload.files)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    return LoadedNPZ(
        X=X,
        labels=labels,
        profile={
            "path": str(source.resolve()),
            "format": "npz",
            "npz_keys": keys,
            "n_samples_original": int(X.shape[0]),
            "n_features_original": int(X.shape[1]),
            "sparse_storage": bool(sp.issparse(X)),
            "labels_loaded_by_outer_runner": labels is not None,
        },
    )


def load_npz_matrix_only(path: str | Path) -> LoadedNPZ:
    """Load only the feature matrix from an NPZ without touching benchmark labels."""
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError(f"V19 selected-dataset adapter requires NPZ input: {source}")
    with np.load(source, allow_pickle=False) as payload:
        sparse_keys = {"data", "indices", "indptr", "shape"}
        if sparse_keys.issubset(payload.files):
            shape = tuple(int(value) for value in np.asarray(payload["shape"]).reshape(-1))
            X: np.ndarray | sp.csr_matrix = sp.csr_matrix(
                (
                    np.asarray(payload["data"]),
                    np.asarray(payload["indices"], dtype=np.int64),
                    np.asarray(payload["indptr"], dtype=np.int64),
                ),
                shape=shape,
            )
        else:
            X = None
            for name in ("X", "x", "features", "data"):
                if name in payload.files:
                    X = np.asarray(payload[name])
                    break
            if X is None:
                raise ValueError(f"NPZ has no X/x/features/data matrix: {source}")
        keys = list(payload.files)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    return LoadedNPZ(
        X=X,
        labels=None,
        profile={
            "path": str(source.resolve()),
            "format": "npz",
            "npz_keys": keys,
            "n_samples_original": int(X.shape[0]),
            "n_features_original": int(X.shape[1]),
            "sparse_storage": bool(sp.issparse(X)),
            "labels_loaded_by_outer_runner": False,
            "labels_accessed": False,
        },
    )


def encode_labels(labels: np.ndarray | None) -> tuple[np.ndarray | None, list[str] | None]:
    if labels is None:
        return None, None
    values = np.asarray(labels).reshape(-1)
    if values.size == 0:
        return None, None
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(values.astype(str)).astype(np.int64)
    return encoded, encoder.classes_.astype(str).tolist()


def _csr_float32(X: np.ndarray | sp.spmatrix) -> sp.csr_matrix:
    matrix = sp.csr_matrix(X, dtype=np.float32)
    matrix.data = np.nan_to_num(matrix.data, nan=0.0, posinf=0.0, neginf=0.0)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _normalize_total_log1p(X: np.ndarray | sp.spmatrix, target_sum: float) -> sp.csr_matrix:
    matrix = _csr_float32(X)
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float32)
    scale = np.divide(
        float(target_sum),
        row_sum,
        out=np.zeros_like(row_sum, dtype=np.float32),
        where=row_sum > 0.0,
    )
    matrix = matrix.multiply(scale[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data).astype(np.float32, copy=False)
    matrix.eliminate_zeros()
    return matrix


def _hvg_subset(
    X: np.ndarray | sp.spmatrix,
    n_top_features: int,
) -> tuple[ad.AnnData, np.ndarray, dict[str, Any]]:
    work = ad.AnnData(X=_csr_float32(X))
    target = min(int(n_top_features), int(work.n_vars))
    if target <= 0:
        raise ValueError("n_top_features must be positive")
    if int(work.n_vars) <= target:
        indices = np.arange(work.n_vars, dtype=np.int64)
        return work, indices, {
            "requested": int(n_top_features),
            "selected": int(work.n_vars),
            "strategy": "all_features_within_limit",
        }
    strategy = "seurat"
    try:
        sc.pp.highly_variable_genes(work, flavor="seurat", n_top_genes=target, subset=False)
        mask = np.asarray(work.var["highly_variable"], dtype=bool)
    except Exception:
        mask = np.zeros(work.n_vars, dtype=bool)
    selected_before_fallback = int(mask.sum())
    if selected_before_fallback == target:
        selected = np.flatnonzero(mask)
    else:
        strategy = "variance_fallback"
        matrix = work.X
        mean = np.asarray(matrix.mean(axis=0)).reshape(-1).astype(np.float64)
        mean_square = np.asarray(matrix.multiply(matrix).mean(axis=0)).reshape(-1).astype(np.float64)
        variance = np.nan_to_num(
            mean_square - mean * mean,
            nan=-np.inf,
            posinf=-np.inf,
            neginf=-np.inf,
        )
        order = np.lexsort((np.arange(work.n_vars, dtype=np.int64), -variance))
        selected = np.sort(order[:target])
    subset = work[:, selected].copy()
    return subset, selected.astype(np.int64), {
        "requested": int(n_top_features),
        "selected": int(subset.n_vars),
        "strategy": strategy,
        "scanpy_selected_before_fallback": selected_before_fallback,
    }


def _dense_finite_float32(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    dense = np.asarray(dense, dtype=np.float32)
    return np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)


def _standardize(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    dense = _dense_finite_float32(X)
    return StandardScaler(copy=False, with_mean=True, with_std=True).fit_transform(dense).astype(
        np.float32, copy=False
    )


def prepare_input(
    X: np.ndarray | sp.spmatrix,
    *,
    dataset_name: str,
    input_protocol: str,
    n_top_features: int = 1000,
    target_sum: float = 10_000.0,
) -> PreparedInput:
    """Apply a fixed, label-free V19 input protocol."""
    if input_protocol not in INPUT_PROTOCOLS:
        raise ValueError(f"input_protocol must be one of {sorted(INPUT_PROTOCOLS)}")
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    key = str(dataset_name).strip().lower()
    source = _csr_float32(X)
    if source.data.size and float(np.min(source.data)) < 0.0:
        raise ValueError("V19 selected sparse/count inputs must be non-negative before scaling")
    zero_fraction_before = float(1.0 - source.nnz / float(source.shape[0] * source.shape[1]))
    count_like = bool(
        source.data.size
        and np.allclose(source.data[: min(source.data.size, 100_000)], np.rint(source.data[: min(source.data.size, 100_000)]), atol=1e-5, rtol=0.0)
    )

    selected = np.arange(source.shape[1], dtype=np.int64)
    hvg_profile: dict[str, Any] = {
        "requested": 0,
        "selected": int(source.shape[1]),
        "strategy": "disabled",
    }
    normalization = "none"
    scale_method = "sklearn_standard_scaler"
    input_kind = BIOLOGICAL_INPUT_KIND.get(key, "sparse_text_features")
    if input_protocol == "rg_native":
        if input_kind not in {"raw_count", "log1p_expression"}:
            raise ValueError(f"rg_native is only registered for biological inputs, got {dataset_name!r}")
        if input_kind == "raw_count":
            if not count_like:
                raise ValueError(f"{dataset_name} rg_native expects integer raw counts")
            source = _normalize_total_log1p(source, target_sum=float(target_sum))
            normalization = f"normalize_total(target_sum={float(target_sum)})_then_log1p"
        else:
            normalization = "already_log1p_no_second_log1p"
        work, selected, hvg_profile = _hvg_subset(source, int(n_top_features))
        sc.pp.scale(work)
        prepared = _dense_finite_float32(work.X)
        scale_method = "scanpy_scale_matching_rg_native"
    elif input_protocol in {"clubench_bridge", "shared_text"}:
        if input_protocol == "shared_text" and input_kind != "sparse_text_features":
            raise ValueError(f"shared_text is only registered for text inputs, got {dataset_name!r}")
        prepared = _standardize(source)
    else:  # pragma: no cover - protected by the protocol validation above
        raise AssertionError(input_protocol)

    profile = {
        "dataset_name": str(dataset_name),
        "input_protocol": input_protocol,
        "input_kind": input_kind,
        "n_samples": int(prepared.shape[0]),
        "n_features_original": int(X.shape[1]),
        "n_features_selected": int(prepared.shape[1]),
        "zero_fraction_before_scaling": zero_fraction_before,
        "count_like_before_scaling": count_like,
        "normalization": normalization,
        "hvg": hvg_profile,
        "scale_method": scale_method,
        "labels_used": False,
        "K_used": False,
        "full_pairwise_matrix_materialized": False,
        "selected_feature_indices_saved_separately": True,
    }
    return PreparedInput(
        X=np.ascontiguousarray(prepared, dtype=np.float32),
        profile=profile,
        selected_feature_indices=selected,
    )
