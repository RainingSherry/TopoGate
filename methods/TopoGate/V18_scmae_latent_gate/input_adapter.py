from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass(frozen=True)
class LoadedData:
    X: np.ndarray | sp.csr_matrix
    labels: np.ndarray | None
    profile: dict[str, Any]


def _first(payload: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in payload.files:
            return np.asarray(payload[name])
    return None


def load_data(path: str | Path) -> LoadedData:
    """Load matrix and optional labels without passing labels to model fitting."""
    source = Path(path)
    if source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            sparse_keys = {"data", "indices", "indptr", "shape"}
            if sparse_keys.issubset(payload.files):
                shape = tuple(int(v) for v in np.asarray(payload["shape"]).reshape(-1))
                X: np.ndarray | sp.csr_matrix = sp.csr_matrix(
                    (np.asarray(payload["data"]), np.asarray(payload["indices"], dtype=np.int64),
                     np.asarray(payload["indptr"], dtype=np.int64)), shape=shape
                )
            else:
                X = _first(payload, ("X", "x", "data", "features"))
                if X is None:
                    raise ValueError(f"NPZ has no X/x/data/features matrix: {source}")
            labels = _first(payload, ("y", "labels", "label"))
        return LoadedData(X, labels, {"path": str(source.resolve()), "format": "npz"})
    if source.suffix == ".npy":
        return LoadedData(np.load(source, allow_pickle=False), None,
                          {"path": str(source.resolve()), "format": "npy"})
    if source.suffix in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if source.suffix == ".tsv" else "," if source.suffix == ".csv" else None
        return LoadedData(np.loadtxt(source, delimiter=delimiter), None,
                          {"path": str(source.resolve()), "format": source.suffix[1:]})
    if source.suffix == ".h5ad":
        try:
            import anndata as ad
        except ImportError as exc:  # pragma: no cover - depends on optional local stack
            raise RuntimeError("loading .h5ad requires anndata") from exc
        adata = ad.read_h5ad(source, backed="r")
        X = adata.X[:]
        labels = None
        label_key = None
        for key in ("cell_type", "celltype", "label", "labels", "cluster", "clusters"):
            if key in adata.obs:
                label_key = key
                labels = np.asarray(adata.obs[key].astype(str))
                break
        return LoadedData(X, labels, {"path": str(source.resolve()), "format": "h5ad", "label_key": label_key})
    raise ValueError(f"unsupported dataset format: {source}")


def encode_labels(labels: np.ndarray | None) -> tuple[np.ndarray | None, list[str] | None]:
    if labels is None:
        return None, None
    values = np.asarray(labels).reshape(-1)
    if values.size == 0:
        return None, None
    encoded = LabelEncoder().fit_transform(values.astype(str)).astype(np.int64)
    names = sorted(set(values.astype(str).tolist()))
    return encoded, names


def prepare_input(
    X: np.ndarray | sp.spmatrix,
    *,
    input_mode: str = "auto",
    standardize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply only input-semantic transforms; no labels or K are consulted."""
    matrix = sp.csr_matrix(X, dtype=np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    dense = np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if input_mode == "auto":
        mode = "nonnegative" if float(np.min(dense)) >= 0.0 else "continuous"
    else:
        mode = input_mode
    if mode in {"count", "nonnegative"} and float(np.min(dense)) < 0.0:
        raise ValueError(f"{mode} input must be non-negative")
    if mode == "count":
        if not np.allclose(dense, np.rint(dense), atol=1e-5, rtol=0.0):
            raise ValueError("count input must contain integer-valued observations")
        dense = np.log1p(dense)
    zero_fraction_after_input_transform = float(np.mean(dense == 0.0))
    if standardize:
        dense = StandardScaler(with_mean=True, with_std=True).fit_transform(dense).astype(np.float32)
    profile = {
        "input_mode_requested": input_mode,
        "input_mode_resolved": mode,
        "standardize": bool(standardize),
        "n_samples": int(dense.shape[0]),
        "n_features": int(dense.shape[1]),
        "zero_fraction_after_input_transform": zero_fraction_after_input_transform,
        "full_pairwise_matrix_materialized": False,
        "labels_used": False,
        "K_used": False,
    }
    return dense, profile
