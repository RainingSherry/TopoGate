from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
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
        raise ValueError(f"V0_RG selected-dataset adapter requires NPZ input: {source}")
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


def _decode_values(values: np.ndarray) -> np.ndarray:
    """Decode HDF5 categorical/string payloads without loading AnnData metadata."""
    raw = np.asarray(values).reshape(-1)
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw],
        dtype=str,
    )


def _h5_csr_matrix(handle: h5py.File, matrix_key: str) -> np.ndarray | sp.csr_matrix:
    if matrix_key not in handle:
        raise ValueError(f"HDF5 matrix key is absent: {matrix_key}")
    node = handle[matrix_key]
    if isinstance(node, h5py.Dataset):
        values = np.asarray(node[:])
        if values.ndim != 2:
            raise ValueError("HDF5 matrix dataset must be two-dimensional")
        return values
    required = {"data", "indices", "indptr"}
    if not isinstance(node, h5py.Group) or not required.issubset(node.keys()):
        raise ValueError("HDF5 matrix must be a dense dataset or CSR group")
    indptr = np.asarray(node["indptr"][:], dtype=np.int64)
    indices = np.asarray(node["indices"][:], dtype=np.int64)
    data = np.asarray(node["data"][:])
    shape = tuple(int(value) for value in np.asarray(node.attrs.get("shape", ())).reshape(-1))
    if len(shape) != 2:
        if "var" not in handle or "_index" not in handle["var"]:
            raise ValueError("CSR H5AD lacks shape metadata and var/_index")
        shape = (int(indptr.size - 1), int(handle["var"]["_index"].shape[0]))
    matrix = sp.csr_matrix((data, indices, indptr), shape=shape)
    if matrix.shape[0] != indptr.size - 1:
        raise ValueError("invalid HDF5 CSR indptr")
    return matrix


def _h5_labels(handle: h5py.File, label_key: str) -> np.ndarray:
    if label_key not in handle:
        raise ValueError(f"HDF5 label key is absent: {label_key}")
    node = handle[label_key]
    if isinstance(node, h5py.Dataset):
        return _decode_values(node[:])
    if not isinstance(node, h5py.Group) or not {"categories", "codes"}.issubset(node.keys()):
        raise ValueError("HDF5 labels must be a dataset or categorical categories/codes group")
    categories = _decode_values(node["categories"][:])
    codes = np.asarray(node["codes"][:], dtype=np.int64).reshape(-1)
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError("HDF5 categorical labels contain missing or invalid codes")
    return categories[codes]


def load_h5_matrix(
    path: str | Path,
    *,
    matrix_key: str = "X",
    label_key: str | None = None,
) -> LoadedNPZ:
    """Read only the HDF5 nodes required by the registered experiment."""
    source = Path(path)
    with h5py.File(source, "r") as handle:
        X = _h5_csr_matrix(handle, matrix_key)
        labels = _h5_labels(handle, label_key) if label_key else None
        profile = {
            "path": str(source.resolve()),
            "format": "h5ad" if source.suffix.lower() == ".h5ad" else "h5",
            "matrix_key": matrix_key,
            "label_key": label_key,
            "top_level_keys": sorted(handle.keys()),
            "matrix_storage": "csr" if sp.issparse(X) else "dense",
            "n_samples_original": int(X.shape[0]),
            "n_features_original": int(X.shape[1]),
            "sparse_storage": bool(sp.issparse(X)),
            "labels_loaded_by_outer_runner": labels is not None,
            "ignored_h5ad_groups": ["uns"] if source.suffix.lower() == ".h5ad" else [],
        }
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("HDF5 X must be a non-empty two-dimensional matrix")
    if labels is not None and labels.size != X.shape[0]:
        raise ValueError("HDF5 label count does not match input rows")
    return LoadedNPZ(X=X, labels=labels, profile=profile)


def load_hrvatin_geo_csv(
    path: str | Path,
    labels_path: str | Path,
    *,
    label_column: str,
) -> LoadedNPZ:
    """Load the canonical gene-by-cell GEO CSV as retained-cell-by-gene expression.

    The explicit ID join and source-column map make the removal of cells with
    missing maintype labels reproducible without creating a persistent raw copy.
    """
    source = Path(path)
    assignments_path = Path(labels_path)
    assignments = pd.read_csv(assignments_path, index_col=0, compression="infer")
    if label_column not in assignments.columns:
        raise ValueError(f"label column is absent: {label_column}")
    if assignments.index.has_duplicates:
        raise ValueError("assignment cell identifiers are not unique")
    labels_series = assignments[label_column]
    keep = labels_series.notna() & labels_series.astype(str).str.strip().ne("")
    retained_ids = labels_series.index[keep].astype(str).to_numpy()
    if retained_ids.size == 0:
        raise ValueError("no non-missing labels remain after filtering")
    header = pd.read_csv(source, nrows=0, compression="infer")
    if header.shape[1] < 2:
        raise ValueError("GEO matrix must contain a feature column and cell columns")
    feature_column = str(header.columns[0])
    source_cell_ids = header.columns[1:].astype(str).to_numpy()
    if len(np.unique(source_cell_ids)) != source_cell_ids.size:
        raise ValueError("GEO matrix cell identifiers are not unique")
    positions = {value: index for index, value in enumerate(source_cell_ids)}
    missing = sorted(set(retained_ids) - set(positions))
    if missing:
        raise ValueError(f"{len(missing)} retained labels are absent from the GEO matrix")
    selected = [feature_column, *retained_ids.tolist()]
    matrix_by_gene = pd.read_csv(
        source,
        usecols=selected,
        index_col=0,
        dtype={cell_id: np.float32 for cell_id in retained_ids},
        compression="infer",
    )
    matrix_by_gene = matrix_by_gene.loc[:, retained_ids]
    X = np.ascontiguousarray(matrix_by_gene.to_numpy(dtype=np.float32, copy=True).T)
    if not np.isfinite(X).all():
        raise ValueError("GEO matrix contains non-finite values")
    labels = labels_series.loc[retained_ids].astype(str).to_numpy()
    profile = {
        "path": str(source.resolve()),
        "format": "geo_csv_gzip",
        "orientation_source": "gene_by_cell",
        "orientation_loaded": "cell_by_gene",
        "labels_path": str(assignments_path.resolve()),
        "label_column": label_column,
        "n_samples_original": int(source_cell_ids.size),
        "n_samples_original_with_assignment": int(assignments.shape[0]),
        "n_samples_retained": int(retained_ids.size),
        "n_samples_excluded_missing_label": int((~keep).sum()),
        "n_features_original": int(matrix_by_gene.shape[0]),
        "source_cell_ids": retained_ids.tolist(),
        "source_cell_column_indices": [int(positions[value]) for value in retained_ids],
        "labels_loaded_by_outer_runner": True,
        "sparse_storage": False,
        "raw_data_hash_policy": "no_full_sha256",
    }
    return LoadedNPZ(X=X, labels=labels, profile=profile)



def load_matrix(
    path: str | Path,
    labels_path: str | Path | None = None,
    *,
    matrix_key: str = "X",
    label_key: str | None = None,
    label_column: str | None = None,
    input_adapter: str | None = None,
) -> LoadedNPZ:
    """Load a registered matrix without silently inferring biological semantics."""
    source = Path(path)
    suffix = source.suffix.lower()
    if input_adapter == "hrvatin_geo_csv":
        if labels_path is None or not label_column:
            raise ValueError("hrvatin_geo_csv requires labels_path and label_column")
        return load_hrvatin_geo_csv(source, labels_path, label_column=label_column)
    if suffix in {".h5", ".h5ad"}:
        loaded = load_h5_matrix(source, matrix_key=matrix_key, label_key=label_key)
    elif suffix == ".npz":
        loaded = load_npz(source)
    elif suffix == ".npy":
        values = np.load(source, allow_pickle=False)
        if values.ndim != 2:
            raise ValueError(f"NPY matrix must be 2-D: {source}")
        loaded = LoadedNPZ(values, None, {"path": str(source.resolve()), "format": "npy",
            "n_samples_original": int(values.shape[0]), "n_features_original": int(values.shape[1]),
            "sparse_storage": False, "labels_loaded_by_outer_runner": False})
    elif suffix in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
        values = np.loadtxt(source, delimiter=delimiter, ndmin=2)
        loaded = LoadedNPZ(values, None, {"path": str(source.resolve()), "format": suffix[1:],
            "n_samples_original": int(values.shape[0]), "n_features_original": int(values.shape[1]),
            "sparse_storage": False, "labels_loaded_by_outer_runner": False})
    else:
        raise ValueError(f"unsupported matrix format: {source}")
    labels = loaded.labels
    if labels_path is not None and label_key is None:
        labels = np.loadtxt(labels_path, dtype=str, ndmin=1).reshape(-1)
    if labels is not None and len(labels) != loaded.X.shape[0]:
        raise ValueError("label count does not match input rows")
    profile = dict(loaded.profile)
    profile["labels_path"] = str(Path(labels_path).resolve()) if labels_path else None
    profile["labels_loaded_by_outer_runner"] = labels is not None
    return LoadedNPZ(loaded.X, labels, profile)

def load_npz_matrix_only(path: str | Path) -> LoadedNPZ:
    """Load only the feature matrix from an NPZ without touching benchmark labels."""
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError(f"V0_RG selected-dataset adapter requires NPZ input: {source}")
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
    import anndata as ad
    import scanpy as sc

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
    """Apply a fixed, label-free V0_RG input protocol."""
    if input_protocol not in INPUT_PROTOCOLS:
        raise ValueError(f"input_protocol must be one of {sorted(INPUT_PROTOCOLS)}")
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix")
    key = str(dataset_name).strip().lower()
    source = _csr_float32(X)
    if input_protocol == "rg_native" and source.data.size and float(np.min(source.data)) < 0.0:
        raise ValueError("V0_RG selected sparse/count inputs must be non-negative before scaling")
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
        import scanpy as sc
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
