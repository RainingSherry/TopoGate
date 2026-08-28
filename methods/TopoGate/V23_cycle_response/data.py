from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


SUPPORTED_PROTOCOLS = frozenset({"clubench_bridge", "shared_text", "scRNA_count"})


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SemanticPreprocessor:
    input_protocol: str
    selected_feature_indices: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    n_features_original: int

    def transform(self, semantic: np.ndarray) -> np.ndarray:
        semantic = np.asarray(semantic, dtype=np.float32)
        if semantic.ndim != 2 or semantic.shape[1] != self.mean.size:
            raise ValueError("semantic matrix does not match fitted preprocessor")
        return np.ascontiguousarray((semantic - self.mean[None, :]) / self.scale[None, :], dtype=np.float32)

    def inverse_transform(self, model_view: np.ndarray) -> np.ndarray:
        model_view = np.asarray(model_view, dtype=np.float32)
        if model_view.ndim != 2 or model_view.shape[1] != self.mean.size:
            raise ValueError("model matrix does not match fitted preprocessor")
        return np.ascontiguousarray(model_view * self.scale[None, :] + self.mean[None, :], dtype=np.float32)


@dataclass(frozen=True)
class PreparedSemanticInput:
    semantic: np.ndarray
    model: np.ndarray
    preprocessor: SemanticPreprocessor
    profile: dict[str, Any]


def _read_matrix_payload(payload: Any) -> np.ndarray | sp.csr_matrix:
    sparse_keys = {"data", "indices", "indptr", "shape"}
    if sparse_keys.issubset(payload.files):
        shape = tuple(int(v) for v in np.asarray(payload["shape"]).reshape(-1))
        return sp.csr_matrix(
            (
                np.asarray(payload["data"], dtype=np.float32),
                np.asarray(payload["indices"], dtype=np.int64),
                np.asarray(payload["indptr"], dtype=np.int64),
            ),
            shape=shape,
            dtype=np.float32,
        )
    for key in ("X", "x", "features"):
        if key in payload.files:
            return np.asarray(payload[key], dtype=np.float32)
    raise ValueError("matrix-only NPZ has no X/x/features or CSR fields")


def load_matrix_only(path: str | Path) -> np.ndarray | sp.csr_matrix:
    """Load only X. This interface never reads or returns labels or K."""

    with np.load(Path(path), allow_pickle=False) as payload:
        matrix = _read_matrix_payload(payload)
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError(f"matrix must be non-empty and 2D, got {matrix.shape}")
    return matrix


def load_labels_outer(path: str | Path) -> np.ndarray:
    labels = np.asarray(np.load(Path(path), allow_pickle=False)).reshape(-1)
    if labels.size == 0:
        raise ValueError("outer label array is empty")
    return labels


def _finite_csr(matrix: np.ndarray | sp.spmatrix) -> sp.csr_matrix:
    result = sp.csr_matrix(matrix, dtype=np.float32)
    result.data = np.nan_to_num(result.data, nan=0.0, posinf=0.0, neginf=0.0)
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _variance_subset(matrix: sp.csr_matrix, feature_cap: int) -> tuple[sp.csr_matrix, np.ndarray]:
    n_features = int(matrix.shape[1])
    if n_features <= feature_cap:
        return matrix, np.arange(n_features, dtype=np.int64)
    mean = np.asarray(matrix.mean(axis=0)).reshape(-1).astype(np.float64)
    mean_square = np.asarray(matrix.multiply(matrix).mean(axis=0)).reshape(-1).astype(np.float64)
    variance = np.nan_to_num(mean_square - mean * mean, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    selected = np.argpartition(-variance, feature_cap - 1)[:feature_cap]
    selected = selected[np.argsort(-variance[selected], kind="stable")].astype(np.int64, copy=False)
    return matrix[:, selected].tocsr(), selected


def fit_semantic_preprocessor(
    matrix: np.ndarray | sp.spmatrix,
    *,
    input_protocol: str,
    feature_cap: int,
) -> PreparedSemanticInput:
    if input_protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported input protocol: {input_protocol}")
    source = _finite_csr(matrix)
    selected, indices = _variance_subset(source, int(feature_cap))
    semantic_sparse = selected.copy()
    if input_protocol == "scRNA_count":
        semantic_sparse.data = np.log1p(np.maximum(semantic_sparse.data, 0.0)).astype(np.float32, copy=False)
    semantic = semantic_sparse.toarray().astype(np.float32, copy=False)
    mean = semantic.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = semantic.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
    preprocessor = SemanticPreprocessor(
        input_protocol=input_protocol,
        selected_feature_indices=indices,
        mean=mean,
        scale=scale,
        n_features_original=int(source.shape[1]),
    )
    model = preprocessor.transform(semantic)
    profile = {
        "input_protocol": input_protocol,
        "semantic_space": (
            "selected_log1p_expression_before_mean_centering"
            if input_protocol == "scRNA_count"
            else "selected_feature_values_before_mean_centering"
        ),
        "n_samples": int(source.shape[0]),
        "n_features_original": int(source.shape[1]),
        "n_features_selected": int(semantic.shape[1]),
        "feature_cap": int(feature_cap),
        "feature_selection": "top_variance_label_free" if source.shape[1] > feature_cap else "none",
        "semantic_zero_fraction": float(np.mean(semantic == 0.0)),
        "model_zero_fraction": float(np.mean(model == 0.0)),
        "corruption_space": "pre_centered_semantic",
        "effective_mask_space": "pre_centered_semantic",
        "zero_corruption_semantics": "set_semantic_coordinate_to_zero",
        "labels_accessible": False,
        "K_accessible": False,
    }
    return PreparedSemanticInput(semantic=semantic, model=model, preprocessor=preprocessor, profile=profile)


def apply_semantic_preprocessor(
    matrix: np.ndarray | sp.spmatrix,
    preprocessor: SemanticPreprocessor,
) -> PreparedSemanticInput:
    source = _finite_csr(matrix)
    if source.shape[1] != preprocessor.n_features_original:
        raise ValueError("source feature count differs from fitted preprocessor")
    selected = source[:, preprocessor.selected_feature_indices].tocsr()
    if preprocessor.input_protocol == "scRNA_count":
        selected.data = np.log1p(np.maximum(selected.data, 0.0)).astype(np.float32, copy=False)
    semantic = selected.toarray().astype(np.float32, copy=False)
    model = preprocessor.transform(semantic)
    return PreparedSemanticInput(
        semantic=semantic,
        model=model,
        preprocessor=preprocessor,
        profile={
            "input_protocol": preprocessor.input_protocol,
            "n_samples": int(source.shape[0]),
            "n_features_original": int(source.shape[1]),
            "n_features_selected": int(semantic.shape[1]),
            "corruption_space": "pre_centered_semantic",
            "effective_mask_space": "pre_centered_semantic",
            "labels_accessible": False,
            "K_accessible": False,
        },
    )


def save_preprocessor(path: str | Path, preprocessor: SemanticPreprocessor) -> None:
    np.savez_compressed(
        Path(path),
        input_protocol=np.asarray(preprocessor.input_protocol),
        selected_feature_indices=preprocessor.selected_feature_indices,
        mean=preprocessor.mean,
        scale=preprocessor.scale,
        n_features_original=np.asarray(preprocessor.n_features_original, dtype=np.int64),
    )


def load_preprocessor(path: str | Path) -> SemanticPreprocessor:
    with np.load(Path(path), allow_pickle=False) as payload:
        return SemanticPreprocessor(
            input_protocol=str(np.asarray(payload["input_protocol"]).item()),
            selected_feature_indices=np.asarray(payload["selected_feature_indices"], dtype=np.int64),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            n_features_original=int(np.asarray(payload["n_features_original"]).item()),
        )
