"""Input adapters and label-boundary utilities for V26."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from . import protocol


@dataclass(frozen=True)
class SparseDataset:
    spec: protocol.DatasetSpec
    x: sp.csr_matrix
    y: np.ndarray
    scale: np.ndarray
    source_metadata: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.x.shape[1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_npz(spec: protocol.DatasetSpec, source: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    with np.load(source, allow_pickle=False) as archive:
        if spec.source_type == "npz_csr":
            required = {"data", "indices", "indptr", "shape", spec.label_field}
            if not required.issubset(archive.files):
                raise ValueError(f"CSR archive fields mismatch for {spec.identifier}: {archive.files}")
            x = sp.csr_matrix(
                (
                    np.asarray(archive["data"], dtype=np.float32),
                    np.asarray(archive["indices"], dtype=np.int64),
                    np.asarray(archive["indptr"], dtype=np.int64),
                ),
                shape=tuple(int(value) for value in archive["shape"]),
            )
        else:
            if spec.matrix_field is None or spec.matrix_field not in archive.files:
                raise ValueError(f"matrix field missing for {spec.identifier}")
            x = sp.csr_matrix(np.asarray(archive[spec.matrix_field], dtype=np.float32))
        if spec.label_field not in archive.files:
            raise ValueError(f"label field missing for {spec.identifier}")
        y = np.asarray(archive[spec.label_field])
    return x, y


def _load_h5ad(spec: protocol.DatasetSpec, source: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise RuntimeError("V26 requires anndata for h5ad sources") from exc
    data = ad.read_h5ad(source, backed="r")
    try:
        matrix = data.X
        if hasattr(matrix, "to_memory"):
            matrix = matrix.to_memory()
        x = sp.csr_matrix(matrix, dtype=np.float32)
        if spec.label_field not in data.obs.columns:
            raise ValueError(f"label column {spec.label_field!r} missing for {spec.identifier}")
        y = np.asarray(data.obs[spec.label_field].astype(str))
    finally:
        data.file.close()
    return x, y


def _zero_preserving_scale(x: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    values = x.astype(np.float32).tocsr()
    # Some upstream CSR archives retain explicit stored zeros.  They are not
    # members of the mathematical non-zero support and SciPy may remove them
    # during diagonal multiplication, so normalize before testing the support
    # invariant rather than treating storage-layout cleanup as a corruption.
    values.eliminate_zeros()
    squared_mean = np.asarray(values.power(2).mean(axis=0)).ravel().astype(np.float64)
    scale = np.sqrt(squared_mean)
    scale[scale < 1e-6] = 1.0
    scaled = (values @ sp.diags(1.0 / scale.astype(np.float32), format="csr")).tocsr().astype(np.float32)
    scaled.eliminate_zeros()
    original_support = values.copy()
    original_support.data = np.ones_like(original_support.data, dtype=np.int8)
    scaled_support = scaled.copy()
    scaled_support.data = np.ones_like(scaled_support.data, dtype=np.int8)
    if (original_support != scaled_support).nnz != 0:
        raise AssertionError("zero-preserving scale changed the sparse support")
    return scaled, scale.astype(np.float32)


def value_only_profiles(x: sp.csr_matrix, quantiles: int) -> np.ndarray:
    """Represent only each row's non-zero value distribution.

    Feature coordinates and the number of active coordinates are deliberately
    excluded: each row is converted to a fixed set of value quantiles.  This
    is the complementary diagnostic to binary support clustering, not a sparse
    matrix embedding with zeros that would leak support information.
    """
    if quantiles < 2:
        raise ValueError("value-only profile needs at least two quantiles")
    matrix = x.tocsr()
    profile = np.zeros((matrix.shape[0], quantiles), dtype=np.float32)
    grid = np.linspace(0.0, 1.0, quantiles, dtype=np.float64)
    for row in range(matrix.shape[0]):
        values = matrix.data[matrix.indptr[row] : matrix.indptr[row + 1]]
        values = values[values != 0.0]
        if values.size:
            profile[row] = np.quantile(values, grid).astype(np.float32, copy=False)
    return profile


def load_dataset(identifier: str, *, hash_source: bool = False) -> SparseDataset:
    spec = protocol.DATASET_BY_ID[identifier]
    source = protocol.resolve_source(spec)
    if not source.exists():
        raise FileNotFoundError(f"missing V26 source: {source}")
    x, y_raw = _load_h5ad(spec, source) if spec.source_type == "h5ad" else _load_npz(spec, source)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2 or not np.isfinite(x.data).all():
        raise ValueError(f"invalid input matrix for {identifier}: {x.shape}")
    labels = LabelEncoder().fit_transform(np.asarray(y_raw).reshape(-1))
    if labels.shape[0] != x.shape[0] or np.unique(labels).size < 2:
        raise ValueError(f"invalid labels for {identifier}")
    scaled, scale = _zero_preserving_scale(x)
    metadata = {
        "source_path": str(source.resolve()),
        "source_size_bytes": int(source.stat().st_size),
        "source_mtime_ns": int(source.stat().st_mtime_ns),
        "source_sha256": sha256_file(source) if hash_source else None,
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "nnz": int(x.nnz),
        "sparsity": float(1.0 - x.nnz / max(1, x.shape[0] * x.shape[1])),
        "n_classes": int(np.unique(labels).size),
        "min_class_size": int(np.min(np.bincount(labels))),
        "zero_pattern_preserved": True,
        "labels_encoded_locally": True,
    }
    return SparseDataset(spec=spec, x=scaled, y=labels.astype(np.int64), scale=scale, source_metadata=metadata)
