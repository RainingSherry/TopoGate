from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any
import zipfile

import numpy as np
import scipy.sparse as sp


class TheoryDomainError(ValueError):
    """Raised when an input does not carry the V16 count-domain certificate."""

    def __init__(self, profile: dict[str, Any]):
        self.profile = profile
        super().__init__("input is outside the V16 sparse-count theory domain")


@dataclass
class PreparedCounts:
    counts: sp.csr_matrix
    profile: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.counts.shape[1])

    def rows(self, indices: np.ndarray) -> np.ndarray:
        """Return a mini-batch in the dense representation consumed by Stage A."""
        values = self.counts[np.asarray(indices, dtype=np.int64)].toarray().astype(np.float32)
        values = np.log1p(values)
        row_sum = values.sum(axis=1, keepdims=True)
        return values / np.clip(row_sum, 1e-8, None)

    def raw_rows(self, indices: np.ndarray) -> np.ndarray:
        return self.counts[np.asarray(indices, dtype=np.int64)].toarray().astype(np.float32)


def _uncompressed_npy_layout(path: str | Any, member: str) -> tuple[tuple[int, ...], np.dtype, int] | None:
    """Return shape, dtype and payload offset for an uncompressed NPZ member.

    The project datasets store ``x.npy`` uncompressed inside NPZ archives.  A
    memmap over that payload lets the loader convert bounded row blocks to CSR
    without first creating a full dense ``n x d`` array.
    """
    archive_path = str(path)
    with zipfile.ZipFile(archive_path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError:
            return None
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        with open(archive_path, "rb") as raw:
            raw.seek(info.header_offset)
            header = raw.read(30)
            if len(header) != 30:
                return None
            fields = struct.unpack("<IHHHHHIIIHH", header)
            filename_length, extra_length = fields[-2:]
            payload_offset = info.header_offset + 30 + filename_length + extra_length
            raw.seek(payload_offset)
            try:
                version = np.lib.format.read_magic(raw)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(raw)
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(raw)
                else:
                    return None
            except (ValueError, OSError):
                return None
            if fortran_order or dtype.hasobject or len(shape) != 2:
                return None
            return tuple(int(value) for value in shape), np.dtype(dtype), int(raw.tell())


def load_npz_matrix(
    path: str | Any,
    *,
    member: str = "x",
    chunk_rows: int = 512,
) -> tuple[np.ndarray | sp.csr_matrix, str]:
    """Load an NPZ matrix with a sparse-memory-aware path.

    Uncompressed numeric NPY members are read through a file-backed memmap and
    converted to CSR one row block at a time.  Compressed or unsupported
    members fall back to a dense array and are explicitly labelled
    ``dense_npz`` so the V16 domain certificate can reject them before fitting.
    """
    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive")
    layout = _uncompressed_npy_layout(path, f"{member}.npy")
    if layout is not None:
        shape, dtype, payload_offset = layout
        mapped = np.memmap(
            str(path),
            mode="r",
            dtype=dtype,
            offset=payload_offset,
            shape=shape,
            order="C",
        )
        blocks: list[sp.csr_matrix] = []
        for start in range(0, shape[0], int(chunk_rows)):
            blocks.append(sp.csr_matrix(np.asarray(mapped[start : start + int(chunk_rows)])))
        matrix = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix(shape=shape, dtype=dtype)
        return matrix, "sparse_npz_chunked"
    with np.load(path, allow_pickle=False) as data:
        if member not in data.files:
            raise ValueError(f"NPZ does not contain member {member!r}: {path}")
        return np.asarray(data[member]), "dense_npz"


def _as_csr(X: np.ndarray | sp.spmatrix) -> tuple[sp.csr_matrix, str]:
    if sp.issparse(X):
        matrix = sp.csr_matrix(X, dtype=np.float64, copy=True)
        storage = "sparse"
    else:
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError(f"X must be a 2D matrix, got {values.shape}")
        matrix = sp.csr_matrix(values.astype(np.float64, copy=False))
        storage = "dense"
    if matrix.ndim != 2:
        raise ValueError("X must be a 2D matrix")
    if matrix.data.size and not np.isfinite(matrix.data).all():
        raise ValueError("X contains non-finite values")
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix, storage


def _recognise_counts(matrix: sp.csr_matrix) -> tuple[str, np.ndarray]:
    values = np.asarray(matrix.data, dtype=np.float64)
    if values.size == 0:
        return "raw_integer", np.zeros(0, dtype=np.int64)
    if np.min(values) < 0.0:
        return "unsupported", np.empty(0, dtype=np.int64)
    if np.allclose(values, np.rint(values), atol=1e-6, rtol=0.0):
        return "raw_integer", np.rint(values).astype(np.int64)
    bounded = values <= 25.0
    if np.all(bounded):
        recovered = np.rint(np.expm1(values))
        reconstructed = np.log1p(recovered)
        if np.allclose(values, reconstructed, atol=2e-5, rtol=2e-5):
            return "log1p_integer", recovered.astype(np.int64)
    return "unsupported", np.empty(0, dtype=np.int64)


def assess_count_domain(
    X: np.ndarray | sp.spmatrix,
    *,
    min_feature_dim: int = 2000,
    min_zero_fraction: float = 0.80,
    min_median_nnz: float = 5.0,
    max_empty_fraction: float = 0.10,
    storage_override: str | None = None,
    require_sparse_input: bool = True,
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    matrix, storage = _as_csr(X)
    storage_label = str(storage_override or storage)
    semantics, recovered = _recognise_counts(matrix)
    n, d = matrix.shape
    nnz = np.diff(matrix.indptr).astype(np.int64)
    size = max(1, int(n) * int(d))
    profile: dict[str, Any] = {
        "n": int(n),
        "d": int(d),
        "input_storage": storage_label,
        "sparse_memory_certificate": "candidate" if not storage_label.startswith("dense") else "not_supported",
        "raw_zero_fraction": float(1.0 - matrix.nnz / size),
        "raw_nnz": int(matrix.nnz),
        "nnz_median": float(np.median(nnz)) if nnz.size else 0.0,
        "nnz_p05": float(np.quantile(nnz, 0.05)) if nnz.size else 0.0,
        "nnz_p95": float(np.quantile(nnz, 0.95)) if nnz.size else 0.0,
        "empty_fraction": float(np.mean(nnz == 0)) if nnz.size else 1.0,
        "count_semantics": semantics,
        "theory_domain": "candidate",
    }
    reasons: list[str] = []
    if require_sparse_input and storage_label.startswith("dense"):
        reasons.append("dense_input_not_supported")
    if semantics == "unsupported":
        reasons.append("count_or_log1p_count_not_identifiable")
    if d < int(min_feature_dim):
        reasons.append("feature_dim_below_threshold")
    if profile["raw_zero_fraction"] < float(min_zero_fraction):
        reasons.append("zero_fraction_below_threshold")
    if profile["nnz_median"] < float(min_median_nnz):
        reasons.append("median_nnz_below_threshold")
    if profile["empty_fraction"] > float(max_empty_fraction):
        reasons.append("empty_rows_above_threshold")
    profile["domain_reasons"] = reasons
    if reasons:
        profile["theory_domain"] = "theory_domain_not_supported"
    if semantics == "unsupported":
        return matrix, profile
    return sp.csr_matrix((recovered, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape), profile


def prepare_counts(
    X: np.ndarray | sp.spmatrix,
    *,
    enforce_domain: bool = True,
    min_feature_dim: int = 2000,
    min_zero_fraction: float = 0.80,
    min_median_nnz: float = 5.0,
    max_empty_fraction: float = 0.10,
    input_storage: str | None = None,
    require_sparse_input: bool = True,
) -> PreparedCounts:
    counts, profile = assess_count_domain(
        X,
        min_feature_dim=min_feature_dim,
        min_zero_fraction=min_zero_fraction,
        min_median_nnz=min_median_nnz,
        max_empty_fraction=max_empty_fraction,
        storage_override=input_storage,
        require_sparse_input=require_sparse_input,
    )
    if enforce_domain and profile["theory_domain"] != "candidate":
        raise TheoryDomainError(profile)
    profile = dict(profile)
    profile["representation"] = "csr_integer_counts"
    return PreparedCounts(counts=counts.astype(np.int64), profile=profile)


def split_counts(
    counts: sp.spmatrix,
    fraction: float = 0.5,
    seed: int = 42,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Poisson/binomial count splitting with exact per-entry conservation."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    source = sp.csr_matrix(counts, dtype=np.int64, copy=True)
    rng = np.random.default_rng(int(seed))
    first_values = rng.binomial(source.data, float(fraction)).astype(np.int64)
    second_values = source.data - first_values
    first = sp.csr_matrix((first_values, source.indices.copy(), source.indptr.copy()), shape=source.shape)
    second = sp.csr_matrix((second_values, source.indices.copy(), source.indptr.copy()), shape=source.shape)
    first.eliminate_zeros()
    second.eliminate_zeros()
    first.sort_indices()
    second.sort_indices()
    return first, second


def repeated_splits(
    counts: sp.spmatrix,
    fraction: float,
    repeats: int,
    seed: int,
) -> list[tuple[sp.csr_matrix, sp.csr_matrix]]:
    return [split_counts(counts, fraction=fraction, seed=int(seed) + r) for r in range(int(repeats))]
