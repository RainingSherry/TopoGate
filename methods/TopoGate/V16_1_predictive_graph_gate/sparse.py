from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any
import zipfile

import numpy as np
import scipy.sparse as sp


class TheoryDomainError(ValueError):
    """Raised when an input lacks the V16.1 count-domain certificate."""

    def __init__(self, profile: dict[str, Any]):
        self.profile = profile
        super().__init__("input is outside the V16.1 sparse-count theory domain")


KNOWN_COUNT_SEMANTICS: dict[str, tuple[str, str]] = {
    "Campbell": ("scRNA_count", "registered local scRNA count source"),
    "Mouse_retina": ("scRNA_count", "registered local scRNA count source"),
    "Baron Human": ("scRNA_count", "registered local scRNA count source"),
    "Quake_Smart-seq2_Lung": ("scRNA_count", "registered local scRNA count source"),
    "hrvatin": ("scRNA_count", "registered local scRNA count source"),
    "hrvatin_filtered": ("scRNA_count", "registered local scRNA count source"),
    "fbis.wc": ("word_count", "registered local word-count source"),
    "tr45.wc": ("word_count", "registered local word-count source"),
}

INPUT_POLICIES = {"strict_legacy", "expanded_count"}


def registered_count_semantics(dataset_name: str) -> tuple[str | None, str | None]:
    """Return the pre-registered source declaration for a dataset name."""
    return KNOWN_COUNT_SEMANTICS.get(str(dataset_name), (None, None))


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
        """Return normalized log-count rows for a bounded dense mini-batch."""
        values = self.counts[np.asarray(indices, dtype=np.int64)].toarray().astype(np.float32)
        values = np.log1p(values)
        row_sum = values.sum(axis=1, keepdims=True)
        return values / np.clip(row_sum, 1e-8, None)

    def raw_rows(self, indices: np.ndarray) -> np.ndarray:
        return self.counts[np.asarray(indices, dtype=np.int64)].toarray().astype(np.float32)


@dataclass(frozen=True)
class DenseNPZReference:
    path: str
    member: str
    shape: tuple[int, ...]
    dtype: str


def _uncompressed_npy_layout(path: str | Any, member: str) -> tuple[tuple[int, ...], np.dtype, int] | None:
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


def _compressed_npy_header(path: str | Any, member: str) -> tuple[tuple[int, ...], np.dtype] | None:
    with zipfile.ZipFile(str(path)) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError:
            return None
        if info.compress_type == zipfile.ZIP_STORED:
            return None
        with archive.open(info) as raw:
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
            return tuple(int(value) for value in shape), np.dtype(dtype)


def load_npz_matrix(
    path: str | Any,
    *,
    member: str = "x",
    chunk_rows: int = 512,
) -> tuple[np.ndarray | sp.csr_matrix | DenseNPZReference, str]:
    """Load an NPZ matrix without materializing an uncompressed full dense member."""
    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive")
    with np.load(path, allow_pickle=False) as data:
        csr_keys = {"data", "indices", "indptr", "shape"}
        if csr_keys.issubset(data.files):
            shape = tuple(int(value) for value in np.asarray(data["shape"]).reshape(-1))
            matrix = sp.csr_matrix(
                (
                    np.asarray(data["data"]),
                    np.asarray(data["indices"], dtype=np.int64),
                    np.asarray(data["indptr"], dtype=np.int64),
                ),
                shape=shape,
            )
            matrix.sort_indices()
            return matrix, "sparse_npz_csr"

    layout = _uncompressed_npy_layout(path, f"{member}.npy")
    if layout is not None:
        shape, dtype, payload_offset = layout
        mapped = np.memmap(
            str(path), mode="r", dtype=dtype, offset=payload_offset, shape=shape, order="C"
        )
        blocks: list[sp.csr_matrix] = []
        for start in range(0, shape[0], int(chunk_rows)):
            blocks.append(sp.csr_matrix(np.asarray(mapped[start : start + int(chunk_rows)])))
        matrix = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix(shape=shape, dtype=dtype)
        return matrix, "sparse_npz_chunked"
    compressed_header = _compressed_npy_header(path, f"{member}.npy")
    if compressed_header is not None:
        shape, dtype = compressed_header
        return DenseNPZReference(str(path), f"{member}.npy", shape, dtype.str), "dense_npz"
    with np.load(path, allow_pickle=False) as data:
        if member not in data.files:
            raise ValueError(f"NPZ does not contain member {member!r}: {path}")
        return np.asarray(data[member]), "dense_npz"


def dense_reference_profile(
    reference: DenseNPZReference,
    *,
    count_semantics: str | None,
    semantics_source: str | None,
    input_policy: str,
) -> dict[str, Any]:
    n, d = reference.shape
    return {
        "n": int(n),
        "d": int(d),
        "input_storage": "dense_npz",
        "sparse_memory_certificate": "not_supported",
        "raw_zero_fraction": None,
        "raw_nnz": None,
        "nnz_median": None,
        "nnz_p05": None,
        "nnz_p95": None,
        "empty_fraction": None,
        "count_semantics_declared": count_semantics,
        "count_semantics_source": semantics_source,
        "count_semantics": "unverified",
        "theory_domain": "theory_domain_not_supported",
        "input_policy": input_policy,
        "bonus_features": {
            "high_dim": d >= 2000,
            "high_zero_fraction": False,
            "enough_nonzero_features": False,
            "few_empty_rows": False,
        },
        "bonus_feature_count": 1 if d >= 2000 else 0,
        "domain_tier": "count_control",
        "domain_reasons": ["dense_input_not_supported"],
        "dense_reference": {
            "path": reference.path,
            "member": reference.member,
            "dtype": reference.dtype,
        },
    }


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


def _recognise_counts(
    matrix: sp.csr_matrix,
    declared_semantics: str | None,
) -> tuple[str, np.ndarray]:
    """Recognise count encoding only after a source declaration is supplied."""
    values = np.asarray(matrix.data, dtype=np.float64)
    if declared_semantics is None:
        return "unverified", np.empty(0, dtype=np.int64)
    declared = str(declared_semantics).strip().lower()
    if declared not in {"scrna_count", "word_count", "raw_count", "log1p_count"}:
        return "unsupported", np.empty(0, dtype=np.int64)
    if values.size == 0 or np.min(values) < 0.0:
        return "unsupported", np.empty(0, dtype=np.int64)
    if np.allclose(values, np.rint(values), atol=1e-6, rtol=0.0):
        recovered = np.rint(values).astype(np.int64)
        # A purely binary/one-hot matrix is not treated as a Poisson count view.
        if recovered.size and int(recovered.max()) <= 1:
            return "binary_or_one_hot", np.empty(0, dtype=np.int64)
        return "raw_integer", recovered
    bounded = values <= 25.0
    if np.all(bounded):
        recovered = np.rint(np.expm1(values))
        reconstructed = np.log1p(recovered)
        if np.allclose(values, reconstructed, atol=2e-5, rtol=2e-5) and recovered.max(initial=0) > 1:
            return "log1p_integer", recovered.astype(np.int64)
    return "unsupported", np.empty(0, dtype=np.int64)


def assess_count_domain(
    X: np.ndarray | sp.spmatrix,
    *,
    count_semantics: str | None = None,
    semantics_source: str | None = None,
    min_feature_dim: int = 2000,
    min_zero_fraction: float = 0.80,
    min_median_nnz: float = 5.0,
    max_empty_fraction: float = 0.10,
    storage_override: str | None = None,
    require_sparse_input: bool = True,
    input_policy: str = "strict_legacy",
) -> tuple[sp.csr_matrix, dict[str, Any]]:
    if input_policy not in INPUT_POLICIES:
        raise ValueError(f"unknown input_policy: {input_policy}")
    matrix, storage = _as_csr(X)
    storage_label = str(storage_override or storage)
    semantics, recovered = _recognise_counts(matrix, count_semantics)
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
        "count_semantics_declared": count_semantics,
        "count_semantics_source": semantics_source,
        "count_semantics": semantics,
        "theory_domain": "candidate",
    }
    bonus_flags = {
        "high_dim": d >= int(min_feature_dim),
        "high_zero_fraction": profile["raw_zero_fraction"] >= float(min_zero_fraction),
        "enough_nonzero_features": profile["nnz_median"] >= float(min_median_nnz),
        "few_empty_rows": profile["empty_fraction"] <= float(max_empty_fraction),
    }
    bonus_count = int(sum(bool(value) for value in bonus_flags.values()))
    if bonus_count == len(bonus_flags):
        domain_tier = "high_sparse_bonus"
    elif bonus_count >= 2:
        domain_tier = "sparse_count_control"
    else:
        domain_tier = "count_control"
    profile["input_policy"] = input_policy
    profile["bonus_features"] = bonus_flags
    profile["bonus_feature_count"] = bonus_count
    profile["domain_tier"] = domain_tier
    reasons: list[str] = []
    if require_sparse_input and storage_label.startswith("dense"):
        reasons.append("dense_input_not_supported")
    if semantics in {"unsupported", "unverified", "binary_or_one_hot"}:
        reasons.append("count_semantics_not_verified")
    if input_policy == "strict_legacy":
        if not bonus_flags["high_dim"]:
            reasons.append("feature_dim_below_threshold")
        if not bonus_flags["high_zero_fraction"]:
            reasons.append("zero_fraction_below_threshold")
        if not bonus_flags["enough_nonzero_features"]:
            reasons.append("median_nnz_below_threshold")
        if not bonus_flags["few_empty_rows"]:
            reasons.append("empty_rows_above_threshold")
    profile["domain_reasons"] = reasons
    if reasons:
        profile["theory_domain"] = "theory_domain_not_supported"
    if semantics not in {"raw_integer", "log1p_integer"}:
        return matrix, profile
    return sp.csr_matrix((recovered, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape), profile


def prepare_counts(
    X: np.ndarray | sp.spmatrix | DenseNPZReference,
    *,
    enforce_domain: bool = True,
    count_semantics: str | None = None,
    semantics_source: str | None = None,
    min_feature_dim: int = 2000,
    min_zero_fraction: float = 0.80,
    min_median_nnz: float = 5.0,
    max_empty_fraction: float = 0.10,
    input_storage: str | None = None,
    require_sparse_input: bool = True,
    input_policy: str = "strict_legacy",
) -> PreparedCounts:
    if isinstance(X, DenseNPZReference):
        profile = dense_reference_profile(
            X,
            count_semantics=count_semantics,
            semantics_source=semantics_source,
            input_policy=input_policy,
        )
        if enforce_domain:
            raise TheoryDomainError(profile)
        raise ValueError("dense NPZ reference cannot be prepared as model input")
    counts, profile = assess_count_domain(
        X,
        count_semantics=count_semantics,
        semantics_source=semantics_source,
        min_feature_dim=min_feature_dim,
        min_zero_fraction=min_zero_fraction,
        min_median_nnz=min_median_nnz,
        max_empty_fraction=max_empty_fraction,
        storage_override=input_storage,
        require_sparse_input=require_sparse_input,
        input_policy=input_policy,
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
    """Binomial split with exact per-entry count conservation."""
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


def summarize_split_views(
    split_views: list[tuple[sp.spmatrix, sp.spmatrix]],
) -> dict[str, Any]:
    """Summarize whether held-out count views contain observable signal."""
    if not split_views:
        return {
            "repeats": 0,
            "joint_nonempty_row_fraction": 0.0,
            "has_nonempty_heldout": False,
            "empty_row_fraction_a": 1.0,
            "empty_row_fraction_b": 1.0,
        }
    joint: list[float] = []
    empty_a: list[float] = []
    empty_b: list[float] = []
    for view_a, view_b in split_views:
        totals_a = np.asarray(sp.csr_matrix(view_a).sum(axis=1)).reshape(-1)
        totals_b = np.asarray(sp.csr_matrix(view_b).sum(axis=1)).reshape(-1)
        joint.append(float(np.mean((totals_a > 0) & (totals_b > 0))))
        empty_a.append(float(np.mean(totals_a == 0)))
        empty_b.append(float(np.mean(totals_b == 0)))
    return {
        "repeats": int(len(split_views)),
        "joint_nonempty_row_fraction": float(np.mean(joint)),
        "min_joint_nonempty_row_fraction": float(np.min(joint)),
        "empty_row_fraction_a": float(np.mean(empty_a)),
        "empty_row_fraction_b": float(np.mean(empty_b)),
        "has_nonempty_heldout": bool(any(value > 0.0 for value in joint)),
    }
