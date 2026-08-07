from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.random_projection import SparseRandomProjection


@dataclass(frozen=True)
class PreparedInput:
    matrix: sp.csr_matrix
    profile: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1])


@dataclass(frozen=True)
class ProjectionViews:
    values: tuple[np.ndarray, ...]
    profile: dict[str, Any]


def load_sparse_npz(path: str) -> sp.csr_matrix:
    """Load a SciPy sparse NPZ or a CSR-field NPZ without dense materialization."""
    try:
        return sp.load_npz(path).tocsr()
    except ValueError as standard_error:
        with np.load(path, allow_pickle=False) as payload:
            required = {"data", "indices", "indptr", "shape"}
            if not required.issubset(payload.files):
                raise standard_error
            shape = tuple(int(value) for value in np.asarray(payload["shape"]).reshape(-1))
            matrix = sp.csr_matrix(
                (
                    np.asarray(payload["data"]),
                    np.asarray(payload["indices"], dtype=np.int64),
                    np.asarray(payload["indptr"], dtype=np.int64),
                ),
                shape=shape,
            )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix.tocsr()


def _resolve_input_mode(matrix: sp.csr_matrix, requested: str) -> str:
    if requested != "auto":
        return requested
    data = matrix.data
    nonnegative = bool(data.size == 0 or np.min(data) >= 0.0)
    if nonnegative:
        return "nonnegative"
    return "continuous"


def prepare_input(X: np.ndarray | sp.spmatrix, *, input_mode: str = "auto") -> PreparedInput:
    """Apply a sparse-safe transform selected only from input semantics."""
    matrix = sp.csr_matrix(X, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D matrix")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.data.size and not np.isfinite(matrix.data).all():
        raise ValueError("X contains non-finite values")
    mode = _resolve_input_mode(matrix, input_mode)
    if mode in {"count", "nonnegative"} and matrix.data.size and np.min(matrix.data) < 0.0:
        raise ValueError(f"{mode} input must be non-negative")
    if mode == "count" and matrix.data.size:
        if not np.allclose(matrix.data, np.rint(matrix.data), atol=1e-6, rtol=0.0):
            raise ValueError("count input must contain integer-valued observations")
        matrix = matrix.copy()
        matrix.data = np.log1p(matrix.data).astype(np.float32, copy=False)
    matrix = normalize(matrix, norm="l2", axis=1, copy=True).tocsr().astype(np.float32)
    row_nnz = np.diff(matrix.indptr)
    profile = {
        "input_mode_requested": input_mode,
        "input_mode_resolved": mode,
        "n_samples": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "zero_fraction": float(1.0 - matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1])),
        "median_row_nnz": float(np.median(row_nnz)) if row_nnz.size else 0.0,
        "empty_row_fraction": float(np.mean(row_nnz == 0)) if row_nnz.size else 1.0,
        "transform": "log1p_then_row_l2" if mode == "count" else "row_l2",
        "storage": "csr",
    }
    return PreparedInput(matrix=matrix, profile=profile)


def build_projection_views(
    prepared: PreparedInput,
    *,
    n_views: int,
    projection_dim: int,
    density: str | float,
    seed: int,
) -> ProjectionViews:
    """Construct fixed sparse random projections; no label or cluster count is used."""
    n_components = min(int(projection_dim), prepared.n_features)
    views: list[np.ndarray] = []
    seeds: list[int] = []
    for view_index in range(int(n_views)):
        view_seed = int(seed) + 104729 * view_index
        projector = SparseRandomProjection(
            n_components=n_components,
            density=density,
            dense_output=True,
            random_state=view_seed,
        )
        projected = projector.fit_transform(prepared.matrix)
        projected = normalize(np.asarray(projected, dtype=np.float32), norm="l2", axis=1, copy=False)
        views.append(np.asarray(projected, dtype=np.float32))
        seeds.append(view_seed)
    return ProjectionViews(
        values=tuple(views),
        profile={
            "kind": "sparse_random_projection",
            "n_views": int(n_views),
            "n_components": int(n_components),
            "density": density,
            "seeds": seeds,
            "full_pairwise_matrix_materialized": False,
        },
    )
