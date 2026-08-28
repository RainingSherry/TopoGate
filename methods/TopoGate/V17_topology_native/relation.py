from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .candidate import CandidateSet


@dataclass(frozen=True)
class RelationResult:
    coefficients: sp.csr_matrix
    candidate_coefficients: np.ndarray
    iterations: np.ndarray
    profile: dict[str, Any]


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Proximal L1 gate; values inside the threshold become exact zeros."""
    array = np.asarray(values)
    return np.sign(array) * np.maximum(np.abs(array) - float(threshold), 0.0)


def _group_huber_residual(residual: np.ndarray, delta: float) -> tuple[np.ndarray, float]:
    """Return the gradient and value of the group-Huber envelope."""
    norm = float(np.linalg.norm(residual))
    if norm <= float(delta):
        return residual, 0.5 * norm * norm
    scale = float(delta) / max(norm, np.finfo(np.float64).eps)
    return residual * scale, float(delta) * norm - 0.5 * float(delta) ** 2


def _row_objective(
    coefficient: np.ndarray,
    targets: list[np.ndarray],
    donors: list[np.ndarray],
    *,
    lambda_l1: float,
    lambda_l2: float,
    lambda_outlier: float,
) -> float:
    robust = 0.0
    for target, donor in zip(targets, donors, strict=True):
        _, value = _group_huber_residual(target - coefficient @ donor, lambda_outlier)
        robust += value
    robust /= max(1, len(targets))
    return float(
        robust
        + float(lambda_l1) * np.abs(coefficient).sum()
        + 0.5 * float(lambda_l2) * np.square(coefficient).sum()
    )


def solve_candidate_self_expression(
    views: tuple[np.ndarray, ...] | list[np.ndarray],
    candidates: CandidateSet,
    *,
    lambda_l1: float,
    lambda_l2: float,
    lambda_outlier: float,
    max_iter: int,
    tolerance: float,
    coefficient_epsilon: float,
) -> RelationResult:
    """Solve shared robust self-expression on the candidate support only.

    Samples are rows, so the implementation uses H ~= C H. This is the
    transpose of the column-sample notation H ~= H C used in the V17 report.
    """
    if not views:
        raise ValueError("at least one projection view is required")
    arrays = [np.asarray(view, dtype=np.float64) for view in views]
    n = candidates.n_nodes
    if any(array.ndim != 2 or array.shape[0] != n for array in arrays):
        raise ValueError("projection views and candidate rows must agree")
    slot_coefficients = np.zeros(candidates.indices.shape, dtype=np.float32)
    iterations = np.zeros(n, dtype=np.int32)
    converged = np.zeros(n, dtype=bool)
    objectives = np.zeros(n, dtype=np.float64)
    for anchor in range(n):
        positions = np.flatnonzero(candidates.valid[anchor])
        if positions.size == 0:
            converged[anchor] = True
            continue
        donor_indices = candidates.indices[anchor, positions]
        if np.any(donor_indices == anchor):
            raise ValueError("candidate support must exclude self edges")
        targets = [array[anchor] for array in arrays]
        donor_views = [array[donor_indices] for array in arrays]
        coefficient = np.zeros(positions.size, dtype=np.float64)
        extrapolated = coefficient.copy()
        momentum = 1.0
        lipschitz = float(lambda_l2)
        for donor in donor_views:
            lipschitz += float(np.linalg.norm(donor, ord=2) ** 2) / len(donor_views)
        step = 1.0 / max(lipschitz, np.finfo(np.float64).eps)
        for iteration in range(1, int(max_iter) + 1):
            gradient = float(lambda_l2) * extrapolated
            for target, donor in zip(targets, donor_views, strict=True):
                residual = target - extrapolated @ donor
                robust_residual, _ = _group_huber_residual(residual, lambda_outlier)
                gradient -= donor @ robust_residual / len(donor_views)
            updated = soft_threshold(extrapolated - step * gradient, step * float(lambda_l1))
            change = float(np.linalg.norm(updated - extrapolated))
            previous = coefficient
            coefficient = updated
            iterations[anchor] = iteration
            if change <= float(tolerance) * (1.0 + float(np.linalg.norm(coefficient))):
                converged[anchor] = True
                break
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            extrapolated = coefficient + ((momentum - 1.0) / next_momentum) * (coefficient - previous)
            momentum = next_momentum
        coefficient[np.abs(coefficient) <= float(coefficient_epsilon)] = 0.0
        slot_coefficients[anchor, positions] = coefficient.astype(np.float32)
        objectives[anchor] = _row_objective(
            coefficient,
            targets,
            donor_views,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_outlier=lambda_outlier,
        )
    rows, positions = np.where(candidates.valid & (slot_coefficients != 0.0))
    cols = candidates.indices[rows, positions]
    data = slot_coefficients[rows, positions]
    coefficients = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    coefficients.sum_duplicates()
    coefficients.setdiag(0.0)
    coefficients.eliminate_zeros()
    valid_edges = int(candidates.valid.sum())
    row_nnz = np.diff(coefficients.indptr)
    profile = {
        "objective": "multi_view_group_huber_elastic_sparse_self_expression",
        "optimizer": "fista_proximal_gradient",
        "sample_convention": "rows_H_approximately_C_times_H",
        "candidate_edges": valid_edges,
        "coefficient_nnz": int(coefficients.nnz),
        "edge_retention_rate": float(coefficients.nnz / max(1, valid_edges)),
        "exact_zero_gate_rate": float(1.0 - coefficients.nnz / max(1, valid_edges)),
        "zero_outgoing_row_fraction": float(np.mean(row_nnz == 0)) if n else 1.0,
        "converged_row_fraction": float(np.mean(converged)) if n else 1.0,
        "mean_iterations": float(np.mean(iterations)) if n else 0.0,
        "mean_row_objective": float(np.mean(objectives)) if n else 0.0,
        "diag_is_zero": bool(np.allclose(coefficients.diagonal(), 0.0)),
    }
    return RelationResult(coefficients, slot_coefficients, iterations, profile)


def affinity_from_coefficients(coefficients: sp.spmatrix) -> sp.csr_matrix:
    """Create the sole V17 affinity A = |C| + |C.T|."""
    coefficient = sp.csr_matrix(coefficients, dtype=np.float32)
    absolute = coefficient.copy()
    absolute.data = np.abs(absolute.data)
    affinity = (absolute + absolute.transpose()).tocsr()
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    return affinity
