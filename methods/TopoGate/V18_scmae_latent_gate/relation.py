from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from .graph import CandidateGraph


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values)
    return np.sign(values) * np.maximum(np.abs(values) - float(threshold), 0.0)


def initialize_relation_fista(
    views: tuple[np.ndarray, ...] | list[np.ndarray],
    graph: CandidateGraph,
    *,
    lambda_l1: float,
    lambda_l2: float,
    max_iter: int,
    tolerance: float,
) -> np.ndarray:
    """Initialize W with candidate-restricted multi-view elastic self-expression."""
    arrays = []
    for view in views:
        array = np.asarray(view, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("latent views must be two-dimensional")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        arrays.append(array / np.clip(norms, 1e-12, None))
    if not arrays:
        raise ValueError("at least one latent view is required")
    n, width = graph.indices.shape
    result = np.zeros((n, width), dtype=np.float32)
    for i in range(n):
        slots = np.flatnonzero(graph.valid[i])
        if slots.size == 0:
            continue
        donors = graph.indices[i, slots]
        gram = np.zeros((slots.size, slots.size), dtype=np.float64)
        target = np.zeros(slots.size, dtype=np.float64)
        for view in arrays:
            donor_view = view[donors]
            gram += donor_view @ donor_view.T / len(arrays)
            target += donor_view @ view[i] / len(arrays)
        gram.flat[:: gram.shape[0] + 1] += float(lambda_l2)
        lipschitz = float(np.linalg.eigvalsh(gram).max())
        step = 1.0 / max(lipschitz, 1e-8)
        coefficient = np.zeros(slots.size, dtype=np.float64)
        extrapolated = coefficient.copy()
        momentum = 1.0
        for _ in range(int(max_iter)):
            previous = coefficient.copy()
            gradient = gram @ extrapolated - target
            coefficient = soft_threshold(extrapolated - step * gradient, step * float(lambda_l1))
            if np.linalg.norm(coefficient - previous) <= float(tolerance) * (1.0 + np.linalg.norm(coefficient)):
                break
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            extrapolated = coefficient + ((momentum - 1.0) / next_momentum) * (coefficient - previous)
            momentum = next_momentum
        result[i, slots] = coefficient.astype(np.float32)
    return result


class EdgeGate(nn.Module):
    """Linear, interpretable HardConcrete edge gate."""

    def __init__(self, n_features: int, *, init_bias: float = -2.0,
                 gamma: float = -0.1, zeta: float = 1.1) -> None:
        super().__init__()
        self.scorer = nn.Linear(int(n_features), 1)
        nn.init.zeros_(self.scorer.weight)
        nn.init.constant_(self.scorer.bias, float(init_bias))
        self.gamma = float(gamma)
        self.zeta = float(zeta)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.scorer(features).squeeze(-1)

    def expected_open_probability(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        stretch_threshold = float(temperature) * np.log(-self.gamma / self.zeta)
        return torch.sigmoid(logits - float(stretch_threshold))

    def forward(
        self,
        features: torch.Tensor,
        *,
        temperature: float,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.logits(features)
        expected = self.expected_open_probability(logits, temperature)
        if sample:
            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            concrete = torch.sigmoid((torch.log(uniform) - torch.log1p(-uniform) + logits) / float(temperature))
            stretched = concrete * (self.zeta - self.gamma) + self.gamma
            gate = stretched.clamp(0.0, 1.0)
        else:
            gate = expected.mul(self.zeta - self.gamma).add(self.gamma).clamp(0.0, 1.0)
            gate = (gate >= 0.5).to(dtype=features.dtype)
        return gate, expected, logits


class SparseRelation(nn.Module):
    """Candidate-slot relation W with C exactly equal to G elementwise W."""

    def __init__(self, graph: CandidateGraph, initial_w: np.ndarray) -> None:
        super().__init__()
        if initial_w.shape != graph.indices.shape:
            raise ValueError("initial W must have the candidate slot shape")
        self.register_buffer("indices", torch.as_tensor(graph.indices, dtype=torch.long))
        self.register_buffer("valid", torch.as_tensor(graph.valid, dtype=torch.bool))
        self.W = nn.Parameter(torch.as_tensor(initial_w, dtype=torch.float32).clone())

    def coefficients(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape != self.W.shape:
            raise ValueError("gate and W must have identical candidate slot shapes")
        return gate * self.W * self.valid.to(dtype=self.W.dtype)

    def reconstruct(self, H: torch.Tensor, coefficients: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        if rows is None:
            rows = torch.arange(H.shape[0], device=H.device)
        donor_h = H[self.indices[rows].clamp_min(0)]
        return torch.sum(coefficients[rows].unsqueeze(-1) * donor_h, dim=1)

    def sparse_coefficients(self, coefficients: np.ndarray, *, epsilon: float = 1e-8) -> sp.csr_matrix:
        values = np.asarray(coefficients, dtype=np.float32)
        keep = self.valid.detach().cpu().numpy() & (np.abs(values) > float(epsilon))
        rows, slots = np.where(keep)
        cols = self.indices.detach().cpu().numpy()[rows, slots]
        matrix = sp.csr_matrix((values[rows, slots], (rows, cols)), shape=(values.shape[0], values.shape[0]))
        matrix.setdiag(0.0)
        matrix.eliminate_zeros()
        return matrix


def group_huber(residual: torch.Tensor, delta: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(residual, dim=-1)
    quadratic = 0.5 * norm.square()
    linear = float(delta) * norm - 0.5 * float(delta) ** 2
    return torch.where(norm <= float(delta), quadratic, linear).mean()


def relation_profile(graph: CandidateGraph, coefficients: np.ndarray, *, epsilon: float) -> dict[str, Any]:
    valid = graph.valid
    active = valid & (np.abs(coefficients) > float(epsilon))
    row_nnz = active.sum(axis=1)
    return {
        "candidate_edges": int(valid.sum()),
        "coefficient_nnz": int(active.sum()),
        "edge_retention_rate": float(active.sum() / max(1, valid.sum())),
        "zero_outgoing_row_fraction": float(np.mean(row_nnz == 0)) if row_nnz.size else 1.0,
        "diag_is_zero": bool(not np.any(np.where(valid, graph.indices, -1) == np.arange(graph.n_nodes)[:, None])),
        "optimizer": "candidate_restricted_fista_initialization_plus_proximal_updates",
    }


def affinity_from_coefficients(coefficients: sp.spmatrix) -> sp.csr_matrix:
    matrix = sp.csr_matrix(coefficients, dtype=np.float32)
    absolute = matrix.copy()
    absolute.data = np.abs(absolute.data)
    affinity = (absolute + absolute.T).tocsr()
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    return affinity
