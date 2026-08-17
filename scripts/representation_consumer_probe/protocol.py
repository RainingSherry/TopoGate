"""Frozen, label-free contracts for the representation-consumer probe.

This module is intentionally small and dependency-light.  It contains only the
S0 contract, graph/loss numerical primitives, and synthetic apparatus checks.
It does not train a model or compute a clustering result.
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


STRESS_DATASETS: tuple[str, ...] = (
    "cnae9",
    "Mouse_retina",
    "sms_spam_collection",
    "Baron Human",
    "Campbell",
    "hate_speech",
)
PILOT_SEEDS: tuple[int, ...] = (42, 123, 7)
HOLDOUT_SEEDS: tuple[int, ...] = (42, 123, 7, 3032, 3033)
LEGAL_GPU_POOL: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
FORBIDDEN_GPU_IDS: tuple[int, ...] = (0, 7)


@dataclass(frozen=True)
class ProtocolConfig:
    project_id: str = "representation_consumer_probe"
    protocol_id: str = "representation_consumer_probe_s0_v1"
    d0: int = 128
    svd_random_state: int = 0
    neighbor_k: int = 20
    retention_ratio: float = 0.4
    # This is a cap, not an exact per-row budget.  Every graph arm derives the
    # same feasible row budget b_i=min(budget_cap, positive_count_i).
    budget_cap: int = 8
    budget_sensitivity: tuple[int, ...] = (4, 8, 12)
    positive_cosine_rule: str = "cosine_gt_0"
    edge_weight_rule: str = "positive_cosine"
    pre_symmetrization: str = "remove_self_loops"
    post_symmetrization: str = "(W+W.T)/2_then_remove_self_loops"
    weight_dtype: str = "float32"
    laplacian: str = "L=D-W"
    normalized_laplacian: str = "L_sym=I-D^-1/2 W D^-1/2"
    zero_degree_inverse: str = "zero"
    spectral_solver: str = "scipy.sparse.linalg.eigsh"
    spectral_which: str = "SM"
    spectral_tol: float = 1e-6
    spectral_drop_first: bool = False
    spectral_zero_eigenvalue_tolerance: float = 1e-8
    gview_alpha: float = 1.0
    gview_eps: float = 1e-8
    loss_eps: float = 1e-8
    encoder_dims: tuple[int, ...] = (64, 32)
    decoder_dims: tuple[int, ...] = (64,)
    v_min: float = 1e-4
    lambda_orth: float = 1.0
    lambda_var: float = 0.1
    loss_reduction: str = "full_graph_sparse_sum_normalized"
    rec_target: str = "H0"
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 80
    batch_mode: str = "full_graph"
    k_source: str = "benchmark_oracle_from_y"
    labels_vector_used_in_fit: bool = False
    primary_readout: str = "clean_embedding_known_k_kmeans"

    @property
    def primary_budget(self) -> int:
        """Backward-compatible spelling; the protocol now calls this a cap."""
        return self.budget_cap

    def validate(self) -> None:
        if self.d0 != 128 or self.svd_random_state != 0:
            raise ValueError("S0 requires d0=128 and svd_random_state=0")
        if self.neighbor_k != 20 or self.budget_cap != 8:
            raise ValueError("S0 requires k=20 and budget cap=8")
        if self.budget_sensitivity != (4, 8, 12):
            raise ValueError("S0 budget sensitivity must be [4,8,12]")
        if not np.isclose(self.retention_ratio, 0.4):
            raise ValueError("S0 retention ratio must be 0.4")
        if self.spectral_drop_first:
            raise ValueError("S0 retains the zero eigenspace")
        if self.batch_mode != "full_graph":
            raise ValueError("graph losses must use full graph computation")
        if self.k_source != "benchmark_oracle_from_y":
            raise ValueError("K source contract mismatch")
        if self.labels_vector_used_in_fit:
            raise ValueError("labels vector cannot enter fit")


CONFIG = ProtocolConfig()
CONFIG.validate()


CONSUMER_CONTRACTS: dict[str, dict[str, bool]] = {
    "Spectral": {
        "K_used_in_representation": True,
        "K_used_in_readout": True,
        "labels_vector_used_in_fit": False,
    },
    "SimpleCut": {
        "K_used_in_representation": False,
        "K_used_in_readout": True,
        "labels_vector_used_in_fit": False,
    },
    "F": {
        "K_used_in_representation": False,
        "K_used_in_readout": True,
        "labels_vector_used_in_fit": False,
    },
}


class IncompleteComputeError(RuntimeError):
    """Raised when a frozen numerical consumer cannot complete."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def resolved_config() -> dict[str, Any]:
    CONFIG.validate()
    value = asdict(CONFIG)
    value.update(
        {
            "stress_datasets": list(STRESS_DATASETS),
            "pilot_seeds": list(PILOT_SEEDS),
            "holdout_seeds": list(HOLDOUT_SEEDS),
            "legal_gpu_pool": list(LEGAL_GPU_POOL),
            "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
            "labels_used_during_fit": False,
            "oracle_non_tuning": True,
            "semantic_fidelity_required": True,
            "consumer_contracts": CONSUMER_CONTRACTS,
        }
    )
    return jsonable(value)


def row_l2_normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-8)


def build_h0(matrix: sp.spmatrix | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Create the one-per-dataset common feature stem."""
    x = sp.csr_matrix(matrix, dtype=np.float32)
    n_samples, n_features = x.shape
    d_eff = min(CONFIG.d0, n_samples - 1, n_features - 1)
    if d_eff < 1:
        raise ValueError(f"cannot construct H0 for shape {x.shape}")
    svd = TruncatedSVD(
        n_components=d_eff,
        random_state=CONFIG.svd_random_state,
        n_iter=5,
    )
    h0 = svd.fit_transform(x).astype(np.float32, copy=False)
    h0 = np.nan_to_num(h0, nan=0.0, posinf=0.0, neginf=0.0)
    profile = {
        "input_shape": [int(n_samples), int(n_features)],
        "d0_requested": CONFIG.d0,
        "d_eff": int(d_eff),
        "svd_random_state": CONFIG.svd_random_state,
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        "zero_rows": int(np.sum(np.linalg.norm(h0, axis=1) <= 1e-8)),
        "h0_sha256": sha256_array(h0),
        "labels_used": False,
    }
    return h0, profile


@dataclass(frozen=True)
class CandidatePool:
    indices: np.ndarray
    cosine: np.ndarray
    positive_counts: np.ndarray
    profile: dict[str, Any]

    @property
    def effective_budget(self) -> np.ndarray:
        """The frozen feasible budget shared by every graph arm."""
        return np.minimum(
            np.asarray(self.positive_counts, dtype=np.int64),
            CONFIG.budget_cap,
        ).astype(np.int64, copy=False)

    @property
    def budget_hash(self) -> str:
        return sha256_array(self.effective_budget)


def budget_profile(pool: CandidatePool) -> dict[str, Any]:
    budget = np.asarray(pool.effective_budget, dtype=np.int64)
    n = int(budget.size)
    return {
        "budget_cap": int(CONFIG.budget_cap),
        "effective_budget_hash": pool.budget_hash,
        "effective_budget_mean": float(np.mean(budget)) if n else 0.0,
        "effective_budget_min": int(np.min(budget)) if n else 0,
        "effective_budget_max": int(np.max(budget)) if n else 0,
        "fraction_budget_below_cap": float(np.mean(budget < CONFIG.budget_cap)) if n else 0.0,
        "fraction_budget_zero": float(np.mean(budget == 0)) if n else 0.0,
        "zero_budget_nodes": int(np.sum(budget == 0)),
    }


def build_candidate_pool(h0: np.ndarray) -> CandidatePool:
    """Build kNN rows and retain only strictly positive-cosine edges."""
    values = row_l2_normalize(h0)
    n_samples = values.shape[0]
    k_eff = min(CONFIG.neighbor_k, max(0, n_samples - 1))
    if k_eff == 0:
        empty = np.empty((n_samples, 0), dtype=np.int64)
        pool = CandidatePool(
            empty,
            np.empty((n_samples, 0), dtype=np.float32),
            np.zeros(n_samples, dtype=np.int64),
            {"k_eff": 0, "k_requested": CONFIG.neighbor_k, "positive_rule": CONFIG.positive_cosine_rule},
        )
        pool.profile.update(budget_profile(pool))
        return pool
    nearest = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
    nearest.fit(values)
    distances, raw = nearest.kneighbors(values)
    indices = np.full((n_samples, k_eff), -1, dtype=np.int64)
    cosine = np.zeros((n_samples, k_eff), dtype=np.float32)
    for row in range(n_samples):
        keep = raw[row] != row
        row_idx = raw[row][keep][:k_eff]
        row_cos = (1.0 - distances[row][keep][:k_eff]).astype(np.float32, copy=False)
        indices[row, : row_idx.size] = row_idx
        cosine[row, : row_cos.size] = row_cos
    positive = (indices >= 0) & (cosine > 0.0)
    positive_counts = positive.sum(axis=1).astype(np.int64)
    profile = {
        "k_requested": CONFIG.neighbor_k,
        "k_effective": int(k_eff),
        "positive_rule": CONFIG.positive_cosine_rule,
        "positive_edge_count": int(positive.sum()),
        "positive_count_min": int(positive_counts.min()) if positive_counts.size else 0,
        "positive_count_median": float(np.median(positive_counts)) if n_samples else 0.0,
        "positive_count_max": int(positive_counts.max()) if positive_counts.size else 0,
        "budget_cap": CONFIG.budget_cap,
        "sensitivity_shortfall_nodes": {
            str(b): int(np.sum(positive_counts < b)) for b in CONFIG.budget_sensitivity
        },
    }
    pool = CandidatePool(indices, cosine, positive_counts, profile)
    profile.update(budget_profile(pool))
    return pool


def _resolve_budget(pool: CandidatePool, budget: int | np.ndarray | None) -> np.ndarray:
    n = int(pool.indices.shape[0])
    if budget is None:
        resolved = pool.effective_budget
    elif np.isscalar(budget):
        if int(budget) < 0:
            raise ValueError("budget must be non-negative")
        resolved = np.full(n, int(budget), dtype=np.int64)
    else:
        resolved = np.asarray(budget, dtype=np.int64)
        if resolved.shape != (n,):
            raise ValueError("budget vector must have one value per node")
        if np.any(resolved < 0):
            raise ValueError("budget must be non-negative")
    if np.any(resolved > pool.positive_counts):
        raise ValueError("budget exceeds available positive candidate edges")
    return resolved


def directed_graph_from_rows(
    pool: CandidatePool,
    selected_mask: np.ndarray,
    *,
    budget: int | np.ndarray | None = None,
) -> sp.csr_matrix:
    mask = np.asarray(selected_mask, dtype=bool)
    if mask.shape != pool.indices.shape:
        raise ValueError("selection mask shape mismatch")
    rows, cols = np.nonzero(mask)
    if rows.size and np.any(pool.cosine[rows, cols] <= 0):
        raise ValueError("selected graph contains a non-positive edge")
    counts = np.bincount(rows, minlength=pool.indices.shape[0])
    expected = _resolve_budget(pool, budget)
    if np.any(counts != expected):
        raise ValueError("selected graph does not satisfy the row-specific budget")
    if rows.size:
        values = pool.cosine[rows, cols].astype(np.float32, copy=False)
        graph = sp.csr_matrix(
            (values, (rows, pool.indices[rows, cols])),
            shape=(mask.shape[0], mask.shape[0]),
        )
    else:
        graph = sp.csr_matrix((mask.shape[0], mask.shape[0]), dtype=np.float32)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def graph_budget_audit(graph: sp.spmatrix, pool: CandidatePool) -> dict[str, Any]:
    """Return the auditable row-budget record shared by an S1 graph arm."""
    value = sp.csr_matrix(graph)
    if value.shape != (pool.indices.shape[0], pool.indices.shape[0]):
        raise ValueError("graph and candidate pool are not aligned")
    counts = np.diff(value.indptr).astype(np.int64, copy=False)
    expected = pool.effective_budget
    if np.any(counts != expected):
        raise ValueError("graph row counts do not match the frozen effective budget")
    return {
        "budget_cap": int(CONFIG.budget_cap),
        "effective_budget_hash": pool.budget_hash,
        "row_counts_match": True,
        "row_count_min": int(counts.min()) if counts.size else 0,
        "row_count_max": int(counts.max()) if counts.size else 0,
        "graph_nnz": int(value.nnz),
    }


def _stable_similarity_order(cosine: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Order candidate slots by frozen H0 cosine, with slot-index tie breaking."""
    if slots.size == 0:
        return slots.astype(np.int64, copy=False)
    return slots[np.lexsort((slots, -np.asarray(cosine[slots], dtype=np.float64)))]


def _positive_slots(pool: CandidatePool, row: int) -> np.ndarray:
    return np.flatnonzero(
        (pool.indices[row] >= 0) & (pool.cosine[row] > 0.0)
    ).astype(np.int64, copy=False)


def build_random_graph(pool: CandidatePool, seed: int) -> sp.csr_matrix:
    """Select each row's feasible positive candidates uniformly without replacement."""
    rng = np.random.default_rng(int(seed))
    mask = np.zeros_like(pool.indices, dtype=bool)
    budget = pool.effective_budget
    for row, b_i in enumerate(budget):
        slots = _positive_slots(pool, row)
        if b_i:
            chosen = rng.choice(slots, size=int(b_i), replace=False)
            mask[row, np.asarray(chosen, dtype=np.int64)] = True
    return directed_graph_from_rows(pool, mask, budget=budget)


def build_ungated_graph(pool: CandidatePool) -> sp.csr_matrix:
    """Keep every strictly positive candidate-pool edge (the S1 U arm).

    Unlike R/O_pool/O_full, U is intentionally not budget-capped: it is the
    descriptive ``ungated candidate graph`` control.  It still uses exactly
    the frozen candidate membership and H0-cosine weights, with no labels.
    """
    mask = (pool.indices >= 0) & (pool.cosine > 0.0)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return sp.csr_matrix((pool.indices.shape[0], pool.indices.shape[0]), dtype=np.float32)
    graph = sp.csr_matrix(
        (
            pool.cosine[rows, cols].astype(np.float32, copy=False),
            (rows, pool.indices[rows, cols]),
        ),
        shape=(pool.indices.shape[0], pool.indices.shape[0]),
    )
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def build_oracle_pool_graph(pool: CandidatePool, labels: np.ndarray) -> sp.csr_matrix:
    """Choose same-class candidate edges first, preserving the common budget and weights."""
    y = np.asarray(labels)
    if y.ndim != 1 or y.size != pool.indices.shape[0]:
        raise ValueError("labels must be a vector aligned with the candidate pool")
    mask = np.zeros_like(pool.indices, dtype=bool)
    budget = pool.effective_budget
    for row, b_i in enumerate(budget):
        if not b_i:
            continue
        slots = _positive_slots(pool, row)
        same = slots[pool.indices[row, slots] != -1]
        same = same[y[pool.indices[row, same]] == y[row]]
        other = slots[y[pool.indices[row, slots]] != y[row]]
        ordered_same = _stable_similarity_order(pool.cosine[row], same)
        ordered_other = _stable_similarity_order(pool.cosine[row], other)
        chosen = np.concatenate((ordered_same, ordered_other))[: int(b_i)]
        mask[row, chosen] = True
    return directed_graph_from_rows(pool, mask, budget=budget)


def _directed_graph_from_pairs(
    n_samples: int,
    row_indices: list[int],
    col_indices: list[int],
    values: list[float],
    budget: np.ndarray,
) -> sp.csr_matrix:
    counts = np.bincount(np.asarray(row_indices, dtype=np.int64), minlength=n_samples)
    if np.any(counts != budget):
        raise ValueError("full-space oracle does not satisfy the common budget")
    if not row_indices:
        return sp.csr_matrix((n_samples, n_samples), dtype=np.float32)
    graph = sp.csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(col_indices, dtype=np.int64),
            ),
        ),
        shape=(n_samples, n_samples),
    )
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def build_oracle_full_graph(
    h0: np.ndarray,
    labels: np.ndarray,
    *,
    pool: CandidatePool | None = None,
) -> sp.csr_matrix:
    """Full-space same-class oracle with the candidate pool's fixed row budgets.

    Labels enter only this diagnostic graph builder.  The H0 cosine weights are
    computed identically to the candidate graph and are never changed by the
    oracle.  Class-specific nearest-neighbour searches make the same/other
    priority exact up to the positive-cosine rule without materialising an
    O(n^2) dense similarity matrix.
    """
    values = row_l2_normalize(np.asarray(h0, dtype=np.float32))
    y = np.asarray(labels)
    if values.ndim != 2 or y.ndim != 1 or y.size != values.shape[0]:
        raise ValueError("h0 and labels are not aligned")
    candidate = pool if pool is not None else build_candidate_pool(values)
    if candidate.indices.shape[0] != values.shape[0]:
        raise ValueError("candidate pool is not aligned with h0")
    budget = candidate.effective_budget
    n_samples = values.shape[0]
    groups: dict[Any, np.ndarray] = {}
    for label in np.unique(y):
        groups[label] = np.flatnonzero(y == label)
    # One NN index per class gives exact top same-class neighbours.  A second
    # index per complement gives exact top异类 candidates for the fallback.
    same_nn: dict[Any, tuple[NearestNeighbors, np.ndarray]] = {}
    other_nn: dict[Any, tuple[NearestNeighbors, np.ndarray] | None] = {}
    max_budget = int(np.max(budget)) if budget.size else 0
    for label, members in groups.items():
        if members.size > 1:
            nn = NearestNeighbors(
                n_neighbors=min(members.size, max_budget + 1), metric="cosine"
            ).fit(values[members])
            same_nn[label] = (nn, members)
        else:
            same_nn[label] = (NearestNeighbors(n_neighbors=1, metric="cosine").fit(values[members]), members)
        complement = np.flatnonzero(y != label)
        if complement.size and max_budget > 0:
            nn_other = NearestNeighbors(
                n_neighbors=min(complement.size, max_budget), metric="cosine"
            ).fit(values[complement])
            other_nn[label] = (nn_other, complement)
        else:
            other_nn[label] = None

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for row in range(n_samples):
        b_i = int(budget[row])
        if b_i == 0:
            continue
        label = y[row]
        nn_same, members = same_nn[label]
        same_dist, same_pos = nn_same.kneighbors(values[row : row + 1])
        same_candidates = members[same_pos[0]]
        keep_self = same_candidates != row
        same_candidates = same_candidates[keep_self]
        same_cos = (1.0 - same_dist[0][keep_self]).astype(np.float32, copy=False)
        same_keep = same_cos > 0.0
        same_candidates = same_candidates[same_keep]
        same_cos = same_cos[same_keep]
        order = np.lexsort((same_candidates, -same_cos.astype(np.float64)))
        same_candidates = same_candidates[order][:b_i]
        same_cos = same_cos[order][:b_i]
        need = b_i - same_candidates.size
        other_candidates = np.empty(0, dtype=np.int64)
        other_cos = np.empty(0, dtype=np.float32)
        if need and other_nn[label] is not None:
            nn_other, complement = other_nn[label]  # type: ignore[misc]
            other_dist, other_pos = nn_other.kneighbors(values[row : row + 1])
            other_candidates = complement[other_pos[0]]
            other_cos = (1.0 - other_dist[0]).astype(np.float32, copy=False)
            keep = other_cos > 0.0
            other_candidates = other_candidates[keep][:need]
            other_cos = other_cos[keep][:need]
        chosen_candidates = np.concatenate((same_candidates, other_candidates))
        chosen_cos = np.concatenate((same_cos, other_cos))
        for col, weight in zip(chosen_candidates, chosen_cos, strict=True):
            rows.append(row)
            cols.append(int(col))
            weights.append(float(weight))
    return _directed_graph_from_pairs(n_samples, rows, cols, weights, budget)


def symmetrize_graph(graph: sp.spmatrix) -> sp.csr_matrix:
    value = (sp.csr_matrix(graph, dtype=np.float32) + sp.csr_matrix(graph, dtype=np.float32).T) * 0.5
    value = value.tocsr()
    value.setdiag(0.0)
    value.eliminate_zeros()
    if value.nnz and float(value.data.min()) < -1e-7:
        raise ValueError("symmetrized graph is not non-negative")
    return value


def laplacians(graph: sp.spmatrix) -> dict[str, Any]:
    w = symmetrize_graph(graph)
    degrees = np.asarray(w.sum(axis=1)).ravel().astype(np.float64)
    d = sp.diags(degrees, offsets=0, format="csr")
    l = (d - w).tocsr()
    inv_sqrt = np.zeros_like(degrees)
    positive = degrees > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    normalized = sp.eye(w.shape[0], dtype=np.float64, format="csr") - (
        sp.diags(inv_sqrt) @ w.astype(np.float64) @ sp.diags(inv_sqrt)
    ).tocsr()
    return {
        "W": w,
        "D": d,
        "L": l,
        "L_sym": normalized.tocsr(),
        "degrees": degrees,
        "isolated_nodes": int(np.sum(~positive)),
    }


def _active_normalized_laplacian(graph: sp.spmatrix) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Return L_sym on the positive-degree induced subgraph only."""
    w = symmetrize_graph(graph).astype(np.float64)
    degrees = np.asarray(w.sum(axis=1)).ravel()
    active = degrees > 0.0
    active_idx = np.flatnonzero(active)
    if active_idx.size == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64), active_idx, degrees
    w_active = w[active_idx][:, active_idx].tocsr()
    d_active = np.asarray(w_active.sum(axis=1)).ravel()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d_active, CONFIG.loss_eps))
    normalized = sp.eye(active_idx.size, dtype=np.float64, format="csr") - (
        sp.diags(inv_sqrt) @ w_active @ sp.diags(inv_sqrt)
    ).tocsr()
    return normalized.tocsr(), active_idx, degrees


def spectral_embedding_with_audit(
    graph: sp.spmatrix,
    n_clusters: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Frozen W -> active L_sym -> eigsh -> row-normalized embedding pipeline."""
    if int(n_clusters) < 1:
        raise ValueError("n_clusters must be positive")
    n_samples = int(graph.shape[0])
    if graph.shape[1] != n_samples:
        raise ValueError("graph must be square")
    l_sym, active_idx, degrees = _active_normalized_laplacian(graph)
    embedding = np.zeros((n_samples, int(n_clusters)), dtype=np.float32)
    metadata: dict[str, Any] = {
        "consumer": "Spectral",
        "K_used_in_representation": True,
        "K_used_in_readout": True,
        "labels_vector_used_in_fit": False,
        "n_clusters": int(n_clusters),
        "active_nodes": int(active_idx.size),
        "isolated_nodes": int(np.sum(degrees <= 0.0)),
        "drop_first": CONFIG.spectral_drop_first,
        "which": CONFIG.spectral_which,
        "tol": CONFIG.spectral_tol,
        "maxiter": max(1000, 10 * int(active_idx.size)),
    }
    if active_idx.size == 0:
        metadata.update({"eigenvalues": [], "status": "completed_no_active_nodes"})
        return embedding, metadata
    if int(n_clusters) >= int(active_idx.size):
        raise IncompleteComputeError(
            f"spectral eigsh requires K < active node count ({n_clusters} >= {active_idx.size})"
        )
    v0 = np.ones(active_idx.size, dtype=np.float64) / np.sqrt(float(active_idx.size))
    try:
        eigenvalues, eigenvectors = eigsh(
            l_sym,
            k=int(n_clusters),
            which=CONFIG.spectral_which,
            tol=CONFIG.spectral_tol,
            maxiter=max(1000, 10 * int(active_idx.size)),
            v0=v0,
        )
    except Exception as exc:  # scipy may expose several ARPACK exception classes
        raise IncompleteComputeError(f"eigsh failed: {exc}") from exc
    order = np.argsort(eigenvalues, kind="stable")
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    active_embedding = np.asarray(eigenvectors[:, order], dtype=np.float32)
    active_embedding = row_l2_normalize(active_embedding)
    embedding[active_idx] = active_embedding
    metadata.update(
        {
            "eigenvalues": eigenvalues.tolist(),
            "status": "completed",
            "active_embedding_finite": bool(np.isfinite(active_embedding).all()),
            "isolated_rows_zero": bool(np.all(embedding[degrees <= 0.0] == 0.0)),
        }
    )
    return embedding, metadata


def spectral_embedding(graph: sp.spmatrix, n_clusters: int) -> np.ndarray:
    """Return the frozen spectral embedding, raising on incomplete eigensolves."""
    embedding, _ = spectral_embedding_with_audit(graph, n_clusters)
    return embedding


def spectral_predict_with_audit(
    graph: sp.spmatrix,
    n_clusters: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run Spectral embedding followed by the frozen known-K KMeans readout."""
    embedding, metadata = spectral_embedding_with_audit(graph, n_clusters)
    model = KMeans(
        n_clusters=int(n_clusters),
        n_init=20,
        random_state=int(seed),
    )
    predictions = model.fit_predict(embedding)
    metadata = dict(metadata)
    metadata.update(
        {
            "kmeans_n_init": 20,
            "kmeans_random_state": int(seed),
            "prediction_unique": int(np.unique(predictions).size),
        }
    )
    return predictions.astype(np.int64, copy=False), embedding, metadata


def spectral_predict(
    graph: sp.spmatrix,
    n_clusters: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return only known-K spectral/KMeans predictions."""
    predictions, _, _ = spectral_predict_with_audit(graph, n_clusters, seed=seed)
    return predictions


def gview(h0: np.ndarray, graph: sp.spmatrix) -> np.ndarray:
    values = np.asarray(h0, dtype=np.float32)
    ops = laplacians(graph)
    degrees = ops["degrees"]
    message = np.asarray(ops["W"].dot(values), dtype=np.float32)
    message /= np.maximum(degrees[:, None], CONFIG.gview_eps).astype(np.float32)
    mixed = values + CONFIG.gview_alpha * message
    return row_l2_normalize(mixed).astype(np.float32, copy=False)


def numerical_loss_contract(h0: np.ndarray, graph: sp.spmatrix) -> dict[str, float | bool]:
    values = np.asarray(h0, dtype=np.float64)
    ops = laplacians(graph)
    z = gview(values.astype(np.float32), graph).astype(np.float64)
    d_z = np.asarray(ops["D"].dot(z), dtype=np.float64)
    l_z = np.asarray(ops["L"].dot(z), dtype=np.float64)
    denom = max(float(np.sum(z * d_z)), CONFIG.loss_eps)
    cut = float(np.sum(z * l_z) / denom)
    gram = (z.T @ d_z) / max(float(np.sum(ops["degrees"])), CONFIG.loss_eps)
    orth = float(np.sum((gram - np.eye(z.shape[1])) ** 2))
    variances = np.var(z, axis=0)
    var_penalty = float(np.sum(np.maximum(0.0, CONFIG.v_min - variances) ** 2))
    finite = bool(np.isfinite([cut, orth, var_penalty]).all())
    return {
        "finite": finite,
        "L_cut": cut,
        "L_orth": orth,
        "L_var": var_penalty,
        "isolated_nodes": int(ops["isolated_nodes"]),
    }


def audit_adapter_semantics(repo_root: str | Path) -> dict[str, Any]:
    """Check whether existing V21/V25 code has a faithful sample-edge adapter.

    The current code intentionally fails this audit: FeatureGate emits feature-coordinate
    scores, while the graph builder emits a candidate graph, but no existing function maps
    the former to sample-edge membership.  Inventing that mapping would violate S0.
    """
    root = Path(repo_root)
    paths = {
        "graph": root / "methods/TopoGate/V21_assignment_adversarial_gate/graph.py",
        "model": root / "methods/TopoGate/V21_assignment_adversarial_gate/model.py",
        "trainer": root / "methods/TopoGate/V21_assignment_adversarial_gate/trainer.py",
    }
    evidence: dict[str, Any] = {"paths": {k: str(v) for k, v in paths.items()}}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        return {"status": "protocol_mismatch", "reason": "missing_source", "missing": missing, **evidence}
    source = {k: p.read_text(encoding="utf-8") for k, p in paths.items()}
    try:
        ast.parse(source["graph"])
        ast.parse(source["model"])
        ast.parse(source["trainer"])
        parse_ok = True
    except SyntaxError as exc:
        parse_ok = False
        evidence["syntax_error"] = str(exc)
    feature_gate_forward = "class FeatureGate" in source["model"] and "def forward" in source["model"]
    has_edge_membership_symbol = any(
        token in source["graph"] + source["model"] + source["trainer"]
        for token in ("edge_membership", "sample_edge_mask", "selection_to_relation_adapter")
    )
    evidence.update(
        {
            "source_parse_ok": parse_ok,
            "feature_gate_forward_present": feature_gate_forward,
            "existing_edge_membership_symbol": has_edge_membership_symbol,
            "feature_gate_semantics": "feature-coordinate scores; not sample-edge membership",
            "semantic_fidelity_required": True,
        }
    )
    if not parse_ok:
        return {"status": "protocol_mismatch", "reason": "source_parse_failed", **evidence}
    if has_edge_membership_symbol:
        return {
            "status": "adapter_review_required",
            "reason": "candidate symbol requires manual semantic review before promotion",
            **evidence,
        }
    return {
        "status": "adapter_not_estimable",
        "reason": "no faithful feature-mask_to_sample-edge mapping exists in the current code boundary",
        "forbidden_repairs": ["new_aggregation", "new_threshold", "trainable_mapping", "human_rule"],
        **evidence,
    }


def _block_graph(labels: np.ndarray, *, contamination: float = 0.0) -> sp.csr_matrix:
    n = labels.size
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for i in range(n):
        same = np.flatnonzero(labels == labels[i])
        same = same[same != i]
        other = np.flatnonzero(labels != labels[i])
        take_other = int(round(contamination * min(4, same.size)))
        chosen_same = same[: max(0, 4 - take_other)]
        chosen_other = other[:take_other]
        for j in np.concatenate([chosen_same, chosen_other]):
            rows.append(i)
            cols.append(int(j))
            values.append(1.0 if labels[int(j)] == labels[i] else 0.5)
    return sp.csr_matrix((values, (rows, cols)), shape=(n, n), dtype=np.float32)


def synthetic_apparatus_sanity() -> dict[str, Any]:
    labels = np.repeat(np.arange(3), 20)
    clean = _block_graph(labels, contamination=0.0)
    contaminated = _block_graph(labels, contamination=0.75)
    ring_rows = np.arange(labels.size)
    ring_cols = np.roll(ring_rows, -1)
    # Unequal positive weights keep the deterministic all-ones v0 from being
    # an exact null eigenvector while retaining a topology-free cycle control.
    ring_values = (1.0 + 0.2 * np.sin(ring_rows * 0.37)).astype(np.float32)
    no_opportunity = sp.csr_matrix(
        (ring_values, (ring_rows, ring_cols)), shape=(labels.size, labels.size)
    )
    clean_ops = laplacians(clean)
    bad_ops = laplacians(contaminated)
    no_ops = laplacians(no_opportunity)
    # Label-aware NCut is a diagnostic only; it is deliberately confined to this apparatus test.
    def label_cut(graph: sp.spmatrix) -> float:
        w = symmetrize_graph(graph)
        total = 0.0
        for group in np.unique(labels):
            idx = np.flatnonzero(labels == group)
            mask = np.ones(labels.size, dtype=bool)
            mask[idx] = False
            total += float(w[idx][:, mask].sum()) / max(float(w[idx].sum()), 1e-8)
        return total

    clean_ncut = label_cut(clean)
    contaminated_ncut = label_cut(contaminated)
    numerical = numerical_loss_contract(np.eye(labels.size, 4, dtype=np.float32), clean)
    clean_finite = bool(np.isfinite(clean_ops["L"].data).all() and np.isfinite(clean_ops["L_sym"].data).all())
    contaminated_finite = bool(np.isfinite(bad_ops["L"].data).all() and np.isfinite(bad_ops["L_sym"].data).all())
    no_opportunity_finite = bool(np.isfinite(no_ops["L"].data).all() and np.isfinite(no_ops["L_sym"].data).all())
    clean_pred, clean_z, clean_spec = spectral_predict_with_audit(clean, 3, seed=42)
    contaminated_pred, contaminated_z, contaminated_spec = spectral_predict_with_audit(
        contaminated, 3, seed=42
    )
    no_pred, no_z, no_spec = spectral_predict_with_audit(no_opportunity, 3, seed=42)
    isolate_graph = sp.csr_matrix(
        (
            np.array([1.0, 0.8, 1.1, 0.9], dtype=np.float32),
            (np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0])),
        ),
        shape=(6, 6),
    )
    isolate_z, isolate_spec = spectral_embedding_with_audit(isolate_graph, 2)
    clean_ari = float(adjusted_rand_score(labels, clean_pred))
    contaminated_ari = float(adjusted_rand_score(labels, contaminated_pred))
    result = {
        "clean_block": {"finite": clean_finite, "ground_truth_ncut": clean_ncut},
        "contaminated_block": {"finite": contaminated_finite, "ground_truth_ncut": contaminated_ncut},
        "no_opportunity": {
            "finite": no_opportunity_finite,
            "embedding_finite": bool(np.isfinite(no_z).all()),
            "prediction_unique": int(np.unique(no_pred).size),
            "ari_against_unrelated_labels": float(adjusted_rand_score(labels, no_pred)),
            "metadata": no_spec,
        },
        "spectral_recovery_sanity": {
            "clean_ari": clean_ari,
            "contaminated_ari": contaminated_ari,
            "clean_embedding_finite": bool(np.isfinite(clean_z).all()),
            "contaminated_embedding_finite": bool(np.isfinite(contaminated_z).all()),
            "clean_ari_threshold": 0.95,
            "clean_beats_contaminated": bool(clean_ari > contaminated_ari),
            "isolated_rows_zero": bool(np.all(isolate_z[4:] == 0.0)),
            "isolates_excluded_from_eigenstructure": bool(
                isolate_spec["active_nodes"] == 4 and isolate_spec["isolated_nodes"] == 2
            ),
            "clean_metadata": clean_spec,
            "contaminated_metadata": contaminated_spec,
            "isolate_metadata": isolate_spec,
        },
        "graph_numerical_sanity": {
            "clean_finite": clean_finite,
            "contaminated_finite": contaminated_finite,
            "no_opportunity_finite": no_opportunity_finite,
            "loss_finite": bool(numerical["finite"]),
        },
        "direction_checks": {
            "clean_cut_better_than_contaminated": bool(clean_ncut < contaminated_ncut),
            "all_laplacians_finite": bool(np.isfinite(clean_ops["L_sym"].data).all() and np.isfinite(bad_ops["L_sym"].data).all() and np.isfinite(no_ops["L_sym"].data).all()),
            "loss_contract_finite": bool(numerical["finite"]),
            "spectral_clean_recovery": bool(clean_ari >= 0.95),
            "spectral_clean_beats_contaminated": bool(clean_ari > contaminated_ari),
            "spectral_embeddings_finite": bool(np.isfinite(clean_z).all() and np.isfinite(contaminated_z).all() and np.isfinite(no_z).all()),
            "spectral_isolate_policy": bool(np.all(isolate_z[4:] == 0.0) and isolate_spec["active_nodes"] == 4),
        },
        "labels_used_in_training": False,
        "purpose": "apparatus_sanity_only",
    }
    result["graph_numerical_sanity_status"] = (
        "PASS" if all(result["graph_numerical_sanity"].values()) else "FAIL"
    )
    result["spectral_recovery_sanity_status"] = (
        "PASS"
        if all(
            result["direction_checks"][key]
            for key in (
                "spectral_clean_recovery",
                "spectral_clean_beats_contaminated",
                "spectral_embeddings_finite",
                "spectral_isolate_policy",
            )
        )
        else "FAIL"
    )
    return result


def contract_manifest(adapter_audit: dict[str, Any], dataset_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {
        "manifest_id": "representation_consumer_probe_s0_freeze_v1",
        "resolved_config": resolved_config(),
        "adapter_audit": jsonable(adapter_audit),
        "datasets": jsonable(list(dataset_rows)),
        "synthetic_apparatus": synthetic_apparatus_sanity(),
        "labels_used_during_fit": False,
        "oracle_non_tuning": True,
        "status": "protocol_frozen_pending_dataset_preflight",
    }
