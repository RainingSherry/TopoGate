"""Candidate graph construction for TopoGate V11.

The graph is an alternating-inference object: raw-space neighbours provide a
stable prior, while EMA-latent neighbours are refreshed periodically.  The
candidate union is discrete, but all edge weights used by the loss are learned
in Torch and remain differentiable within a refresh interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class CandidateGraph:
    indices: np.ndarray
    raw_similarity: np.ndarray
    mutual: np.ndarray
    snn: np.ndarray
    valid: np.ndarray
    source: str
    knn_backend: str = "exact"
    raw_knn_indices: np.ndarray | None = None
    tda_prior: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return int(self.indices.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.indices.shape[1])


def pca_embedding(X: np.ndarray, max_dim: int, variance: float, seed: int) -> tuple[np.ndarray, int]:
    X = np.asarray(X, dtype=np.float32)
    upper = min(int(max_dim), X.shape[0] - 1, X.shape[1])
    if upper <= 0:
        return X.copy(), int(X.shape[1])
    pca = PCA(n_components=upper, random_state=seed, whiten=True)
    Z = np.nan_to_num(pca.fit_transform(X), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    cumulative = np.cumsum(np.nan_to_num(pca.explained_variance_ratio_, nan=0.0))
    needed = int(np.searchsorted(cumulative, float(variance)) + 1)
    needed = max(2, min(needed, upper)) if upper >= 2 else upper
    return Z[:, :needed].astype(np.float32), int(needed)


def _normalise_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-8, None)


def _remove_self(
    indices: np.ndarray,
    similarities: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove self by node identity, including duplicate-row/tie cases."""
    n = int(indices.shape[0])
    out_i = np.empty((n, k), dtype=np.int64)
    out_s = np.empty((n, k), dtype=np.float32)
    for row in range(n):
        keep = indices[row] != row
        row_indices = indices[row][keep]
        row_similarities = similarities[row][keep]
        if row_indices.size < k:
            raise RuntimeError(f"failed to find {k} non-self neighbours for node {row}")
        out_i[row] = row_indices[:k]
        out_s[row] = row_similarities[:k]
    return out_i, out_s


def _knn(
    X: np.ndarray,
    k: int,
    *,
    backend: Literal["auto", "exact", "faiss_hnsw"] = "exact",
    exact_max_nodes: int = 5000,
    hnsw_m: int = 32,
    hnsw_ef_search: int = 64,
) -> tuple[np.ndarray, np.ndarray, str]:
    n = int(X.shape[0])
    if n <= 1:
        return (
            np.zeros((n, 1), dtype=np.int64),
            np.zeros((n, 1), dtype=np.float32),
            "none",
        )
    k = max(1, min(int(k), n - 1))
    Z = _normalise_rows(np.asarray(X, dtype=np.float32))
    if backend not in {"auto", "exact", "faiss_hnsw"}:
        raise ValueError(f"unknown kNN backend: {backend}")
    selected = "exact" if backend == "auto" and n <= int(exact_max_nodes) else backend
    if selected == "auto":
        selected = "faiss_hnsw"
    query_width = min(n, k + 2)
    if selected == "exact":
        # Ask for spares, then remove self explicitly by node id. With duplicate
        # rows/ties sklearn does not guarantee that self is column 0.
        nn = NearestNeighbors(n_neighbors=query_width, metric="cosine")
        nn.fit(Z)
        distances, indices = nn.kneighbors(Z, return_distance=True)
        return _remove_self(indices, 1.0 - distances, k) + ("exact",)
    try:
        import faiss
    except Exception as error:
        if backend == "faiss_hnsw":
            raise ImportError("faiss is required for knn_backend='faiss_hnsw'") from error
        # The fallback remains correct and explicit in the saved provenance.
        nn = NearestNeighbors(n_neighbors=query_width, metric="cosine")
        nn.fit(Z)
        distances, indices = nn.kneighbors(Z, return_distance=True)
        return _remove_self(indices, 1.0 - distances, k) + ("exact_fallback_no_faiss",)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(
        int(Z.shape[1]), max(4, int(hnsw_m)), faiss.METRIC_INNER_PRODUCT
    )
    index.hnsw.efConstruction = max(int(hnsw_ef_search), 2 * (k + 1))
    index.hnsw.efSearch = max(int(hnsw_ef_search), k + 1)
    index.add(np.ascontiguousarray(Z, dtype=np.float32))
    similarities, indices = index.search(np.ascontiguousarray(Z, dtype=np.float32), query_width)
    cleaned_i, cleaned_s = _remove_self(indices, similarities, k)
    return cleaned_i, cleaned_s, "faiss_hnsw"


def _row_set(indices: np.ndarray, row: int) -> set[int]:
    return set(int(x) for x in indices[row].tolist() if int(x) != row)


def _union_candidates(raw_i: np.ndarray, latent_i: np.ndarray | None, candidate_k: int) -> tuple[np.ndarray, np.ndarray]:
    n = raw_i.shape[0]
    width = max(1, min(int(candidate_k), max(1, n - 1)))
    out = np.zeros((n, width), dtype=np.int64)
    valid = np.zeros((n, width), dtype=bool)
    for i in range(n):
        seen: set[int] = set()
        candidates: list[int] = []
        for source in (raw_i, latent_i):
            if source is None:
                continue
            for value in source[i].tolist():
                value = int(value)
                if value != i and value not in seen:
                    seen.add(value)
                    candidates.append(value)
                if len(candidates) >= width:
                    break
            if len(candidates) >= width:
                break
        if not candidates:
            candidates = [int((i + 1) % n)] if n > 1 else [i]
        count = min(len(candidates), width)
        out[i, :count] = np.asarray(candidates[:count], dtype=np.int64)
        valid[i, :count] = True
        if count < width:
            out[i, count:] = out[i, count - 1]
    return out, valid


def _snn_score(indices: np.ndarray, candidates: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n, width = candidates.shape
    sets = [_row_set(indices, i) for i in range(n)]
    scores = np.zeros((n, width), dtype=np.float32)
    for i in range(n):
        for j in range(width):
            if not valid[i, j]:
                continue
            neighbour = int(candidates[i, j])
            a, b = sets[i], sets[neighbour]
            denom = len(a | b)
            scores[i, j] = float(len(a & b) / denom) if denom else 0.0
    return scores


def build_candidate_graph(
    raw_embedding: np.ndarray,
    latent_embedding: np.ndarray | None,
    neighbor_k: int,
    candidate_k: int,
    *,
    knn_backend: Literal["auto", "exact", "faiss_hnsw"] = "exact",
    knn_exact_max_nodes: int = 5000,
    knn_hnsw_m: int = 32,
    knn_hnsw_ef_search: int = 64,
) -> CandidateGraph:
    """Build a raw/latent union graph and return edge-side prior statistics."""
    options = {
        "backend": knn_backend,
        "exact_max_nodes": knn_exact_max_nodes,
        "hnsw_m": knn_hnsw_m,
        "hnsw_ef_search": knn_hnsw_ef_search,
    }
    raw_i, _, raw_backend = _knn(raw_embedding, neighbor_k, **options)
    latent_i = None
    latent_backend = None
    if latent_embedding is not None:
        latent_i, _, latent_backend = _knn(latent_embedding, neighbor_k, **options)
    candidates, valid = _union_candidates(raw_i, latent_i, candidate_k)
    n, width = candidates.shape
    raw_norm = _normalise_rows(np.asarray(raw_embedding, dtype=np.float32))
    raw_sim = np.sum(raw_norm[:, None, :] * raw_norm[candidates], axis=2).astype(np.float32)
    mutual_sets = [_row_set(raw_i, i) for i in range(n)]
    mutual = np.zeros((n, width), dtype=np.float32)
    for i in range(n):
        for j in range(width):
            if valid[i, j] and i in mutual_sets[int(candidates[i, j])]:
                mutual[i, j] = 1.0
    snn = _snn_score(raw_i, candidates, valid)
    backend_used = raw_backend if latent_backend is None else f"raw:{raw_backend}|latent:{latent_backend}"
    return CandidateGraph(
        indices=candidates,
        raw_similarity=raw_sim,
        mutual=mutual,
        snn=snn,
        valid=valid,
        source="raw+latent" if latent_embedding is not None else "raw",
        knn_backend=backend_used,
        raw_knn_indices=raw_i,
    )


def graph_change_fraction(previous: CandidateGraph | None, current: CandidateGraph) -> float:
    if previous is None or previous.indices.shape != current.indices.shape:
        return 1.0
    changed = (previous.indices != current.indices) | (previous.valid != current.valid)
    return float(np.mean(changed))


def edge_recurrence_against(
    current: CandidateGraph,
    previous: CandidateGraph | None,
) -> np.ndarray:
    """Return a label-free edge recurrence target for ``current``.

    The value for edge ``i -> j`` is one exactly when the same directed edge
    was present in the *previous dynamic* candidate graph. It deliberately
    does not depend on current gate features or responsibilities, so it can
    supervise the gate without self-confirmation.
    """
    recurrence = np.zeros_like(current.raw_similarity, dtype=np.float32)
    if previous is None or previous.n_nodes != current.n_nodes:
        return recurrence
    previous_sets = [
        set(int(value) for value in previous.indices[row, previous.valid[row]])
        for row in range(previous.n_nodes)
    ]
    for row in range(current.n_nodes):
        for column in np.flatnonzero(current.valid[row]):
            recurrence[row, column] = float(int(current.indices[row, column]) in previous_sets[row])
    return recurrence
