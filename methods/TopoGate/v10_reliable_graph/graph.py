"""Deterministic graph construction and non-redundant edge features for V10.

The module intentionally depends only on NumPy, scikit-learn, and PyTorch.  It
does not use Scanpy, so the same graph semantics are available for tabular,
text, and single-cell inputs after task-appropriate preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "similarity",
    "mutual",
    "snn",
    "density_compatibility",
    "stability",
)


@dataclass(slots=True)
class KNNGraph:
    """Fixed-width directed kNN graph with an explicit validity mask.

    All edge-valued arrays have shape ``[n_nodes, max_neighbors]``.  Padded
    positions use index ``-1`` and are excluded by ``valid_mask``.  ``density``
    is a density-*compatibility* score in ``[0, 1]`` rather than a second copy
    of distance.  ``stability`` is edge recurrence across independent graph
    views and is zero when no auxiliary view was supplied.
    """

    indices: np.ndarray
    similarity: np.ndarray
    mutual: np.ndarray
    snn: np.ndarray
    density: np.ndarray
    stability: np.ndarray
    embedding: np.ndarray
    valid_mask: np.ndarray
    profile: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        edge_shape = tuple(self.indices.shape)
        if self.indices.ndim != 2:
            raise ValueError(f"indices must be 2D, got {edge_shape}.")
        for name in ("similarity", "mutual", "snn", "density", "stability", "valid_mask"):
            value = np.asarray(getattr(self, name))
            if tuple(value.shape) != edge_shape:
                raise ValueError(f"{name} must have shape {edge_shape}, got {value.shape}.")
        if self.embedding.ndim != 2 or self.embedding.shape[0] != self.indices.shape[0]:
            raise ValueError("embedding must have shape [n_nodes, embedding_dim].")
        valid = np.asarray(self.valid_mask, dtype=bool)
        if np.any(self.indices[valid] < 0) or np.any(self.indices[valid] >= self.n_nodes):
            raise ValueError("Valid neighbor indices must be in [0, n_nodes).")
        if np.any(self.indices[~valid] != -1):
            raise ValueError("Padded neighbor positions must use index -1.")

    @property
    def n_nodes(self) -> int:
        """Number of graph nodes."""

        return int(self.indices.shape[0])

    @property
    def k(self) -> int:
        """Maximum number of stored neighbors per node."""

        return int(self.indices.shape[1])

    @property
    def edge_features(self) -> np.ndarray:
        """Return edge features in the canonical V10 order."""

        return compute_edge_features(self)


# Public semantic name used by the V10 runner.
GraphState = KNNGraph


def _as_float_matrix(data: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {array.shape}.")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature column.")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def _row_normalize(embedding: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    return (embedding / np.clip(norms, 1e-12, None)).astype(np.float32, copy=False)


def _project(data: np.ndarray, pca_dim: int | None, seed: int) -> np.ndarray:
    n_nodes, n_features = data.shape
    max_rank = min(n_nodes, n_features)
    if pca_dim is None or pca_dim <= 0 or pca_dim >= max_rank:
        return _row_normalize(data.copy())
    dim = max(1, min(int(pca_dim), max_rank - 1 if max_rank > 1 else 1))
    projected = PCA(
        n_components=dim,
        svd_solver="randomized",
        random_state=int(seed),
    ).fit_transform(data)
    return _row_normalize(projected.astype(np.float32, copy=False))


def _remove_self_from_candidates(
    candidate_indices: np.ndarray,
    candidate_similarities: np.ndarray,
    k_eff: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = candidate_indices.shape[0]
    indices = np.empty((n_nodes, k_eff), dtype=np.int64)
    similarities = np.empty((n_nodes, k_eff), dtype=np.float32)
    for node in range(n_nodes):
        keep = candidate_indices[node] != node
        row_indices = candidate_indices[node, keep][:k_eff]
        row_similarities = candidate_similarities[node, keep][:k_eff]
        if row_indices.shape[0] != k_eff:
            raise RuntimeError("kNN search did not return enough non-self neighbors.")
        indices[node] = row_indices
        similarities[node] = row_similarities
    return indices, similarities


def _knn_exact(embedding: np.ndarray, k_eff: int) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = embedding.shape[0]
    search = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", algorithm="brute")
    search.fit(embedding)
    candidate_distances, candidate_indices = search.kneighbors(embedding, return_distance=True)
    candidate_similarities = (1.0 - candidate_distances).astype(np.float32, copy=False)
    return _remove_self_from_candidates(candidate_indices, candidate_similarities, k_eff)


def _knn_faiss_hnsw(
    embedding: np.ndarray,
    k_eff: int,
    *,
    hnsw_m: int,
    hnsw_ef_search: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import faiss
    except Exception as error:  # optional scalable backend
        raise ImportError("faiss is required for knn_backend='faiss_hnsw'.") from error

    matrix = np.ascontiguousarray(embedding, dtype=np.float32)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(
        int(matrix.shape[1]),
        max(4, int(hnsw_m)),
        faiss.METRIC_INNER_PRODUCT,
    )
    index.hnsw.efConstruction = max(int(hnsw_ef_search), 2 * (k_eff + 1))
    index.hnsw.efSearch = max(int(hnsw_ef_search), k_eff + 1)
    index.add(matrix)
    candidate_similarities, candidate_indices = index.search(matrix, k_eff + 1)
    return _remove_self_from_candidates(candidate_indices, candidate_similarities, k_eff)


def _knn(
    embedding: np.ndarray,
    k: int,
    *,
    backend: Literal["exact", "faiss_hnsw", "auto"] = "exact",
    exact_max_nodes: int = 5000,
    hnsw_m: int = 32,
    hnsw_ef_search: int = 64,
) -> tuple[np.ndarray, np.ndarray, str]:
    n_nodes = int(embedding.shape[0])
    k_eff = min(max(int(k), 0), max(n_nodes - 1, 0))
    if k_eff == 0:
        return (
            np.empty((n_nodes, 0), dtype=np.int64),
            np.empty((n_nodes, 0), dtype=np.float32),
            "none",
        )
    if backend not in {"exact", "faiss_hnsw", "auto"}:
        raise ValueError(f"Unknown kNN backend: {backend!r}.")
    selected = "exact" if backend == "auto" and n_nodes <= int(exact_max_nodes) else backend
    if selected == "auto":
        selected = "faiss_hnsw"
    if selected == "exact":
        indices, similarities = _knn_exact(embedding, k_eff)
        return indices, similarities, "exact"
    try:
        indices, similarities = _knn_faiss_hnsw(
            embedding,
            k_eff,
            hnsw_m=hnsw_m,
            hnsw_ef_search=hnsw_ef_search,
        )
        return indices, similarities, "faiss_hnsw"
    except ImportError:
        if backend != "auto":
            raise
        indices, similarities = _knn_exact(embedding, k_eff)
        return indices, similarities, "exact_fallback_no_faiss"


def _topology_features(indices: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_nodes, width = indices.shape
    neighbor_sets = [set(indices[i, valid_mask[i]].tolist()) for i in range(n_nodes)]
    mutual = np.zeros((n_nodes, width), dtype=np.float32)
    snn = np.zeros((n_nodes, width), dtype=np.float32)
    for i in range(n_nodes):
        set_i = neighbor_sets[i]
        for position in np.flatnonzero(valid_mask[i]):
            j = int(indices[i, position])
            set_j = neighbor_sets[j]
            mutual[i, position] = float(i in set_j)
            union_size = len(set_i | set_j)
            snn[i, position] = len(set_i & set_j) / float(max(union_size, 1))
    return mutual, snn


def _density_compatibility(
    indices: np.ndarray,
    similarity: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compare local scales without duplicating cosine distance as a feature."""

    n_nodes, width = indices.shape
    distances = np.maximum(1.0 - similarity, 1e-6)
    counts = np.maximum(valid_mask.sum(axis=1), 1)
    local_scale = (distances * valid_mask).sum(axis=1) / counts
    log_scale = np.log(np.maximum(local_scale, 1e-6))
    result = np.zeros((n_nodes, width), dtype=np.float32)
    safe_indices = np.maximum(indices, 0)
    compatibility = np.exp(-np.abs(log_scale[:, None] - log_scale[safe_indices]))
    result[valid_mask] = compatibility[valid_mask].astype(np.float32, copy=False)
    return result


def _edge_recurrence(
    indices: np.ndarray,
    valid_mask: np.ndarray,
    auxiliary_neighbor_sets: Sequence[Sequence[set[int]]],
) -> np.ndarray:
    stability = np.zeros(indices.shape, dtype=np.float32)
    if not auxiliary_neighbor_sets:
        return stability
    for i in range(indices.shape[0]):
        for position in np.flatnonzero(valid_mask[i]):
            j = int(indices[i, position])
            hits = sum(float(j in view[i]) for view in auxiliary_neighbor_sets)
            stability[i, position] = hits / float(len(auxiliary_neighbor_sets))
    return stability


def build_knn_graph(
    data: np.ndarray,
    k: int,
    pca_dim: int | None = None,
    seed: int = 42,
    *,
    stability_views: Sequence[np.ndarray] | None = None,
    backend: Literal["exact", "faiss_hnsw", "auto"] = "exact",
    exact_max_nodes: int = 5000,
    hnsw_m: int = 32,
    hnsw_ef_search: int = 64,
) -> KNNGraph:
    """Build a deterministic cosine kNN graph without using class labels.

    Parameters
    ----------
    data:
        Preprocessed feature matrix ``[n_nodes, n_features]``.
    k:
        Requested out-degree.  It is clipped to ``n_nodes - 1``.
    pca_dim:
        Optional PCA dimension used only for graph initialization.  ``None``
        or a non-positive value disables projection.
    seed:
        Random state for randomized PCA.
    stability_views:
        Optional independently augmented feature matrices.  Edge stability is
        the fraction of their kNN graphs in which each primary edge recurs.
    backend:
        ``"exact"`` uses deterministic sklearn brute-force cosine kNN.
        ``"faiss_hnsw"`` uses optional FAISS HNSW approximate inner-product
        search on row-normalized embeddings. ``"auto"`` selects exact search
        through ``exact_max_nodes`` and HNSW above that size, falling back to
        exact search if FAISS is unavailable.
    """

    matrix = _as_float_matrix(data, "data")
    n_nodes = int(matrix.shape[0])
    if n_nodes == 0:
        raise ValueError("data must contain at least one row.")
    embedding = _project(matrix, pca_dim, seed)
    indices, similarity, backend_used = _knn(
        embedding,
        k,
        backend=backend,
        exact_max_nodes=exact_max_nodes,
        hnsw_m=hnsw_m,
        hnsw_ef_search=hnsw_ef_search,
    )
    valid_mask = np.ones(indices.shape, dtype=bool)
    mutual, snn = _topology_features(indices, valid_mask)
    density = _density_compatibility(indices, similarity, valid_mask)

    view_sets: list[list[set[int]]] = []
    views: Sequence[np.ndarray] = () if stability_views is None else stability_views
    for view_number, view in enumerate(views):
        view_matrix = _as_float_matrix(view, f"stability_views[{view_number}]")
        if view_matrix.shape[0] != n_nodes:
            raise ValueError("Every stability view must have the same number of rows as data.")
        view_embedding = _project(view_matrix, pca_dim, seed + view_number + 1)
        view_indices, _, _ = _knn(
            view_embedding,
            indices.shape[1],
            backend=backend,
            exact_max_nodes=exact_max_nodes,
            hnsw_m=hnsw_m,
            hnsw_ef_search=hnsw_ef_search,
        )
        view_sets.append([set(row.tolist()) for row in view_indices])
    stability = _edge_recurrence(indices, valid_mask, view_sets)

    profile: dict[str, Any] = {
        "n_nodes": n_nodes,
        "neighbor_k": int(indices.shape[1]),
        "projection_dim": int(embedding.shape[1]),
        "stability_view_count": len(view_sets),
        "knn_backend": backend_used,
        "mean_similarity": float(similarity.mean()) if similarity.size else 0.0,
        "mean_mutual": float(mutual.mean()) if mutual.size else 0.0,
        "mean_snn": float(snn.mean()) if snn.size else 0.0,
        "mean_density_compatibility": float(density.mean()) if density.size else 0.0,
        "mean_stability": float(stability.mean()) if stability.size else 0.0,
    }
    return KNNGraph(
        indices=indices,
        similarity=similarity,
        mutual=mutual,
        snn=snn,
        density=density,
        stability=stability,
        embedding=embedding,
        valid_mask=valid_mask,
        profile=profile,
    )


def build_consensus_graph(
    input_graph: KNNGraph,
    latent_graph: KNNGraph,
    mode: Literal["intersection", "union"] = "intersection",
    k: int | None = None,
) -> KNNGraph:
    """Combine input- and latent-space graphs with deterministic edge ranking.

    Common edges receive stability ``1``; edges found in only one graph receive
    ``0.5`` and are included only in ``union`` mode.  Ranking prioritizes
    stability, then mean similarity, then node id, making graph refreshes fully
    reproducible for fixed embeddings.
    """

    if input_graph.n_nodes != latent_graph.n_nodes:
        raise ValueError("input_graph and latent_graph must contain the same nodes.")
    if mode not in {"intersection", "union"}:
        raise ValueError(f"mode must be 'intersection' or 'union', got {mode!r}.")
    n_nodes = input_graph.n_nodes
    requested_width = max(input_graph.k, latent_graph.k) if k is None else max(int(k), 0)
    requested_width = min(requested_width, max(n_nodes - 1, 0))

    selected: list[list[tuple[int, float, float, float]]] = []
    maximum_width = 0
    for i in range(n_nodes):
        source_maps: list[dict[int, tuple[float, float]]] = []
        for graph in (input_graph, latent_graph):
            source_maps.append(
                {
                    int(graph.indices[i, p]): (
                        float(graph.similarity[i, p]),
                        float(graph.density[i, p]),
                    )
                    for p in np.flatnonzero(graph.valid_mask[i])
                }
            )
        left, right = source_maps
        candidates = (left.keys() & right.keys()) if mode == "intersection" else (left.keys() | right.keys())
        row: list[tuple[int, float, float, float]] = []
        for j in candidates:
            present = int(j in left) + int(j in right)
            values = [source[j] for source in source_maps if j in source]
            mean_similarity = float(np.mean([value[0] for value in values]))
            mean_density = float(np.mean([value[1] for value in values]))
            row.append((int(j), mean_similarity, mean_density, present / 2.0))
        row.sort(key=lambda item: (-item[3], -item[1], item[0]))
        if requested_width > 0:
            row = row[:requested_width]
        selected.append(row)
        maximum_width = max(maximum_width, len(row))

    width = min(maximum_width, requested_width) if requested_width > 0 else 0
    indices = np.full((n_nodes, width), -1, dtype=np.int64)
    similarity = np.zeros((n_nodes, width), dtype=np.float32)
    density = np.zeros((n_nodes, width), dtype=np.float32)
    stability = np.zeros((n_nodes, width), dtype=np.float32)
    valid_mask = np.zeros((n_nodes, width), dtype=bool)
    for i, row in enumerate(selected):
        for position, (j, sim, density_score, stable) in enumerate(row[:width]):
            indices[i, position] = j
            similarity[i, position] = sim
            density[i, position] = density_score
            stability[i, position] = stable
            valid_mask[i, position] = True
    mutual, snn = _topology_features(indices, valid_mask)
    embedding = latent_graph.embedding
    valid_count = int(valid_mask.sum())
    profile: dict[str, Any] = {
        "n_nodes": n_nodes,
        "neighbor_k": width,
        "consensus_mode": mode,
        "valid_edge_count": valid_count,
        "mean_valid_degree": float(valid_mask.sum(axis=1).mean()) if n_nodes else 0.0,
        "mean_stability": float(stability.sum() / max(valid_count, 1)),
    }
    return KNNGraph(
        indices=indices,
        similarity=similarity,
        mutual=mutual,
        snn=snn,
        density=density,
        stability=stability,
        embedding=embedding,
        valid_mask=valid_mask,
        profile=profile,
    )


def edge_recurrence_against(graph: KNNGraph, reference: KNNGraph | None) -> np.ndarray:
    """Return an edge recurrence target from an *independent* prior graph.

    This differs from ``graph.stability``: the latter is an input/latent
    candidate-edge feature for the current refresh, whereas this function
    checks whether the current candidate edge already existed in a prior graph
    snapshot.  Keeping the target separate prevents a gate from learning the
    identity shortcut ``gate = current_stability_feature``.

    When no prior graph exists (the first dynamic refresh), all targets are
    zero.  Callers should omit the corresponding loss for that refresh rather
    than treat the zeros as negative labels.
    """

    result = np.zeros(graph.indices.shape, dtype=np.float32)
    if reference is None:
        return result
    if graph.n_nodes != reference.n_nodes:
        raise ValueError("graph and reference must contain the same number of nodes.")
    reference_sets = [
        set(reference.indices[node, reference.valid_mask[node]].tolist())
        for node in range(reference.n_nodes)
    ]
    for node in range(graph.n_nodes):
        for position in np.flatnonzero(graph.valid_mask[node]):
            result[node, position] = float(int(graph.indices[node, position]) in reference_sets[node])
    return result


def compute_edge_features(graph: KNNGraph) -> np.ndarray:
    """Stack the five canonical edge features as ``float32``."""

    features = np.stack(
        (graph.similarity, graph.mutual, graph.snn, graph.density, graph.stability),
        axis=-1,
    ).astype(np.float32, copy=False)
    return features * graph.valid_mask[..., None]


def edge_features_tensor(
    graph: KNNGraph,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert canonical edge features to a PyTorch tensor."""

    return torch.as_tensor(compute_edge_features(graph), device=device, dtype=dtype)
