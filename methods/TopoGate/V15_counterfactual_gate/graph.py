from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from .sparse import PreparedInput


@dataclass
class CandidateGraph:
    indices: np.ndarray
    features: np.ndarray
    valid: np.ndarray
    raw_indices: np.ndarray
    latent_indices: np.ndarray
    raw_embedding: np.ndarray
    latent_embedding: np.ndarray
    profile: dict

    @property
    def n_nodes(self) -> int:
        return int(self.indices.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.indices.shape[1])

    def features_for(self, indices: np.ndarray) -> np.ndarray:
        return self.features[np.asarray(indices, dtype=np.int64)]

    def donor_indices(self, indices: np.ndarray) -> np.ndarray:
        return self.indices[np.asarray(indices, dtype=np.int64)]

    def edge_purity(self, labels: np.ndarray) -> float:
        y = np.asarray(labels).reshape(-1)
        rows, cols = np.where(self.valid)
        if rows.size == 0:
            return 0.0
        return float(np.mean(y[rows] == y[self.indices[rows, cols]]))

    def candidate_recall(self, labels: np.ndarray) -> float:
        """Budget-normalized recall of same-label local candidates.

        Dividing by every point in a class would make recall mechanically tiny
        for large datasets even when a kNN budget is entirely useful. The
        denominator is therefore the smaller of the candidate budget and the
        number of same-label alternatives for that anchor.
        """
        y = np.asarray(labels).reshape(-1)
        recalls: list[float] = []
        for i in range(self.n_nodes):
            same = np.flatnonzero(y == y[i])
            same = same[same != i]
            if same.size == 0:
                continue
            chosen = self.indices[i, self.valid[i]]
            denominator = min(chosen.size, same.size)
            if denominator > 0:
                recalls.append(float(np.intersect1d(chosen, same, assume_unique=False).size / denominator))
        return float(np.mean(recalls)) if recalls else 0.0


def restrict_candidate_scope(graph: CandidateGraph, scope: str) -> CandidateGraph:
    """Apply a graph-source scope without rebuilding either kNN view.

    Candidate construction remains a recall stage.  This function is an
    explicit, reversible graph ablation that can retain the union, keep only
    raw-supported or latent-supported edges, or require the two views to agree.
    The underlying indices and features are preserved so every scope can be
    replayed and audited against the same checkpoint.
    """
    if scope not in {"all", "both_views", "raw_supported", "latent_supported"}:
        raise ValueError(
            "candidate scope must be all, both_views, raw_supported, or latent_supported"
        )
    source = graph.features[:, :, 2]
    if scope == "all":
        scope_mask = np.ones_like(graph.valid, dtype=bool)
    elif scope == "both_views":
        scope_mask = np.abs(source) < 0.5
    elif scope == "raw_supported":
        scope_mask = source <= 0.5
    else:
        scope_mask = source >= -0.5
    valid = np.asarray(graph.valid, dtype=bool) & scope_mask
    profile = dict(graph.profile)
    profile["candidate_scope"] = scope
    profile["mean_valid_candidates"] = float(valid.sum(axis=1).mean()) if valid.size else 0.0
    profile["scope_empty_row_fraction"] = float(np.mean(~valid.any(axis=1))) if valid.size else 0.0
    return replace(graph, valid=valid, profile=profile)


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def _latent_graph_view(embedding: np.ndarray, max_dim: int | None, seed: int) -> np.ndarray:
    """Keep the EMA graph inexpensive when the model latent is wide."""
    values = np.asarray(embedding, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"latent embedding must be 2D, got {values.shape}")
    if max_dim is None or max_dim <= 0 or values.shape[1] <= int(max_dim):
        return _normalise_rows(values).astype(np.float32)
    upper = min(int(max_dim), values.shape[0] - 1, values.shape[1])
    if upper <= 0:
        return _normalise_rows(values).astype(np.float32)
    reduced = PCA(n_components=upper, random_state=seed, whiten=True).fit_transform(values)
    return _normalise_rows(np.asarray(reduced, dtype=np.float32)).astype(np.float32)


def graph_embedding(data: PreparedInput, max_dim: int, seed: int) -> np.ndarray:
    max_dim = max(2, int(max_dim))
    if data.sparse:
        upper = min(max_dim, data.n_samples - 1, data.n_features - 1)
        if upper <= 0:
            return data.matrix.toarray().astype(np.float32)
        model = TruncatedSVD(n_components=upper, random_state=seed)
        embedded = model.fit_transform(data.matrix)
    else:
        upper = min(max_dim, data.n_samples - 1, data.n_features)
        if upper <= 0:
            embedded = np.asarray(data.matrix, dtype=np.float32)
        else:
            model = PCA(n_components=upper, random_state=seed, whiten=True)
            embedded = model.fit_transform(np.asarray(data.matrix, dtype=np.float32))
    return _normalise_rows(np.nan_to_num(embedded, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float32)


def _knn(embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(embedding.shape[0])
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0), dtype=np.float32)
    k_eff = min(int(k), n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", algorithm="auto")
    nn.fit(embedding)
    distances, indices = nn.kneighbors(embedding)
    out_i = np.empty((n, k_eff), dtype=np.int64)
    out_s = np.empty((n, k_eff), dtype=np.float32)
    for row in range(n):
        keep = indices[row] != row
        row_i = indices[row][keep]
        row_s = (1.0 - distances[row][keep]).astype(np.float32)
        out_i[row] = row_i[:k_eff]
        out_s[row] = row_s[:k_eff]
    return out_i, out_s


def _rank_map(row: np.ndarray) -> dict[int, int]:
    return {int(value): int(pos + 1) for pos, value in enumerate(row.tolist())}


def _build_union(
    raw_indices: np.ndarray,
    raw_similarity: np.ndarray,
    latent_indices: np.ndarray,
    latent_similarity: np.ndarray,
    raw_embedding: np.ndarray,
    latent_embedding: np.ndarray,
    cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n = int(raw_indices.shape[0])
    width = min(int(cap), max(raw_indices.shape[1], latent_indices.shape[1], 1) * 2)
    indices = np.full((n, width), -1, dtype=np.int64)
    # Six compact retrieval/support features are the only scorer inputs.
    # Counterfactual utility components remain detached targets, not features.
    features = np.zeros((n, width, 6), dtype=np.float32)
    valid = np.zeros((n, width), dtype=bool)
    raw_sets = [set(row.tolist()) for row in raw_indices]
    latent_sets = [set(row.tolist()) for row in latent_indices]
    raw_radius = 1.0 - np.mean(raw_similarity, axis=1) if raw_similarity.size else np.ones(n, dtype=np.float32)
    latent_radius = 1.0 - np.mean(latent_similarity, axis=1) if latent_similarity.size else np.ones(n, dtype=np.float32)
    for i in range(n):
        raw_rank = _rank_map(raw_indices[i])
        latent_rank = _rank_map(latent_indices[i])
        both = sorted(
            set(raw_rank) & set(latent_rank),
            key=lambda j: (raw_rank[j] + latent_rank[j], int(j)),
        )
        raw_only = sorted(
            set(raw_rank) - set(latent_rank),
            key=lambda j: (raw_rank[j], int(j)),
        )
        latent_only = sorted(
            set(latent_rank) - set(raw_rank),
            key=lambda j: (latent_rank[j], int(j)),
        )
        # Reserve capacity for both retrieval routes. Candidate generation is
        # a recall problem; raw-only edges must not consume the whole retained
        # budget before latent-only candidates are even inspected.
        consensus_budget = (width + 1) // 2
        selected = both[:consensus_budget]
        remaining = width - len(selected)
        raw_budget = (remaining + 1) // 2
        latent_budget = remaining - raw_budget
        selected.extend(raw_only[:raw_budget])
        selected.extend(latent_only[:latent_budget])
        selected_set = set(selected)
        for candidate in both + raw_only + latent_only:
            if len(selected) >= width:
                break
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
        for pos, j in enumerate(selected):
            indices[i, pos] = int(j)
            valid[i, pos] = True
            rr = raw_rank.get(j, raw_indices.shape[1] + 1)
            lr = latent_rank.get(j, latent_indices.shape[1] + 1)
            in_raw = j in raw_rank
            in_latent = j in latent_rank
            # One scalar still identifies the source path: raw-only=-1,
            # consensus=0, latent-only=+1.
            source_indicator = float(int(in_latent) - int(in_raw))
            mutual = float((i in raw_sets[j]) or (i in latent_sets[j])) if j < n else 0.0
            union_size = max(1, len(raw_sets[i] | latent_sets[i] | raw_sets[j] | latent_sets[j]))
            overlap = len((raw_sets[i] | latent_sets[i]) & (raw_sets[j] | latent_sets[j])) / union_size
            density_ratio = np.log((float(raw_radius[j]) + 1e-4) / (float(raw_radius[i]) + 1e-4))
            # Slot 5 is filled at training/evaluation time with detached
            # teacher assignment support.
            features[i, pos] = np.asarray(
                [
                    min(1.0, rr / max(1, raw_indices.shape[1])),
                    min(1.0, lr / max(1, latent_indices.shape[1])),
                    source_indicator,
                    0.5 * (mutual + overlap),
                    float(np.clip(density_ratio, -5.0, 5.0) / 5.0),
                    0.0,
                ],
                dtype=np.float32,
            )
    profile = {
        "candidate_width": int(width),
        "mean_valid_candidates": float(valid.sum(axis=1).mean()) if n else 0.0,
        "raw_radius_mean": float(np.mean(raw_radius)) if n else 0.0,
        "latent_radius_mean": float(np.mean(latent_radius)) if n else 0.0,
        "union_source_both_fraction": float(np.mean(features[:, :, 2][valid] == 0.0)) if valid.any() else 0.0,
        "complete_union_requested": bool(
            int(cap) >= int(raw_indices.shape[1] + latent_indices.shape[1])
        ),
        "mean_distance_cv": float(np.mean(np.std(raw_similarity, axis=1) / np.clip(np.abs(np.mean(raw_similarity, axis=1)), 1e-6, None))) if raw_similarity.size else 0.0,
            "feature_names": [
                "raw_rank",
                "latent_rank",
                "source_indicator",
                "mutual_snn_overlap",
                "density_ratio",
                "teacher_assignment_agreement",
            ],
    }
    return indices, features, valid, profile


def build_candidate_graph(
    data: PreparedInput,
    *,
    k_raw: int,
    k_latent: int,
    candidate_cap: int,
    raw_svd_dim: int,
    latent_embedding: np.ndarray,
    latent_graph_dim: int | None = None,
    seed: int,
) -> CandidateGraph:
    raw_embedding = graph_embedding(data, raw_svd_dim, seed)
    raw_indices, raw_similarity = _knn(raw_embedding, k_raw)
    latent_embedding = _latent_graph_view(latent_embedding, latent_graph_dim, seed)
    latent_indices, latent_similarity = _knn(latent_embedding, k_latent)
    indices, features, valid, profile = _build_union(
        raw_indices,
        raw_similarity,
        latent_indices,
        latent_similarity,
        raw_embedding,
        latent_embedding,
        candidate_cap,
    )
    profile.update(
        {
            "k_raw": int(k_raw),
            "k_latent": int(k_latent),
            "latent_graph_dim": int(latent_embedding.shape[1]),
            "seed": int(seed),
        }
    )
    return CandidateGraph(
        indices=indices,
        features=features,
        valid=valid,
        raw_indices=raw_indices,
        latent_indices=latent_indices,
        raw_embedding=raw_embedding,
        latent_embedding=latent_embedding,
        profile=profile,
    )


def refresh_latent_graph(
    graph: CandidateGraph,
    latent_embedding: np.ndarray,
    k_latent: int,
    candidate_cap: int,
    latent_graph_dim: int | None = None,
    seed: int = 42,
) -> CandidateGraph:
    latent_embedding = _latent_graph_view(latent_embedding, latent_graph_dim, seed)
    latent_indices, latent_similarity = _knn(latent_embedding, k_latent)
    raw_similarity = np.einsum("ij,ikj->ik", graph.raw_embedding, graph.raw_embedding[graph.raw_indices]).astype(np.float32)
    indices, features, valid, profile = _build_union(
        graph.raw_indices,
        raw_similarity,
        latent_indices,
        latent_similarity,
        graph.raw_embedding,
        latent_embedding,
        candidate_cap,
    )
    profile.update(
        {
            "k_raw": int(graph.raw_indices.shape[1]),
            "k_latent": int(k_latent),
            "latent_graph_dim": int(latent_embedding.shape[1]),
            "refresh": True,
        }
    )
    return CandidateGraph(
        indices=indices,
        features=features,
        valid=valid,
        raw_indices=graph.raw_indices,
        latent_indices=latent_indices,
        raw_embedding=graph.raw_embedding,
        latent_embedding=latent_embedding,
        profile=profile,
    )


def replace_candidate_edges(graph: CandidateGraph, fraction: float, seed: int) -> CandidateGraph:
    """Replace a controlled fraction of donors with random non-self nodes.

    This is a stress-test hook, disabled by default. Features for replaced
    edges are neutralized rather than pretending that the original rank/source
    statistics still describe the random donor.
    """
    fraction = float(fraction)
    if fraction <= 0.0:
        return graph
    rng = np.random.default_rng(seed)
    indices = graph.indices.copy()
    features = graph.features.copy()
    valid = graph.valid.copy()
    for row in range(graph.n_nodes):
        existing = set(int(value) for value in indices[row, valid[row]])
        for col in np.flatnonzero(valid[row]):
            if rng.random() >= fraction:
                continue
            choices = np.asarray(
                [candidate for candidate in range(graph.n_nodes) if candidate != row and candidate not in existing],
                dtype=np.int64,
            )
            if choices.size == 0:
                choices = np.asarray([candidate for candidate in range(graph.n_nodes) if candidate != row], dtype=np.int64)
            if choices.size == 0:
                continue
            existing.discard(int(indices[row, col]))
            replacement = int(rng.choice(choices))
            indices[row, col] = replacement
            existing.add(replacement)
            features[row, col] = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    profile = dict(graph.profile)
    profile.update(
        {
            "graph_replacement_fraction": fraction,
            "graph_replacement_seed": int(seed),
        }
    )
    return CandidateGraph(
        indices=indices,
        features=features,
        valid=valid,
        raw_indices=graph.raw_indices,
        latent_indices=graph.latent_indices,
        raw_embedding=graph.raw_embedding,
        latent_embedding=graph.latent_embedding,
        profile=profile,
    )
