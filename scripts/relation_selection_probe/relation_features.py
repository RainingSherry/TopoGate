"""Frozen relation features for :mod:`relation_selection_probe`.

The feature extractor consumes only the audited S0 ``H0`` and candidate-pool
artifacts.  Labels are deliberately absent from this module.  Diagnostic target
construction lives in ``rs1_information_probe.py`` and is kept separate from
feature extraction and selector scoring.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
S0_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S0_freeze"
S1_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S1_oracle_v2"

DATASETS: tuple[str, ...] = (
    "cnae9",
    "Mouse_retina",
    "sms_spam_collection",
    "Baron Human",
    "Campbell",
    "hate_speech",
)
PRIMARY_DATASETS: tuple[str, ...] = ("cnae9", "Campbell", "sms_spam_collection")
PILOT_SEEDS: tuple[int, ...] = (42, 123, 7)
HOLDOUT_SEEDS: tuple[int, ...] = (42, 123, 7, 3032, 3033)
NEIGHBOR_K = 20
VIEW_SEEDS: tuple[int, ...] = (17, 31, 47, 61, 73, 89, 101, 113)
VIEW_DIM = 96
BUDGET_CAP = 8
MATERIALITY_DELTA = 0.03
RS1_DELTA_AP = 0.10
RS1_LIFT = 1.5
RS2_CAPTURE = 0.25

GEOMETRY_FEATURES: tuple[str, ...] = (
    "cosine",
    "cosine_rank",
    "cosine_percentile",
    "margin_to_budget",
    "margin_to_next_budget",
    "distance",
    "distance_local_mean_ratio",
    "distance_local_median_ratio",
)
TOPOLOGY_FEATURES: tuple[str, ...] = (
    "mutual",
    "snn_count",
    "jaccard",
    "common_neighbor_ratio",
    "target_indegree",
    "degree_asymmetry",
    "local_hubness",
)
STABILITY_FEATURES: tuple[str, ...] = (
    "stability_recurrence",
    "stability_cosine_std",
)
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "G": GEOMETRY_FEATURES,
    "T": TOPOLOGY_FEATURES,
    "S": STABILITY_FEATURES,
    "G+T": GEOMETRY_FEATURES + TOPOLOGY_FEATURES,
    "G+S": GEOMETRY_FEATURES + STABILITY_FEATURES,
    "T+S": TOPOLOGY_FEATURES + STABILITY_FEATURES,
    "G+T+S": GEOMETRY_FEATURES + TOPOLOGY_FEATURES + STABILITY_FEATURES,
}


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_h0_and_pool(dataset: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    root = S0_ROOT / "datasets" / dataset
    h0_path = root / "H0.npy"
    pool_path = root / "candidate_pool.npz"
    if not h0_path.exists() or not pool_path.exists():
        raise FileNotFoundError(f"missing audited S0 artifacts for {dataset}: {root}")
    h0 = np.asarray(np.load(h0_path), dtype=np.float32)
    with np.load(pool_path, allow_pickle=False) as archive:
        pool = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"indices", "cosine", "positive_counts", "effective_budget"}
    if not required.issubset(pool):
        raise ValueError(f"candidate pool missing {sorted(required - set(pool))}")
    if pool["indices"].shape != pool["cosine"].shape:
        raise ValueError("candidate indices/cosine shape mismatch")
    if h0.shape[0] != pool["indices"].shape[0]:
        raise ValueError("H0 and candidate pool row mismatch")
    return h0, pool


@dataclass(frozen=True)
class EdgeTable:
    """Flattened positive candidate edges and label-free features."""

    n_samples: int
    rows: np.ndarray
    cols: np.ndarray
    cosine: np.ndarray
    budget: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]

    def feature(self, name: str) -> np.ndarray:
        try:
            index = self.feature_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.features[:, index]

    def family(self, family: str) -> np.ndarray:
        names = FEATURE_FAMILIES[family]
        indices = [self.feature_names.index(name) for name in names]
        return self.features[:, indices]


def _valid_pool_edges(pool: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(pool["indices"], dtype=np.int64)
    cosine = np.asarray(pool["cosine"], dtype=np.float32)
    valid = (indices >= 0) & (cosine > 0.0)
    rows, slots = np.nonzero(valid)
    return rows.astype(np.int64), slots.astype(np.int64), indices[rows, slots].astype(np.int64)


def _row_sets(indices: np.ndarray, cosine: np.ndarray) -> list[set[int]]:
    return [
        set(int(v) for v in row[(row >= 0) & (cosine_row > 0.0)])
        for row, cosine_row in zip(indices, cosine, strict=True)
    ]


def _geometry_features(
    rows: np.ndarray,
    slots: np.ndarray,
    cosine: np.ndarray,
    pool: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    indices = np.asarray(pool["indices"], dtype=np.int64)
    all_cos = np.asarray(pool["cosine"], dtype=np.float32)
    budgets = np.asarray(pool["effective_budget"], dtype=np.int64)
    n_edges = rows.size
    out = {name: np.zeros(n_edges, dtype=np.float32) for name in GEOMETRY_FEATURES}
    for row in range(indices.shape[0]):
        edge_ids = np.flatnonzero(
            (indices[row] >= 0) & (all_cos[row] > 0.0)
        )
        if edge_ids.size == 0:
            continue
        ordered = edge_ids[np.lexsort((edge_ids, -all_cos[row, edge_ids].astype(np.float64)))]
        rank = {int(slot): position + 1 for position, slot in enumerate(ordered)}
        row_start = int(np.searchsorted(rows, row, side="left"))
        row_end = int(np.searchsorted(rows, row, side="right"))
        distances = 1.0 - all_cos[row, edge_ids].astype(np.float64)
        mean_distance = max(float(np.mean(distances)), 1e-8)
        median_distance = max(float(np.median(distances)), 1e-8)
        b_i = int(budgets[row])
        ordered_cos = all_cos[row, ordered]
        threshold_b = float(ordered_cos[b_i - 1]) if b_i > 0 and b_i <= ordered_cos.size else 0.0
        threshold_next = float(ordered_cos[b_i]) if b_i >= 0 and b_i < ordered_cos.size else 0.0
        for edge_id in range(row_start, row_end):
            slot = int(slots[edge_id])
            value = float(cosine[edge_id])
            rank_value = rank[slot]
            distance = 1.0 - value
            out["cosine"][edge_id] = value
            out["cosine_rank"][edge_id] = float(rank_value)
            out["cosine_percentile"][edge_id] = float((ordered.size - rank_value + 1) / ordered.size)
            out["margin_to_budget"][edge_id] = value - threshold_b
            out["margin_to_next_budget"][edge_id] = value - threshold_next
            out["distance"][edge_id] = distance
            out["distance_local_mean_ratio"][edge_id] = distance / mean_distance
            out["distance_local_median_ratio"][edge_id] = distance / median_distance
    return out


def _topology_features(
    rows: np.ndarray,
    cols: np.ndarray,
    pool: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    indices = np.asarray(pool["indices"], dtype=np.int64)
    cosine = np.asarray(pool["cosine"], dtype=np.float32)
    sets = _row_sets(indices, cosine)
    degrees = np.asarray([len(value) for value in sets], dtype=np.float64)
    indegree = np.zeros(len(sets), dtype=np.float64)
    for values in sets:
        for target in values:
            indegree[target] += 1.0
    max_indegree = max(float(np.max(indegree)), 1.0) if indegree.size else 1.0
    out = {name: np.zeros(rows.size, dtype=np.float32) for name in TOPOLOGY_FEATURES}
    for edge_id, (row, col) in enumerate(zip(rows, cols, strict=True)):
        left = sets[int(row)]
        right = sets[int(col)]
        common = len(left.intersection(right))
        union = len(left.union(right))
        min_degree = min(len(left), len(right))
        total_degree = max(len(left) + len(right), 1)
        out["mutual"][edge_id] = float(int(int(row) in right))
        out["snn_count"][edge_id] = float(common)
        out["jaccard"][edge_id] = float(common / union) if union else 0.0
        out["common_neighbor_ratio"][edge_id] = float(common / min_degree) if min_degree else 0.0
        out["target_indegree"][edge_id] = float(indegree[int(col)])
        out["degree_asymmetry"][edge_id] = float((len(left) - len(right)) / total_degree)
        out["local_hubness"][edge_id] = float(indegree[int(col)] / max_indegree)
    return out


def _view_neighbor_sets(h0: np.ndarray, seed: int) -> tuple[list[set[int]], np.ndarray]:
    rng = np.random.default_rng(int(seed))
    dim = min(VIEW_DIM, h0.shape[1])
    dimensions = np.sort(rng.choice(h0.shape[1], size=dim, replace=False))
    view = np.asarray(h0[:, dimensions], dtype=np.float32)
    norms = np.linalg.norm(view, axis=1, keepdims=True)
    view = view / np.maximum(norms, 1e-8)
    k_eff = min(NEIGHBOR_K, max(h0.shape[0] - 1, 0))
    if k_eff == 0:
        return [set() for _ in range(h0.shape[0])], view
    nearest = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", n_jobs=-1)
    nearest.fit(view)
    _, raw = nearest.kneighbors(view)
    sets: list[set[int]] = []
    for row, values in enumerate(raw):
        filtered = [int(value) for value in values if int(value) != row][:k_eff]
        sets.append(set(filtered))
    return sets, view


def _stability_features(rows: np.ndarray, cols: np.ndarray, h0: np.ndarray) -> dict[str, np.ndarray]:
    recurrence = np.zeros(rows.size, dtype=np.float64)
    cosine_values: list[np.ndarray] = []
    for seed in VIEW_SEEDS:
        sets, view = _view_neighbor_sets(h0, seed)
        recurrence += np.asarray([int(int(col) in sets[int(row)]) for row, col in zip(rows, cols, strict=True)])
        cosine_values.append(np.asarray([float(np.dot(view[int(row)], view[int(col)])) for row, col in zip(rows, cols, strict=True)]))
    stacked = np.vstack(cosine_values)
    return {
        "stability_recurrence": (recurrence / len(VIEW_SEEDS)).astype(np.float32),
        "stability_cosine_std": np.std(stacked, axis=0).astype(np.float32),
    }


def extract_edge_features(h0: np.ndarray, pool: dict[str, np.ndarray]) -> EdgeTable:
    """Extract all frozen label-free relation features for positive pool edges."""
    h0 = np.asarray(h0, dtype=np.float32)
    rows, slots, cols = _valid_pool_edges(pool)
    cosine = np.asarray(pool["cosine"], dtype=np.float32)[rows, slots]
    budgets = np.asarray(pool["effective_budget"], dtype=np.int64)
    values: dict[str, np.ndarray] = {}
    values.update(_geometry_features(rows, slots, cosine, pool))
    values.update(_topology_features(rows, cols, pool))
    values.update(_stability_features(rows, cols, h0))
    names = GEOMETRY_FEATURES + TOPOLOGY_FEATURES + STABILITY_FEATURES
    features = np.column_stack([values[name] for name in names]).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("relation feature extraction produced non-finite values")
    metadata = {
        "feature_names": list(names),
        "feature_families": {key: list(value) for key, value in FEATURE_FAMILIES.items()},
        "n_samples": int(h0.shape[0]),
        "candidate_edge_count": int(rows.size),
        "candidate_pool_sha256": sha256_array(np.asarray(pool["indices"], dtype=np.int64)),
        "budget_hash": sha256_array(np.asarray(pool["effective_budget"], dtype=np.int64)),
        "h0_sha256": sha256_array(h0),
        "neighbor_k": NEIGHBOR_K,
        "view_seeds": list(VIEW_SEEDS),
        "view_dim": VIEW_DIM,
        "labels_used": False,
    }
    return EdgeTable(
        n_samples=int(h0.shape[0]),
        rows=rows,
        cols=cols,
        cosine=cosine,
        budget=budgets,
        features=features,
        feature_names=names,
        metadata=metadata,
    )


def save_edge_table(path: str | Path, table: EdgeTable) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rows=table.rows,
        cols=table.cols,
        cosine=table.cosine,
        budget=table.budget,
        features=table.features,
        feature_names=np.asarray(table.feature_names),
        metadata=np.asarray(json.dumps(jsonable(table.metadata), sort_keys=True)),
    )


def load_edge_table(path: str | Path) -> EdgeTable:
    with np.load(path, allow_pickle=False) as archive:
        names = tuple(str(value) for value in archive["feature_names"].tolist())
        metadata = json.loads(str(archive["metadata"].item()))
        features = np.asarray(archive["features"], dtype=np.float32)
        return EdgeTable(
            n_samples=int(metadata["n_samples"]),
            rows=np.asarray(archive["rows"], dtype=np.int64),
            cols=np.asarray(archive["cols"], dtype=np.int64),
            cosine=np.asarray(archive["cosine"], dtype=np.float32),
            budget=np.asarray(archive["budget"], dtype=np.int64),
            features=features,
            feature_names=names,
            metadata=metadata,
        )
