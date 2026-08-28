from __future__ import annotations

"""Shared protocol helpers for the V9 conditional-effect study.

This module deliberately keeps dataset discovery, X-only diagnostics and
post-fit metric handling separate.  The training runner receives ``y=None``;
labels are loaded only by the outer benchmark process for K and final metrics.
"""

import csv
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (REPO_ROOT / "datasets").resolve()
AHDPC_ROOT = DATA_ROOT / "AHDPC" / "processed"
V9_CONFIG_DIR = REPO_ROOT / "methods" / "TopoGate" / "learnable_gate" / "configs"
DEFAULT_RESULT_ROOT = REPO_ROOT / "result" / "v9_regime_2026-08-06"
DEFAULT_TMP_ROOT = Path("/tmp/v9_regime_20260806")
DEFAULT_SEEDS = (42, 123, 7)
CASE_SEEDS = (42, 123, 7, 2026, 31415)
PROTOCOL_ID = "v9_regime_protocol_v1"
SPLIT_SEED = 20260806
MAX_ELEMENTS = 80_000_000
MAX_SAMPLES = 20_000
MIN_SAMPLES = 100
MIN_CLUSTERS = 2
MAX_CLUSTERS = 50

BASE_OVERRIDES: dict[str, Any] = {
    "variant": "learnable_gate_v9_adaptive",
    "epochs": 80,
    "mask_ratio": 0.3,
    "neighbor_k": 5,
    "mix_neighbors": 4,
    "warmup_epochs": 20,
    "ramp_epochs": 10,
    "n_top_features": 0,
    "knn_pca_mode": "adaptive",
    "knn_pca_dim": 2000,
    "scale_input": False,
    "hidden_size": 128,
    "batch_size": 256,
    "legacy_labels_output": False,
    "config_dir": str(V9_CONFIG_DIR),
}

VARIANT_OVERRIDES: dict[str, dict[str, Any]] = {
    "full": {"gate_mode": "learned", "mix_mode": "reliability", "pseudo_weight": 0.3},
    "nomix": {"gate_mode": "learned", "mix_mode": "none", "pseudo_weight": 0.0},
    # Vanilla scMAE task baseline.  This is intentionally separate from the
    # frozen Full/NoMix estimand: it disables the gate module itself, while
    # retaining the V9 input/training protocol and the same backbone contract.
    "scmae": {"gate_mode": "none", "mix_mode": "none", "pseudo_weight": 0.0},
    "static": {"gate_mode": "topology", "mix_mode": "reliability", "pseudo_weight": 0.3},
    "random": {"gate_mode": "learned", "mix_mode": "random", "pseudo_weight": 0.3},
    "far": {"gate_mode": "learned", "mix_mode": "far", "pseudo_weight": 0.3},
}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=json_default), encoding="utf-8")


def normalize_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def dataset_id_for(path: Path, source_kind: str) -> str:
    stem = path.stem
    return f"{source_kind}__{normalize_name(stem)}"


def _npy_header_from_member(handle: Any) -> tuple[int, ...]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, _, _ = np.lib.format.read_array_header_1_0(handle)
    elif version in {(2, 0), (3, 0)}:
        shape, _, _ = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"unsupported NPY header version {version}")
    return tuple(int(dim) for dim in shape)


def npz_member_name(path: Path, key: str) -> str | None:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name == f"{key}.npy" or name.endswith(f"/{key}.npy"):
                return name
    return None


def read_npz_shape(path: Path) -> tuple[int, int]:
    """Read X shape without materialising the matrix."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        member = next(
            (name for key in ("x", "X", "data") for name in names if name == f"{key}.npy" or name.endswith(f"/{key}.npy")),
            None,
        )
        if member is None:
            raise ValueError(f"{path} has no x/X/data member")
        with archive.open(member) as handle:
            shape = _npy_header_from_member(handle)
    if len(shape) != 2:
        raise ValueError(f"{path} feature matrix is not 2-D: {shape}")
    return shape[0], shape[1]


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as data:
        x_key = next((key for key in ("x", "X", "data") if key in data.files), None)
        y_key = next((key for key in ("y", "Y", "labels", "label") if key in data.files), None)
        if x_key is None:
            raise ValueError(f"{path} has no x/X/data key")
        x = np.asarray(data[x_key], dtype=np.float32)
        y = None if y_key is None else np.asarray(data[y_key]).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"{path} X must be 2-D, got {x.shape}")
    if y is not None and len(y) != x.shape[0]:
        raise ValueError(f"{path} X/y mismatch: {x.shape} vs {y.shape}")
    return x, y


def load_x(path: Path) -> np.ndarray:
    x, _ = load_xy(path)
    return x


def deterministic_rows(x: np.ndarray, y: np.ndarray | None, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x, y, None
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(x.shape[0], size=int(max_samples), replace=False))
    return x[indices], None if y is None else y[indices], indices


def standardize_x(x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the single label-free preprocessing used by this protocol."""
    x = np.asarray(x, dtype=np.float32)
    finite_before = np.isfinite(x)
    nonfinite_fraction = float(1.0 - np.mean(finite_before))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler(with_mean=True, with_std=True)
    z = scaler.fit_transform(x).astype(np.float32)
    zero_fraction = float(np.mean(x == 0.0))
    return z, {
        "input_preprocessing": "nan_to_num_then_column_standard_scaler",
        "nonfinite_fraction": nonfinite_fraction,
        "zero_fraction": zero_fraction,
        "constant_feature_fraction": float(np.mean(np.asarray(scaler.scale_) == 1.0)),
    }


def adaptive_pca_dim(x: np.ndarray, max_dim: int = 2000, seed: int = SPLIT_SEED) -> int:
    actual_max = min(int(max_dim), x.shape[0] - 1, x.shape[1])
    if actual_max <= 0:
        return max(1, actual_max)
    pca = PCA(n_components=actual_max, random_state=int(seed))
    pca.fit(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    selected = int(np.searchsorted(cumulative, 0.95) + 1)
    return max(1, min(selected, actual_max))


def _graph_component_fraction(indices: np.ndarray) -> tuple[int, float]:
    n, k = indices.shape
    if n <= 1 or k == 0:
        return n, 1.0 / max(1, n)
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = indices.reshape(-1)
    matrix = csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )
    count, labels = connected_components(matrix, directed=False)
    sizes = np.bincount(labels, minlength=count)
    return int(count), float(np.max(sizes) / n)


def build_x_only_features(
    x: np.ndarray,
    *,
    seed: int = SPLIT_SEED,
    max_analysis_samples: int = 4000,
    max_analysis_features: int = 512,
) -> dict[str, Any]:
    """Compute pre-training, label-free features using the V9 graph recipe."""
    from methods.TopoGate.learnable_gate.neighbor_graph import build_pca_knn_graph, compute_edge_reliability

    original_n, original_d = x.shape
    z, prep = standardize_x(x)
    z, _, row_indices = deterministic_rows(z, None, max_analysis_samples, seed)
    feature_indices = None
    if z.shape[1] > max_analysis_features:
        rng = np.random.default_rng(int(seed) + 17)
        feature_indices = np.sort(rng.choice(z.shape[1], size=int(max_analysis_features), replace=False))
        z = z[:, feature_indices]
    pca_dim = adaptive_pca_dim(z, max_dim=2000, seed=seed)
    graph = build_pca_knn_graph(z, k=min(5, z.shape[0] - 1), pca_dim=pca_dim, tau=0.2, seed=seed)
    _, weights, edge_summary = compute_edge_reliability(
        graph,
        mode="sim_mutual_snn_distance",
        gamma_sim=1.0,
        gamma_mutual=1.0,
        gamma_snn=1.0,
        gamma_distance=1.0,
    )
    distances = np.asarray(graph.distance, dtype=np.float64)
    if distances.size:
        row_distance = distances[:, 0]
        mean_distance = float(np.mean(distances))
        p95_distance = float(np.quantile(distances, 0.95))
        cv_distance = float(np.std(distances) / max(np.mean(distances), 1e-8))
        entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, None)), axis=1)
        effective = np.exp(entropy)
        neighbor_mean = np.einsum("ij,ijk->ik", weights, z[graph.indices])
        perturb = np.linalg.norm(neighbor_mean - z, axis=1) / np.maximum(np.linalg.norm(z, axis=1), 1e-8)
    else:
        row_distance = np.zeros(z.shape[0])
        mean_distance = p95_distance = cv_distance = 0.0
        entropy = effective = perturb = np.zeros(z.shape[0])
    components, largest_fraction = _graph_component_fraction(graph.indices)
    return {
        "analysis_n": int(z.shape[0]),
        "analysis_d": int(z.shape[1]),
        "analysis_row_sampled": row_indices is not None,
        "analysis_feature_sampled": feature_indices is not None,
        "analysis_pca_dim": int(pca_dim),
        "analysis_pca_dim_ratio": float(pca_dim / max(1, z.shape[1])),
        "mean_1nn_distance": float(np.mean(row_distance)) if row_distance.size else 0.0,
        "median_1nn_distance": float(np.median(row_distance)) if row_distance.size else 0.0,
        "mean_knn_distance": mean_distance,
        "p95_knn_distance": p95_distance,
        "cv_knn_distance": cv_distance,
        "mean_mutual_ratio": float(np.mean(graph.mutual)) if graph.mutual.size else 0.0,
        "mean_snn": float(np.mean(graph.snn)) if graph.snn.size else 0.0,
        "graph_components": int(components),
        "graph_largest_component_fraction": largest_fraction,
        "reliability_entropy": float(np.mean(entropy)) if entropy.size else 0.0,
        "effective_neighbor_count": float(np.mean(effective)) if effective.size else 0.0,
        "neighbor_perturbation_norm": float(np.mean(perturb)) if perturb.size else 0.0,
        **prep,
        "n": int(original_n),
        "d": int(original_d),
    }


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"protocol_id": PROTOCOL_ID, "datasets": payload}
    if "datasets" not in payload:
        raise ValueError(f"manifest {path} has no datasets list")
    return payload


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_family(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("retina", "campbell", "baron", "quake", "hrvatin", "lung", "mouse")):
        return "scrna"
    if any(token in lower for token in ("text", "enron", "reuters", "news", "spam", "wos", "cnae", "imdb", "hate")):
        return "text"
    if any(token in lower for token in ("mnist", "cifar", "coil", "face", "image", "pcam", "indian", "satellite", "olivetti")):
        return "image"
    if any(token in lower for token in ("2d_", "flame", "aggregation", "twodiamond", "unbalance", "asymmetric", "smile")):
        return "synthetic"
    return "tabular"


def get_record(manifest: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    for record in manifest["datasets"]:
        if record.get("dataset_id") == dataset_id:
            return record
    raise KeyError(f"dataset_id not found in manifest: {dataset_id}")
