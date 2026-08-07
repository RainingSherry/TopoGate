#!/usr/bin/env python3
"""Training entry point for TopoGate V10: dynamic reliable-graph clustering.

V10 deliberately separates data reconstruction from graph regularisation:

* two independently corrupted views reconstruct the same clean sample;
* an EMA encoder periodically rebuilds a latent mutual-kNN graph;
* input/latent graph recurrence is explicit edge evidence for a candidate graph;
* an edge-level gate controls assignment consistency without NumPy detaches;
* clustering assignments are trained jointly with entropy balancing.

The graph terms use one scalar schedule only: zero during warm-up, followed by
one linear ramp to their configured weights.  Neighbour-mixed inputs are not a
reconstruction target in V10.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent for parent in (CURRENT_DIR, *CURRENT_DIR.parents) if (parent / "methods" / "TopoGate").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.TopoGate.v10_reliable_graph import (
    EdgeGate,
    V10AutoEncoder,
    V10LossWeights,
    apply_mask_corruption,
    build_consensus_graph,
    build_knn_graph,
    combine_v10_losses,
    edge_assignment_js_loss,
    edge_features_tensor,
    edge_recurrence_against,
    entropy_balance_loss,
    gate_budget_loss,
    gate_stability_loss,
    view_consistency_loss,
)


DEFAULT_CONFIG = CURRENT_DIR / "configs" / "topogate_v10_reliable_graph.yaml"


@dataclass
class PCASelection:
    dimension: int
    cumulative_variance: float
    threshold_reached: bool
    cap: int


class PrototypeHead(nn.Module):
    """Cosine prototype head used by the train-time clustering objective."""

    def __init__(self, latent_dim: int, n_clusters: int, temperature: float):
        super().__init__()
        if float(temperature) <= 0:
            raise ValueError("cluster temperature must be positive")
        self.prototypes = nn.Parameter(torch.empty(n_clusters, latent_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.temperature = float(temperature)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return F.softmax(z @ prototypes.t() / max(self.temperature, 1e-6), dim=-1)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _label_value(value: Any) -> Any:
    """Convert LabelEncoder classes to portable JSON scalar values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _load_config(path: str | Path | None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"V10 config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"V10 config must be a YAML mapping: {config_path}")
    config["config_path"] = str(config_path.resolve())
    return config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(gpu: int, no_cuda: bool = False) -> torch.device:
    if no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    if int(gpu) not in {1, 4, 5}:
        raise ValueError("Project rules allow only physical GPUs 1, 4, or 5.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        physical = [int(part.strip()) for part in visible.split(",") if part.strip()]
        if int(gpu) not in physical:
            raise ValueError(f"GPU {gpu} is not present in CUDA_VISIBLE_DEVICES={visible!r}")
        return torch.device(f"cuda:{physical.index(int(gpu))}")
    if int(gpu) >= torch.cuda.device_count():
        raise ValueError(f"GPU {gpu} is unavailable; detected {torch.cuda.device_count()} CUDA devices")
    return torch.device(f"cuda:{int(gpu)}")


def load_data(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        X = payload.get("X", payload.get("x", payload.get("data")))
        y = payload.get("y", payload.get("labels", payload.get("label", None)))
        if X is None:
            raise ValueError(f"{path} must contain X, x, or data")
        return np.asarray(X, dtype=np.float64), None if y is None else np.asarray(y).ravel()
    if path.is_dir():
        import zlib

        with (path / "data.bin").open("rb") as handle:
            X = np.asarray(json.loads(zlib.decompress(handle.read()).decode("utf-8")), dtype=np.float64)
        label_path = path / "label.bin"
        if not label_path.exists():
            return X, None
        with label_path.open("rb") as handle:
            y = np.asarray(json.loads(zlib.decompress(handle.read()).decode("utf-8"))).ravel()
        return X, y
    raise ValueError(f"Unsupported dataset path: {path}")


def _preprocess(X: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError(f"X must have shape (n>=2, d>=1), received {X.shape}")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if config.get("input_mode", "raw") == "log1p":
        if np.min(X) < 0:
            raise ValueError("input_mode=log1p requires non-negative observations")
        X = np.log1p(X)

    original_features = int(X.shape[1])
    n_top = int(config.get("n_top_features", 0))
    selected_features: np.ndarray | None = None
    if 0 < n_top < X.shape[1]:
        # Feature selection intentionally precedes StandardScaler; otherwise
        # every non-constant feature has approximately unit variance.
        variance = np.var(X, axis=0)
        selected_features = np.argsort(variance)[-n_top:]
        X = X[:, selected_features]

    if bool(config.get("scale_input", True)):
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    metadata = {
        "original_n_features": original_features,
        "model_n_features": int(X.shape[1]),
        "n_top_features": n_top,
        "feature_selection_before_scaling": True,
        "selected_feature_indices": selected_features,
        "input_mode": config.get("input_mode", "raw"),
        "scale_input": bool(config.get("scale_input", True)),
    }
    return X, metadata


def _select_pca_dimension(
    X: np.ndarray,
    max_dim: int,
    variance_threshold: float,
    min_dim: int,
    seed: int,
) -> tuple[PCASelection, np.ndarray]:
    if not 0.0 < float(variance_threshold) <= 1.0:
        raise ValueError("knn_pca_variance must be in (0, 1]")
    if int(max_dim) < 1 or int(min_dim) < 1:
        raise ValueError("PCA dimensions must be positive")
    cap = max(1, min(int(max_dim), X.shape[0] - 1, X.shape[1]))
    if cap >= min(X.shape):
        return PCASelection(cap, 1.0, True, cap), X.astype(np.float32, copy=False)
    # Reuse the fitted transform rather than fitting PCA a second time inside
    # graph construction.
    fitted = PCA(n_components=cap, svd_solver="auto", random_state=seed)
    transformed = fitted.fit_transform(X)
    cumulative = np.cumsum(fitted.explained_variance_ratio_)
    hit = np.flatnonzero(cumulative >= float(variance_threshold))
    if hit.size:
        dim = max(1, min(cap, max(int(min_dim), int(hit[0]) + 1)))
        reached = True
    else:
        dim = cap
        reached = False
    return (
        PCASelection(dim, float(cumulative[dim - 1]), reached, cap),
        transformed[:, :dim].astype(np.float32, copy=False),
    )


def _graph_scale(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """Exactly one linear graph schedule; no second multiplication elsewhere."""
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch - warmup_epochs) / float(ramp_epochs)))


@torch.no_grad()
def _ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for name, ema_value in ema_state.items():
        value = model_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(value, alpha=1.0 - decay)
        else:
            ema_value.copy_(value)


@torch.no_grad()
def _encode_all(model: V10AutoEncoder, X_cpu: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, X_cpu.shape[0], batch_size):
        z = model.encode(X_cpu[start : start + batch_size].to(device))
        chunks.append(z.detach().cpu().numpy())
    return np.nan_to_num(np.concatenate(chunks, axis=0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _valid_edges(
    graph: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = np.asarray(graph.indices, dtype=np.int64)
    n, k = indices.shape
    valid = np.asarray(getattr(graph, "valid_mask", np.ones((n, k), dtype=bool)), dtype=bool)
    valid &= indices >= 0
    valid &= indices < n
    sources = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], (n, k))
    if not np.any(valid):
        raise RuntimeError("The input/latent candidate graph contains no valid edges")
    features = edge_features_tensor(graph, device=torch.device("cpu"), dtype=torch.float32)
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)
    return (
        torch.from_numpy(sources[valid].copy()),
        torch.from_numpy(indices[valid].copy()),
        features[torch.from_numpy(valid)],
        torch.from_numpy(np.asarray(graph.stability, dtype=np.float32)[valid].copy()),
        torch.from_numpy(valid),
    )


def _masked_reconstruction(
    model: V10AutoEncoder,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return model.reconstruction_loss(reconstruction, target, mask)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    aligned = np.zeros_like(y_pred)
    rows, cols = linear_sum_assignment(-cm)
    for row, col in zip(rows, cols):
        aligned[y_pred == col] = row
    return {
        "acc": float(accuracy_score(y_true, aligned)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, aligned, average="macro", zero_division=0)),
        "fmi": float(fowlkes_mallows_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
    }


def _mean_history(sums: dict[str, float], count: int) -> dict[str, float]:
    return {name: value / max(1, count) for name, value in sums.items()}


def train_v10(
    X: np.ndarray,
    y: np.ndarray | None,
    n_clusters: int | None,
    save_dir: str | Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Train the complete V10 objective and persist all audit artefacts."""
    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    device = _resolve_device(int(config.get("gpu", 1)), bool(config.get("no_cuda", False)))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X_np, preprocessing = _preprocess(X, config)
    label_mapping: dict[str, Any] | None = None
    if y is None:
        y_encoded = None
        if n_clusters is None:
            raise ValueError("n_clusters is required when labels are unavailable")
        K = int(n_clusters)
        k_source = "explicit_n_clusters"
    else:
        label_encoder = LabelEncoder().fit(np.asarray(y).ravel())
        y_encoded = label_encoder.transform(np.asarray(y).ravel()).astype(np.int64)
        label_mapping = {
            str(index): _label_value(value)
            for index, value in enumerate(label_encoder.classes_)
        }
        detected_k = int(np.unique(y_encoded).size)
        if n_clusters is not None and int(n_clusters) != detected_k:
            raise ValueError(
                f"Provided n_clusters={n_clusters} disagrees with unique(y)={detected_k}; "
                "project rules require automatic K detection from y"
            )
        K = detected_k
        k_source = "labels_unique"
    if K < 2 or K > X_np.shape[0]:
        raise ValueError(f"n_clusters must be in [2, n_samples], received {K}")

    graph_enabled = bool(config.get("graph_enabled", True))
    dynamic_graph = bool(config.get("dynamic_graph", True))
    neighbor_k = int(config.get("neighbor_k", 10))
    if graph_enabled and neighbor_k < 1:
        raise ValueError("neighbor_k must be at least 1 when graph_enabled=True")
    knn_options = {
        "backend": str(config.get("knn_backend", "auto")),
        "exact_max_nodes": int(config.get("knn_exact_max_nodes", 5000)),
        "hnsw_m": int(config.get("knn_hnsw_m", 32)),
        "hnsw_ef_search": int(config.get("knn_hnsw_ef_search", 64)),
    }
    if knn_options["backend"] not in {"exact", "faiss_hnsw", "auto"}:
        raise ValueError("knn_backend must be one of: exact, faiss_hnsw, auto")
    if any(knn_options[name] < 1 for name in ("exact_max_nodes", "hnsw_m", "hnsw_ef_search")):
        raise ValueError("kNN backend size parameters must be positive")

    if graph_enabled:
        pca_selection, input_graph_embedding = _select_pca_dimension(
            X_np,
            max_dim=int(config.get("knn_pca_max_dim", 200)),
            variance_threshold=float(config.get("knn_pca_variance", 0.95)),
            min_dim=int(config.get("knn_pca_min_dim", 2)),
            seed=seed,
        )
    else:
        pca_selection = PCASelection(dimension=0, cumulative_variance=0.0, threshold_reached=False, cap=0)
        input_graph_embedding = None
    input_graph = (
        build_knn_graph(
            input_graph_embedding,
            k=neighbor_k,
            pca_dim=None,
            seed=seed,
            **knn_options,
        )
        if graph_enabled
        else None
    )

    X_cpu = torch.from_numpy(X_np)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(X_cpu),
        batch_size=int(config.get("batch_size", 256)),
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    model = V10AutoEncoder(
        input_dim=X_np.shape[1],
        latent_dim=int(config.get("latent_dim", 64)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        decoder_rank=int(config.get("decoder_rank", 64)),
        dropout=float(config.get("dropout", 0.1)),
        condition_on_mask=bool(config.get("condition_on_mask", False)),
        reconstruction_kind=str(config.get("reconstruction_kind", "mse")),
        masked_weight=float(config.get("masked_weight", 1.0)),
        visible_weight=float(config.get("unmasked_weight", 0.2)),
    ).to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    cluster_head = PrototypeHead(
        int(config.get("latent_dim", 64)),
        K,
        float(config.get("cluster_temperature", 0.2)),
    ).to(device)
    ema_cluster_head = copy.deepcopy(cluster_head).to(device).eval()
    for parameter in ema_cluster_head.parameters():
        parameter.requires_grad_(False)
    edge_gate = EdgeGate(
        feature_dim=5,
        hidden_dim=int(config.get("gate_hidden_dim", 32)),
        gate_min=float(config.get("gate_min", 0.0)),
        gate_max=float(config.get("gate_max", 1.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": float(config.get("lr", 1e-3))},
            {"params": cluster_head.parameters(), "lr": float(config.get("cluster_lr", config.get("lr", 1e-3)))},
            {"params": edge_gate.parameters(), "lr": float(config.get("gate_lr", config.get("lr", 1e-3)))},
        ],
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    warmup_epochs = int(config.get("warmup_epochs", 20))
    ramp_epochs = int(config.get("ramp_epochs", 10))
    refresh_interval = max(1, int(config.get("refresh_interval", 5)))
    epochs = int(config.get("epochs", 100))
    edge_batch_size = max(1, int(config.get("edge_batch_size", config.get("batch_size", 256))))
    ema_decay = float(config.get("ema_decay", 0.99))
    mask_ratio = float(config.get("mask_ratio", 0.3))
    mask_strategy = str(config.get("mask_strategy", "zero"))
    if epochs < 1 or warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError("epochs must be positive and warmup/ramp epochs must be non-negative")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1]")
    confidence_start = float(config.get("confidence_fraction_start", 0.2))
    confidence_end = float(config.get("confidence_fraction_end", 1.0))
    if not 0.0 <= confidence_start <= confidence_end <= 1.0:
        raise ValueError("Require 0 <= confidence_fraction_start <= confidence_fraction_end <= 1")
    graph_rng = np.random.default_rng(seed + 1009)
    objective_weights = V10LossWeights(
        reconstruction=1.0,
        view_consistency=float(config.get("lambda_view", 0.1)),
        edge_assignment=float(config.get("lambda_edge", 0.2)),
        entropy_balance=float(config.get("lambda_entropy", 0.05)),
        gate_budget=float(config.get("lambda_gate_budget", 0.05)),
        gate_temporal=float(config.get("lambda_gate_temporal", 0.05)),
    )

    trusted_sources: torch.Tensor | None = None
    trusted_targets: torch.Tensor | None = None
    trusted_features: torch.Tensor | None = None
    trusted_stability: torch.Tensor | None = None
    trusted_temporal_target: torch.Tensor | None = None
    temporal_target_available = False
    previous_latent_graph: Any | None = None
    graph_history: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    prototype_initialization_epoch: int | None = None
    prototype_initialization_method: str | None = None
    cluster_prior: torch.Tensor | None = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        graph_scale = _graph_scale(epoch, warmup_epochs, ramp_epochs) if graph_enabled else 0.0
        ema_clean_embedding: np.ndarray | None = None
        prototype_initialization_event = False
        if graph_scale > 0.0 and prototype_initialization_epoch is None:
            ema_clean_embedding = _encode_all(
                ema_model,
                X_cpu,
                int(config.get("eval_batch_size", max(512, int(config.get("batch_size", 256)) * 2))),
                device,
            )
            normalized_for_initialization = ema_clean_embedding / np.clip(
                np.linalg.norm(ema_clean_embedding, axis=1, keepdims=True),
                1e-12,
                None,
            )
            initialization = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(
                normalized_for_initialization
            )
            centers = initialization.cluster_centers_.astype(np.float32)
            centers /= np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
            centers_t = torch.from_numpy(centers).to(device)
            with torch.no_grad():
                cluster_head.prototypes.copy_(centers_t)
                ema_cluster_head.prototypes.copy_(centers_t)
            prior_mode = str(config.get("cluster_prior_mode", "warmup_kmeans"))
            if prior_mode == "warmup_kmeans":
                smoothing = float(config.get("cluster_prior_smoothing", 1e-3))
                if smoothing < 0:
                    raise ValueError("cluster_prior_smoothing must be non-negative")
                counts = np.bincount(initialization.labels_, minlength=K).astype(np.float32)
                counts += smoothing
                cluster_prior = torch.from_numpy(counts / counts.sum()).to(device)
            elif prior_mode == "uniform":
                cluster_prior = torch.full((K,), 1.0 / K, device=device)
            else:
                raise ValueError("cluster_prior_mode must be 'warmup_kmeans' or 'uniform'")
            prototype_initialization_epoch = epoch
            prototype_initialization_method = "kmeans_on_normalized_ema_clean_embedding_n_init20"
            prototype_initialization_event = True

        refresh_due = graph_scale > 0.0 and (
            trusted_sources is None
            or (
                dynamic_graph
                and (epoch - warmup_epochs - 1) % refresh_interval == 0
            )
        )
        if refresh_due:
            latent = (
                ema_clean_embedding
                if ema_clean_embedding is not None
                else _encode_all(
                    ema_model,
                    X_cpu,
                    int(config.get("eval_batch_size", max(512, int(config.get("batch_size", 256)) * 2))),
                    device,
                )
            )
            latent_graph = build_knn_graph(
                latent,
                k=neighbor_k,
                pca_dim=None,
                seed=seed + epoch,
                **knn_options,
            )
            requested_consensus_mode = str(config.get("consensus_mode", "union"))
            consensus = build_consensus_graph(
                # Union is the candidate set.  Recurrence (1.0 shared, 0.5
                # single-view) and mutuality remain non-constant gate evidence.
                input_graph,
                latent_graph,
                mode=requested_consensus_mode,
                k=neighbor_k,
            )
            consensus_fallback = None
            if not np.any(consensus.valid_mask):
                empty_policy = str(config.get("empty_graph_policy", "union_fallback"))
                if requested_consensus_mode == "intersection" and empty_policy == "union_fallback":
                    consensus = build_consensus_graph(
                        input_graph,
                        latent_graph,
                        mode="union",
                        k=neighbor_k,
                    )
                    consensus_fallback = "intersection_to_union"
                else:
                    raise RuntimeError(
                        "The candidate graph is empty; use consensus_mode=union or "
                        "empty_graph_policy=union_fallback."
                    )
            temporal_target = edge_recurrence_against(
                consensus,
                previous_latent_graph if dynamic_graph else None,
            )
            temporal_target_available = bool(dynamic_graph and previous_latent_graph is not None)
            (
                trusted_sources,
                trusted_targets,
                trusted_features,
                trusted_stability,
                valid_edge_mask,
            ) = _valid_edges(consensus)
            trusted_temporal_target = torch.from_numpy(temporal_target)[valid_edge_mask]
            if dynamic_graph:
                previous_latent_graph = latent_graph
            profile = dict(getattr(consensus, "profile", {}))
            graph_history.append(
                {
                    "epoch": epoch,
                    "candidate_edges": int(trusted_sources.numel()),
                    "effective_neighbors_per_node": float(trusted_sources.numel() / X_np.shape[0]),
                    "mean_edge_stability": float(trusted_stability.mean()) if trusted_stability.numel() else 0.0,
                    "temporal_target_available": temporal_target_available,
                    "mean_temporal_recurrence": (
                        float(trusted_temporal_target.mean())
                        if temporal_target_available and trusted_temporal_target.numel()
                        else None
                    ),
                    "consensus_fallback": consensus_fallback,
                    "profile": profile,
                }
            )

        model.train()
        cluster_head.train()
        edge_gate.train()
        sums = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "view_consistency": 0.0,
            "edge_js": 0.0,
            "entropy_balance": 0.0,
            "gate_budget": 0.0,
            "gate_temporal": 0.0,
            "gate_mean": 0.0,
            "gate_low_fraction": 0.0,
            "gate_high_fraction": 0.0,
            "accepted_edge_fraction": 0.0,
        }
        n_batches = 0

        for (x_cpu_batch,) in loader:
            x = x_cpu_batch.to(device, non_blocking=True)
            view1, mask1 = apply_mask_corruption(x, mask_ratio, strategy=mask_strategy)
            view2, mask2 = apply_mask_corruption(x, mask_ratio, strategy=mask_strategy)
            z1, reconstruction1 = model(view1, corruption_mask=mask1)
            z2, reconstruction2 = model(view2, corruption_mask=mask2)
            reconstruction = 0.5 * (
                _masked_reconstruction(
                    model,
                    reconstruction1,
                    x,
                    mask1,
                )
                + _masked_reconstruction(
                    model,
                    reconstruction2,
                    x,
                    mask2,
                )
            )
            view_loss = view_consistency_loss(z1, z2)
            if graph_scale > 0.0:
                batch_assignments = cluster_head(0.5 * (z1 + z2))
                if cluster_prior is None:
                    raise RuntimeError("cluster prior was not initialized before graph training")
                balance = entropy_balance_loss(batch_assignments, prior=cluster_prior)
            else:
                # Keep the prototype optimizer state untouched during warmup;
                # it is initialized from EMA clean embeddings at first graph use.
                balance = torch.zeros((), device=device)

            edge_js = torch.zeros((), device=device)
            gate_budget = torch.zeros((), device=device)
            gate_temporal = torch.zeros((), device=device)
            gate_mean = torch.zeros((), device=device)
            gate_low_fraction = torch.zeros((), device=device)
            gate_high_fraction = torch.zeros((), device=device)
            accepted_edge_fraction = torch.zeros((), device=device)
            if graph_scale > 0.0:
                if (
                    trusted_sources is None
                    or trusted_targets is None
                    or trusted_features is None
                    or trusted_stability is None
                    or trusted_temporal_target is None
                ):
                    raise RuntimeError("Graph schedule is active but no consensus graph has been built")
                sample_size = min(edge_batch_size, int(trusted_sources.numel()))
                choice = torch.from_numpy(
                    graph_rng.choice(int(trusted_sources.numel()), size=sample_size, replace=False)
                ).long()
                src = trusted_sources[choice]
                dst = trusted_targets[choice]
                edge_x_src = X_cpu[src].to(device, non_blocking=True)
                edge_x_dst = X_cpu[dst].to(device, non_blocking=True)
                edge_src_view, _ = apply_mask_corruption(edge_x_src, mask_ratio, strategy=mask_strategy)
                edge_dst_view, _ = apply_mask_corruption(edge_x_dst, mask_ratio, strategy=mask_strategy)
                q_src = cluster_head(model.encode(edge_src_view))
                q_dst = cluster_head(model.encode(edge_dst_view))
                edge_features = trusted_features[choice].to(device, non_blocking=True)
                gates = edge_gate(edge_features)
                confidence_progress = min(
                    1.0,
                    max(0.0, (epoch - warmup_epochs - 1) / max(1, int(config.get("confidence_ramp_epochs", 40)))),
                )
                confidence_fraction = (
                    confidence_start
                    + confidence_progress
                    * (
                        confidence_end
                        - confidence_start
                    )
                )
                confidence_fraction = min(1.0, max(0.0, confidence_fraction))
                with torch.no_grad():
                    teacher_src = ema_cluster_head(ema_model.encode(edge_x_src))
                    teacher_dst = ema_cluster_head(ema_model.encode(edge_x_dst))
                    pair_confidence = torch.minimum(
                        teacher_src.max(dim=1).values,
                        teacher_dst.max(dim=1).values,
                    )
                if confidence_fraction >= 1.0:
                    accepted = torch.ones_like(pair_confidence, dtype=torch.bool)
                elif confidence_fraction <= 0.0:
                    accepted = torch.zeros_like(pair_confidence, dtype=torch.bool)
                else:
                    threshold = torch.quantile(pair_confidence.detach(), 1.0 - confidence_fraction)
                    accepted = pair_confidence >= threshold
                accepted_edge_fraction = accepted.to(dtype=gates.dtype).mean()
                edge_js = edge_assignment_js_loss(q_src, q_dst, gates, valid_mask=accepted)
                gate_budget = gate_budget_loss(
                    gates,
                    target=float(config.get("gate_budget_target", 0.5)),
                    mode=str(config.get("gate_budget_mode", "upper_bound")),
                )
                if temporal_target_available:
                    gate_temporal = gate_stability_loss(
                        gates,
                        trusted_temporal_target[choice].to(device),
                    )
                gate_mean = gates.mean()
                gate_low_fraction = (gates <= float(config.get("gate_saturation_low", 0.05))).float().mean()
                gate_high_fraction = (gates >= float(config.get("gate_saturation_high", 0.95))).float().mean()

            # graph_scale appears exactly once.  Individual graph weights are
            # not scheduled inside their modules.
            loss = combine_v10_losses(
                {
                    "reconstruction": reconstruction,
                    "view_consistency": view_loss,
                    "edge_assignment": edge_js,
                    "entropy_balance": balance,
                    "gate_budget": gate_budget,
                    "gate_temporal": gate_temporal,
                },
                objective_weights,
                graph_scale=graph_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip = float(config.get("gradient_clip", 5.0))
            if clip > 0:
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(cluster_head.parameters()) + list(edge_gate.parameters()), clip
                )
            optimizer.step()
            _ema_update(ema_model, model, ema_decay)
            _ema_update(ema_cluster_head, cluster_head, ema_decay)

            values = {
                "loss": loss,
                "reconstruction": reconstruction,
                "view_consistency": view_loss,
                "edge_js": edge_js,
                "entropy_balance": balance,
                "gate_budget": gate_budget,
                "gate_temporal": gate_temporal,
                "gate_mean": gate_mean,
                "gate_low_fraction": gate_low_fraction,
                "gate_high_fraction": gate_high_fraction,
                "accepted_edge_fraction": accepted_edge_fraction,
            }
            for name, value in values.items():
                sums[name] += float(value.detach().cpu())
            n_batches += 1

        epoch_values = _mean_history(sums, n_batches)
        training_history.append(
            {
                "epoch": epoch,
                "graph_scale": graph_scale,
                "graph_refreshed": refresh_due,
                "prototype_initialization_event": prototype_initialization_event,
                "prototype_initialization_method": (
                    prototype_initialization_method if prototype_initialization_event else None
                ),
                "candidate_edges": 0 if trusted_sources is None else int(trusted_sources.numel()),
                **epoch_values,
            }
        )
        if epoch == 1 or epoch == epochs or epoch % 10 == 0 or refresh_due:
            print(
                f"[V10] epoch={epoch:03d}/{epochs} loss={epoch_values['loss']:.5f} "
                f"graph_scale={graph_scale:.2f} candidate_edges="
                f"{0 if trusted_sources is None else trusted_sources.numel()} gate={epoch_values['gate_mean']:.3f}",
                flush=True,
            )

    train_seconds = time.time() - t0
    embedding = _encode_all(
        ema_model,
        X_cpu,
        int(config.get("eval_batch_size", max(512, int(config.get("batch_size", 256)) * 2))),
        device,
    )

    final_gate_diagnostics: dict[str, float] | None = None
    final_gate_values: np.ndarray | None = None
    if graph_enabled and trusted_features is not None and trusted_sources is not None:
        edge_gate.eval()
        gate_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, trusted_features.shape[0], edge_batch_size):
                batch_features = trusted_features[start : start + edge_batch_size].to(device)
                gate_chunks.append(edge_gate(batch_features).cpu().numpy())
        final_gate_values = np.concatenate(gate_chunks).astype(np.float32)
        per_node_open = np.bincount(
            trusted_sources.numpy(),
            weights=(final_gate_values >= 0.5).astype(np.float32),
            minlength=X_np.shape[0],
        )
        final_gate_diagnostics = {
            "mean": float(final_gate_values.mean()),
            "minimum": float(final_gate_values.min()),
            "maximum": float(final_gate_values.max()),
            "low_fraction": float(
                np.mean(final_gate_values <= float(config.get("gate_saturation_low", 0.05)))
            ),
            "high_fraction": float(
                np.mean(final_gate_values >= float(config.get("gate_saturation_high", 0.95)))
            ),
            "open_fraction_at_0_5": float(np.mean(final_gate_values >= 0.5)),
            "mean_open_neighbors_at_0_5": float(per_node_open.mean()),
        }
    # The standard KMeans readout is primary for strict comparability with the
    # feature-only control.  A prototype-initialised readout is saved as a
    # secondary, clustering-objective-aligned diagnostic.
    normalized_embedding = embedding / np.clip(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12, None)
    predictions = KMeans(n_clusters=K, n_init=20, random_state=seed).fit_predict(normalized_embedding).astype(np.int64)
    metrics = {} if y_encoded is None else _compute_metrics(y_encoded, predictions)

    cluster_probabilities: np.ndarray | None = None
    prototype_kmeans_predictions: np.ndarray | None = None
    prototype_predictions: np.ndarray | None = None
    prototype_kmeans_metrics: dict[str, float] = {}
    prototype_metrics: dict[str, float] = {}
    if prototype_initialization_epoch is not None:
        with torch.no_grad():
            probabilities: list[np.ndarray] = []
            ema_cluster_head.eval()
            for start in range(0, embedding.shape[0], int(config.get("eval_batch_size", 512))):
                z = torch.from_numpy(embedding[start : start + int(config.get("eval_batch_size", 512))]).to(device)
                probabilities.append(ema_cluster_head(z).cpu().numpy())
            cluster_probabilities = np.concatenate(probabilities, axis=0).astype(np.float32)
            prototype_init = F.normalize(ema_cluster_head.prototypes, dim=1).cpu().numpy()
        prototype_kmeans_predictions = KMeans(
            n_clusters=K,
            init=prototype_init,
            n_init=1,
            random_state=seed,
        ).fit_predict(normalized_embedding).astype(np.int64)
        prototype_predictions = np.argmax(cluster_probabilities, axis=1).astype(np.int64)
        prototype_kmeans_metrics = (
            {} if y_encoded is None else _compute_metrics(y_encoded, prototype_kmeans_predictions)
        )
        prototype_metrics = {} if y_encoded is None else _compute_metrics(y_encoded, prototype_predictions)

    np.save(save_dir / "embedding_final.npy", embedding)
    np.save(save_dir / "predictions.npy", predictions)
    if final_gate_values is not None:
        assert trusted_sources is not None
        assert trusted_targets is not None
        assert trusted_stability is not None
        assert trusted_temporal_target is not None
        np.savez_compressed(
            save_dir / "final_graph_edges.npz",
            source=trusted_sources.numpy(),
            target=trusted_targets.numpy(),
            gate=final_gate_values,
            input_latent_stability=trusted_stability.numpy(),
            temporal_target=trusted_temporal_target.numpy(),
        )
    prototype_artifacts = (
        "prototype_kmeans_predictions.npy",
        "prototype_predictions.npy",
        "cluster_probabilities.npy",
        "prototype_kmeans_metrics.json",
        "prototype_metrics.json",
    )
    if prototype_initialization_epoch is not None:
        assert cluster_probabilities is not None
        assert prototype_kmeans_predictions is not None
        assert prototype_predictions is not None
        np.save(save_dir / "prototype_kmeans_predictions.npy", prototype_kmeans_predictions)
        np.save(save_dir / "prototype_predictions.npy", prototype_predictions)
        np.save(save_dir / "cluster_probabilities.npy", cluster_probabilities)
    else:
        # A reused feature-only directory must not retain stochastic prototype
        # diagnostics written by an earlier graph-enabled run.
        for artifact in prototype_artifacts:
            (save_dir / artifact).unlink(missing_ok=True)
    if y_encoded is not None:
        np.save(save_dir / "labels_true.npy", y_encoded)
        _save_json(save_dir / "label_mapping.json", label_mapping)
    _save_json(save_dir / "metrics.json", metrics)
    if prototype_initialization_epoch is not None:
        _save_json(save_dir / "prototype_kmeans_metrics.json", prototype_kmeans_metrics)
        _save_json(save_dir / "prototype_metrics.json", prototype_metrics)
    _save_json(save_dir / "history.json", training_history)
    _save_json(save_dir / "graph_history.json", graph_history)
    _save_json(save_dir / "config_resolved.json", config)
    summary = {
        "method": "TopoGate",
        "variant": str(config.get("variant_name", "topogate_v10_reliable_graph")),
        "seed": seed,
        "device": str(device),
        "n_samples": int(X_np.shape[0]),
        "n_features": int(X_np.shape[1]),
        "n_clusters": K,
        "known_k": True,
        "k_source": k_source,
        "train_seconds": train_seconds,
        "pca": asdict(pca_selection),
        "preprocessing": preprocessing,
        "graph_enabled": graph_enabled,
        "dynamic_graph": dynamic_graph,
        "confidence_teacher": (
            "ema_encoder_and_ema_prototype_head_on_clean_edges" if graph_enabled else None
        ),
        "prototype_initialization_epoch": prototype_initialization_epoch,
        "prototype_initialization_method": prototype_initialization_method,
        "cluster_prior_mode": str(config.get("cluster_prior_mode", "warmup_kmeans")),
        "cluster_prior": None if cluster_prior is None else cluster_prior.detach().cpu().tolist(),
        "consensus_mode": str(config.get("consensus_mode", "union")),
        "input_graph_profile": None if input_graph is None else dict(input_graph.profile),
        "final_graph": graph_history[-1] if graph_history else None,
        "model_parameters": {
            **model.parameter_profile(),
            "prototype_head_parameters": int(sum(parameter.numel() for parameter in cluster_head.parameters())),
            "edge_gate_parameters": int(sum(parameter.numel() for parameter in edge_gate.parameters())),
            "trainable_total_parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
                + sum(parameter.numel() for parameter in cluster_head.parameters())
                + sum(parameter.numel() for parameter in edge_gate.parameters())
            ),
        },
        "final_gate_diagnostics": final_gate_diagnostics,
        "metrics": metrics,
        "prototype_kmeans_metrics": prototype_kmeans_metrics if prototype_initialization_epoch is not None else None,
        "prototype_metrics": prototype_metrics if prototype_initialization_epoch is not None else None,
        "output_contract": {
            "predictions": "predictions.npy",
            "labels_true_encoded": "labels_true.npy" if y_encoded is not None else None,
            "label_mapping": "label_mapping.json" if y_encoded is not None else None,
            "prototype_kmeans_predictions": (
                "prototype_kmeans_predictions.npy" if prototype_initialization_epoch is not None else None
            ),
            "prototype_predictions": "prototype_predictions.npy" if prototype_initialization_epoch is not None else None,
            "cluster_probabilities": "cluster_probabilities.npy" if prototype_initialization_epoch is not None else None,
            "embedding": "embedding_final.npy",
            "history": "history.json",
            "graph_history": "graph_history.json",
            "final_graph_edges": "final_graph_edges.npz" if final_gate_values is not None else None,
        },
    }
    _save_json(save_dir / "summary.json", summary)
    return predictions, train_seconds, metrics


def run_v10(
    X: np.ndarray,
    n_clusters: int | None = None,
    y: np.ndarray | None = None,
    gpu: int = 1,
    seed: int = 42,
    save_dir: str | Path | None = None,
    return_metrics: bool = False,
    config_path: str | Path | None = None,
    **overrides: Any,
):
    """Programmatic V10 API used by multi-seed and benchmark scripts."""
    import tempfile

    config = _load_config(config_path)
    config.update(overrides)
    config.update({"gpu": int(gpu), "seed": int(seed)})
    cleanup = save_dir is None
    output = Path(tempfile.mkdtemp(prefix="topogate_v10_")) if cleanup else Path(save_dir)
    try:
        predictions, elapsed, metrics = train_v10(X, y, n_clusters, output, config)
    finally:
        if cleanup:
            import shutil

            shutil.rmtree(output, ignore_errors=True)
    if return_metrics:
        return predictions, elapsed, metrics
    return predictions, elapsed


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = pre.parse_known_args(argv)
    defaults = _load_config(known.config)

    parser = argparse.ArgumentParser(description="TopoGate V10 dynamic reliable-graph clustering")
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 42)))
    parser.add_argument("--gpu", type=int, choices=[1, 4, 5], default=int(defaults.get("gpu", 1)))
    parser.add_argument("--no_cuda", action=argparse.BooleanOptionalAction, default=bool(defaults.get("no_cuda", False)))
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs", 100)))
    parser.add_argument("--batch_size", type=int, default=int(defaults.get("batch_size", 256)))
    parser.add_argument("--warmup_epochs", type=int, default=int(defaults.get("warmup_epochs", 20)))
    parser.add_argument("--ramp_epochs", type=int, default=int(defaults.get("ramp_epochs", 10)))
    parser.add_argument("--refresh_interval", type=int, default=int(defaults.get("refresh_interval", 5)))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _load_config(args.config)
    for key in ("seed", "gpu", "no_cuda", "epochs", "batch_size", "warmup_epochs", "ramp_epochs", "refresh_interval"):
        config[key] = getattr(args, key)
    config["dataset_name"] = args.dataset_name or Path(args.data_path).stem
    X, y = load_data(args.data_path)
    train_v10(X, y, args.n_clusters, args.save_dir, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
