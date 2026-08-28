#!/usr/bin/env python3
"""Independent public runner for TopoGate V11."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import sklearn
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import V11Config, load_config
from .graph import pca_embedding
from .trainer import V11Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu: int, no_cuda: bool = False) -> torch.device:
    if no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    forbidden = {0, 7}
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible:
        if any(int(item) in forbidden for item in visible):
            raise ValueError("CUDA_VISIBLE_DEVICES contains forbidden GPU 0 or 7")
        if str(gpu) in visible:
            return torch.device(f"cuda:{visible.index(str(gpu))}")
        if 0 <= gpu < len(visible):
            return torch.device(f"cuda:{gpu}")
        raise ValueError(f"GPU {gpu} is not present in CUDA_VISIBLE_DEVICES={visible}")
    if int(gpu) in forbidden:
        raise ValueError("physical GPU 0 and GPU 7 are forbidden by project policy")
    return torch.device(f"cuda:{int(gpu)}")


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path)
    X = data.get("X", data.get("x", data.get("data")))
    y = data.get("y", data.get("labels", data.get("label")))
    if X is None:
        raise ValueError(f"{path} does not contain X/x/data")
    return np.asarray(X), None if y is None else np.asarray(y).ravel()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(X: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(X)
    header = f"{contiguous.shape}|{contiguous.dtype}".encode("utf-8")
    return _sha256_bytes(header + contiguous.tobytes())


def preprocess(X: np.ndarray, cfg: V11Config) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got {X.shape}")
    original_features = int(X.shape[1])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if cfg.input_mode == "log1p":
        if np.min(X) < 0:
            raise ValueError("log1p input_mode requires non-negative inputs")
        X = np.log1p(X)
    elif cfg.input_mode != "raw":
        raise ValueError(f"unknown input_mode: {cfg.input_mode}")

    selected = None
    if 0 < cfg.n_top_features < X.shape[1]:
        # Selection deliberately occurs before StandardScaler; selecting by
        # variance after scaling would make every non-constant feature tie.
        variance = np.var(X, axis=0)
        selected = np.argsort(variance)[-int(cfg.n_top_features):]
        X = X[:, selected]

    if cfg.reconstruction_distribution == "poisson":
        if cfg.scale_input:
            raise ValueError("Poisson reconstruction requires scale_input=false")
        if np.min(X) < 0:
            raise ValueError("Poisson reconstruction requires non-negative inputs")
    if cfg.reconstruction_distribution == "bernoulli" and cfg.scale_input:
        raise ValueError("Bernoulli reconstruction requires scale_input=false and targets in [0,1]")
    if cfg.reconstruction_distribution == "bernoulli" and (np.min(X) < 0 or np.max(X) > 1):
        raise ValueError("Bernoulli reconstruction targets must lie in [0,1]")

    if cfg.scale_input:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    X = np.asarray(X, dtype=np.float32)
    return X, {
        "input_mode": cfg.input_mode,
        "scale_input": bool(cfg.scale_input),
        "original_features": original_features,
        "selected_feature_count": int(X.shape[1]),
        "selected_feature_indices": None if selected is None else selected.astype(int).tolist(),
    }


def align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    matrix = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    for i, true_value in enumerate(true_values):
        for j, pred_value in enumerate(pred_values):
            matrix[i, j] = int(np.sum((y_true == true_value) & (y_pred == pred_value)))
    rows, cols = linear_sum_assignment(-matrix)
    mapping = {pred_values[col]: true_values[row] for row, col in zip(rows, cols)}
    return np.asarray([mapping.get(value, value) for value in y_pred])


def external_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    aligned = align_labels(y_true, y_pred)
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


def _json_dump(payload: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def fit_v11(
    X: np.ndarray,
    n_clusters: int,
    y: np.ndarray | None = None,
    *,
    config: V11Config,
    save_dir: str | Path,
    dataset_name: str,
    gpu: int = 1,
    no_cuda: bool = False,
    source_path: str | Path | None = None,
    k_protocol: str = "explicit",
) -> tuple[np.ndarray, float, dict]:
    set_seed(config.seed)
    raw_X = np.asarray(X)
    X_processed, preprocessing = preprocess(raw_X, config)
    raw_embedding, actual_pca_dim = pca_embedding(
        X_processed, config.pca_dim, config.pca_variance, config.seed
    )
    device = get_device(gpu, no_cuda)
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainer = V11Trainer(X_processed, raw_embedding, n_clusters, config, device)
    result = trainer.fit()

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=config.seed)
    kmeans_pred = kmeans.fit_predict(result.embedding).astype(np.int64)
    primary_pred = result.predictions if config.use_cluster_head else kmeans_pred
    primary_source = "student_t_mixture_head" if config.use_cluster_head else "kmeans_ablation"
    metrics: dict[str, Any] = {
        "prediction_source": primary_source,
        "silhouette": float(silhouette_score(result.embedding, primary_pred))
        if 1 < len(np.unique(primary_pred)) < len(primary_pred)
        else None,
    }
    if y is not None:
        label_encoder = LabelEncoder().fit(np.asarray(y).ravel())
        encoded_y = label_encoder.transform(np.asarray(y).ravel()).astype(np.int64)
        metrics.update(external_metrics(encoded_y, primary_pred))
        metrics["head"] = external_metrics(encoded_y, result.predictions)
        metrics["kmeans"] = external_metrics(encoded_y, kmeans_pred)
    else:
        label_encoder = None
        encoded_y = None

    np.save(output / "embedding_final.npy", result.embedding)
    np.save(output / "cluster_probabilities.npy", result.probabilities)
    np.save(output / "predictions.npy", primary_pred)
    if encoded_y is not None:
        np.save(output / "labels_true.npy", encoded_y)
        _json_dump(
            {str(index): str(value) for index, value in enumerate(label_encoder.classes_)},
            output / "label_mapping.json",
        )
    _json_dump(metrics, output / "metrics.json")
    source_hash = file_sha256(source_path) if source_path is not None else array_sha256(raw_X)
    summary = {
        "method": "TopoGate",
        "variant": "V11",
        "dataset": dataset_name,
        "source_path": None if source_path is None else str(Path(source_path).resolve()),
        "source_sha256": source_hash,
        "seed": int(config.seed),
        "n_samples": int(X_processed.shape[0]),
        "n_features": int(X_processed.shape[1]),
        "n_clusters": int(n_clusters),
        "k_protocol": k_protocol,
        "labels_used_during_fit": False,
        "actual_pca_dim": int(actual_pca_dim),
        "device": str(device),
        "train_seconds": float(result.train_seconds),
        "config": config.to_dict(),
        "preprocessing": preprocessing,
        "metrics": metrics,
        "history": result.history,
        "graph_history": result.graph_history,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda,
        },
        "output_files": {
            "embedding": "embedding_final.npy",
            "probabilities": "cluster_probabilities.npy",
            "predictions": "predictions.npy",
            "labels_true": "labels_true.npy" if encoded_y is not None else None,
            "label_mapping": "label_mapping.json" if encoded_y is not None else None,
        },
    }
    _json_dump(config.to_dict(), output / "args.json")
    _json_dump(summary, output / "summary.json")
    return primary_pred, result.train_seconds, metrics


def run_v11(
    X: np.ndarray,
    n_clusters: int,
    y: np.ndarray | None = None,
    *,
    config_path: str | Path | None = None,
    save_dir: str | Path,
    dataset_name: str = "adhoc",
    gpu: int = 1,
    no_cuda: bool = False,
    seed: int = 42,
    source_path: str | Path | None = None,
    k_protocol: str = "explicit",
    **overrides: Any,
) -> tuple[np.ndarray, float, dict]:
    overrides = {**overrides, "seed": int(seed)}
    config = load_config(config_path, overrides)
    return fit_v11(
        X,
        int(n_clusters),
        y,
        config=config,
        save_dir=save_dir,
        dataset_name=dataset_name,
        gpu=gpu,
        no_cuda=no_cuda,
        source_path=source_path,
        k_protocol=k_protocol,
    )


def _parse_override(values: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"override must have key=value form: {value}")
        key, raw = value.split("=", 1)
        output[key] = yaml.safe_load(raw)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoGate V11")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "topogate_v11.yaml"))
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    X, y = load_npz(args.data_path)
    if args.n_clusters is None:
        if y is None:
            raise ValueError("--n_clusters is required when labels are absent")
        n_clusters = int(np.unique(y).size)
        k_protocol = "benchmark_oracle_from_y"
    else:
        n_clusters = int(args.n_clusters)
        k_protocol = "explicit"
    overrides = _parse_override(args.overrides)
    overrides["seed"] = int(args.seed)
    cfg = load_config(args.config, overrides)
    fit_v11(
        X,
        n_clusters,
        y,
        config=cfg,
        save_dir=args.save_dir,
        dataset_name=args.dataset_name or Path(args.data_path).stem,
        gpu=args.gpu,
        no_cuda=args.no_cuda,
        source_path=args.data_path,
        k_protocol=k_protocol,
    )


if __name__ == "__main__":
    main()
