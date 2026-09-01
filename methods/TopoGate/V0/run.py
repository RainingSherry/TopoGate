#!/usr/bin/env python
"""Command-line runner for the unified TopoGate V0 (scVICAR) model.

The runner accepts the historical F/T aliases, but both routes call the same
``trainer.fit_predict`` implementation.  Labels are loaded only by this outer
benchmark layer for K selection and post-fit metrics; they are never passed to
the model, graph, corruption operator, or optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import V0Config, load_config
from .diagnostics import embedding_geometry
from .trainer import ALLOWED_PHYSICAL_GPUS, fit_predict


LABEL_CANDIDATES = (
    "resolved_label",
    "maintype",
    "cell_type",
    "Celltype",
    "celltype",
    "label",
    "labels",
    "cell_label",
    "Cluster",
    "cluster",
    "clusters",
    "Seurat_clusters",
)


@dataclass(frozen=True)
class InputBundle:
    # labels 仅由外层 benchmark runner 保存；训练器收到的只有 X。
    X: np.ndarray
    labels: np.ndarray | None
    label_values: list[str] | None
    profile: dict[str, Any]
    preprocess_profile: dict[str, Any]
    adata: Any | None = None
    gene_names: np.ndarray | None = None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8"
    )


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in {"1", "true", "t", "yes", "y"}:
        return True
    if key in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(
        f"{array.shape}|{array.dtype}".encode("utf-8") + array.tobytes()
    ).hexdigest()


def _first_npz(payload: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in payload.files:
            return np.asarray(payload[name])
    return None


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    # 同时兼容普通 dense NPZ 和 scipy CSR 的四数组存储，不改变输入数值语义。
    with np.load(path, allow_pickle=False) as payload:
        sparse_keys = {"data", "indices", "indptr", "shape"}
        if sparse_keys.issubset(payload.files):
            shape = tuple(int(value) for value in np.asarray(payload["shape"]).reshape(-1))
            X: np.ndarray | sp.csr_matrix = sp.csr_matrix(
                (
                    np.asarray(payload["data"]),
                    np.asarray(payload["indices"], dtype=np.int64),
                    np.asarray(payload["indptr"], dtype=np.int64),
                ),
                shape=shape,
            )
        else:
            X = _first_npz(payload, ("X", "x", "features", "data"))
            if X is None:
                raise ValueError(f"NPZ has no X/x/features/data matrix: {path}")
        # 标签可以被读取用于 K/最终指标，但从这里开始只留在外层 runner。
        labels = _first_npz(payload, ("y", "labels", "label"))
        keys = list(payload.files)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
        raise ValueError("X must contain at least two samples and one feature")
    return np.asarray(X.toarray() if sp.issparse(X) else X), labels, {
        "path": str(path.resolve()),
        "format": "npz",
        "npz_keys": keys,
        "n_samples_original": int(X.shape[0]),
        "n_features_original": int(X.shape[1]),
        "sparse_storage": bool(sp.issparse(X)),
        "labels_loaded_by_outer_runner": labels is not None,
    }


def _encode_labels(labels: np.ndarray | None) -> tuple[np.ndarray | None, list[str] | None]:
    if labels is None:
        return None, None
    # 评估指标需要整数标签；LabelEncoder 只改变外层表示，不参与模型拟合。
    values = np.asarray(labels).reshape(-1)
    if values.size == 0:
        return None, None
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(values.astype(str)).astype(np.int64)
    return encoded, encoder.classes_.astype(str).tolist()


def _is_count_like(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    finite = finite[: min(100_000, finite.size)]
    return bool(np.all(finite >= 0.0) and np.allclose(finite, np.rint(finite), atol=1e-5))


def _prepare_array(
    values: np.ndarray,
    *,
    config: V0Config,
    dataset_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply deterministic, label-free preprocessing to an NPZ matrix."""

    # NPZ 预处理完全由 X 决定：计数归一化/log1p -> 方差特征选择 -> 可选标准化。
    # 这里明确不接收 labels，防止标签泄漏到模型输入或特征选择。
    raw = np.asarray(values, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] == 0:
        raise ValueError("input matrix must have shape [n_samples>=2, n_features>=1]")
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if np.min(raw) < 0.0:
        raise ValueError("V0 input must be non-negative before raw/log1p preprocessing")
    count_like = _is_count_like(raw)
    mode = config.input_mode
    if mode == "auto":
        mode = "raw" if count_like else "log1p"
    if mode == "raw" and not count_like:
        raise ValueError("input_mode=raw requires integer-like non-negative counts")
    work = raw.astype(np.float32, copy=True)
    normalization = "none"
    if mode == "raw":
        # 原始计数先按每个细胞的总量归一化，再 log1p；零计数行保持全零。
        row_sum = work.sum(axis=1, keepdims=True)
        scale = np.divide(
            float(config.target_sum),
            row_sum,
            out=np.zeros_like(row_sum),
            where=row_sum > 0.0,
        )
        work = np.log1p(work * scale).astype(np.float32)
        normalization = f"normalize_total(target_sum={float(config.target_sum)})_then_log1p"
    elif mode == "log1p":
        normalization = "already_log1p_or_continuous"
    else:  # protected by V0Config validation
        raise AssertionError(mode)

    original_features = int(work.shape[1])
    selected = np.arange(original_features, dtype=np.int64)
    selection_strategy = "disabled"
    if int(config.n_top_features) > 0 and original_features > int(config.n_top_features):
        # 只按无标签方差排序；lexsort 的原始索引 tie-break 保证确定性。
        variance = np.nan_to_num(np.var(work, axis=0), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        order = np.lexsort((np.arange(original_features, dtype=np.int64), -variance))
        selected = np.sort(order[: int(config.n_top_features)]).astype(np.int64)
        work = work[:, selected]
        selection_strategy = "variance_top_features"
    if config.scale_input:
        # 标准化参数也只从当前输入 X 拟合，避免使用标签或外部 oracle。
        work = StandardScaler(with_mean=True, with_std=True).fit_transform(work).astype(np.float32)
        scale_method = "sklearn_standard_scaler"
    else:
        scale_method = "none"
    work = np.nan_to_num(np.asarray(work, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    profile = {
        "dataset_name": str(dataset_name),
        "input_mode_requested": config.input_mode,
        "input_mode_used": mode,
        "count_like_before_scaling": count_like,
        "normalization": normalization,
        "scale_input": bool(config.scale_input),
        "scale_method": scale_method,
        "n_samples": int(work.shape[0]),
        "n_features_original": original_features,
        "n_features_selected": int(work.shape[1]),
        "selected_feature_indices": selected.astype(int).tolist(),
        "feature_selection_strategy": selection_strategy,
        "labels_used": False,
        "K_used": False,
    }
    return np.ascontiguousarray(work, dtype=np.float32), profile


def _load_h5ad(path: Path, config: V0Config, label_key: str) -> InputBundle:
    """Load the historical scMAE h5ad format, with labels kept outside fit."""

    try:
        import scanpy as sc
        import methods.DeepLearning.scMAE_family as family
    except ImportError as exc:  # pragma: no cover - depends on optional h5ad stack
        raise RuntimeError("h5ad input requires scanpy and its anndata dependencies") from exc

    adata = sc.read_h5ad(path)
    # h5ad 的 count/raw 选择、HVG 和 scaling 沿用 family 工具；obs 标签仅在
    # 外层读取后保存，不会传给 fit_predict。
    source_x, gene_names, var, source_desc, inferred_mode = family.select_count_source(
        adata, config.input_mode
    )
    work = sc.AnnData(X=family.ensure_csr(source_x), obs=adata.obs.copy(), var=var.copy())
    work.var_names = gene_names.copy()
    if inferred_mode == "raw":
        work.X = family.normalize_total_log1p(work.X, target_sum=config.target_sum)
    if config.n_top_features > 0:
        work, hvg = family.strict_hvg_subset(work, config.n_top_features)
    else:
        hvg = {"requested": 0, "selected": int(work.n_vars), "strategy": "disabled"}
    if config.scale_input:
        sc.pp.scale(work)
    data = family.dense_float32(work.X)
    labels = None
    label_values = None
    resolved_label_key = None
    try:
        labels, names, resolved_label_key = family.resolve_labels(work, label_key)
        label_values = [str(item) for item in names.tolist()]
    except KeyError:
        # Unlabelled deployment is valid; K must then be supplied explicitly.
        if label_key != "auto":
            raise
        labels = None
    profile = {
        "path": str(path.resolve()),
        "format": "h5ad",
        "source": source_desc,
        "inferred_input_mode": inferred_mode,
        "n_samples_original": int(adata.n_obs),
        "n_features_original": int(adata.n_vars),
        "n_samples": int(data.shape[0]),
        "n_features": int(data.shape[1]),
        "label_key": resolved_label_key,
        "labels_loaded_by_outer_runner": labels is not None,
    }
    preprocess = {
        "input_mode_requested": config.input_mode,
        "input_mode_used": inferred_mode,
        "n_features_selected": int(data.shape[1]),
        "hvg": hvg,
        "scale_input": bool(config.scale_input),
        "labels_used": False,
        "K_used": False,
    }
    return InputBundle(
        X=np.ascontiguousarray(data, dtype=np.float32),
        labels=None if labels is None else np.asarray(labels, dtype=np.int64),
        label_values=label_values,
        profile=profile,
        preprocess_profile=preprocess,
        adata=work,
        gene_names=np.asarray(work.var_names).astype(str),
    )


def load_input(path: str | Path, config: V0Config, label_key: str = "auto") -> InputBundle:
    # 根据后缀选择输入适配器；两条路径都返回统一的 InputBundle 契约。
    source = Path(path)
    if source.suffix.lower() == ".npz":
        raw, raw_labels, profile = _load_npz(source)
        X, preprocess = _prepare_array(raw, config=config, dataset_name=source.stem)
        labels, values = _encode_labels(raw_labels)
        if labels is not None and labels.shape[0] != X.shape[0]:
            raise ValueError("label count does not match input rows")
        return InputBundle(X, labels, values, profile, preprocess)
    if source.suffix.lower() == ".h5ad":
        return _load_h5ad(source, config, label_key)
    raise ValueError(f"V0 runner accepts .npz or .h5ad input, got {source}")


def resolve_runtime_device(device: str, gpu: int) -> str:
    """Resolve ``auto|cpu|cuda`` and reject physical GPU 0/7."""

    # CLI 的 gpu 是物理卡候选；若 CUDA_VISIBLE_DEVICES 已设置，则转换为
    # torch 需要的逻辑序号，同时拒绝包含 0/7 的可见列表。
    choice = str(device).lower()
    if choice not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if choice == "cpu":
        return "cpu"
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible:
        try:
            physical = [int(item) for item in visible]
        except ValueError as exc:
            raise ValueError("CUDA_VISIBLE_DEVICES must contain integer ids") from exc
        if set(physical).intersection({0, 7}):
            raise ValueError("CUDA_VISIBLE_DEVICES includes forbidden physical GPU 0 or 7")
        if choice == "auto" and not torch.cuda.is_available():
            return "cpu"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if int(gpu) in physical:
            return f"cuda:{physical.index(int(gpu))}"
        if len(physical) == 1:
            return "cuda:0"
        if 0 <= int(gpu) < len(physical):
            return f"cuda:{int(gpu)}"
        raise ValueError(f"GPU {gpu} is not present in CUDA_VISIBLE_DEVICES={visible}")
    if int(gpu) not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(
            f"physical GPU {gpu} is forbidden or unavailable; allowed GPUs are {sorted(ALLOWED_PHYSICAL_GPUS)}"
        )
    if choice == "auto" and not torch.cuda.is_available():
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return f"cuda:{int(gpu)}"


def _mapped_predictions(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    # 聚类标签本身无序，Hungarian matching 只用于报告 ACC/F1，不参与训练或选参。
    true_values = np.unique(y_true)
    pred_values = np.unique(predictions)
    width = max(len(true_values), len(pred_values))
    counts = np.zeros((width, width), dtype=np.int64)
    for row, true_value in enumerate(true_values):
        for column, predicted_value in enumerate(pred_values):
            counts[row, column] = int(np.sum((y_true == true_value) & (predictions == predicted_value)))
    rows, columns = linear_sum_assignment(-counts)
    mapped = np.full_like(predictions, fill_value=-1, dtype=np.int64)
    for row, column in zip(rows, columns):
        if row < len(true_values) and column < len(pred_values):
            mapped[predictions == pred_values[column]] = true_values[row]
    return mapped


def clustering_metrics(labels: np.ndarray | None, predictions: np.ndarray) -> dict[str, Any]:
    # 这是 fit 完成后的外层评估；无标签输入仍可输出 KMeans 协议说明。
    if labels is None:
        return {"labels_available": False, "cluster_method": "kmeans_known_k"}
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
    mapped = _mapped_predictions(y, pred)
    return {
        "labels_available": True,
        "acc": float(np.mean(mapped == y)),
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "ami": float(adjusted_mutual_info_score(y, pred)),
        "f1_macro": float(f1_score(y, mapped, average="macro", zero_division=0)),
        "fmi": float(fowlkes_mallows_score(y, pred)),
        "n_pred_clusters": int(np.unique(pred).size),
        "cluster_method": "kmeans_known_k",
        "uses_known_k": True,
    }


def run_one(
    data_path: str | Path,
    save_dir: str | Path,
    *,
    config: V0Config,
    seed: int = 42,
    device: str = "cpu",
    n_clusters: int | None = None,
    label_key: str = "auto",
    dataset_name: str | None = None,
    no_save_h5ad: bool = True,
) -> dict[str, Any]:
    """Run one auditable V0 fit and write a self-contained output directory."""

    # run_one 负责输入、K 协议、标签隔离和产物落盘；真正的无标签训练在
    # trainer.fit_predict 中完成。
    source = Path(data_path)
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    name = dataset_name or source.stem
    started = time.time()
    run_key = f"{name}::topogate_v0::{config.parameterization}::seed{int(seed)}"
    record: dict[str, Any] = {
        "status": "running",
        "run_key": run_key,
        "protocol_id": config.protocol_id,
        "dataset": name,
        "source_path": str(source.resolve()),
        "parameterization": config.parameterization,
        "seed": int(seed),
        "labels_used_during_fit": False,
        "labels_used_during_preprocessing": False,
        "labels_used_for_graph": False,
        "labels_used_for_gate": False,
        "labels_used_for_loss": False,
        "labels_used_for_selection": False,
    }
    _write_json(output / "run_record.json", record)
    _write_json(output / "status.json", {"status": "running", "run_key": run_key})

    bundle = load_input(source, config, label_key=label_key)
    labels = bundle.labels
    if n_clusters is None:
        if labels is None:
            raise ValueError("n_clusters is required for an input without benchmark labels")
        # benchmark 标签只能提供 oracle K；标签值本身不会传入 fit_predict。
        K = int(np.unique(labels).size)
        k_source = "benchmark_oracle_from_y"
    else:
        K = int(n_clusters)
        k_source = "explicit_n_clusters"
    if K <= 0 or K > bundle.X.shape[0]:
        raise ValueError("n_clusters must be in [1, n_samples]")

    # resolved_config 同时记录输入 hash、seed、设备和 K 来源，是本次运行的协议快照。
    resolved = config.resolved_dict()
    resolved.update(
        {
            "seed": int(seed),
            "device": str(device),
            "dataset_name": name,
            "n_clusters": K,
            "K_source": k_source,
            "source_sha256": _file_sha256(source),
            "input_array_sha256": _array_sha256(bundle.X),
        }
    )
    _write_json(output / "resolved_config.json", resolved)
    _write_json(output / "dataset_profile.json", bundle.profile)
    _write_json(output / "preprocess_profile.json", bundle.preprocess_profile)
    if bundle.gene_names is not None:
        np.save(output / "gene_names.npy", bundle.gene_names.astype(str))
    if bundle.labels is not None:
        np.save(output / "labels_true.npy", bundle.labels.astype(np.int64))
        label_values = bundle.label_values or []
        _write_json(
            output / "label_mapping.json",
            {str(index): str(value) for index, value in enumerate(label_values)},
        )

    # 关键隔离边界：这里只传 bundle.X；labels 仅留在本函数用于 readout 后评估。
    predictions, embedding, diagnostics = fit_predict(
        bundle.X,
        n_clusters=K,
        config=config,
        seed=int(seed),
        device=device,
    )
    if predictions is None:  # K is always supplied by this outer runner
        raise AssertionError("V0 runner expected a KMeans prediction")
    predictions = np.asarray(predictions, dtype=np.int64)
    # 预测、embedding、图/gate 数组和 JSON 诊断组成可复核的输出契约。
    np.save(output / "predictions.npy", predictions)
    np.save(output / "embedding_final.npy", embedding.astype(np.float32))
    if labels is not None:
        np.save(output / "predictions_mapped.npy", _mapped_predictions(labels, predictions))
    for key in (
        "neighbor_indices",
        "neighbor_base_probs",
        "neighbor_similarity",
        "neighbor_distance",
        "edge_reliability",
        "edge_weights",
        "node_gate",
        "pseudo_perturbation",
    ):
        np.save(output / f"{key}.npy", np.asarray(diagnostics[key]))
    _write_json(output / "training_history.json", diagnostics["training_history"])
    _write_json(output / "neighbor_graph_profile.json", diagnostics["graph_profile"])
    _write_json(output / "edge_weight_summary.json", diagnostics["edge_summary"])
    _write_json(output / "gate_summary.json", diagnostics["gate_summary"])
    _write_json(output / "embedding_geometry.json", embedding_geometry(embedding))
    _write_json(output / "unsupervised_diagnostics.json", diagnostics["unsupervised_diagnostics"])
    torch.save(
        {
            "model_state": diagnostics["model_state"],
            "resolved_config": resolved,
            "graph_profile": diagnostics["graph_profile"],
            "gate_summary": diagnostics["gate_summary"],
        },
        output / "model.pt",
    )
    # 指标在模型完成后计算；任何指标都不会反向影响模型参数或配置选择。
    metrics = clustering_metrics(labels, predictions)
    _write_json(output / "metrics.json", metrics)
    summary: dict[str, Any] = {
        **diagnostics["core_summary"],
        "status": "completed",
        "run_key": run_key,
        "dataset": name,
        "source_path": str(source.resolve()),
        "source_sha256": resolved["source_sha256"],
        "input_array_sha256": resolved["input_array_sha256"],
        "seed": int(seed),
        "n_samples": int(bundle.X.shape[0]),
        "n_features": int(bundle.X.shape[1]),
        "n_clusters": K,
        "K_source": k_source,
        "benchmark_oracle_from_y": k_source == "benchmark_oracle_from_y",
        "labels_present": labels is not None,
        "labels_used_during_fit": False,
        "labels_used_during_preprocessing": False,
        "labels_used_for_graph": False,
        "labels_used_for_gate": False,
        "labels_used_for_loss": False,
        "labels_used_for_selection": False,
        "label_values": bundle.label_values,
        "metrics": metrics,
        "preprocess_profile": bundle.preprocess_profile,
        "graph_profile": diagnostics["graph_profile"],
        "edge_weight_summary": diagnostics["edge_summary"],
        "gate_summary": diagnostics["gate_summary"],
        "unsupervised_diagnostics": diagnostics["unsupervised_diagnostics"],
        "wall_seconds": float(time.time() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "output_files": {
            "predictions": "predictions.npy",
            "predictions_mapped": "predictions_mapped.npy" if labels is not None else None,
            "labels_true": "labels_true.npy" if labels is not None else None,
            "label_mapping": "label_mapping.json" if labels is not None else None,
            "embedding_final": "embedding_final.npy",
            "summary": "summary.json",
            "metrics": "metrics.json",
        },
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "status.json", {"status": "completed", "run_key": run_key})
    record.update({"status": "completed", "wall_seconds": summary["wall_seconds"], "summary": "summary.json"})
    _write_json(output / "run_record.json", record)

    if bundle.adata is not None and not no_save_h5ad:
        # h5ad 写回是可选的展示产物，失败只记录 warning，不改变已完成的核心结果。
        try:
            import methods.shared_utils as shared_utils

            bundle.adata.obsm["X_topogate_v0"] = embedding
            bundle.adata.uns["topogate_v0"] = summary
            shared_utils.sanitize_anndata_for_write(bundle.adata)
            bundle.adata.write_h5ad(output / "adata_topogate_v0.h5ad", compression="gzip")
        except Exception as exc:  # pragma: no cover - optional output path
            _write_json(output / "h5ad_write_warning.json", {"error": str(exc)})
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TopoGate V0 unified scVICAR runner")
    parser.add_argument("--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", required=True)
    parser.add_argument("--config", default=None)
    parameterization = parser.add_mutually_exclusive_group()
    parameterization.add_argument(
        "--parameterization",
        "--variant",
        dest="parameterization",
        default=None,
        help="F/fixed or T/topology; historical short aliases are -f/-F and -t/-T",
    )
    # 保留历史单字母开关，便于直接替换旧 NeighborMix runner；它们写入同一个
    # 原始参数值，随后由 load_config 统一归一化为 fixed/topology。
    parameterization.add_argument(
        "-f", "-F", dest="parameterization", action="store_const", const="fixed"
    )
    parameterization.add_argument(
        "-t", "-T", dest="parameterization", action="store_const", const="topology"
    )
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", default=None)
    parser.add_argument("--label-key", "--label_key", dest="label_key", default="auto")
    parser.add_argument("--n-clusters", "--n_clusters", dest="n_clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--hidden-size", "--hidden_size", dest="hidden_size", type=int, default=None)
    # Common legacy flags remain available as explicit config overrides. The
    # YAML files are still the recommended way to freeze a formal protocol.
    for option, dest, kind in (
        (("--lr",), "lr", float),
        (("--mask-ratio", "--mask_ratio"), "mask_ratio", float),
        (("--pseudo-weight", "--pseudo_weight"), "pseudo_weight", float),
        (("--alpha",), "alpha", float),
        (("--neighbor-k", "--neighbor_k"), "neighbor_k", int),
        (("--mix-neighbors", "--mix_neighbors"), "mix_neighbors", int),
        (("--knn-pca-dim", "--knn_pca_dim"), "knn_pca_dim", int),
        (("--tau",), "tau", float),
        (("--masked-data-weight", "--masked_data_weight"), "masked_data_weight", float),
        (("--mask-loss-weight", "--mask_loss_weight"), "mask_loss_weight", float),
        (("--dropout",), "dropout", float),
        (("--n-top-features", "--n_top_features", "--n-top-genes", "--n_top_genes"), "n_top_features", int),
        (("--target-sum", "--target_sum"), "target_sum", float),
        (("--num-workers", "--num_workers"), "num_workers", int),
        (("--gate-min", "--gate_min"), "gate_min", float),
        (("--gate-max", "--gate_max"), "gate_max", float),
        (("--gamma-sim", "--gamma_sim"), "gamma_sim", float),
        (("--gamma-mutual", "--gamma_mutual"), "gamma_mutual", float),
        (("--gamma-snn", "--gamma_snn"), "gamma_snn", float),
        (("--gamma-distance", "--gamma_distance"), "gamma_distance", float),
        (("--beta-mutual", "--beta_mutual"), "beta_mutual", float),
        (("--beta-snn", "--beta_snn"), "beta_snn", float),
        (("--beta-perturb", "--beta_perturb"), "beta_perturb", float),
        (("--beta-uncertainty", "--beta_uncertainty"), "beta_uncertainty", float),
    ):
        parser.add_argument(*option, dest=dest, type=kind, default=None)
    parser.add_argument(
        "--neighbor-estimator",
        "--neighbor_estimator",
        dest="neighbor_estimator",
        choices=["current", "uniform_sample", "full"],
        default=None,
    )
    parser.add_argument(
        "--edge-reliability-mode",
        "--edge_reliability_mode",
        dest="edge_reliability_mode",
        default=None,
    )
    parser.add_argument(
        "--input-mode", "--input_mode", dest="input_mode", choices=["auto", "raw", "log1p"], default=None
    )
    parser.add_argument("--use-pseudo", dest="use_pseudo", type=_str2bool, default=None)
    parser.add_argument("--no-pseudo", dest="use_pseudo", action="store_false")
    parser.add_argument("--scale-input", dest="scale_input", type=_str2bool, default=None)
    parser.add_argument("--no-scale-input", dest="scale_input", action="store_false")
    parser.add_argument("--drop-last", dest="drop_last", type=_str2bool, default=None)
    parser.add_argument("--no-drop-last", dest="drop_last", action="store_false")
    parser.add_argument("--no-save-h5ad", action="store_true")
    parser.add_argument("--evaluate-unsupervised", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {
        "parameterization": args.parameterization,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "evaluate_unsupervised": True if args.evaluate_unsupervised else None,
    }
    for name in (
        "lr",
        "mask_ratio",
        "pseudo_weight",
        "alpha",
        "neighbor_k",
        "mix_neighbors",
        "knn_pca_dim",
        "tau",
        "masked_data_weight",
        "mask_loss_weight",
        "dropout",
        "n_top_features",
        "target_sum",
        "num_workers",
        "gate_min",
        "gate_max",
        "gamma_sim",
        "gamma_mutual",
        "gamma_snn",
        "gamma_distance",
        "beta_mutual",
        "beta_snn",
        "beta_perturb",
        "beta_uncertainty",
        "neighbor_estimator",
        "edge_reliability_mode",
        "input_mode",
        "use_pseudo",
        "scale_input",
        "drop_last",
    ):
        overrides[name] = getattr(args, name)
    config = load_config(args.config, overrides)
    output = Path(args.save_dir)
    runtime_device = "unresolved"
    try:
        runtime_device = resolve_runtime_device(args.device, args.gpu)
        summary = run_one(
            args.data_path,
            output,
            config=config,
            seed=args.seed,
            device=runtime_device,
            n_clusters=args.n_clusters,
            label_key=args.label_key,
            dataset_name=args.dataset_name,
            no_save_h5ad=args.no_save_h5ad,
        )
        print(json.dumps(summary, ensure_ascii=True), flush=True)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "incomplete_compute",
            "protocol_id": config.protocol_id,
            "parameterization": config.parameterization,
            "seed": int(args.seed),
            "device": runtime_device,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json(output / "status.json", failure)
        record_path = output / "run_record.json"
        if record_path.exists():
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                record = {}
            record.update(failure)
            _write_json(record_path, record)
        raise


if __name__ == "__main__":
    main()
