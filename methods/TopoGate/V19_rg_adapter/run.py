from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
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

from .config import INPUT_PROTOCOLS, VARIANTS, V19Config, load_config
from .input_adapter import encode_labels, load_npz, prepare_input
from .trainer import fit_predict


ALLOWED_PHYSICAL_GPUS = frozenset({1, 2, 3, 4, 5, 6})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _mapped_predictions(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    true_values = np.unique(y_true)
    pred_values = np.unique(predictions)
    width = max(len(true_values), len(pred_values))
    counts = np.zeros((width, width), dtype=np.int64)
    for row, true_value in enumerate(true_values):
        for column, predicted_value in enumerate(pred_values):
            counts[row, column] = int(
                np.sum((y_true == true_value) & (predictions == predicted_value))
            )
    rows, columns = linear_sum_assignment(-counts)
    mapped = np.full_like(predictions, fill_value=-1, dtype=np.int64)
    for row, column in zip(rows, columns, strict=True):
        if row < len(true_values) and column < len(pred_values):
            mapped[predictions == pred_values[column]] = true_values[row]
    return mapped


def clustering_metrics(labels: np.ndarray | None, predictions: np.ndarray) -> dict[str, Any]:
    if labels is None:
        return {"labels_available": False}
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


def resolve_runtime_device(device: str, gpu: int) -> str:
    if device == "cpu":
        return "cpu"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        physical = [item.strip() for item in visible.split(",") if item.strip()]
        if set(physical).intersection({"0", "7"}):
            raise ValueError("CUDA_VISIBLE_DEVICES includes forbidden physical GPU 0 or 7")
        if not physical:
            return "cpu" if device == "auto" else "cuda:0"
        if str(gpu) in physical:
            logical = physical.index(str(gpu))
        elif len(physical) == 1:
            logical = 0
        elif 0 <= int(gpu) < len(physical):
            logical = int(gpu)
        else:
            raise ValueError(f"GPU {gpu} is not available in CUDA_VISIBLE_DEVICES={visible!r}")
        if device == "auto" and not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{logical}"
    if int(gpu) not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(
            f"physical GPU {gpu} is forbidden; allowed GPUs are {sorted(ALLOWED_PHYSICAL_GPUS)}"
        )
    # Isolate the requested physical card before the first CUDA availability
    # query; changing CUDA_VISIBLE_DEVICES after initialization is ineffective.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    if not torch.cuda.is_available():
        if device == "auto":
            return "cpu"
        raise RuntimeError("CUDA was requested but is unavailable")
    return "cuda:0"


def run_one(
    data_path: str | Path,
    save_dir: str | Path,
    *,
    config: V19Config,
    input_protocol: str,
    seed: int,
    device: str,
    dataset_name: str | None = None,
    dataset_id: str | None = None,
    n_clusters: int | None = None,
    max_samples: int = 0,
) -> dict[str, Any]:
    if input_protocol not in INPUT_PROTOCOLS:
        raise ValueError(f"input_protocol must be one of {sorted(INPUT_PROTOCOLS)}")
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    name = dataset_name or Path(data_path).stem
    identifier = dataset_id or f"{name}__{input_protocol}"
    started = time.time()
    run_key = f"{identifier}::{config.variant}::seed{int(seed)}"
    run_record: dict[str, Any] = {
        "status": "running",
        "run_key": run_key,
        "protocol_id": config.protocol_id,
        "dataset_id": identifier,
        "dataset": name,
        "source_path": str(Path(data_path).resolve()),
        "input_protocol": input_protocol,
        "variant": config.variant,
        "seed": int(seed),
        "labels_used_during_fit": False,
        "labels_used_during_preprocessing": False,
        "labels_used_for_graph": False,
        "labels_used_for_gate": False,
        "labels_used_for_loss": False,
        "labels_used_for_selection": False,
    }
    _write_json(output / "run_record.json", run_record)
    _write_json(
        output / "status.json",
        {"status": "running", "protocol_id": config.protocol_id, "run_key": run_key},
    )

    loaded = load_npz(data_path)
    raw_X = loaded.X
    raw_labels = loaded.labels
    original_n_samples = int(raw_X.shape[0])
    row_indices = None
    if int(max_samples) > 0 and raw_X.shape[0] > int(max_samples):
        sampling_rng = np.random.default_rng(int(seed) + 9109)
        row_indices = np.sort(
            sampling_rng.choice(raw_X.shape[0], size=int(max_samples), replace=False)
        )
        raw_X = raw_X[row_indices]
        if raw_labels is not None:
            raw_labels = np.asarray(raw_labels).reshape(-1)[row_indices]
    dataset_profile = dict(loaded.profile)
    dataset_profile.update(
        {
            "dataset_name": name,
            "dataset_id": identifier,
            "n_samples_used": int(raw_X.shape[0]),
            "row_sampling": row_indices is not None,
            "row_sampling_seed": int(seed) + 9109 if row_indices is not None else None,
            "max_samples": int(max_samples),
            "labels_passed_to_preprocessing": False,
            "labels_passed_to_fit": False,
        }
    )
    prepared = prepare_input(
        raw_X,
        dataset_name=name,
        input_protocol=input_protocol,
        n_top_features=config.n_top_features,
        target_sum=config.target_sum,
    )
    labels, label_values = encode_labels(raw_labels)
    if labels is not None and labels.shape[0] != prepared.X.shape[0]:
        raise ValueError("label count does not match the number of input rows")
    if n_clusters is None:
        if labels is None:
            raise ValueError("n_clusters is required when the NPZ has no benchmark labels")
        K = int(np.unique(labels).size)
        k_source = "benchmark_oracle_from_y"
    else:
        K = int(n_clusters)
        k_source = "explicit_n_clusters"
    if K <= 0:
        raise ValueError("n_clusters must be positive")

    resolved = config.resolved_dict()
    resolved.update(
        {
            "seed": int(seed),
            "device": str(device),
            "input_protocol": input_protocol,
            "dataset_name": name,
            "dataset_id": identifier,
        }
    )
    _write_json(output / "resolved_config.json", resolved)
    _write_json(output / "dataset_profile.json", dataset_profile)
    _write_json(output / "preprocess_profile.json", prepared.profile)
    np.save(output / "selected_feature_indices.npy", prepared.selected_feature_indices)

    predictions, embedding, diagnostics = fit_predict(
        prepared.X,
        n_clusters=K,
        config=config,
        seed=int(seed),
        device=device,
    )
    np.save(output / "predictions.npy", predictions.astype(np.int64))
    if labels is not None:
        np.save(output / "labels_true.npy", labels.astype(np.int64))
    np.save(output / "embedding_final.npy", embedding.astype(np.float32))
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
        np.save(output / f"{key}.npy", diagnostics[key])
    _write_json(output / "training_history.json", diagnostics["training_history"])
    _write_json(output / "neighbor_graph_profile.json", diagnostics["graph_profile"])
    _write_json(output / "edge_weight_summary.json", diagnostics["edge_summary"])
    _write_json(output / "gate_summary.json", diagnostics["gate_summary"])
    metrics = clustering_metrics(labels, predictions)
    _write_json(output / "metrics.json", metrics)
    comparison_scope = (
        "archived_sota_bridge_eligible"
        if input_protocol in {"clubench_bridge", "shared_text"}
        else "internal_rg_native_only"
    )
    summary = dict(diagnostics["core_summary"])
    summary.update(
        {
            "status": "completed",
            "protocol_id": config.protocol_id,
            "run_key": run_key,
            "dataset": name,
            "dataset_id": identifier,
            "source_path": str(Path(data_path).resolve()),
            "input_protocol": input_protocol,
            "comparison_scope": comparison_scope,
            "seed": int(seed),
            "n_samples_original": original_n_samples,
            "n_samples": int(prepared.X.shape[0]),
            "n_features": int(prepared.X.shape[1]),
            "n_clusters": int(K),
            "K_source": k_source,
            "benchmark_oracle_from_y": k_source == "benchmark_oracle_from_y",
            "K_used_only_in_readout": True,
            "labels_used_during_fit": False,
            "labels_used_during_preprocessing": False,
            "labels_used_for_graph": False,
            "labels_used_for_gate": False,
            "labels_used_for_loss": False,
            "labels_used_for_selection": False,
            "label_values": label_values,
            "metrics": metrics,
            "preprocess_profile": prepared.profile,
            "graph_profile": diagnostics["graph_profile"],
            "edge_weight_summary": diagnostics["edge_summary"],
            "gate_summary": diagnostics["gate_summary"],
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
                "labels_true": "labels_true.npy" if labels is not None else None,
                "embedding_final": "embedding_final.npy",
                "neighbor_indices": "neighbor_indices.npy",
                "edge_reliability": "edge_reliability.npy",
                "node_gate": "node_gate.npy",
                "metrics": "metrics.json",
                "summary": "summary.json",
            },
        }
    )
    _write_json(output / "summary.json", summary)
    status = {
        "status": "completed",
        "protocol_id": config.protocol_id,
        "run_key": run_key,
        "dataset_id": identifier,
    }
    _write_json(output / "status.json", status)
    run_record.update(
        {
            "status": "completed",
            "wall_seconds": summary["wall_seconds"],
            "K_source": k_source,
            "metrics": metrics,
            "summary": "summary.json",
        }
    )
    _write_json(output / "run_record.json", run_record)
    launcher_log = output / "launcher.log"
    if not launcher_log.exists():
        launcher_log.write_text("direct V19 invocation; stdout was not captured\n", encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TopoGate V19 RG-NeighborMix-scMAE NPZ adapter")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--input-protocol", choices=sorted(INPUT_PROTOCOLS), required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="rg_full")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(
        args.config,
        {
            "variant": args.variant,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
        },
    )
    output = Path(args.save_dir)
    runtime_device = "unresolved"
    try:
        runtime_device = resolve_runtime_device(args.device, args.gpu)
        summary = run_one(
            args.data_path,
            output,
            config=config,
            input_protocol=args.input_protocol,
            seed=args.seed,
            device=runtime_device,
            dataset_name=args.dataset_name,
            dataset_id=args.dataset_id,
            n_clusters=args.n_clusters,
            max_samples=args.max_samples,
        )
        print(json.dumps(summary, ensure_ascii=True), flush=True)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "incomplete_compute",
            "protocol_id": config.protocol_id,
            "variant": config.variant,
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
