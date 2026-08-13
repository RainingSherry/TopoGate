from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import load_config
from .input_adapter import load_npz, prepare_dual_input
from .readout import select_readout
from .trainer import ALLOWED_PHYSICAL_GPUS, fit_v21, resolve_device


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _mapped_accuracy(y_true: np.ndarray, pred: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    classes = np.unique(y_true)
    predicted = np.unique(pred)
    matrix = np.zeros((classes.size, predicted.size), dtype=np.int64)
    for i, actual in enumerate(classes):
        for j, value in enumerate(predicted):
            matrix[i, j] = int(np.sum((y_true == actual) & (pred == value)))
    rows, cols = linear_sum_assignment(-matrix)
    return float(matrix[rows, cols].sum() / max(1, y_true.size))


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    _unique, encoded = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return encoded.astype(np.int64)


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda":
        if args.gpu is None:
            raise ValueError("--gpu is required for CUDA runs; allowed physical GPUs are 1..6")
        if int(args.gpu) not in ALLOWED_PHYSICAL_GPUS:
            raise ValueError(f"GPU {args.gpu} is forbidden; allowed={sorted(ALLOWED_PHYSICAL_GPUS)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    elif args.gpu is not None:
        raise ValueError("--gpu cannot be used with --device cpu")
    if args.device == "cpu":
        cpu_threads = max(1, int(os.environ.get("TOPOGATE_CPU_THREADS", "1")))
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(2, cpu_threads)))
    base_config = load_config(args.config)
    overrides: dict[str, Any] = {}
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
        if args.warmup_epochs is None:
            overrides["warmup_epochs"] = int(
                args.epochs if base_config.variant == "scmae_only" else max(0, args.epochs // 2)
            )
    if args.warmup_epochs is not None:
        overrides["warmup_epochs"] = int(args.warmup_epochs)
    if args.gate_lr is not None:
        overrides["gate_lr"] = float(args.gate_lr)
    if args.assignment_weight is not None:
        overrides["assignment_weight"] = float(args.assignment_weight)
    if args.infomax_weight is not None:
        overrides["infomax_weight"] = float(args.infomax_weight)
    config = load_config(args.config, overrides)

    loaded = load_npz(args.data)
    labels = loaded.labels
    if args.n_clusters is not None:
        n_clusters = int(args.n_clusters)
        k_source = "explicit_n_clusters"
    elif labels is not None:
        n_clusters = int(np.unique(labels).size)
        k_source = "benchmark_oracle_from_y"
    else:
        raise ValueError("--n-clusters is required when the NPZ has no labels")
    if n_clusters <= 1:
        raise ValueError("n_clusters must be greater than one")

    prepared = prepare_dual_input(loaded.X, dataset_name=args.dataset_name, input_protocol=args.input_protocol)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime_device = resolve_device(args.device, args.gpu)
    model_k = n_clusters if config.uses_cluster_head else None
    embedding, probabilities, diagnostics = fit_v21(
        prepared.X_model,
        prepared.X_graph if config.uses_topology_gate else None,
        n_clusters=model_k,
        config=config,
        seed=args.seed,
        device=runtime_device,
        stats_cache_dir=output / "cache" if config.uses_topology_gate else None,
    )

    predictions, head_predictions, readout_profile = select_readout(
        embedding,
        probabilities,
        n_clusters=n_clusters,
        mode=config.readout_mode,
        kmeans_n_init=config.kmeans_n_init,
        seed=args.seed,
    )
    cluster_method = str(readout_profile["primary_method"])

    _write_json(
        output / "resolved_config.json",
        config.to_dict()
        | {
            "seed": int(args.seed),
            "device": str(runtime_device),
            "dataset": args.dataset_name,
            "input_protocol": args.input_protocol,
            "n_clusters": int(n_clusters),
            "K_source": k_source,
        },
    )
    _write_json(
        output / "preprocess_profile.json",
        prepared.profile
        | {
            "labels_used_during_fit": False,
            "K_used_during_fit": bool(config.uses_cluster_head),
            "K_source": k_source,
        },
    )
    _write_json(output / "graph_profile.json", diagnostics["graph_profile"])
    _write_json(output / "stats_profile.json", diagnostics["stats_profile"])
    _write_json(output / "training_history.json", diagnostics["history"])
    _write_json(output / "readout_profile.json", readout_profile)
    np.save(output / "embedding_final.npy", embedding)
    np.save(output / "predictions.npy", predictions)
    np.save(output / "selected_feature_indices.npy", prepared.selected_feature_indices)
    if probabilities is not None:
        np.save(output / "cluster_probabilities.npy", probabilities)
    if head_predictions is not None:
        np.save(output / "student_t_predictions.npy", head_predictions)

    metrics: dict[str, Any] = {
        "labels_available": labels is not None,
        "n_clusters": int(n_clusters),
        "cluster_method": cluster_method,
        "K_source": k_source,
        "labels_used_during_fit": False,
        "K_used_during_fit": bool(config.uses_cluster_head),
    }
    if labels is not None:
        encoded = _encode_labels(labels)
        np.save(output / "labels_true.npy", encoded)
        metrics.update(
            {
                "ari": float(adjusted_rand_score(encoded, predictions)),
                "nmi": float(normalized_mutual_info_score(encoded, predictions)),
                "acc": _mapped_accuracy(encoded, predictions),
            }
        )
        if head_predictions is not None:
            metrics["student_t_training_head"] = {
                "ari": float(adjusted_rand_score(encoded, head_predictions)),
                "nmi": float(normalized_mutual_info_score(encoded, head_predictions)),
                "acc": _mapped_accuracy(encoded, head_predictions),
                "labels_used_after_fit_only": True,
            }
    _write_json(output / "metrics.json", metrics)

    model = diagnostics.pop("model")
    head = diagnostics.pop("cluster_head")
    gate = diagnostics.pop("gate")
    torch.save(
        {
            "model": model.state_dict(),
            "cluster_head": None if head is None else head.state_dict(),
            "gate": None if gate is None else gate.state_dict(),
        },
        output / "checkpoint.pt",
    )
    summary = {
        "status": "completed",
        "evidence_level": "engineering_smoke" if config.epochs < 10 else "experiment",
        "protocol_id": config.protocol_id,
        "variant": config.variant,
        "seed": int(args.seed),
        "dataset": args.dataset_name,
        "dataset_path": str(Path(args.data).resolve()),
        "input_protocol": args.input_protocol,
        "n_samples": int(prepared.X_model.shape[0]),
        "n_features": int(prepared.X_model.shape[1]),
        "labels_used_during_fit": False,
        "K_used_during_fit": bool(config.uses_cluster_head),
        "K_source": k_source,
        "prediction_semantics": cluster_method,
        "readout": readout_profile,
        "mask_semantics": {
            "reconstruction": f"{config.random_mask_mode}_random_mask; {config.mask_target_mode}_positions_train_scMAE",
            "assignment": "exact_ratio_within_donor_different_positions; every_selected_position_changes_value"
            if config.uses_cluster_head
            else None,
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    _write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one V21 assignment-adversarial clustering experiment")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-protocol", choices=("clubench_bridge", "shared_text"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "v21_topology_assignment_adversarial.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 7), default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--gate-lr", type=float, default=None)
    parser.add_argument("--assignment-weight", type=float, default=None)
    parser.add_argument("--infomax-weight", type=float, default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_one(parse_args())
