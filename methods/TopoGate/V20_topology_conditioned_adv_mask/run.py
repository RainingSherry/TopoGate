from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import load_config
from .input_adapter import load_npz, prepare_dual_input
from .trainer import ALLOWED_PHYSICAL_GPUS, fit_full, fit_scmae_only, resolve_device


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


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu is not None and int(args.gpu) not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(f"GPU {args.gpu} is forbidden; allowed={sorted(ALLOWED_PHYSICAL_GPUS)}")
    if args.gpu is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(int(args.gpu)))
    overrides: dict[str, Any] = {}
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
        overrides["warmup_epochs"] = int(args.warmup_epochs if args.warmup_epochs is not None else min(40, max(0, args.epochs // 2)))
    elif args.warmup_epochs is not None:
        overrides["warmup_epochs"] = int(args.warmup_epochs)
    if args.gate_lr is not None:
        overrides["gate_lr"] = float(args.gate_lr)
    if args.tau_ste is not None:
        overrides["tau_ste"] = float(args.tau_ste)
    config = load_config(args.config, overrides)
    loaded = load_npz(args.data)
    prepared = prepare_dual_input(loaded.X, dataset_name=args.dataset_name, input_protocol=args.input_protocol)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime_device = resolve_device(args.device, args.gpu)
    if config.variant == "scmae_only":
        embedding, diagnostics = fit_scmae_only(
            prepared.X_model,
            config=config,
            seed=args.seed,
            device=runtime_device,
        )
    else:
        embedding, diagnostics = fit_full(
            prepared.X_model,
            prepared.X_graph,
            config=config,
            seed=args.seed,
            device=runtime_device,
            stats_cache_dir=output / "cache",
        )
    labels = loaded.labels
    if labels is None and args.n_clusters is None:
        raise ValueError("n_clusters is required when the NPZ has no labels")
    k = int(args.n_clusters) if args.n_clusters is not None else int(np.unique(labels).size)
    predictions = KMeans(n_clusters=k, n_init=int(config.kmeans_n_init), random_state=int(args.seed)).fit_predict(embedding).astype(np.int64)
    _write_json(output / "resolved_config.json", config.to_dict() | {"seed": int(args.seed), "device": str(runtime_device), "dataset": args.dataset_name, "input_protocol": args.input_protocol})
    _write_json(output / "preprocess_profile.json", prepared.profile | {"labels_used": False, "K_used": False})
    _write_json(output / "graph_profile.json", diagnostics["graph_profile"])
    _write_json(output / "stats_profile.json", diagnostics["stats_profile"])
    _write_json(output / "training_history.json", diagnostics["history"])
    np.save(output / "embedding_final.npy", embedding)
    np.save(output / "predictions.npy", predictions)
    np.save(output / "selected_feature_indices.npy", prepared.selected_feature_indices)
    metrics: dict[str, Any] = {"labels_available": labels is not None, "n_clusters": k, "cluster_method": "kmeans_known_k"}
    if labels is not None:
        labels_encoded = np.asarray(labels).astype(str)
        _unique, encoded = np.unique(labels_encoded, return_inverse=True)
        np.save(output / "labels_true.npy", encoded.astype(np.int64))
        metrics.update({"ari": float(adjusted_rand_score(encoded, predictions)), "nmi": float(normalized_mutual_info_score(encoded, predictions)), "acc": _mapped_accuracy(encoded, predictions), "labels_used_during_fit": False, "K_source": "benchmark_oracle_from_y"})
    _write_json(output / "metrics.json", metrics)
    model = diagnostics.pop("model")
    gate = diagnostics.pop("gate")
    import torch

    torch.save({"model": model.state_dict(), "gate": None if gate is None else gate.state_dict()}, output / "checkpoint.pt")
    summary = {"status": "completed", "protocol_id": config.protocol_id, "variant": config.variant, "seed": int(args.seed), "dataset": args.dataset_name, "dataset_path": str(Path(args.data).resolve()), "input_protocol": args.input_protocol, "n_samples": int(prepared.X_model.shape[0]), "n_features": int(prepared.X_model.shape[1]), "K_used_only_in_readout": True, "labels_used_during_fit": False, "mask_semantics": f"{config.mask_target_mode}_mask_is_training_target; effective_value_changes_are_recorded", "metrics": metrics, "diagnostics": diagnostics}
    _write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V20 Full on one NPZ dataset")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-protocol", choices=("clubench_bridge", "shared_text"), required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "v20_full.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 7), default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--gate-lr", type=float, default=None)
    parser.add_argument("--tau-ste", type=float, default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_one(parse_args())
