from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import V22_VARIANTS, load_config
from .input_adapter import load_npz, prepare_dual_input
from .trainer import ALLOWED_PHYSICAL_GPUS, fit_v22, resolve_device


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

    overrides: dict[str, Any] = {}
    base_config = load_config(args.config)
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.variant is not None:
        overrides["variant"] = str(args.variant)
    config = load_config(args.config, overrides)
    if args.reuse_topology_cache and not config.uses_topology_gate:
        raise ValueError("--reuse-topology-cache requires a topology Gate variant")
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

    prepared = prepare_dual_input(
        loaded.X,
        dataset_name=args.dataset_name,
        input_protocol=args.input_protocol,
        feature_cap=config.feature_cap,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime_device = resolve_device(args.device, args.gpu)
    topology_cache_dir = Path(args.topology_cache_dir) if args.topology_cache_dir is not None else output / "cache"
    if args.reuse_topology_cache and not topology_cache_dir.is_dir():
        raise FileNotFoundError(f"topology cache directory is missing: {topology_cache_dir}")
    embedding, diagnostics = fit_v22(
        prepared.X_model,
        prepared.X_graph if config.uses_topology_gate else None,
        prepared.X_support if config.uses_topology_gate else None,
        config=config,
        seed=args.seed,
        device=runtime_device,
        stats_cache_dir=topology_cache_dir if config.uses_topology_gate else None,
        reuse_topology_cache=bool(args.reuse_topology_cache),
    )
    predictions = KMeans(n_clusters=n_clusters, n_init=config.kmeans_n_init, random_state=args.seed).fit_predict(embedding)

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
            "source_sha256": getattr(args, "source_sha256", None),
            "topology_cache_reused": bool(args.reuse_topology_cache),
            "topology_cache_dir": str(topology_cache_dir.resolve()) if args.reuse_topology_cache else None,
        },
    )
    _write_json(
        output / "preprocess_profile.json",
        prepared.profile
        | {
            "labels_used_during_fit": False,
            "K_used_during_fit": False,
            "K_source": k_source,
        },
    )
    _write_json(output / "graph_profile.json", diagnostics["graph_profile"])
    _write_json(output / "stats_profile.json", diagnostics["stats_profile"])
    _write_json(output / "training_history.json", diagnostics["history"])
    np.save(output / "embedding_final.npy", embedding)
    np.save(output / "predictions.npy", predictions)
    np.save(output / "selected_feature_indices.npy", prepared.selected_feature_indices)

    metrics: dict[str, Any] = {
        "labels_available": labels is not None,
        "n_clusters": int(n_clusters),
        "cluster_method": "kmeans_clean_embedding",
        "K_source": k_source,
        "labels_used_during_fit": False,
        "K_used_during_fit": False,
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
    _write_json(output / "metrics.json", metrics)

    model = diagnostics.pop("model")
    discriminator = diagnostics.pop("discriminator")
    gate = diagnostics.pop("gate")
    torch.save(
        {
            "model": model.state_dict(),
            "discriminator": None if discriminator is None else discriminator.state_dict(),
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
        "source_sha256": getattr(args, "source_sha256", None),
        "input_protocol": args.input_protocol,
        "n_samples": int(prepared.X_model.shape[0]),
        "n_features_original": int(prepared.profile["n_features_original"]),
        "n_features": int(prepared.X_model.shape[1]),
        "labels_used_during_fit": False,
        "K_used_during_fit": False,
        "K_source": k_source,
        "prediction_semantics": "kmeans_clean_embedding",
        "mask_semantics": {
            "reconstruction": "random_topk_donor_corruption; effective changed positions train scMAE",
            "adversarial": (
                "cooperative keep Gate; selected keep coordinates remain visible and the complementary changed coordinates are reconstructed"
                if config.gate_reward_mode == "cooperative_keep"
                else "hard topk donor corruption; M=1 means coordinate is reconstructed"
            ),
            "discriminator": "matched real/fake coordinate pairs; no mask or hint input",
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    _write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one V22 topology-discriminator hard-mask experiment")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-protocol", choices=("clubench_bridge", "shared_text", "scRNA_count"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "v22_topology_discriminator_hard_gate.yaml",
    )
    parser.add_argument("--variant", choices=tuple(sorted(V22_VARIANTS)), default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 7), default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--reuse-topology-cache",
        action="store_true",
        help="reuse a complete topology_statistics.dat from the selected cache directory",
    )
    parser.add_argument(
        "--topology-cache-dir",
        type=Path,
        default=None,
        help="cache directory to read when --reuse-topology-cache is enabled",
    )
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument(
        "--source-sha256",
        default=None,
        help="provenance hash from the frozen dataset manifest; does not affect fitting",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_one(parse_args())
