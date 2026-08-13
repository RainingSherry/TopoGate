from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
import torch
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

from .config import V18Config, load_config
from .input_adapter import encode_labels, load_data, prepare_input
from .model import VARIANTS, fit_v18


def _metrics(labels_true: np.ndarray | None, predictions: np.ndarray, abstained: np.ndarray) -> dict[str, Any]:
    if labels_true is None:
        return {"labels_available": False, "abstention_rate": float(np.mean(abstained)) if abstained.size else 0.0}
    y = np.asarray(labels_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(predictions, dtype=np.int64).reshape(-1)
    active = ~np.asarray(abstained, dtype=bool)
    result: dict[str, Any] = {
        "labels_available": True,
        "abstention_rate": float(np.mean(~active)) if active.size else 0.0,
        "active_nodes": int(active.sum()),
        "total_nodes": int(active.size),
    }
    if active.sum() >= 2 and np.unique(pred[active]).size >= 2 and np.unique(y[active]).size >= 2:
        result.update({
            "ari_active": float(adjusted_rand_score(y[active], pred[active])),
            "nmi_active": float(normalized_mutual_info_score(y[active], pred[active])),
            "ami_active": float(adjusted_mutual_info_score(y[active], pred[active])),
        })
    else:
        result.update({"ari_active": None, "nmi_active": None, "ami_active": None})
    result.update({
        "ari_all_with_abstention_label": float(adjusted_rand_score(y, pred)),
        "nmi_all_with_abstention_label": float(normalized_mutual_info_score(y, pred)),
    })
    return result


def _write_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def run_one(data_path: str | Path, save_dir: str | Path, *, config: V18Config, variant: str,
            n_clusters: int | None, dataset_name: str | None = None, dataset_id: str | None = None,
            max_samples: int = 0) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaded = load_data(data_path)
    raw_X = loaded.X
    raw_labels = loaded.labels
    row_indices = None
    if max_samples > 0 and raw_X.shape[0] > max_samples:
        rng = np.random.default_rng(int(config.seed) + 9109)
        row_indices = np.sort(rng.choice(raw_X.shape[0], size=int(max_samples), replace=False))
        raw_X = raw_X[row_indices]
        raw_labels = None if raw_labels is None else np.asarray(raw_labels).reshape(-1)[row_indices]
    X, input_profile = prepare_input(raw_X, input_mode=config.input_mode, standardize=True)
    input_profile["row_sampling"] = row_indices is not None
    input_profile["row_sampling_seed"] = int(config.seed) + 9109
    input_profile["max_samples"] = int(max_samples)
    y, label_values = encode_labels(raw_labels)
    if variant == "v18_leiden" and n_clusters is None:
        k = None
        k_source = "not_applicable_leiden"
    elif n_clusters is None:
        if y is None:
            raise ValueError("--n-clusters is required when the input has no labels")
        k = int(np.unique(y).size)
        k_source = "benchmark_oracle_from_y"
    else:
        k = int(n_clusters)
        k_source = "explicit_n_clusters"
    if k is not None and k <= 0:
        raise ValueError("n_clusters must be positive")
    _write_json(input_profile, output / "input_profile.json")
    result = fit_v18(X, k, config=config, variant=variant, save_dir=output,
                     dataset_name=dataset_name or Path(data_path).stem, source_path=data_path)
    if y is not None:
        if y.shape[0] != X.shape[0]:
            raise ValueError("label count does not match input rows")
        np.save(output / "labels_true.npy", y)
    metrics = _metrics(y, result.predictions, result.abstained)
    _write_json(metrics, output / "metrics.json")
    summary = dict(result.summary)
    summary.update({
        "status": "completed", "protocol_id": config.protocol_id,
        "dataset": dataset_name or Path(data_path).stem,
        "dataset_id": dataset_id or dataset_name or Path(data_path).stem,
        "variant": variant, "seed": int(config.seed),
        "device": str(result.summary.get("device", config.device)),
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "K_used_only_in_readout": variant != "v18_leiden",
        "data_path": str(Path(data_path).resolve()), "input_profile": input_profile,
        "labels_used_during_fit": False, "label_values": label_values,
        "n_clusters": None if k is None else int(k),
        "K_source": k_source, "benchmark_oracle_from_y": k_source == "benchmark_oracle_from_y",
        "metrics": metrics,
        "output_files": {"predictions": "predictions.npy", "labels_true": "labels_true.npy" if y is not None else None,
                          "embedding_final": "embedding_final.npy", "latent_final": "latent_final.npy",
                          "latent_mae": "latent_mae.npy", "metrics": "metrics.json", "summary": "summary.json"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                         "sklearn": sklearn.__version__, "torch": torch.__version__},
    })
    _write_json(summary, output / "summary.json")
    _write_json({"status": "completed", "protocol_id": config.protocol_id,
                 "dataset_id": summary["dataset_id"],
                 "run_key": f"{summary['dataset_id']}::{variant}::seed{config.seed}"}, output / "status.json")
    print(json.dumps(summary, ensure_ascii=True))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TopoGate V18 scMAE latent edge-gate runner")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="v18_full")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-mode", choices=["auto", "count", "nonnegative", "continuous"], default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--epochs-mae", type=int, default=None)
    parser.add_argument("--epochs-gate", type=int, default=None)
    parser.add_argument("--epochs-joint", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--candidate-width", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {"seed": args.seed, "input_mode": args.input_mode, "device": args.device,
                 "hidden_size": args.hidden_size, "mask_ratio": args.mask_ratio,
                 "epochs_mae": args.epochs_mae, "epochs_gate": args.epochs_gate,
                 "epochs_joint": args.epochs_joint, "batch_size": args.batch_size,
                 "candidate_k": args.candidate_k, "candidate_width": args.candidate_width}
    config = load_config(args.config, overrides)
    try:
        run_one(args.data_path, args.save_dir, config=config, variant=args.variant,
                n_clusters=args.n_clusters, dataset_name=args.dataset_name, dataset_id=args.dataset_id,
                max_samples=args.max_samples)
    except Exception as exc:
        output = Path(args.save_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_json({"status": "incomplete_compute", "protocol_id": config.protocol_id,
                     "error_type": type(exc).__name__, "error": str(exc)}, output / "status.json")
        raise


if __name__ == "__main__":
    main()
