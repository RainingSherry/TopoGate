from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import sklearn
import torch
import yaml
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from .config import V15Config, load_config
from .sparse import prepare_input
from .trainer import V15Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu: int, no_cuda: bool) -> torch.device:
    if no_cuda or not torch.cuda.is_available():
        return torch.device("cpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if set(ids).intersection({"0", "7"}):
            raise ValueError("CUDA_VISIBLE_DEVICES includes forbidden GPU 0 or 7")
        if len(ids) == 1:
            return torch.device("cuda:0")
        if str(gpu) in ids:
            return torch.device(f"cuda:{ids.index(str(gpu))}")
        if 0 <= gpu < len(ids):
            return torch.device(f"cuda:{gpu}")
        raise ValueError(f"gpu={gpu} is not in CUDA_VISIBLE_DEVICES={visible}")
    if gpu in {0, 7}:
        raise ValueError("physical GPU 0 and GPU 7 are forbidden")
    return torch.device(f"cuda:{gpu}")


def load_npz(path: str | Path) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        x_key = "x" if "x" in keys else "X" if "X" in keys else None
        sparse_keys = {"data", "indices", "indptr", "shape"}
        if x_key is None and sparse_keys.issubset(keys):
            X = sp.csr_matrix(
                (payload["data"], payload["indices"], payload["indptr"]),
                shape=tuple(int(v) for v in payload["shape"]),
                dtype=np.float32,
            )
        elif x_key is None:
            raise KeyError(f"NPZ must contain x or X; found {sorted(keys)}")
        else:
            X = np.asarray(payload[x_key], dtype=np.float32)
        y_key = "y" if "y" in keys else "labels" if "labels" in keys else None
        y = None if y_key is None else np.asarray(payload[y_key]).reshape(-1)
    return X, y


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_input(X: np.ndarray | sp.spmatrix) -> str:
    digest = hashlib.sha256()
    if sp.issparse(X):
        matrix = sp.csr_matrix(X, dtype=np.float32)
        for value in (matrix.data, matrix.indices, matrix.indptr, np.asarray(matrix.shape, dtype=np.int64)):
            digest.update(np.asarray(value).tobytes())
    else:
        digest.update(np.asarray(X, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    files = [
        root / "config.py",
        root / "sparse.py",
        root / "graph.py",
        root / "model.py",
        root / "trainer.py",
        root / "run.py",
    ]
    return {str(path.relative_to(root.parent.parent.parent)): _hash_file(path) for path in files}


def _json_dump(payload: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def external_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(y_true, prediction)),
        "nmi": float(normalized_mutual_info_score(y_true, prediction)),
        "ami": float(adjusted_mutual_info_score(y_true, prediction)),
    }


def fit_v15(
    X: np.ndarray | sp.spmatrix,
    n_clusters: int,
    y: np.ndarray | None = None,
    *,
    config: V15Config,
    save_dir: str | Path,
    dataset_name: str = "adhoc",
    source_path: str | Path | None = None,
    k_protocol: str = "explicit",
    run_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, float, dict]:
    set_seed(config.seed)
    prepared = prepare_input(X, config.sparse_zero_threshold, config.sparse_transform)
    device = get_device(config.gpu, config.no_cuda)
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainer = V15Trainer(prepared, int(n_clusters), config, device)
    result = trainer.fit()
    primary_pred = result.predictions
    metrics: dict[str, Any] = {
        "prediction_source": (
            "student_clean"
            if config.final_prediction_source == "student_clean"
            else "assignment_transport"
            if config.output_mode == "assignment"
            else config.cluster_head
        ),
        "silhouette": float(silhouette_score(result.embedding, primary_pred))
        if 1 < np.unique(primary_pred).size < len(primary_pred)
        else None,
    }
    encoded_y = None
    if y is not None:
        values, encoded_y = np.unique(np.asarray(y).reshape(-1), return_inverse=True)
        encoded_y = encoded_y.astype(np.int64)
        metrics.update(external_metrics(encoded_y, primary_pred))
        metrics["labels_unique"] = int(values.size)
    np.save(output / "embedding_final.npy", result.embedding)
    np.save(output / "embedding_self.npy", result.gate_diagnostics["final_embedding_self"])
    np.save(output / "embedding_transport.npy", result.gate_diagnostics["final_embedding_transport"])
    np.save(output / "cluster_probabilities.npy", result.probabilities)
    np.save(output / "predictions.npy", primary_pred)
    np.save(output / "teacher_embedding.npy", result.teacher_diagnostics["embedding"])
    np.save(output / "teacher_probabilities_clean.npy", result.teacher_diagnostics["probabilities_clean"])
    np.save(
        output / "teacher_probabilities_augmented.npy",
        result.teacher_diagnostics["probabilities_augmented"],
    )
    np.save(output / "teacher_probabilities_epoch0.npy", result.teacher_diagnostics["probabilities_epoch0"])
    np.save(
        output / "teacher_probabilities_epoch_last.npy",
        result.teacher_diagnostics["probabilities_epoch_last"],
    )
    np.save(
        output / "teacher_probabilities_shuffled.npy",
        result.teacher_diagnostics["probabilities_shuffled"],
    )
    np.save(
        output / "teacher_probabilities_raw.npy",
        result.teacher_diagnostics["probabilities_raw_view"],
    )
    np.save(
        output / "teacher_probabilities_raw_aligned.npy",
        result.teacher_diagnostics["probabilities_raw_aligned"],
    )
    np.save(
        output / "teacher_probabilities_reference.npy",
        result.teacher_diagnostics["probabilities_reference"],
    )
    np.save(
        output / "teacher_reference_agreement.npy",
        result.teacher_diagnostics["reference_agreement"],
    )
    np.save(
        output / "teacher_reference_disagreement.npy",
        result.teacher_diagnostics["reference_disagreement"],
    )
    _json_dump(result.teacher_selection, output / "teacher_selection.json")
    np.savez_compressed(
        output / "gate_diagnostics.npz",
        anchor_indices=result.gate_diagnostics["anchor_indices"],
        candidate_indices=result.graph.indices,
        raw_candidate_indices=result.graph.raw_indices,
        latent_candidate_indices=result.graph.latent_indices,
        candidate_features=result.graph.features,
        candidate_valid=result.graph.valid,
        utility_target=result.gate_diagnostics["utility_target"],
        utility_hat=result.gate_diagnostics["utility_hat"],
        predicted_pi=result.gate_diagnostics["predicted_pi"],
        utility_valid=result.gate_diagnostics["valid"],
        gate_valid=result.gate_diagnostics["gate_valid"],
        utility_features=result.gate_diagnostics["features"],
        utility_semantic_help=result.gate_diagnostics["semantic_help"],
        utility_reconstruction_damage=result.gate_diagnostics["reconstruction_damage"],
        teacher_reference_agreement=result.gate_diagnostics["reference_agreement"],
        teacher_reference_disagreement=result.gate_diagnostics["reference_disagreement"],
        utility_independent_cluster_gain=result.gate_diagnostics["independent_cluster_gain"],
        utility_probe_self_prediction=result.gate_diagnostics["probe_self_prediction"],
        utility_probe_edge_prediction=result.gate_diagnostics["probe_edge_prediction"],
        utility_train_anchor=result.gate_diagnostics["train_anchor"],
        final_predicted_pi=result.gate_diagnostics["final_predicted_pi"],
        final_utility_hat=result.gate_diagnostics["final_utility_hat"],
        final_utility_features=result.gate_diagnostics["final_utility_features"],
        final_gate_valid=result.gate_diagnostics["final_gate_valid"],
        final_probe_self_prediction=result.gate_diagnostics["final_probe_self_prediction"],
        final_probe_edge_prediction=result.gate_diagnostics["final_probe_edge_prediction"],
        final_embedding_self=result.gate_diagnostics["final_embedding_self"],
        final_q_self=result.gate_diagnostics["final_q_self"],
        final_q_edge=result.gate_diagnostics["final_q_edge"],
        final_edge_embedding=result.gate_diagnostics["final_edge_embedding"],
        final_embedding_transport=result.gate_diagnostics["final_embedding_transport"],
        final_gate_readout_probabilities=result.gate_diagnostics["final_gate_readout_probabilities"],
        final_student_probabilities=result.gate_diagnostics["final_student_probabilities"],
    )
    if encoded_y is not None:
        np.save(output / "labels_true.npy", encoded_y)
    graph_profile = dict(result.graph.profile)
    if encoded_y is not None:
        graph_profile["posthoc_edge_purity"] = result.graph.edge_purity(encoded_y)
        graph_profile["posthoc_candidate_recall"] = result.graph.candidate_recall(encoded_y)
    gate_diag = result.gate_diagnostics
    valid_diag = gate_diag["valid"].astype(bool)
    predicted_pi = gate_diag.get("final_predicted_pi", gate_diag["predicted_pi"])
    final_utility_hat = gate_diag.get("final_utility_hat", gate_diag["utility_hat"])
    if predicted_pi.size:
        gate_metrics = {
            "final_null_mass": float(np.mean(predicted_pi[:, 0])),
            "final_edge_mass": float(np.mean(predicted_pi[:, 1:].sum(axis=1))),
            "final_effective_neighbors": float(
                np.mean(np.exp(-(predicted_pi[:, 1:] * np.log(np.clip(predicted_pi[:, 1:], 1e-8, None))).sum(axis=1)))
            ),
            "final_utility_positive_rate": float(np.mean(final_utility_hat > 0.0)) if final_utility_hat.size else 0.0,
        }
    else:
        gate_metrics = {
            "final_null_mass": None,
            "final_edge_mass": None,
            "final_effective_neighbors": None,
            "final_utility_positive_rate": None,
        }
    metrics["gate"] = gate_metrics
    source_hash = _hash_file(source_path) if source_path is not None else _hash_input(X)
    summary = {
        "method": "TopoGate",
        "variant": "V15_counterfactual_gate",
        "dataset": dataset_name,
        "source_path": None if source_path is None else str(Path(source_path).resolve()),
        "source_sha256": source_hash,
        "source_files_sha256": _source_hashes(),
        "seed": int(config.seed),
        "n_samples": int(prepared.n_samples),
        "n_features": int(prepared.n_features),
        "n_clusters": int(n_clusters),
        "K": int(n_clusters),
        "k_protocol": k_protocol,
        "cluster_count_source": k_protocol,
        "benchmark_oracle_from_y": k_protocol == "benchmark_oracle_from_y",
        "labels_used_during_fit": False,
        "device": str(device),
        "train_seconds": float(result.train_seconds),
        "config": config.to_dict(),
        "preprocessing": prepared.profile,
        "graph_profile": graph_profile,
        "metrics": metrics,
        "history": result.history,
        "graph_history": result.graph_history,
        "cluster_frequency_ema": result.cluster_frequency_ema.tolist(),
        "teacher_selection": result.teacher_selection,
        "run_metadata": {} if run_metadata is None else run_metadata,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda,
        },
        "output_files": {
            "embedding": "embedding_final.npy",
            "embedding_self": "embedding_self.npy",
            "embedding_transport": "embedding_transport.npy",
            "probabilities": "cluster_probabilities.npy",
            "predictions": "predictions.npy",
            "labels_true": "labels_true.npy" if encoded_y is not None else None,
            "gate_diagnostics": "gate_diagnostics.npz",
            "teacher_embedding": "teacher_embedding.npy",
            "teacher_probabilities_clean": "teacher_probabilities_clean.npy",
            "teacher_probabilities_augmented": "teacher_probabilities_augmented.npy",
            "teacher_probabilities_epoch0": "teacher_probabilities_epoch0.npy",
            "teacher_probabilities_epoch_last": "teacher_probabilities_epoch_last.npy",
            "teacher_probabilities_shuffled": "teacher_probabilities_shuffled.npy",
            "teacher_probabilities_raw": "teacher_probabilities_raw.npy",
            "teacher_probabilities_raw_aligned": "teacher_probabilities_raw_aligned.npy",
            "teacher_probabilities_reference": "teacher_probabilities_reference.npy",
            "teacher_reference_agreement": "teacher_reference_agreement.npy",
            "teacher_reference_disagreement": "teacher_reference_disagreement.npy",
            "teacher_selection": "teacher_selection.json",
        },
    }
    _json_dump(config.to_dict(), output / "resolved_config.json")
    _json_dump(metrics, output / "metrics.json")
    _json_dump(summary, output / "summary.json")
    return primary_pred, result.train_seconds, metrics


def run_v15(
    X: np.ndarray | sp.spmatrix,
    n_clusters: int,
    y: np.ndarray | None = None,
    *,
    config_path: str | Path | None = None,
    save_dir: str | Path,
    dataset_name: str = "adhoc",
    seed: int = 42,
    source_path: str | Path | None = None,
    run_metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> tuple[np.ndarray, float, dict]:
    overrides = {**overrides, "seed": int(seed)}
    config = load_config(config_path, overrides)
    return fit_v15(
        X,
        int(n_clusters),
        y,
        config=config,
        save_dir=save_dir,
        dataset_name=dataset_name,
        source_path=source_path,
        k_protocol="explicit",
        run_metadata=run_metadata,
    )


def _parse_overrides(values: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"override must be key=value: {value}")
        key, raw = value.split("=", 1)
        output[key] = yaml.safe_load(raw)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoGate V15 counterfactual gate")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "topogate_v15.yaml"))
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    X, y = load_npz(args.data_path)
    if args.n_clusters is None:
        if y is None:
            raise ValueError("--n_clusters is required when NPZ has no labels")
        n_clusters = int(np.unique(y).size)
        k_protocol = "benchmark_oracle_from_y"
    else:
        n_clusters = int(args.n_clusters)
        k_protocol = "explicit"
    overrides = _parse_overrides(args.overrides)
    overrides.update({"seed": args.seed, "gpu": args.gpu, "no_cuda": args.no_cuda})
    config = load_config(args.config, overrides)
    fit_v15(
        X,
        n_clusters,
        y,
        config=config,
        save_dir=args.save_dir,
        dataset_name=args.dataset_name or Path(args.data_path).stem,
        source_path=args.data_path,
        k_protocol=k_protocol,
    )


if __name__ == "__main__":
    main()
