"""Minimal independent transductive execution primitive for V0_RG.

All unlabeled rows are used for preprocessing and representation fitting. Labels
are retained only for validation/test scoring by the caller.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import torch
from .config import V0_RGConfig
from .input_adapter import load_matrix, encode_labels
from .tuning import SplitPreprocessor
from .trainer import fit_predict
from .run import clustering_metrics

PROTOCOL_ID = "topogate_transductive_perdataset_64x3_v1"
SEEDS = (42, 123, 7)


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)

def run_transductive(data_path, output_dir, *, config: V0_RGConfig, n_clusters: int,
                     labels_path=None, seed: int, input_kind: str = "general",
                     feature_limit: int = 2000, device: str = "cpu", score_indices=None,
                     score_name: str = "validation") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    loaded = load_matrix(data_path, labels_path=labels_path)
    X = loaded.X
    labels = None if loaded.labels is None else np.asarray(loaded.labels).reshape(-1)
    labels, label_classes = encode_labels(labels)
    if labels is not None and len(labels) != X.shape[0]:
        raise ValueError("label count does not match X rows")
    prep = SplitPreprocessor(input_kind, feature_limit).fit(X)
    X_all = prep.transform(X)
    pred, embedding, diagnostics = fit_predict(X_all, fit_X=X_all, n_clusters=int(n_clusters), config=config, seed=int(seed), device=device)
    if not np.isfinite(embedding).all() or not np.isfinite(pred).all():
        raise ValueError("non-finite transductive output")
    record = {"protocol_id": PROTOCOL_ID, "evaluation_mode": "transductive",
              "feature_scope": "all_samples_unlabeled", "graph_scope": "all_samples_unlabeled",
              "labels_used_during_fit": False, "labels_used_for_selection": True,
              "selection_labels": "validation_only", "test_used_for_selection": False,
              "seed": int(seed), "config": asdict(config), "preprocessing_hash": prep.fingerprint,
              "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]), "status": "completed"}
    if labels is not None:
        idx = np.arange(len(labels)) if score_indices is None else np.asarray(score_indices, dtype=int)
        record[f"{score_name}_metrics"] = clustering_metrics(labels[idx], pred[idx])
    _atomic_numpy(out / f"predictions_seed_{seed}.npy", pred.astype(np.int64))
    _atomic_numpy(out / f"embedding_seed_{seed}.npy", embedding.astype(np.float32))
    _atomic_torch(out / f"model_seed_{seed}.pt", diagnostics["model_state_dict"])
    _atomic_npz(
        out / f"preprocessor_seed_{seed}.npz",
        features=np.asarray(prep.features, dtype=np.int64),
        mean=np.asarray(prep.scaler.mean_, dtype=np.float64),
        scale=np.asarray(prep.scaler.scale_, dtype=np.float64),
    )
    centers = diagnostics.get("cluster_centers")
    if centers is not None:
        _atomic_numpy(out / f"cluster_centers_seed_{seed}.npy", np.asarray(centers, dtype=np.float32))
    _atomic_json(out / f"preprocessor_seed_{seed}.json", {
        "input_kind": input_kind,
        "feature_limit": int(feature_limit),
        "width": int(prep.width),
        "selection": prep.selection,
        "target_sum": float(prep.target_sum),
        "fingerprint": prep.fingerprint,
    })
    _atomic_json(out / f"training_history_seed_{seed}.json", diagnostics["training_history"])
    _atomic_json(out / f"graph_gate_summary_seed_{seed}.json", {
        "core_summary": diagnostics["core_summary"],
        "graph_profile": diagnostics["graph_profile"],
        "edge_summary": diagnostics["edge_summary"],
        "gate_summary": diagnostics["gate_summary"],
    })
    _atomic_json(out / f"record_seed_{seed}.json", record)
    return record

__all__ = ["run_transductive", "PROTOCOL_ID", "SEEDS"]
