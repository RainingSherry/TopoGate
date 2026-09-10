"""Minimal independent transductive execution primitive for V0_RG.

All unlabeled rows are used for preprocessing and representation fitting. Labels
are retained only for validation/test scoring by the caller.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from .config import V0_RGConfig
from .input_adapter import load_matrix
from .tuning import SplitPreprocessor
from .trainer import fit_predict
from .run import clustering_metrics

PROTOCOL_ID = "topogate_transductive_perdataset_64x3_v1"
SEEDS = (42, 123, 7)

def run_transductive(data_path, output_dir, *, config: V0_RGConfig, n_clusters: int,
                     labels_path=None, seed: int, input_kind: str = "general",
                     feature_limit: int = 2000, device: str = "cpu") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    loaded = load_matrix(data_path, labels_path=labels_path)
    X = loaded.X
    labels = None if loaded.labels is None else np.asarray(loaded.labels).reshape(-1)
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
    if labels is not None: record["metrics"] = clustering_metrics(labels, pred)
    np.save(out / f"predictions_seed_{seed}.npy", pred.astype(np.int64)); np.save(out / f"embedding_seed_{seed}.npy", embedding.astype(np.float32))
    (out / f"record_seed_{seed}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record

__all__ = ["run_transductive", "PROTOCOL_ID", "SEEDS"]
