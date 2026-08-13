#!/usr/bin/env python
"""Register a fixed second sparse/high-dimensional V19 extension panel.

The candidate list is deliberately declared in source before any first-panel
performance is read.  Labels are recorded only as outer benchmark metadata;
they are never used to decide eligibility, preprocessing, or model settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "result" / "V19" / "v19_rg_extended_sparse_batch2_manifest_20260811.json"
DATA_ROOT = Path(os.environ.get("TOPOGATE_DATA_ROOT", ROOT / "datasets")).expanduser()


CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "dataset_id": "20newsgroups__local_highdim_text",
        "name": "20newsgroups",
        "source_path": DATA_ROOT / "20newsgroups.npz",
        "input_protocol": "shared_text",
        "family": "highdim_text",
        "selection_basis": "fixed second-panel high-dimensional text candidate",
    },
    {
        "dataset_id": "reuters__local_highdim_text",
        "name": "reuters",
        "source_path": DATA_ROOT / "reuters.npz",
        "input_protocol": "shared_text",
        "family": "highdim_text",
        "selection_basis": "fixed second-panel high-dimensional text candidate",
    },
    {
        "dataset_id": "enron__local_highdim_text",
        "name": "enron",
        "source_path": DATA_ROOT / "enron.npz",
        "input_protocol": "shared_text",
        "family": "highdim_text",
        "selection_basis": "fixed second-panel high-dimensional text candidate",
    },
    {
        "dataset_id": "wos__local_highdim_text",
        "name": "wos",
        "source_path": DATA_ROOT / "wos.npz",
        "input_protocol": "shared_text",
        "family": "highdim_text",
        "selection_basis": "fixed second-panel high-dimensional text candidate",
    },
    {
        "dataset_id": "isolet__local_highdim_control",
        "name": "ISOLET",
        "source_path": DATA_ROOT / "ISOLET.npz",
        "input_protocol": "clubench_bridge",
        "family": "highdim_control",
        "selection_basis": "fixed second-panel high-dimensional benchmark candidate",
    },
    {
        "dataset_id": "secom__local_sparse_control",
        "name": "secom",
        "source_path": DATA_ROOT / "secom.npz",
        "input_protocol": "clubench_bridge",
        "family": "sparse_highdim_control",
        "selection_basis": "fixed second-panel sparse high-dimensional benchmark candidate",
    },
    {
        "dataset_id": "webdata_wxa__openml_sparse_web",
        "name": "webdata_wXa",
        "source_path": DATA_ROOT / "external" / "v9_related_20260806" / "processed" / "webdata_wXa.npz",
        "input_protocol": "clubench_bridge",
        "family": "web_sparse",
        "selection_basis": "fixed second-panel reused provenance-audited sparse web candidate",
        "source_kind": "OpenML",
        "source_identity": "openml:did=350;version=1;file_id=52253",
        "source_manifest": "datasets/external/v9_related_20260806/processed/v9_external_manifest.json",
    },
)


def _matrix_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        x_key = next((key for key in ("X", "x", "features", "data") if key in archive), None)
        y_key = next((key for key in ("y", "labels", "label") if key in archive), None)
        if x_key is None:
            raise ValueError(f"missing feature array in {path}")
        matrix = np.asarray(archive[x_key])
        labels = None if y_key is None else np.asarray(archive[y_key]).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"invalid feature matrix shape in {path}: {matrix.shape}")
    if labels is not None and labels.shape[0] != matrix.shape[0]:
        raise ValueError(f"label length mismatch in {path}")
    return matrix, labels


def build_manifest() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        path = Path(candidate["source_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        matrix, labels = _matrix_and_labels(path)
        finite = bool(np.isfinite(matrix).all())
        if not finite:
            raise ValueError(f"non-finite matrix in {path}")
        profile = {
            "n_samples": int(matrix.shape[0]),
            "n_features": int(matrix.shape[1]),
            "zero_fraction": float(np.mean(matrix == 0)),
            "labels_unique": None if labels is None else int(np.unique(labels).size),
            "storage": "dense_npz",
        }
        record = {
            **candidate,
            "source_path": str(path),
            "status": "eligible",
            "source_provenance_status": "reused_local_registry" if "source_manifest" not in candidate else "reused_v9_external_manifest",
            "source_hash": "unavailable",
            "profile": profile,
            "selection_uses_labels_or_outcomes": False,
            "comparison_scope": "external_highdim_bridge_only",
        }
        records.append(record)
    return {
        "manifest_id": "v19_rg_extended_sparse_batch2_manifest_20260811",
        "protocol_id": "v19_rg_extended_sparse_batch2_v1",
        "description": "Pre-registered second sparse/high-dimensional extension panel for matched RG/scMAE",
        "selection_policy": {
            "selection_uses_labels_or_outcomes": False,
            "selection_basis": "fixed candidate list declared in prepare_extended_sparse_batch2.py before first-panel results",
            "activation_rule": "run the complete panel only if the primary panel does not meet the five-dataset success criterion",
        },
        "variants": ["scmae_only", "rg_full"],
        "seeds": [42, 123, 7],
        "datasets": records,
        "comparison_scope": "external_highdim_bridge_only; no archived SOTA row is implied",
    }


def main() -> int:
    payload = build_manifest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"existing batch-2 manifest differs: {OUTPUT}")
    else:
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(OUTPUT), "datasets": len(payload["datasets"]), "protocol_id": payload["protocol_id"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
