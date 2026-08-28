#!/usr/bin/env python3
"""Build label-free capped ACCG inputs and record their provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V21_assignment_adversarial_gate.input_adapter import load_npz  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _feature_variance(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    if sp.issparse(X):
        matrix = X.tocsr().astype(np.float64)
        mean = np.asarray(matrix.mean(axis=0)).reshape(-1)
        mean_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).reshape(-1)
        return np.maximum(mean_sq - np.square(mean), 0.0)
    values = np.asarray(X, dtype=np.float64)
    return np.var(values, axis=0)


def _save_input(path: Path, X: np.ndarray | sp.spmatrix, labels: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sp.issparse(X):
        matrix = X.tocsr().astype(np.float32)
        payload: dict[str, Any] = {
            "data": matrix.data,
            "indices": matrix.indices.astype(np.int64),
            "indptr": matrix.indptr.astype(np.int64),
            "shape": np.asarray(matrix.shape, dtype=np.int64),
        }
    else:
        payload = {"X": np.asarray(X, dtype=np.float32)}
    if labels is not None:
        payload["y"] = np.asarray(labels)
    np.savez_compressed(path, **payload)


def build_inputs(spec_path: Path, output_root: Path, feature_cap: int) -> dict[str, Any]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    rows = spec.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("input spec must contain a non-empty datasets list")
    records = []
    dataset_spec_rows = []
    for row in rows:
        source = Path(str(row["source_path"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        loaded = load_npz(source)
        X = loaded.X
        variance = _feature_variance(X)
        order = np.argsort(-variance, kind="mergesort")
        selected = np.sort(order[: min(int(feature_cap), int(X.shape[1]))]).astype(np.int64)
        capped = X[:, selected]
        output = output_root / f"{row['dataset_id']}.npz"
        _save_input(output, capped, loaded.labels)
        record = {
            **row,
            "source_path": str(source),
            "source_sha256": _sha256(source),
            "output_path": str(output.resolve()),
            "output_sha256": _sha256(output),
            "n_samples_original": int(X.shape[0]),
            "n_features_original": int(X.shape[1]),
            "n_features_selected": int(selected.size),
            "selected_feature_indices_sha256": _sha256_array(selected),
            "selection_rule": "top_variance_non_label_with_stable_index_tie_break",
            "labels_used_for_feature_selection": False,
            "labels_saved_for_outer_evaluation_only": loaded.labels is not None,
            "input_protocol": row["input_protocol"],
            "preprocessing_contract": "accg_v2_label_free_feature_cap_2000",
        }
        records.append(record)
        dataset_spec_row = dict(row)
        dataset_spec_row["source_path"] = str(output.resolve())
        dataset_spec_rows.append(dataset_spec_row)
    return {
        "manifest_id": "accg_v2_label_free_capped_inputs_v1",
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": _sha256(spec_path),
        "feature_cap": int(feature_cap),
        "selection_uses_labels_or_outcomes": False,
        "development_subset": [str(value) for value in spec.get("development_subset", [])],
        "records": records,
        "dataset_spec_rows": dataset_spec_rows,
        "formal_training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--feature-cap", type=int, default=2000)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--dataset-spec-out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_inputs(args.spec, args.output_root, args.feature_cap)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.dataset_spec_out.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_spec_out.write_text(
        yaml.safe_dump(
            {
                "development_subset": payload["development_subset"],
                "datasets": payload["dataset_spec_rows"],
            },
            sort_keys=False,
        ),
    )
    print(json.dumps({"status": "capped_inputs_built_not_run", "records": len(payload["records"]), "training_started": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
