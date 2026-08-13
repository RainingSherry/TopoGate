#!/usr/bin/env python
"""Run fixed external-method controls on every preregistered RG winner.

The extension summary decides only which already-registered datasets are
winners (positive mean ARI delta).  This runner never ranks or truncates that
set.  AHDPC, DPC-GFNN, and GCC are invoked with explicit benchmark K and fixed
parameters; labels are read only outside each estimator for K and post-fit
metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

DEFAULT_SUMMARY = ROOT / "result" / "V19" / "v19_rg_extended_sparse_ari_v1" / "summary" / "extension_summary.json"
DEFAULT_MANIFEST = ROOT / "result" / "V19" / "v19_rg_extended_sparse_manifest_20260811.json"
DEFAULT_OUTPUT = ROOT / "result" / "V19" / "v19_rg_extended_winner_baselines_v1"
READY_METHODS = ("AHDPC", "DPC_GFNN", "GCC")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        y_key = next((key for key in ("y", "labels", "label") if key in archive), None)
        if y_key is None:
            raise ValueError(f"winner baseline input must contain X and y: {path}")
        sparse_keys = {"data", "indices", "indptr", "shape"}
        if sparse_keys.issubset(archive.files):
            shape = tuple(int(value) for value in np.asarray(archive["shape"]).reshape(-1))
            X: np.ndarray | sp.csr_matrix = sp.csr_matrix(
                (
                    np.asarray(archive["data"]),
                    np.asarray(archive["indices"], dtype=np.int64),
                    np.asarray(archive["indptr"], dtype=np.int64),
                ),
                shape=shape,
            )
        else:
            x_key = next((key for key in ("X", "x", "features", "data") if key in archive), None)
            if x_key is None:
                raise ValueError(f"winner baseline input must contain X and y: {path}")
            X = np.asarray(archive[x_key])
        y = np.asarray(archive[y_key]).reshape(-1)
    if X.ndim != 2 or y.shape[0] != X.shape[0]:
        raise ValueError(f"invalid X/y shapes in {path}: {X.shape}, {y.shape}")
    return X, y


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    encoded = LabelEncoder().fit_transform(np.asarray(y).astype(str))
    pred = np.asarray(prediction, dtype=np.int64).reshape(-1)
    true_values = np.unique(encoded)
    pred_values = np.unique(pred)
    counts = np.zeros((true_values.size, pred_values.size), dtype=np.int64)
    for i, true_value in enumerate(true_values):
        for j, pred_value in enumerate(pred_values):
            counts[i, j] = int(np.count_nonzero((encoded == true_value) & (pred == pred_value)))
    rows, cols = linear_sum_assignment(-counts)
    mapping = {pred_values[col]: true_values[row] for row, col in zip(rows, cols, strict=True)}
    mapped = np.asarray([mapping.get(value, -1) for value in pred], dtype=np.int64)
    return {
        "ari": float(adjusted_rand_score(encoded, pred)),
        "nmi": float(normalized_mutual_info_score(encoded, pred)),
        "acc": float(np.mean(mapped == encoded)),
        "n_true_clusters": int(np.unique(encoded).size),
        "n_pred_clusters": int(np.unique(pred).size),
        "labels_used_during_fit": False,
    }


def _prepared_input(X: np.ndarray, record: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from methods.TopoGate.V19_rg_adapter.input_adapter import prepare_input

    prepared = prepare_input(
        X,
        dataset_name=str(record["name"]),
        input_protocol=str(record["input_protocol"]),
        n_top_features=1000,
        target_sum=10000.0,
    )
    return prepared.X, prepared.profile


def _run_one_method(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    output: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if method == "AHDPC":
            from methods.AHDPC.run import run_ahdpc

            prediction, _, details = run_ahdpc(X, K, y=None)
            method_config = {
                "epsilon": 0.1,
                "normalization_mode": "prose_consistent",
                "external_preprocessing": "V19 fixed label-free input adapter",
            }
        elif method == "DPC_GFNN":
            from methods.DPC_GFNN import DPCGFNN

            model = DPCGFNN()
            prediction = model.fit_predict(X, n_clusters=K)
            method_config = {
                "n_neighbors": "default sqrt(n) clipped to [2,50]",
                "noise_lambda": 3.0,
                "minmax_normalize": True,
                "n_clusters_source": "explicit_n_clusters",
            }
        elif method == "GCC":
            from methods.GCC import GravityCenterClustering

            model = GravityCenterClustering(
                n_clusters=K,
                random_state=42,
                max_samples=20000,
                standardize=False,
            )
            prediction = model.fit_predict(X)
            method_config = {
                "n_clusters_source": "explicit_n_clusters",
                "max_samples": 20000,
                "standardize": False,
                "seed": 42,
            }
        else:  # pragma: no cover - protected by CLI validation
            raise ValueError(method)
        prediction = np.asarray(prediction, dtype=np.int64)
        metrics = _metrics(y, prediction)
        np.save(output / "predictions.npy", prediction)
        np.save(output / "labels_true.npy", LabelEncoder().fit_transform(np.asarray(y).astype(str)))
        summary = {
            "status": "completed",
            "method": method,
            "dataset_id": str(record["dataset_id"]),
            "dataset": str(record["name"]),
            "source_path": str(Path(record["source_path"]).resolve()),
            "input_protocol": str(record["input_protocol"]),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_clusters": int(K),
            "K_source": "benchmark_oracle_from_y",
            "benchmark_oracle_from_y": True,
            "labels_used_during_fit": False,
            "method_config": method_config,
            "metrics": metrics,
            "wall_seconds": float(time.perf_counter() - started),
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "output_files": {"predictions": "predictions.npy", "labels_true": "labels_true.npy"},
        }
        _write(output / "summary.json", summary)
        _write(output / "status.json", {"status": "completed", "method": method, "dataset_id": record["dataset_id"]})
        return summary
    except Exception as exc:
        failure = {
            "status": "incomplete_compute",
            "method": method,
            "dataset_id": str(record["dataset_id"]),
            "dataset": str(record["name"]),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "labels_used_during_fit": False,
            "wall_seconds": float(time.perf_counter() - started),
        }
        _write(output / "summary.json", failure)
        _write(output / "status.json", failure)
        return failure


def _run_dataset(record: dict[str, Any], output_root: Path, methods: tuple[str, ...]) -> list[dict[str, Any]]:
    X_raw, y = _load_npz(Path(record["source_path"]))
    X, preprocess_profile = _prepared_input(X_raw, record)
    K = int(np.unique(y).size)
    dataset_root = output_root / str(record["dataset_id"])
    _write(
        dataset_root / "input_profile.json",
        {
            "dataset_id": record["dataset_id"],
            "dataset": record["name"],
            "input_protocol": record["input_protocol"],
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "preprocess_profile": preprocess_profile,
            "labels_used_for_preprocessing": False,
            "K_source": "benchmark_oracle_from_y",
        },
    )
    rows = []
    for method in methods:
        rows.append(_run_one_method(method, X, y, K, dataset_root / method, record))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", default=list(READY_METHODS))
    parser.add_argument("--min-wins", type=int, default=5)
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        default=None,
        help="optionally restrict execution to these already-selected winner dataset IDs",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--wait-for-summary", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=30)
    args = parser.parse_args()
    methods = tuple(str(method) for method in args.methods)
    if not methods or set(methods) - set(READY_METHODS):
        raise ValueError(f"methods must be a subset of {READY_METHODS}")
    if args.wait_for_summary:
        while not args.extension_summary.is_file():
            time.sleep(max(5, int(args.wait_seconds)))
    if not args.extension_summary.is_file():
        raise FileNotFoundError(args.extension_summary)
    summary = _read(args.extension_summary)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    if summary.get("audit_ok") is not True:
        status = {"status": "blocked_incomplete_extension", "extension_summary": str(args.extension_summary.resolve())}
        _write(output_root / "baseline_status.json", status)
        return 2
    manifest = _read(args.manifest)
    records = {str(row["dataset_id"]): row for row in manifest.get("datasets", []) if row.get("status") == "eligible"}
    winners = [row for row in summary.get("datasets", []) if row.get("promotion_rg_win_by_mean_ari") is True]
    if len(winners) < int(args.min_wins):
        status = {
            "status": "insufficient_primary_wins",
            "n_winners": len(winners),
            "minimum_required": int(args.min_wins),
            "activation_rule": "run second pre-registered panel before any SOTA comparison",
            "winners": [row.get("dataset_id") for row in winners],
        }
        _write(output_root / "baseline_status.json", status)
        return 2
    selected_records = []
    requested_ids = None if args.dataset_ids is None else {str(value) for value in args.dataset_ids}
    for row in winners:
        dataset_id = str(row["dataset_id"])
        if requested_ids is not None and dataset_id not in requested_ids:
            continue
        if dataset_id not in records:
            raise KeyError(f"winner missing from manifest: {dataset_id}")
        selected_records.append(records[dataset_id])
    if requested_ids is not None:
        missing_requested = sorted(requested_ids - {str(row["dataset_id"]) for row in selected_records})
        if missing_requested:
            raise ValueError(f"requested dataset IDs are not selected winners: {missing_requested}")
    _write(
        output_root / "baseline_protocol.json",
        {
            "protocol_id": "v19_rg_extended_winner_baselines_v1",
            "extension_summary": str(args.extension_summary.resolve()),
            "manifest": str(args.manifest.resolve()),
            "selection_rule": "all extension rows with mean RG ARI strictly above matched scMAE-only ARI",
            "selection_scope": "requested winner subset" if requested_ids is not None else "all selected winners",
            "winner_count": len(selected_records),
            "winner_dataset_ids": [row["dataset_id"] for row in selected_records],
            "methods": list(methods),
            "preprocessing": "V19 fixed label-free prepare_input; n_top_features=1000; target_sum=10000",
            "K_source": "benchmark_oracle_from_y",
            "labels_used_during_fit": False,
            "parameters_fixed_before_outcomes": True,
        },
    )
    rows: list[dict[str, Any]] = []
    worker_count = max(1, min(int(args.workers), len(selected_records)))
    if worker_count == 1:
        for record in selected_records:
            rows.extend(_run_dataset(record, output_root, methods))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run_dataset, record, output_root, methods) for record in selected_records]
            for future in as_completed(futures):
                rows.extend(future.result())
    completed = sum(row.get("status") == "completed" for row in rows)
    expected = len(selected_records) * len(methods)
    result = {
        "status": "completed" if completed == expected else "incomplete_compute",
        "protocol_id": "v19_rg_extended_winner_baselines_v1",
        "expected_runs": expected,
        "completed_runs": completed,
        "winner_count": len(selected_records),
        "methods": list(methods),
        "labels_used_during_fit": False,
        "rows": rows,
    }
    _write(output_root / "baseline_summary.json", result)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
