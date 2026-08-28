#!/usr/bin/env python3
"""Run AHDPC, HDPC and V9 on the complete CLUBench dataset list.

The benchmark protocol is deliberately explicit:

* input is loaded through CLUBench's official ``load_data`` helper (column-wise
  z-score);
* ``K = int(np.unique(y).size)`` is used only for the benchmark contract and
  post-fit metrics;
* AHDPC/HDPC use one preregistered, label-free ``epsilon`` for every dataset;
* V9 receives the already-standardized matrix with ``scale_input=False`` and
  is called with ``y=None`` so labels cannot enter training;
* all methods write ``predictions.npy`` and ``labels_true.npy`` separately.

The runner is resumable.  A completed method/dataset pair is skipped when its
``benchmark_summary.json`` is present; failures are retained as explicit error
records and can be retried with ``--retry-errors``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result" / "clubench_ahdpc_hdpc_v9_2026-08-02"
V9_CONFIG_DIR = REPO_ROOT / "methods" / "TopoGate" / "learnable_gate" / "configs"
GPU_POOL = (2, 3, 6)
FORBIDDEN_GPU = (0, 7)
DEFAULT_SEED = 42
DEFAULT_EPSILON = 1.0

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

for _path in (REPO_ROOT / "baseline" / "CLUBench", REPO_ROOT / "baseline" / "AHDPC"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from CLUBench import clustering_evaluation, load_data  # noqa: E402
from CLUBench.configs import DATASETS  # noqa: E402
from ahdpc import AHDPC  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _normalise_name(name: str) -> str:
    return name if name.endswith(".npz") else f"{name}.npz"


def _metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Return CLUBench metrics plus the AHDPC paper metrics."""

    from ahdpc.metrics import evaluate_clustering

    club = clustering_evaluation(y, pred)
    paper = evaluate_clustering(y, pred)
    return {
        "ACC": float(club["acc"]),
        "NMI": float(club["nmi"]),
        "ARI": float(club["ari"]),
        "AMI": float(paper["AMI"]),
        "RI": float(paper["RI"]),
        "FMI": float(paper["FMI"]),
    }


def _completed(summary_path: Path, retry_errors: bool) -> bool:
    if not summary_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    status = payload.get("status")
    return status == "completed" or (status == "error" and not retry_errors)


def _load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, Path]:
    filename = _normalise_name(name)
    if filename not in DATASETS:
        raise KeyError(f"{filename!r} is not in CLUBench DATASETS")
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    X, y = load_data(filename)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{filename}: X/y sample mismatch: {X.shape} vs {y.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{filename}: non-finite values after CLUBench z-score")
    return X, y, path


def _base_record(
    *,
    dataset: str,
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    source_path: Path,
    source_sha256: str,
    seed: int,
    k_source: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "status": "running",
        "seed": int(seed),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_clusters": int(np.unique(y).size),
        "k_source": k_source,
        "labels_used_during_fit": False,
        "input_preprocessing": "CLUBench.load_data z-score",
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
    }


def _run_density(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    *,
    epsilon: float,
    block_size: int,
    seed: int,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    adaptive = method == "AHDPC"
    K = int(np.unique(y).size)
    record = _base_record(
        dataset=source_path.stem,
        method=method,
        X=X,
        y=y,
        source_path=source_path,
        source_sha256=source_sha256,
        seed=seed,
        k_source="labels_unique",
    )
    record.update(
        {
            "epsilon": float(epsilon),
            "adaptive": bool(adaptive),
            "adaptive_distance_rule": "table_reproduction",
            "normalization": "paper_semantic",
            "block_size": int(block_size),
        }
    )
    started = time.perf_counter()
    model = AHDPC(
        n_clusters=K,
        epsilon=float(epsilon),
        adaptive=adaptive,
        adaptive_distance_rule="table_reproduction",
        normalization="paper_semantic",
        block_size=int(block_size),
        store_distance_matrix=False,
        store_alpha_matrix=False,
    )
    pred = np.asarray(model.fit_predict(X)).astype(np.int64)
    elapsed = time.perf_counter() - started
    metrics = _metric_row(y, pred)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "predictions.npy", pred)
    np.save(out_dir / "labels_true.npy", y)
    record.update(
        {
            "status": "completed",
            "elapsed_seconds": float(elapsed),
            "metrics": metrics,
            "cutoff_distance": float(model.cutoff_distance_),
            "neighbor_ratio": float(model.neighbor_ratio_),
            "cutoff_converged": bool(model.cutoff_converged_),
            "cluster_centers_indices": np.asarray(model.cluster_centers_indices_),
        }
    )
    _write_json(out_dir / "benchmark_summary.json", record)
    return record


def _run_v9(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    *,
    gpu: int,
    epochs: int,
    batch_size: int,
    seed: int,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    from methods.TopoGate.learnable_gate.run_npz import run_topogate

    K = int(np.unique(y).size)
    record = _base_record(
        dataset=source_path.stem,
        method="V9",
        X=X,
        y=y,
        source_path=source_path,
        source_sha256=source_sha256,
        seed=seed,
        k_source="labels_unique",
    )
    record.update(
        {
            "variant": "learnable_gate_v9_adaptive",
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "gpu": int(gpu),
            "scale_input": False,
            "knn_pca_mode": "adaptive",
            "knn_pca_dim": 2000,
            "n_top_features": 0,
            "fit_labels_argument": None,
        }
    )
    started = time.perf_counter()
    pred, elapsed, _ = run_topogate(
        X,
        n_clusters=K,
        y=None,
        gpu=int(gpu),
        variant="learnable_gate_v9_adaptive",
        seed=int(seed),
        return_metrics=True,
        save_dir=str(out_dir),
        config_dir=str(V9_CONFIG_DIR),
        epochs=int(epochs),
        batch_size=int(batch_size),
        scale_input=False,
        n_top_features=0,
        knn_pca_mode="adaptive",
        knn_pca_dim=2000,
    )
    pred = np.asarray(pred).astype(np.int64)
    elapsed = float(elapsed or (time.perf_counter() - started))
    metrics = _metric_row(y, pred)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "predictions.npy", pred)
    np.save(out_dir / "labels_true.npy", y)
    record.update(
        {
            "status": "completed",
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "fit_labels_argument": None,
        }
    )
    _write_json(out_dir / "benchmark_summary.json", record)
    return record


def _run_one(
    dataset_name: str,
    method: str,
    output_dir: Path,
    *,
    epsilon: float,
    block_size: int,
    gpu: int,
    epochs: int,
    batch_size: int,
    seed: int,
    retry_errors: bool,
) -> dict[str, Any]:
    method_dir = output_dir / dataset_name.removesuffix(".npz") / method
    summary_path = method_dir / "benchmark_summary.json"
    if _completed(summary_path, retry_errors):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    X, y, source_path = _load_dataset(dataset_name)
    source_hash = _sha256(source_path)
    try:
        if method in {"AHDPC", "HDPC"}:
            return _run_density(
                method,
                X,
                y,
                method_dir,
                epsilon=epsilon,
                block_size=block_size,
                seed=seed,
                source_path=source_path,
                source_sha256=source_hash,
            )
        if method == "V9":
            return _run_v9(
                X,
                y,
                method_dir,
                gpu=gpu,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                source_path=source_path,
                source_sha256=source_hash,
            )
        raise ValueError(f"unknown method {method!r}")
    except Exception as exc:
        record = _base_record(
            dataset=source_path.stem,
            method=method,
            X=X,
            y=y,
            source_path=source_path,
            source_sha256=source_hash,
            seed=seed,
            k_source="labels_unique",
        )
        record.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        method_dir.mkdir(parents=True, exist_ok=True)
        _write_json(summary_path, record)
        return record
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _collect(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*/*/benchmark_summary.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row = {
            "dataset": record.get("dataset"),
            "method": record.get("method"),
            "status": record.get("status"),
            "seed": record.get("seed"),
            "n_samples": record.get("n_samples"),
            "n_features": record.get("n_features"),
            "n_clusters": record.get("n_clusters"),
            "ACC": (record.get("metrics") or {}).get("ACC"),
            "NMI": (record.get("metrics") or {}).get("NMI"),
            "ARI": (record.get("metrics") or {}).get("ARI"),
            "AMI": (record.get("metrics") or {}).get("AMI"),
            "RI": (record.get("metrics") or {}).get("RI"),
            "FMI": (record.get("metrics") or {}).get("FMI"),
            "elapsed_seconds": record.get("elapsed_seconds"),
            "error_type": record.get("error_type"),
            "error": record.get("error"),
        }
        rows.append(row)
    return rows


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    fields = [
        "dataset",
        "method",
        "status",
        "seed",
        "n_samples",
        "n_features",
        "n_clusters",
        "ACC",
        "NMI",
        "ARI",
        "AMI",
        "RI",
        "FMI",
        "elapsed_seconds",
        "error_type",
        "error",
    ]
    with (output_dir / "comparison_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), {})[str(row["method"])] = row
    wide_fields = ["dataset", "n_samples", "n_features", "n_clusters"]
    for method in ("AHDPC", "HDPC", "V9"):
        for metric in ("ACC", "NMI", "ARI", "AMI", "RI", "FMI"):
            wide_fields.append(f"{method}_{metric}")
    wide_rows = []
    for dataset in sorted(by_dataset):
        entries = by_dataset[dataset]
        base = next(iter(entries.values()))
        row = {
            "dataset": dataset,
            "n_samples": base.get("n_samples"),
            "n_features": base.get("n_features"),
            "n_clusters": base.get("n_clusters"),
        }
        for method in ("AHDPC", "HDPC", "V9"):
            entry = entries.get(method, {})
            for metric in ("ACC", "NMI", "ARI", "AMI", "RI", "FMI"):
                row[f"{method}_{metric}"] = entry.get(metric)
        wide_rows.append(row)
    with (output_dir / "comparison_wide.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(wide_rows)

    manifest = {
        "created_at": "2026-08-02",
        "dataset_source": "CLUBench DATASETS (131 entries)",
        "selected_datasets": args.datasets,
        "methods": args.methods,
        "seed": int(args.seed),
        "k_protocol": "K=int(np.unique(y).size), labels used only for benchmark K and post-fit metrics",
        "input_preprocessing": "CLUBench.load_data z-score",
        "ahdpc_hdpc": {
            "epsilon": float(args.epsilon),
            "adaptive_distance_rule": "table_reproduction",
            "normalization": "paper_semantic",
            "block_size": int(args.block_size),
        },
        "v9": {
            "variant": "learnable_gate_v9_adaptive",
            "scale_input": False,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "gpu": int(args.gpu),
            "gpu_pool_declared": list(GPU_POOL),
            "forbidden_gpus": list(FORBIDDEN_GPU),
        },
        "completed_records": len(rows),
        "completed": sum(1 for row in rows if row.get("status") == "completed"),
        "errors": sum(1 for row in rows if row.get("status") == "error"),
    }
    _write_json(output_dir / "MANIFEST.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None, help="Names with or without .npz; default: all 131 CLUBench datasets")
    parser.add_argument("--methods", nargs="+", choices=["AHDPC", "HDPC", "V9"], default=["AHDPC", "HDPC", "V9"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.gpu in FORBIDDEN_GPU:
        raise ValueError(f"GPU {args.gpu} is forbidden; physical GPU 0 and 7 cannot be used.")
    if args.gpu not in GPU_POOL:
        raise ValueError(f"GPU {args.gpu} is outside the declared CLUBench GPU pool {GPU_POOL}.")
    selected = [_normalise_name(item) for item in (args.datasets or DATASETS)]
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise KeyError(f"unknown CLUBench datasets: {unknown}")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Datasets={len(selected)} Methods={args.methods} Seed={args.seed}")
    print(f"Output={args.output_dir}")
    print(
        f"Protocol: CLUBench z-score; K=unique(y); epsilon={args.epsilon}; "
        f"V9 epochs={args.epochs}; GPU={args.gpu}"
    )
    if args.dry_run:
        return 0

    total = len(selected) * len(args.methods)
    done = 0
    for dataset in selected:
        for method in args.methods:
            done += 1
            print(f"[{done}/{total}] {dataset} / {method}", flush=True)
            started = time.perf_counter()
            result = _run_one(
                dataset,
                method,
                args.output_dir,
                epsilon=args.epsilon,
                block_size=args.block_size,
                gpu=args.gpu,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                retry_errors=args.retry_errors,
            )
            if result.get("status") == "completed":
                metrics = result.get("metrics", {})
                print(
                    "  completed "
                    f"ACC={metrics.get('ACC', float('nan')):.4f} "
                    f"NMI={metrics.get('NMI', float('nan')):.4f} "
                    f"ARI={metrics.get('ARI', float('nan')):.4f} "
                    f"wall={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"  {result.get('status')} {result.get('error_type')}: "
                    f"{result.get('error', '')}",
                    flush=True,
                )
            _write_outputs(args.output_dir, _collect(args.output_dir), args)

    rows = _collect(args.output_dir)
    _write_outputs(args.output_dir, rows, args)
    print(
        f"Finished records={len(rows)} completed={sum(r.get('status') == 'completed' for r in rows)} "
        f"errors={sum(r.get('status') == 'error' for r in rows)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
