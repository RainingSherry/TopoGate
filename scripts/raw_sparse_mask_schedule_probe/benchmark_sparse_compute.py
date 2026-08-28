"""Label-free sparse-vs-dense first-projection feasibility benchmark."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import protocol, raw_adapter, run_main


def projection_equivalence(x: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    from scipy import sparse

    value = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    dense = value @ w
    sparse_value = sparse.csr_matrix(value)
    sparse_result = np.asarray(sparse_value @ w, dtype=np.float32)
    error = float(np.max(np.abs(dense - sparse_result))) if dense.size else 0.0
    return dense, sparse_result, error


def _time_call(fn: Any, repeats: int = 50, warmup: int = 10) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(times, dtype=np.float64)
    return float(np.median(values)), float(np.percentile(values, 90))


def benchmark_dataset(dataset: str, *, seed: int = 42, batches: tuple[int, ...] = (64, 256, 512)) -> dict[str, Any]:
    from scipy import sparse

    data = raw_adapter.load_dataset(dataset)
    rng = np.random.default_rng(seed)
    value = data.x0
    weight = rng.normal(0.0, 1.0 / np.sqrt(max(value.shape[1], 1)), size=(value.shape[1], 64)).astype(np.float32)
    sparse_full = sparse.csr_matrix(value)
    dense_metrics: list[dict[str, Any]] = []
    for requested in batches:
        bs = min(int(requested), value.shape[0])
        if bs <= 0:
            continue
        batch = np.asarray(value[:bs], dtype=np.float32)
        csr = sparse.csr_matrix(batch)
        dense_ms, dense_p90 = _time_call(lambda: batch @ weight)
        sparse_ms, sparse_p90 = _time_call(lambda: csr @ weight)
        _, sparse_result, error = projection_equivalence(batch, weight)
        dense_bytes = int(batch.nbytes + weight.nbytes)
        sparse_bytes = int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes + weight.nbytes)
        dense_metrics.append({
            "batch_size": bs,
            "dense_ms_p50": dense_ms,
            "dense_ms_p90": dense_p90,
            "sparse_ms_p50": sparse_ms,
            "sparse_ms_p90": sparse_p90,
            "speedup_dense_over_sparse": float(dense_ms / max(sparse_ms, 1e-12)),
            "peak_mem_dense": dense_bytes,
            "peak_mem_sparse": sparse_bytes,
            "memory_reduction_ratio": float(dense_bytes / max(sparse_bytes, 1)),
            "max_abs_error": error,
            "equivalence_ok": bool(error <= 1e-5),
            "nnz": int(csr.nnz),
        })
    best = max(dense_metrics, key=lambda row: (row["speedup_dense_over_sparse"], row["memory_reduction_ratio"])) if dense_metrics else {}
    leverage = bool(best and best["equivalence_ok"] and (best["speedup_dense_over_sparse"] >= 1.5 or best["memory_reduction_ratio"] >= 2.0))
    return {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "dataset": dataset,
        "source_sha256": data.manifest["source_sha256"],
        "sparsity": float(np.mean(value == 0.0)),
        "nnz_per_row_p50": float(np.percentile(data.active.sum(axis=1), 50)),
        "nnz_per_row_p90": float(np.percentile(data.active.sum(axis=1), 90)),
        "rows": dense_metrics,
        "best_row": best,
        "compute_leverage_dataset": leverage,
        "labels_used": False,
        "status": "completed_valid",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=protocol.DATASETS)
    parser.add_argument("--output-root", type=Path, default=protocol.COMPUTE_ROOT)
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else list(protocol.DATASETS)
    rows = [benchmark_dataset(dataset) for dataset in datasets]
    result = {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "rows": rows, "status": "completed_valid", "datasets_with_leverage": sum(r["compute_leverage_dataset"] for r in rows), "project_compute_leverage": sum(r["compute_leverage_dataset"] and r["sparsity"] >= 0.90 for r in rows) >= 2}
    run_main.write_json_atomic(args.output_root / "summary.json", result)
    print(json.dumps({"status": result["status"], "datasets": len(rows), "project_compute_leverage": result["project_compute_leverage"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
