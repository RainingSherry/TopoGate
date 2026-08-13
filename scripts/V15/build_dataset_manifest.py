#!/usr/bin/env python3
"""Build the auditable Stage-0 V15 dataset manifest.

The script is descriptive only. Labels are used, when present, for K and
post-hoc graph diagnostics; they are never passed to a trainer. CLM metadata is
optional and is marked unranked unless it comes from an explicitly supplied,
locally verified JSON file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances

from methods.TopoGate.V15_counterfactual_gate.graph import build_candidate_graph, graph_embedding
from methods.TopoGate.V15_counterfactual_gate.sparse import prepare_input


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        x_key = "x" if "x" in keys else "X" if "X" in keys else None
        if x_key is not None:
            X: np.ndarray | sp.csr_matrix = np.asarray(payload[x_key], dtype=np.float32)
        elif {"data", "indices", "indptr", "shape"}.issubset(keys):
            X = sp.csr_matrix(
                (payload["data"], payload["indices"], payload["indptr"]),
                shape=tuple(int(v) for v in payload["shape"]),
                dtype=np.float32,
            )
        else:
            raise KeyError(f"{path} has no x/X or CSR fields; keys={sorted(keys)}")
        y_key = "y" if "y" in keys else "labels" if "labels" in keys else None
        y = None if y_key is None else np.asarray(payload[y_key]).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"{path} contains a non-matrix input: {X.shape}")
    return X, y


def _raw_profile(X: np.ndarray | sp.spmatrix) -> tuple[dict[str, Any], np.ndarray]:
    if sp.issparse(X):
        matrix = sp.csr_matrix(X, dtype=np.float32)
        n, d = matrix.shape
        nnz = np.diff(matrix.indptr).astype(np.int64)
        zero_fraction = float(1.0 - matrix.nnz / max(1, n * d))
        sample_n = min(n, 512)
        sample = matrix[:sample_n].toarray().astype(np.float32)
    else:
        matrix = np.asarray(X, dtype=np.float32)
        n, d = matrix.shape
        nnz = np.count_nonzero(matrix, axis=1).astype(np.int64)
        zero_fraction = float(np.mean(matrix == 0.0)) if matrix.size else 0.0
        sample_n = min(n, 512)
        sample = matrix[:sample_n]
    if sample_n > 1:
        distances = pairwise_distances(sample, metric="euclidean", n_jobs=1)
        values = distances[np.triu_indices(sample_n, k=1)]
        mean_distance = float(np.mean(values)) if values.size else 0.0
        distance_concentration = float(np.std(values) / max(abs(mean_distance), 1e-8)) if values.size else 0.0
    else:
        mean_distance = 0.0
        distance_concentration = 0.0
    profile = {
        "n": int(n),
        "d": int(d),
        "raw_zero_fraction": zero_fraction,
        "density": float(1.0 - zero_fraction),
        "nnz_min": int(np.min(nnz)) if nnz.size else 0,
        "nnz_p05": float(np.quantile(nnz, 0.05)) if nnz.size else 0.0,
        "nnz_median": float(np.median(nnz)) if nnz.size else 0.0,
        "nnz_p95": float(np.quantile(nnz, 0.95)) if nnz.size else 0.0,
        "nnz_max": int(np.max(nnz)) if nnz.size else 0,
        "distance_sample_n": int(sample_n),
        "mean_pairwise_distance": mean_distance,
        "distance_concentration": distance_concentration,
    }
    return profile, sample


def _clm_record(name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    candidates = [name, name.lower(), Path(name).stem, Path(name).stem.lower()]
    record: dict[str, Any] | None = None
    for key in candidates:
        candidate = metadata.get(key)
        if isinstance(candidate, dict):
            record = candidate
            break
    if record is None:
        return {
            "clm_source": None,
            "clm_metric": None,
            "clm_value": None,
            "clm_rank": None,
            "clm_stratum": "CLM-unranked",
        }
    raw_value = record.get("value", record.get("clm_value"))
    try:
        value = None if raw_value is None else float(raw_value)
    except (TypeError, ValueError):
        value = None
    if value is None or not math.isfinite(value):
        stratum = "CLM-unranked"
    elif value >= 0.70:
        stratum = "CLM-high"
    elif value >= 0.30:
        stratum = "CLM-middle"
    else:
        stratum = "CLM-low"
    return {
        "clm_source": record.get("source"),
        "clm_metric": record.get("metric", record.get("clm_metric")),
        "clm_value": value,
        "clm_rank": record.get("rank", record.get("clm_rank")),
        "clm_stratum": stratum,
    }


def _label_recall(indices: np.ndarray, valid: np.ndarray, y: np.ndarray) -> float:
    values = np.asarray(y).reshape(-1)
    recalls: list[float] = []
    for i in range(indices.shape[0]):
        same = np.flatnonzero(values == values[i])
        same = same[same != i]
        if same.size == 0:
            continue
        chosen = indices[i, valid[i]]
        denominator = min(chosen.size, same.size)
        if denominator > 0:
            recalls.append(float(np.intersect1d(chosen, same).size / denominator))
    return float(np.mean(recalls)) if recalls else 0.0


def _edge_purity(indices: np.ndarray, valid: np.ndarray, y: np.ndarray) -> float:
    labels = np.asarray(y).reshape(-1)
    rows, cols = np.where(valid)
    if rows.size == 0:
        return 0.0
    return float(np.mean(labels[rows] == labels[indices[rows, cols]]))


def _graph_audit(
    X: np.ndarray | sp.spmatrix,
    y: np.ndarray | None,
    *,
    raw_svd_dim: int,
    latent_dim: int,
    k_raw: int,
    k_latent: int,
    candidate_cap: int,
    seed: int,
) -> dict[str, Any]:
    prepared = prepare_input(X)
    # Stage 0 has no trained EMA representation. The latent path is therefore
    # explicitly labelled a raw-SVD proxy and must not be used as V15 evidence.
    latent_proxy = graph_embedding(prepared, latent_dim, seed)
    graph = build_candidate_graph(
        prepared,
        k_raw=k_raw,
        k_latent=k_latent,
        candidate_cap=candidate_cap,
        raw_svd_dim=raw_svd_dim,
        latent_embedding=latent_proxy,
        latent_graph_dim=latent_dim,
        seed=seed,
    )
    result: dict[str, Any] = {
        "latent_source": "raw_svd_proxy",
        "candidate_width": graph.n_candidates,
        "mean_valid_candidates": float(graph.valid.sum(axis=1).mean()) if graph.n_nodes else 0.0,
        "raw_graph_candidate_count": int(graph.raw_indices.shape[1]),
        "latent_graph_candidate_count": int(graph.latent_indices.shape[1]),
    }
    if y is not None:
        labels = np.asarray(y).reshape(-1)
        result.update(
            {
                "union_candidate_recall": graph.candidate_recall(labels),
                "union_edge_purity": graph.edge_purity(labels),
                "raw_candidate_recall": _label_recall(
                    graph.raw_indices,
                    np.ones_like(graph.raw_indices, dtype=bool),
                    labels,
                ),
                "latent_candidate_recall": _label_recall(
                    graph.latent_indices,
                    np.ones_like(graph.latent_indices, dtype=bool),
                    labels,
                ),
                "raw_edge_purity": _edge_purity(
                    graph.raw_indices,
                    np.ones_like(graph.raw_indices, dtype=bool),
                    labels,
                ),
                "latent_edge_purity": _edge_purity(
                    graph.latent_indices,
                    np.ones_like(graph.latent_indices, dtype=bool),
                    labels,
                ),
            }
        )
    return result


def build_record(
    path: Path,
    root: Path,
    clm_metadata: dict[str, Any],
    *,
    candidate_audit: bool,
    raw_svd_dim: int,
    latent_dim: int,
    k_raw: int,
    k_latent: int,
    candidate_cap: int,
    seed: int,
) -> dict[str, Any]:
    X, y = load_npz(path)
    profile, _ = _raw_profile(X)
    name = path.stem
    record: dict[str, Any] = {
        "dataset": name,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        **profile,
        "K": None if y is None else int(np.unique(y).size),
        "k_source": "benchmark_oracle_from_y" if y is not None else "unavailable",
        **_clm_record(name, clm_metadata),
    }
    if candidate_audit:
        record["candidate_graph"] = _graph_audit(
            X,
            y,
            raw_svd_dim=raw_svd_dim,
            latent_dim=latent_dim,
            k_raw=k_raw,
            k_latent=k_latent,
            candidate_cap=candidate_cap,
            seed=seed,
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "result" / "V15" / "dataset_manifest.json")
    parser.add_argument("--clm-json", type=Path, default=None)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset stem to include; repeatable")
    parser.add_argument("--candidate-audit", action="store_true")
    parser.add_argument("--raw-svd-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--k-raw", type=int, default=10)
    parser.add_argument("--k-latent", type=int, default=10)
    parser.add_argument("--candidate-cap", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    wanted = None if args.dataset is None else {str(value).lower() for value in args.dataset}
    paths = sorted(dataset_root.rglob("*.npz"))
    if wanted is not None:
        selected: list[Path] = []
        for name in sorted(wanted):
            matches = [path for path in paths if path.stem.lower() == name]
            if not matches:
                continue
            direct = dataset_root / f"{name}.npz"
            selected.append(direct if direct.exists() else matches[0])
        paths = selected
    if not paths:
        raise SystemExit(f"No NPZ files found under {dataset_root}")
    clm_metadata: dict[str, Any] = {}
    clm_verified = False
    if args.clm_json is not None:
        with args.clm_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("--clm-json must contain a mapping keyed by dataset name")
        clm_metadata = payload.get("datasets", payload)
        if not isinstance(clm_metadata, dict):
            raise ValueError("CLM metadata datasets field must be a mapping")
        clm_verified = True
    records = []
    for path in paths:
        try:
            records.append(
                build_record(
                    path,
                    dataset_root,
                    clm_metadata,
                    candidate_audit=args.candidate_audit,
                    raw_svd_dim=args.raw_svd_dim,
                    latent_dim=args.latent_dim,
                    k_raw=args.k_raw,
                    k_latent=args.k_latent,
                    candidate_cap=args.candidate_cap,
                    seed=args.seed,
                )
            )
        except Exception as exc:  # keep the manifest auditable without hiding failures
            records.append(
                {
                    "dataset": path.stem,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "V15-stage0-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "clm_verified": clm_verified,
        "clm_note": "CLM-unranked unless supplied through a locally verified metadata file",
        "records": records,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    keys = sorted({key for record in records for key in record.keys() if not isinstance(record.get(key), dict)})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in keys})
    errors = sum(record.get("status") == "error" for record in records)
    print(json.dumps({"output": str(output), "csv": str(csv_path), "records": len(records), "errors": errors}))


if __name__ == "__main__":
    main()
