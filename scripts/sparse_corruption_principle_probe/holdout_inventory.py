"""Freeze an outcome-independent scRNA holdout membership manifest.

The inventory reads only source metadata, matrix shape and a bounded matrix
sample for sparsity/intrinsic-dimension proxies.  Label columns are recorded
as a provenance boundary but never inspected for selection.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import protocol


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_matrix(x: Any, rows: int = 256) -> np.ndarray:
    sample = x[:rows]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    return np.asarray(sample, dtype=np.float32)


def _intrinsic_proxy(sample: np.ndarray) -> float:
    if sample.ndim != 2 or min(sample.shape) < 2:
        return float(min(sample.shape)) if sample.ndim == 2 else 0.0
    # Bounded SVD is deliberately a structural proxy, not a tuned model input.
    centered = sample - np.mean(sample, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = np.square(singular)
    total = float(np.sum(energy))
    if total <= 1e-12:
        return 0.0
    cumulative = np.cumsum(energy) / total
    return float(np.searchsorted(cumulative, 0.9) + 1)


def _inspect_npz(path: Path) -> dict[str, Any]:
    with np.load(path, mmap_mode="r", allow_pickle=False) as archive:
        if "x" not in archive:
            raise ValueError(f"candidate NPZ has no x array: {path}")
        x = archive["x"]
        n, d = map(int, x.shape)
        sample = np.asarray(x[: min(256, n)], dtype=np.float32)
        label_columns = ["y"] if "y" in archive else []
    return {
        "n": n,
        "d": d,
        "x_dtype": str(x.dtype),
        "x_storage": "npz_dense",
        "sample_rows": int(sample.shape[0]),
        "estimated_sparsity": float(1.0 - np.count_nonzero(sample) / max(sample.size, 1)),
        "estimated_intrinsic_dimension_proxy": _intrinsic_proxy(sample),
        "label_columns_present": label_columns,
        "labels_used_for_selection": False,
    }


def _inspect_h5ad(path: Path) -> dict[str, Any]:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("holdout inventory requires anndata for h5ad sources") from exc
    data = ad.read_h5ad(path, backed="r")
    n, d = map(int, data.shape)
    sample = _sample_matrix(data.X, rows=min(256, n))
    label_columns = [str(column) for column in data.obs.columns if any(token in str(column).lower() for token in ("label", "celltype", "cell_type", "cluster", "cell_type"))]
    return {
        "n": n,
        "d": d,
        "x_dtype": str(getattr(data.X, "dtype", "unknown")),
        "x_storage": type(data.X).__name__,
        "sample_rows": int(sample.shape[0]),
        "estimated_sparsity": float(1.0 - np.count_nonzero(sample) / max(sample.size, 1)),
        "estimated_intrinsic_dimension_proxy": _intrinsic_proxy(sample),
        "label_columns_present": label_columns,
        "labels_used_for_selection": False,
    }


def inspect_source(path: Path, *, relative_path: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "relative_path": relative_path,
            "path": str(path),
            "status": "missing_source",
            "labels_used_for_selection": False,
        }
    metadata = _inspect_h5ad(path) if path.suffix == ".h5ad" else _inspect_npz(path)
    source_family = relative_path.split("/", 1)[0]
    record = {
        "relative_path": relative_path,
        "path": str(path.resolve()),
        "source_family": source_family,
        "source_size_bytes": int(path.stat().st_size),
        "source_mtime_ns": int(path.stat().st_mtime_ns),
        "source_sha256": _sha256_file(path),
        "status": "candidate_valid",
        **metadata,
    }
    return record


def _feature_vector(record: dict[str, Any], families: list[str]) -> np.ndarray:
    continuous = np.array(
        [
            np.log1p(float(record["n"])),
            np.log1p(float(record["d"])),
            float(record["estimated_sparsity"]),
            np.log1p(float(record["estimated_intrinsic_dimension_proxy"])),
        ],
        dtype=np.float64,
    )
    one_hot = np.zeros(len(families), dtype=np.float64)
    one_hot[families.index(str(record["source_family"]))] = 1.0
    return np.concatenate([continuous, one_hot])


def _select_maximin(records: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    valid = [record for record in records if record.get("status") == "candidate_valid"]
    if not valid:
        return []
    valid = sorted(valid, key=lambda record: str(record["relative_path"]))
    target = min(int(target), len(valid))
    families = sorted({str(record["source_family"]) for record in valid})
    vectors = np.stack([_feature_vector(record, families) for record in valid])
    scale = np.std(vectors, axis=0)
    scale[scale < 1e-12] = 1.0
    vectors = vectors / scale
    selected = [0]
    while len(selected) < target:
        distances = np.linalg.norm(vectors[:, None, :] - vectors[np.asarray(selected)][None, :, :], axis=2)
        min_distance = np.min(distances, axis=1)
        min_distance[np.asarray(selected)] = -np.inf
        best_value = float(np.max(min_distance))
        candidates = np.flatnonzero(np.isclose(min_distance, best_value, rtol=0.0, atol=1e-12))
        selected.append(int(candidates[0]))
    return [valid[index] for index in selected]


def run(output_dir: Path) -> dict[str, Any]:
    protocol.validate_contract()
    candidate_records: list[dict[str, Any]] = []
    for relative in protocol.HOLDOUT_CANDIDATE_RELATIVE_PATHS:
        # Relative paths are anchored at the scCluBench data root.  The source
        # universe roots are recorded separately so a missing path is visible.
        base = Path("/data/luolie/biopipeline/scCluBench/data")
        candidate_records.append(inspect_source(base / relative, relative_path=relative))
    selected = _select_maximin(candidate_records, protocol.HOLDOUT_TARGET_DATASETS)
    selected_paths = {record["relative_path"] for record in selected}
    development_names = set(protocol.DEVELOPMENT_PANEL)
    overlap = [
        record["relative_path"]
        for record in selected
        if any(name.lower().replace(" ", "_") in record["relative_path"].lower() for name in development_names)
    ]
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "C0_holdout_inventory",
        "selection_status": "frozen_before_C2_results",
        "selection_rule": "label-free maximin over log(n), log(d), sampled sparsity, sampled SVD-90 intrinsic proxy and source-family one-hot",
        "outcome_features_used": [],
        "label_values_read": False,
        "candidate_universe_roots": list(protocol.HOLDOUT_UNIVERSE_ROOTS),
        "candidate_records": candidate_records,
        "selected_records": selected,
        "selected_relative_paths": sorted(selected_paths),
        "target_count": protocol.HOLDOUT_TARGET_DATASETS,
        "minimum_count": protocol.HOLDOUT_MIN_DATASETS,
        "selected_count": len(selected),
        "shortfall": max(0, protocol.HOLDOUT_MIN_DATASETS - len(selected)),
        "development_overlap": overlap,
        "development_overlap_check": "pass" if not overlap else "fail",
        "run_before_C2_matrix": True,
        "holdout_runs_authorized": False,
        "publication_scope": "metadata/hash manifest only; source matrices and labels remain external/local",
    }
    latest = output_dir / "holdout_manifest.json"
    timestamped = output_dir / f"holdout_manifest_{stamp}.json"
    _write_json(timestamped, inventory)
    latest.write_bytes(timestamped.read_bytes())
    audit = {
        "audit_ok": bool(
            inventory["selection_status"] == "frozen_before_C2_results"
            and inventory["outcome_features_used"] == []
            and inventory["label_values_read"] is False
            and inventory["development_overlap_check"] == "pass"
            and inventory["selected_count"] >= protocol.HOLDOUT_MIN_DATASETS
        ),
        "selected_count": inventory["selected_count"],
        "minimum_count": protocol.HOLDOUT_MIN_DATASETS,
        "shortfall": inventory["shortfall"],
        "holdout_runs_authorized": False,
        "labels_used_for_selection": False,
        "source_hashes_recorded": all("source_sha256" in row for row in selected),
    }
    _write_json(output_dir / "holdout_audit.json", audit)
    _write_json(output_dir / "holdout_manifest_resolved_config.json", protocol.resolved_config())
    return {"manifest": inventory, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=protocol.RESULT_ROOT / "C0_holdout_inventory")
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()

