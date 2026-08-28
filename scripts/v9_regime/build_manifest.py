#!/usr/bin/env python3
from __future__ import annotations

"""Build the frozen V9 dataset manifest.

The local scan is deterministic and does not calculate hashes.  An optional
external JSON file can add already downloaded, provenance-audited NPZ files;
network retrieval is intentionally separate so unresolved sources cannot be
silently replaced by similar data.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.v9_regime.protocol import (
    AHDPC_ROOT,
    DATA_ROOT,
    MAX_ELEMENTS,
    MAX_CLUSTERS,
    MIN_CLUSTERS,
    MIN_SAMPLES,
    PROTOCOL_ID,
    dataset_id_for,
    infer_family,
    json_default,
    read_npz_shape,
    write_json,
)


def _metadata(path: Path) -> tuple[tuple[int, int], int | None, str | None]:
    shape = read_npz_shape(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            y_key = next((key for key in ("y", "Y", "labels", "label") if key in data.files), None)
            if y_key is None:
                return shape, None, "missing_labels"
            y = np.asarray(data[y_key]).reshape(-1)
            if len(y) != shape[0]:
                return shape, None, "label_length_mismatch"
            return shape, int(np.unique(y).size), None
    except Exception as exc:
        return shape, None, f"metadata_error:{type(exc).__name__}:{exc}"


def _status(n: int, d: int, k: int | None, error: str | None, max_elements: int) -> tuple[str, str | None]:
    if error:
        return "unresolved", error
    if n < MIN_SAMPLES:
        return "ineligible", f"n<{MIN_SAMPLES}"
    if k is None:
        return "unresolved", "missing_or_invalid_K"
    if k < MIN_CLUSTERS or k > MAX_CLUSTERS:
        return "ineligible", f"K_outside_{MIN_CLUSTERS}_{MAX_CLUSTERS}"
    if n * d > max_elements:
        return "ineligible", f"dense_element_cap:{n}x{d}>{max_elements}"
    return "eligible", None


def _local_record(path: Path, source_kind: str, max_elements: int) -> dict[str, Any]:
    shape, k, error = _metadata(path)
    n, d = shape
    status, reason = _status(n, d, k, error, max_elements)
    source_rel = str(path.resolve().relative_to(DATA_ROOT.parent)) if path.resolve().is_relative_to(DATA_ROOT.parent) else str(path.resolve())
    name = path.stem
    return {
        "dataset_id": dataset_id_for(path, source_kind),
        "name": name,
        "source_kind": source_kind,
        "source_path": str(path.resolve()),
        "source_relpath": source_rel,
        "source_identity": f"{source_kind}:{source_rel}",
        "source_version": "local_snapshot_2026-08-06",
        "license": "see_source_manifest",
        "citation": "see_source_manifest",
        "family": infer_family(name),
        "n": int(n),
        "d": int(d),
        "n_clusters": k,
        "status": status,
        "status_reason": reason,
        "row_sampling_required": bool(n > 20_000),
        "preprocessing": "nan_to_num_then_column_standard_scaler",
        "labels_used_during_fit": False,
        "source_hash_policy": "not_recomputed_by_experiment_runner",
    }


def _external_records(path: Path, max_elements: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("datasets", [])
    records: list[dict[str, Any]] = []
    for item in payload:
        item = dict(item)
        source_path = Path(item.get("source_path", ""))
        if not source_path.is_absolute():
            source_path = (path.parent / source_path).resolve()
        if not source_path.exists():
            item.update({"status": "unresolved", "status_reason": "source_path_missing"})
            item.setdefault("dataset_id", f"external__{item.get('name', 'unknown')}")
            records.append(item)
            continue
        record = _local_record(source_path, "external", max_elements)
        record.update({key: value for key, value in item.items() if key not in {"source_path", "n", "d", "n_clusters", "status", "status_reason"}})
        record["source_path"] = str(source_path)
        record["dataset_id"] = item.get("dataset_id", record["dataset_id"])
        record["source_identity"] = item.get("source_identity", record["source_identity"])
        record["source_version"] = item.get("source_version", "external_manifest")
        record["status"] = item.get("status", record["status"])
        record["status_reason"] = item.get("status_reason", record["status_reason"])
        records.append(record)
    return records


def build_manifest(max_elements: int, external_json: Path | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    # Prefer the main local dataset mapping when a prepared AHDPC variant has
    # the same canonical name.  Distinct AHDPC names remain in the pool.
    for path in sorted(DATA_ROOT.glob("*.npz")):
        record = _local_record(path, "local", max_elements)
        seen_names.add(record["name"].lower())
        records.append(record)
    if AHDPC_ROOT.exists():
        for path in sorted(AHDPC_ROOT.glob("*.npz")):
            if path.stem.lower() in seen_names:
                continue
            records.append(_local_record(path, "ahdpc_prepared", max_elements))

    if external_json is not None:
        records.extend(_external_records(external_json, max_elements))

    # Dataset order is part of the protocol and makes worker sharding stable.
    records.sort(key=lambda row: str(row.get("dataset_id", "")))
    return {
        "protocol_id": PROTOCOL_ID,
        "created_at": "2026-08-06",
        "dataset_selection": {
            "local_sources": ["CLUBench_root", "AHDPC_prepared"],
            "external_sources": ["OpenML", "UCI", "scRNAseq_GEO", "public_text_image"],
            "external_manifest": str(external_json) if external_json else None,
            "selection_uses_labels": False,
            "hashes_recomputed_during_experiments": False,
        },
        "protocol": {
            "max_dense_elements": int(max_elements),
            "min_samples": MIN_SAMPLES,
            "cluster_range": [MIN_CLUSTERS, MAX_CLUSTERS],
            "preprocessing": "nan_to_num_then_column_standard_scaler",
            "v9_config": "learnable_gate_v9_adaptive",
        },
        "datasets": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-json", type=Path, default=None)
    parser.add_argument("--max-elements", type=int, default=MAX_ELEMENTS)
    args = parser.parse_args()
    payload = build_manifest(args.max_elements, args.external_json)
    write_json(args.output, payload)
    counts: dict[str, int] = {}
    for record in payload["datasets"]:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print(json.dumps({"output": str(args.output), "datasets": len(payload["datasets"]), "status": counts}, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

