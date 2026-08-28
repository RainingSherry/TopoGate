"""Raw zero-preserving adapter with an explicit post-fit label boundary."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import protocol, provenance


@dataclass
class RawDataset:
    dataset: str
    x0: np.ndarray
    active: np.ndarray
    manifest: dict[str, Any]
    scale: np.ndarray


def _load_npz_field(path: Path, field: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if field not in archive.files:
            raise KeyError(f"required field {field!r} missing from {path}")
        return np.asarray(archive[field])


def load_raw_numeric(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    source = provenance.resolve_dataset(dataset)
    x = _load_npz_field(Path(source["source_path"]), source["matrix_field"])
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.number):
        raise ValueError(f"raw matrix for {dataset} must be a numeric 2D array")
    if list(x.shape) != list(source["shape_from_e3"]):
        raise ValueError(f"raw shape drift for {dataset}: {x.shape} != {source['shape_from_e3']}")
    if not np.isfinite(x).all():
        raise ValueError(f"raw matrix for {dataset} contains non-finite values")
    # Explicitly do not access the y field here.  The fit path receives x only.
    return np.asarray(x), source


def zero_preserving_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(x)
    if value.ndim != 2:
        raise ValueError("x must be 2D")
    # Accumulate in float64 for stable scale factors while preserving exact
    # zeros.  No centering, clipping or feature selection is allowed.
    scale = np.sqrt(np.mean(np.square(value.astype(np.float64, copy=False)), axis=0, dtype=np.float64))
    scale = np.asarray(scale, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    x0 = (value.astype(np.float32, copy=False) / scale.astype(np.float32)).astype(np.float32, copy=False)
    if not np.array_equal(value == 0, x0 == 0):
        raise AssertionError("zero-preserving scale changed the zero pattern")
    if not np.isfinite(x0).all():
        raise ValueError("scaled matrix contains non-finite values")
    return x0, scale


def load_dataset(dataset: str) -> RawDataset:
    x, source = load_raw_numeric(dataset)
    x0, scale = zero_preserving_scale(x)
    active = x0 != 0.0
    manifest = {
        **source,
        "scaled_dtype": str(x0.dtype),
        "scaled_shape": list(x0.shape),
        "zero_pattern_preserved": bool(np.array_equal(x == 0, x0 == 0)),
        "scale_hash": provenance.sha256_array(scale),
        "scale_min": float(np.min(scale)) if scale.size else 0.0,
        "scale_max": float(np.max(scale)) if scale.size else 0.0,
        "active_count_total": int(active.sum()),
        "zero_budget_rows": int(np.sum(active.sum(axis=1) == 0)),
        "labels_loaded": False,
    }
    return RawDataset(dataset=dataset, x0=x0, active=active, manifest=manifest, scale=scale)


def load_labels_after_fit(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load labels only after a fit has completed."""
    source = provenance.resolve_dataset(dataset)
    path = Path(source["source_path"])
    labels = _load_npz_field(path, source["label_field"])
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("benchmark labels must be one-dimensional")
    return labels, {
        "labels_loaded_after_fit": True,
        "labels_path": str(path.resolve()),
        "labels_sha256": provenance.sha256_file(path),
        "labels_unique": int(np.unique(labels).size),
    }


def write_adapter_manifest(path: Path) -> dict[str, Any]:
    rows = []
    for dataset in protocol.DATASETS:
        data = load_dataset(dataset)
        rows.append(data.manifest)
        del data
    manifest = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "zero_preserving_scale": "sqrt(mean(X^2)); floor 1e-6; no centering",
        "rows": rows,
        "labels_loaded": False,
        "code_sha256": provenance.code_sha256(),
        "status": "completed_valid",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
