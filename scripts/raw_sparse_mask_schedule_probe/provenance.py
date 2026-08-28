"""Input and provenance audit for the raw sparse masking probe."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from . import protocol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(np.asarray(array))
    return sha256_bytes(value.tobytes(order="C"))


def code_sha256() -> str:
    """Fingerprint the frozen project code used to produce a run."""
    digest = hashlib.sha256()
    root = protocol.PROJECT_ROOT / "scripts/raw_sparse_mask_schedule_probe"
    for path in sorted(root.glob("*.py")):
        digest.update(str(path.relative_to(protocol.PROJECT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(args, cwd=protocol.PROJECT_ROOT, text=True, capture_output=True, check=False, timeout=10)
        except Exception:
            return False, ""
        return result.returncode == 0, result.stdout.strip()

    ok_head, head = run("git", "rev-parse", "HEAD")
    ok_remote, remote = run("git", "remote", "-v")
    if not ok_head or not ok_remote:
        return {"git_provenance": "unverifiable_local_git_metadata", "claimed_head": None, "remote": None}
    return {"git_provenance": "verified", "claimed_head": head, "remote": remote}


def _load_e3_summary() -> list[dict[str, Any]]:
    if not protocol.E3_SUMMARY.exists():
        raise FileNotFoundError(f"required E3 summary missing: {protocol.E3_SUMMARY}")
    value = json.loads(protocol.E3_SUMMARY.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("E3 summary must be a list")
    return value


def resolve_dataset(dataset: str) -> dict[str, Any]:
    rows = [row for row in _load_e3_summary() if row.get("dataset") == dataset]
    if len(rows) != 1:
        raise ValueError(f"E3 must resolve exactly one row for {dataset}, got {len(rows)}")
    row = rows[0]
    raw_source = str(row.get("raw_source", ""))
    source_path = protocol.PROJECT_ROOT / raw_source
    if not source_path.exists():
        # E3 paths are repository-relative; the datasets symlink is the canonical
        # input root in this checkout.
        source_path = protocol.DATA_ROOT / (dataset + ".npz")
    if not source_path.exists():
        raise FileNotFoundError(f"raw source missing for {dataset}: {raw_source}")
    if row.get("raw_sha256") and sha256_file(source_path) != row["raw_sha256"]:
        raise ValueError(f"raw source hash drift for {dataset}")
    shape = list(row.get("raw_shape", []))
    return {
        "dataset": dataset,
        "domain_role": protocol.ROLE_BY_DATASET[dataset],
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "source_format": "npz",
        "matrix_field": "x",
        "label_field": "y",
        "shape_from_e3": shape,
        "dtype_from_e3": row.get("raw_dtype"),
        "nnz_from_e3": row.get("nnz"),
        "sparsity_from_e3": row.get("sparsity_zero_fraction"),
        "zero_rows_from_e3": row.get("zero_rows"),
        "labels_loaded": False,
        "e3_summary_sha256": sha256_file(protocol.E3_SUMMARY),
    }


def build_manifest() -> dict[str, Any]:
    rows = [resolve_dataset(dataset) for dataset in protocol.DATASETS]
    return {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "git": git_provenance(),
        "e3_summary": str(protocol.E3_SUMMARY.resolve()),
        "rows": rows,
        "labels_loaded": False,
        "status": "completed_valid",
        "code_sha256": code_sha256(),
    }


def write_manifest(path: Path) -> dict[str, Any]:
    manifest = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_manifest(), indent=2, sort_keys=True))
