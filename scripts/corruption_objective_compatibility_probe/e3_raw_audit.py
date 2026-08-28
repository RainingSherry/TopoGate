"""CPU-only raw-input descriptive audit; never feeds raw support into fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import protocol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_row(dataset: str) -> dict[str, Any]:
    manifest = json.loads((protocol.PROJECT_ROOT / "result/representation_consumer_probe/S0_freeze/dataset_manifest.json").read_text())
    return next(row for row in manifest if row["dataset"] == dataset)


def audit_dataset(dataset: str) -> dict[str, Any]:
    raw_path = protocol.RAW_ROOT / f"{dataset}.npz"
    manifest = _manifest_row(dataset)
    with np.load(raw_path, allow_pickle=False) as archive:
        if "x" not in archive.files:
            raise ValueError(f"raw archive has no x key: {dataset}")
        raw = np.asarray(archive["x"])
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError(f"raw matrix invalid: {dataset}")
    n, d = raw.shape
    nnz = int(np.count_nonzero(raw))
    h0 = np.load(protocol.INPUT_ROOT / dataset / "H0.npy", mmap_mode="r")
    return {
        "dataset": dataset,
        "role": protocol.ROLE_BY_DATASET[dataset],
        "raw_source": f"datasets/{dataset}.npz",
        "raw_sha256": _sha256(raw_path),
        "raw_shape": [int(n), int(d)],
        "raw_dtype": str(raw.dtype),
        "storage": "dense_npz_array",
        "nnz": nnz,
        "sparsity_zero_fraction": float(1.0 - nnz / max(raw.size, 1)),
        "zero_rows": int(np.sum(np.count_nonzero(raw, axis=1) == 0)),
        "raw_memory_bytes_estimate": int(raw.nbytes),
        "H0_shape": [int(v) for v in h0.shape],
        "H0_sha256": manifest["H0_sha256"],
        "H0_d_eff": int(h0.shape[1]),
        "H0_input_protocol": manifest.get("input_protocol"),
        "H0_labels_used_during_fit": manifest.get("labels_vector_used_in_fit") is False,
        "raw_zero_nonzero_support_descriptive_only": True,
        "raw_support_used_in_fit": False,
        "labels_not_loaded": True,
        "normalization_note": "Raw archive is audited descriptively; E1/E2 fit consumes frozen standardized dense H0 only.",
    }


def run(output_dir: Path = protocol.RESULT_ROOT / "E3_raw_audit") -> dict[str, Any]:
    protocol.validate_contract()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [audit_dataset(dataset) for dataset in protocol.DEVELOPMENT_PANEL]
    audit = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "E3_raw_audit",
        "audit_ok": len(rows) == len(protocol.DEVELOPMENT_PANEL) and all(row["labels_not_loaded"] for row in rows),
        "rows": len(rows),
        "labels_not_loaded": True,
        "gpu_runs_started": 0,
        "does_not_change_fit_or_gates": True,
        "support_semantics": "raw_X_zero_nonzero_descriptive_only; H0_threshold_support remains separate",
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(protocol.resolved_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"audit": audit, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=protocol.RESULT_ROOT / "E3_raw_audit")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result["audit"], sort_keys=True))
    return 0 if result["audit"]["audit_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
