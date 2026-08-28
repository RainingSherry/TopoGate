"""Run the non-training D0 inheritance and lock audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import protocol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_freeze(output_dir: Path) -> dict[str, Any]:
    protocol.validate_contract()
    output_dir.mkdir(parents=True, exist_ok=True)
    m1_decision_path = protocol.M1_ROOT / "decision.json"
    m1_audit_path = protocol.M1_ROOT / "audit.json"
    if not m1_decision_path.exists() or not m1_audit_path.exists():
        raise FileNotFoundError("support_target_validation_probe M1 compact artifacts are required")
    m1_decision = json.loads(m1_decision_path.read_text())
    m1_audit = json.loads(m1_audit_path.read_text())
    h0_rows: list[dict[str, Any]] = []
    for dataset in protocol.DEVELOPMENT_PANEL:
        h0_path = protocol.H0_ROOT / dataset / "H0.npy"
        if not h0_path.exists():
            raise FileNotFoundError(h0_path)
        shape = tuple(int(v) for v in np.load(h0_path, mmap_mode="r").shape)
        h0_rows.append(
            {
                "dataset": dataset,
                "shape": list(shape),
                "H0_sha256": sha256_file(h0_path),
                "source": f"representation_consumer_probe/S0_freeze/datasets/{dataset}/H0.npy",
            }
        )
    checks = {
        "m1_status_is_estimability_terminal": m1_decision.get("status") == "magnitude_match_not_estimable",
        "m1_gpu_runs_zero": int(m1_decision.get("gpu_runs_started", -1)) == 0,
        "m1_audit_ok": bool(m1_audit.get("audit_ok")),
        "m1_model_training_false": bool(m1_audit.get("model_training_started")) is False,
        "three_h0_sources_present": len(h0_rows) == 3,
        "d2_locked": protocol.D2_LOCKED,
        "raw_x_bridge_locked": protocol.RAW_X_BRIDGE_LOCKED,
        "holdout_locked": protocol.HOLDOUT_LOCKED,
        "adaptive_locked": protocol.ADAPTIVE_LOCKED,
        "gan_locked": protocol.GAN_LOCKED,
        "labels_not_loaded": True,
    }
    audit = {
        "audit_ok": bool(all(checks.values())),
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.D0_PROTOCOL_ID,
        "stage": "D0_common_dose_freeze",
        "checks": checks,
        "inherited_m1": {
            "decision_sha256": sha256_file(m1_decision_path),
            "audit_sha256": sha256_file(m1_audit_path),
            "status": m1_decision.get("status"),
            "gpu_runs_started": m1_decision.get("gpu_runs_started"),
        },
        "h0_sources": h0_rows,
        "d2_gpu_runs_started": 0,
        "labels_not_loaded": True,
        "publication_scope": "D0/D1 protocol, compact feasibility summaries and source hashes only",
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(protocol.resolved_config(), indent=2, sort_keys=True) + "\n")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    status = "passed" if audit["audit_ok"] else "failed"
    (output_dir / "D0_FREEZE.md").write_text(
        "# D0 Freeze\n\n"
        f"Status: `{status}`.\n\n"
        "D0 inherits the closed C2/M1 evidence read-only and authorizes only a CPU\n"
        "constructive common-dose feasibility map. D2 GPU runs remain locked.\n"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=protocol.RESULT_ROOT / "D0_freeze")
    args = parser.parse_args()
    audit = run_freeze(args.output)
    print(json.dumps({"audit_ok": audit["audit_ok"], "output": str(args.output)}, sort_keys=True))
    return 0 if audit["audit_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
