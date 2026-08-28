"""No-training M1 magnitude-estimability preflight.

The preflight audits all 30 epochs for every proposed dataset×seed control
before any model is constructed.  A tolerance failure is an estimability
No-Go, never a performance result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import protocol
from .m0_freeze import _json, _write_json, sha256_file
from .replay import build_magnitude_matched_epoch, replay_p2_epoch


def preflight_job(dataset: str, seed: int) -> dict[str, Any]:
    h0_path = protocol.H0_ROOT / dataset / "H0.npy"
    h0 = np.asarray(np.load(h0_path), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    p2_total = 0.0
    mm_total = 0.0
    row_relative: list[float] = []
    epoch_total_relative: list[float] = []
    exact_budget = True
    support_zero = True
    multiset_mismatch = 0
    match_failures = 0
    started = time.perf_counter()
    for _epoch in range(protocol.EPOCHS):
        p2_raw, p2_audit = replay_p2_epoch(h0, rng)
        _mm_raw, mm_audit = build_magnitude_matched_epoch(h0, p2_raw, p2_audit)
        p2_total += float(mm_audit["p2_total_absolute_change"])
        mm_total += float(mm_audit["total_absolute_change"])
        row_relative.extend(float(row["relative_mismatch"]) for row in mm_audit["row_records"])
        epoch_total_relative.append(float(mm_audit["dataset_total_relative_mismatch"]))
        exact_budget = exact_budget and bool(mm_audit["exact_budget"])
        support_zero = support_zero and bool(mm_audit["support_change_rate"] == 0.0)
        multiset_mismatch += int(mm_audit["row_value_multiset_mismatch_count"])
        match_failures += int(mm_audit["match_failure_count"])
        # This is the exact C2 post-corruption batch-order RNG consumption.
        rng.permutation(h0.shape[0])
    total_relative = abs(mm_total - p2_total) / max(abs(p2_total), protocol.ROW_REL_EPS)
    median_relative = float(np.median(np.asarray(row_relative, dtype=np.float64))) if row_relative else 0.0
    estimable = bool(
        total_relative <= protocol.TOTAL_L1_REL_TOLERANCE
        and median_relative <= protocol.MEDIAN_ROW_REL_TOLERANCE
        and match_failures == 0
    )
    return {
        "dataset": dataset,
        "seed": int(seed),
        "epochs": protocol.EPOCHS,
        "H0_sha256": sha256_file(h0_path),
        "budget_manifest_sha256": sha256_file(protocol.H0_ROOT / dataset / "budget_manifest.json"),
        "dataset_total_relative_mismatch": total_relative,
        "median_row_relative_mismatch": median_relative,
        "max_epoch_total_relative_mismatch": float(max(epoch_total_relative) if epoch_total_relative else 0.0),
        "max_row_relative_mismatch": float(max(row_relative) if row_relative else 0.0),
        "exact_changed_coordinate_budget": exact_budget,
        "support_change_rate_exact_zero": support_zero,
        "row_value_multiset_mismatch_count": multiset_mismatch,
        "match_failure_count": match_failures,
        "magnitude_match_estimable": estimable,
        "labels_used": False,
        "model_training_started": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    structural = all(
        row["exact_changed_coordinate_budget"]
        and row["support_change_rate_exact_zero"]
        and row["row_value_multiset_mismatch_count"] == 0
        and row["match_failure_count"] == 0
        for row in records
    )
    non_estimable = [
        {"dataset": row["dataset"], "seed": row["seed"], "dataset_total_relative_mismatch": row["dataset_total_relative_mismatch"], "median_row_relative_mismatch": row["median_row_relative_mismatch"]}
        for row in records
        if not row["magnitude_match_estimable"]
    ]
    return {
        "status": "magnitude_match_not_estimable" if non_estimable else "magnitude_match_estimable",
        "expected_rows": len(protocol.DEVELOPMENT_PANEL) * len(protocol.PRIMARY_SEEDS),
        "completed_rows": len(records),
        "structural_contract_passed": structural,
        "estimable_rows": sum(bool(row["magnitude_match_estimable"]) for row in records),
        "non_estimable_rows": len(non_estimable),
        "non_estimable": non_estimable,
        "formal_m1_gpu_runs_authorized": bool(structural and not non_estimable),
        "gpu_runs_started": 0,
        "labels_used": False,
        "later_stages_locked": True,
    }


def run(output_root: Path = protocol.RESULT_ROOT / "M1_preflight") -> dict[str, Any]:
    protocol.validate_contract()
    m0_audit_path = protocol.RESULT_ROOT / "M0_freeze" / "audit.json"
    if not m0_audit_path.exists() or _json(m0_audit_path).get("audit_ok") is not True:
        raise RuntimeError("M0 exact replay audit must pass before M1 preflight")
    records = [
        preflight_job(dataset, seed)
        for dataset in protocol.DEVELOPMENT_PANEL
        for seed in protocol.PRIMARY_SEEDS
    ]
    decision = evaluate(records)
    output_root.mkdir(parents=True, exist_ok=True)
    audit = {
        "audit_ok": bool(decision["completed_rows"] == decision["expected_rows"] and decision["structural_contract_passed"]),
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.M1_PROTOCOL_ID,
        "stage": "M1_magnitude_estimability_preflight",
        "decision": decision,
        "records": records,
        "m0_audit_sha256": sha256_file(m0_audit_path),
        "model_training_started": False,
        "raw_arrays_persisted": False,
        "labels_used": False,
    }
    _write_json(output_root / "preflight_records.json", {"records": records})
    _write_json(output_root / "decision.json", decision)
    _write_json(output_root / "audit.json", audit)
    _write_json(output_root / "resolved_config.json", {
        **protocol.resolved_config(),
        "stage": "M1_magnitude_estimability_preflight",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_runs_started": 0,
    })
    lines = [
        "# M1 Magnitude Estimability Preflight",
        "",
        f"Status: `{decision['status']}`; rows: `{decision['completed_rows']}/{decision['expected_rows']}`; GPU runs started: `0`.",
        "",
        "This is a no-training structural preflight. A tolerance failure is `magnitude_match_not_estimable`, not a performance negative.",
        "",
        "| Dataset | Seed | Total L1 mismatch | Median row mismatch | Estimable |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['dataset_total_relative_mismatch']:.6f} | {row['median_row_relative_mismatch']:.6f} | {row['magnitude_match_estimable']} |")
    lines.extend([
        "",
        f"- Frozen total-L1 tolerance: `{protocol.TOTAL_L1_REL_TOLERANCE}`.",
        f"- Frozen median-row tolerance: `{protocol.MEDIAN_ROW_REL_TOLERANCE}`.",
        f"- Formal M1 GPU authorization: `{decision['formal_m1_gpu_runs_authorized']}`.",
        "- No M1 model was constructed or trained; M2/M3/M4/adaptive/GAN remain locked.",
        "",
        f"> {protocol.resolved_config()['support_interpretation_firewall']}",
    ])
    (output_root / "M1_PREFLIGHT_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"decision": decision, "audit": audit}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=protocol.RESULT_ROOT / "M1_preflight")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), indent=2, sort_keys=True, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

