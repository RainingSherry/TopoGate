"""Run the CPU-only D1 constructive common-dose feasibility audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import matching, protocol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(dataset: str, seed: int, rows: list[dict[str, Any]], source_hash: str) -> dict[str, Any]:
    positive = [row for row in rows if row["nonzero_budget"]]
    common = [row for row in positive if row["common_exists"]]
    matched = [row for row in common if row.get("match_ok", False)]
    mismatches = [float(row["row_relative_mismatch"]) for row in matched]
    cross_doses = [float(row["cross_dose"]) for row in matched]
    preserve_doses = [float(row["preserve_dose"]) for row in matched]
    cross_support_failures = sum(not row["cross_audit"]["support_expectation_ok"] for row in matched)
    preserve_support_failures = sum(not row["preserve_audit"]["support_expectation_ok"] for row in matched)
    exact_budget_failures = sum(
        not (row["cross_audit"]["exact_changed_count"] and row["preserve_audit"]["exact_changed_count"])
        for row in matched
    )
    multiset_failures = sum(
        not (row["cross_audit"]["row_value_multiset_ok"] and row["preserve_audit"]["row_value_multiset_ok"])
        for row in matched
    )
    total_cross = float(np.sum(cross_doses, dtype=np.float64)) if cross_doses else 0.0
    total_preserve = float(np.sum(preserve_doses, dtype=np.float64)) if preserve_doses else 0.0
    total_mismatch = abs(total_cross - total_preserve) / max(total_cross, protocol.DOSE_EPS)
    common_fraction = len(common) / max(1, len(positive))
    median_mismatch = float(np.median(mismatches)) if mismatches else float("inf")
    gate = {
        "positive_budget_rows": len(positive),
        "common_interval_rows": len(common),
        "common_interval_fraction": float(common_fraction),
        "all_common_rows_constructed": len(matched) == len(common),
        "dataset_total_relative_mismatch_ok": bool(total_mismatch <= protocol.TOTAL_DOSE_REL_TOLERANCE),
        "median_row_relative_mismatch_ok": bool(median_mismatch <= protocol.MEDIAN_ROW_DOSE_REL_TOLERANCE),
        "exact_changed_count_ok": exact_budget_failures == 0,
        "cross_support_change_positive_ok": cross_support_failures == 0,
        "preserve_support_change_zero_ok": preserve_support_failures == 0,
        "row_value_multiset_ok": multiset_failures == 0,
        "range_constructive_failure_count": sum(bool(row.get("range_failure")) for row in positive),
    }
    gate["pass"] = bool(
        common_fraction >= protocol.MIN_COMMON_BUDGET_ROW_FRACTION
        and all(gate[key] for key in (
            "all_common_rows_constructed",
            "dataset_total_relative_mismatch_ok",
            "median_row_relative_mismatch_ok",
            "exact_changed_count_ok",
            "cross_support_change_positive_ok",
            "preserve_support_change_zero_ok",
            "row_value_multiset_ok",
        ))
    )
    return {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.D1_PROTOCOL_ID,
        "dataset": dataset,
        "seed": int(seed),
        "H0_sha256": source_hash,
        "labels_not_loaded": True,
        "gpu_runs_started": 0,
        "gate": gate,
        "cross_total_dose": total_cross,
        "preserve_total_dose": total_preserve,
        "dataset_total_relative_mismatch": float(total_mismatch),
        "median_row_relative_mismatch": median_mismatch,
        "row_relative_mismatch_max": float(max(mismatches)) if mismatches else float("inf"),
        "common_width_median": float(np.median([row["common_width"] for row in positive])) if positive else 0.0,
        "common_width_p10": float(np.quantile([row["common_width"] for row in positive], 0.10)) if positive else 0.0,
        "range_kind": "constructive_min_max_witnesses",
        "failure_counts": {
            "no_common_interval": len(positive) - len(common),
            "range_constructive_failure": sum(bool(row.get("range_failure")) for row in positive),
            "target_match_failure": len(common) - len(matched),
            "exact_budget": exact_budget_failures,
            "cross_support": cross_support_failures,
            "preserve_support": preserve_support_failures,
            "row_value_multiset": multiset_failures,
        },
    }


def run_dataset_seed(dataset: str, seed: int, *, retain_rows: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    h0_path = protocol.H0_ROOT / dataset / "H0.npy"
    clean = np.asarray(np.load(h0_path, mmap_mode="r"), dtype=np.float32)
    support = matching.support_mask(clean, reference=clean)
    _, pair_counts = matching.dataset_budget(clean)
    rows: list[dict[str, Any]] = []
    for row_idx in range(clean.shape[0]):
        pair_count = int(pair_counts[row_idx])
        ranges = matching.row_constructive_ranges(
            clean[row_idx], row=row_idx, seed=seed, pair_count=pair_count
        )
        if pair_count <= 0:
            result = {
                "target_dose": 0.0,
                "cross_dose": 0.0,
                "preserve_dose": 0.0,
                "row_relative_mismatch": 0.0,
                "match_ok": True,
                "cross_audit": matching.audit_swap(
                    clean[row_idx], clean[row_idx], reference_support=support[row_idx], requested_changed_count=0,
                    expect_support_change=False
                ),
                "preserve_audit": matching.audit_swap(
                    clean[row_idx], clean[row_idx], reference_support=support[row_idx], requested_changed_count=0,
                    expect_support_change=False
                ),
            }
        elif ranges["common_exists"]:
            result = matching.build_common_dose_row(
                clean[row_idx], row=row_idx, seed=seed, pair_count=pair_count, ranges=ranges
            )
        else:
            result = {"target_dose": None, "match_ok": False, "reason": "no_common_interval"}
        row_record = {
            "row": row_idx,
            **ranges,
            **result,
        }
        rows.append(row_record)
    summary = _summary(dataset, seed, rows, sha256_file(h0_path))
    return summary, rows if retain_rows else []


def run_matrix(output_dir: Path) -> dict[str, Any]:
    protocol.validate_contract()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    records: dict[str, list[dict[str, Any]]] = {}
    for dataset in protocol.DEVELOPMENT_PANEL:
        for seed in protocol.PRIMARY_SEEDS:
            summary, rows = run_dataset_seed(dataset, seed)
            summaries.append(summary)
            records[f"{dataset}__seed{seed}"] = rows
    pass_rows = [summary for summary in summaries if summary["gate"]["pass"]]
    computation_audit_ok = bool(
        len(summaries) == 9
        and all(summary.get("labels_not_loaded") is True for summary in summaries)
        and all(int(summary.get("gpu_runs_started", -1)) == 0 for summary in summaries)
    )
    d1_gate_pass = bool(len(pass_rows) == len(summaries))
    audit = {
        # A valid feasibility audit may fail the scientific estimability gate;
        # keep those states separate so a completed No-Go is not mislabeled as
        # incomplete computation.
        "audit_ok": computation_audit_ok,
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.D1_PROTOCOL_ID,
        "stage": "D1_constructive_common_dose_feasibility",
        "labels_not_loaded": True,
        "gpu_runs_started": 0,
        "dataset_seed_count": len(summaries),
        "passing_dataset_seed_count": len(pass_rows),
        "all_dataset_seed_gates_pass": d1_gate_pass,
        "d1_gate_pass": d1_gate_pass,
        "d2_authorized": False,
        "summaries": summaries,
        "range_kind": "constructive_min_max_witnesses",
        "claim_boundary": (
            "D1 only tests a frozen constructive matching contract; it does not establish a universal full feasible range "
            "or a causal support effect."
        ),
    }
    decision = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "common_dose_estimable" if d1_gate_pass else "common_dose_not_estimable",
        "audit_ok": audit["audit_ok"],
        "d1_gate_pass": d1_gate_pass,
        "d2_gpu_runs_started": 0,
        "d2_authorized": False,
        "adaptive_locked": True,
        "gan_locked": True,
        "raw_x_bridge_locked": True,
        "holdout_locked": True,
        "interpretation": (
            "The D1 constructive contract passes; D2 would compare Cross vs Preserve at a frozen matched dose."
            if d1_gate_pass
            else "The D1 contract is not constructively estimable under the frozen witness/tolerance rules; stop before D2."
        ),
        "status_definition": "This status is sufficient-for-this-contract, not a universal infeasibility theorem.",
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(protocol.resolved_config(), indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    (output_dir / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    return {"audit": audit, "decision": decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=protocol.RESULT_ROOT / "D1_feasibility")
    args = parser.parse_args()
    result = run_matrix(args.output)
    print(json.dumps(result["decision"], sort_keys=True))
    # Gate failure is a valid terminal scientific outcome; only a malformed
    # or incomplete computation returns non-zero.
    return 0 if result["audit"]["audit_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
