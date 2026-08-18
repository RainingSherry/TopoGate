"""Deterministic integrity audit for the compact C0/C1/C2-contract artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import protocol


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_csv(path: Path) -> tuple[bool, int, list[str]]:
    if not path.exists():
        return False, 0, []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric_bad = []
    for row_idx, row in enumerate(rows):
        for key, value in row.items():
            if value == "":
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not np.isfinite(number):
                numeric_bad.append(f"row{row_idx}:{key}")
    return not numeric_bad, len(rows), numeric_bad


def run(result_root: Path) -> dict[str, Any]:
    protocol.validate_contract()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    holdout = result_root / "C0_holdout_inventory"
    holdout_manifest = json.loads((holdout / "holdout_manifest.json").read_text(encoding="utf-8"))
    holdout_audit = json.loads((holdout / "holdout_audit.json").read_text(encoding="utf-8"))
    checks["holdout_audit_ok"] = holdout_audit.get("audit_ok") is True
    checks["holdout_count_minimum"] = int(holdout_manifest.get("selected_count", 0)) >= protocol.HOLDOUT_MIN_DATASETS
    checks["holdout_no_development_overlap"] = holdout_manifest.get("development_overlap") == []
    checks["holdout_outcome_features_empty"] = holdout_manifest.get("outcome_features_used") == []
    checks["holdout_runs_locked"] = holdout_manifest.get("holdout_runs_authorized") is False
    details["holdout_selected_count"] = holdout_manifest.get("selected_count")

    toy_path = result_root / "C2_toy_sanity" / "toy_sanity.json"
    toy = json.loads(toy_path.read_text(encoding="utf-8"))
    checks["toy_completed_valid"] = toy.get("status") == "completed_valid"
    checks["toy_no_fit_labels"] = toy.get("labels_used_during_corruption") is False
    checks["toy_all_rows_valid"] = all(row.get("status") == "completed_valid" for row in toy.get("rows", []))
    checks["toy_expected_row_count"] = len(toy.get("rows", [])) == 18
    details["toy_rows"] = len(toy.get("rows", []))

    c1 = json.loads((result_root / "C1_mechanism_audit" / "c1_structural_summary.json").read_text(encoding="utf-8"))
    c1_csv_ok, c1_rows, c1_bad = _finite_csv(result_root / "C1_mechanism_audit" / "c1_structural_rows.csv")
    checks["c1_completed_valid"] = c1.get("status") == "completed_valid"
    checks["c1_expected_row_count"] = c1_rows == len(protocol.DEVELOPMENT_PANEL) * 6 * len(protocol.PRIMARY_SEEDS)
    checks["c1_zero_fit"] = c1.get("fit_runs") == 0
    checks["c1_no_labels"] = c1.get("labels_loaded") is False
    checks["c1_csv_finite"] = c1_csv_ok and not c1_bad
    checks["c1_b1_metrics_are_post_fit"] = c1.get("b1_metrics_are_post_fit") is True
    details["c1_rows"] = c1_rows
    details["c1_nonfinite_fields"] = c1_bad

    # No formal matrix or adaptive implementation may appear under the new
    # result root at this gate.
    forbidden_tokens = ("adaptive", "gan", "generator", "checkpoint", "embedding", "prediction")
    forbidden_paths = []
    for path in result_root.rglob("*"):
        if not path.is_file():
            continue
        if any(token in path.name.lower() for token in forbidden_tokens):
            forbidden_paths.append(str(path.relative_to(result_root)))
    checks["no_forbidden_performance_artifacts"] = not forbidden_paths
    details["forbidden_performance_artifacts"] = forbidden_paths

    result = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "audit_type": "local_deterministic_contract_audit",
        "audit_ok": all(checks.values()),
        "checks": checks,
        "details": details,
        "external_review_status": "review_unavailable_no_score",
        "scientific_performance_claim": False,
        "c2_matrix_authorized": False,
    }
    _write_json(result_root / "C0_C1_CONTRACT_AUDIT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=protocol.RESULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.result_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
