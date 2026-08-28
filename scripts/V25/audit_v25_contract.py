#!/usr/bin/env python3
"""Audit the V25 registry, triage, and optional E1 engineering artifact."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "result" / "V25_systematic_mechanism_study"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv_fields(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    return {str(field) for field in header if field}


def audit(root: Path, e1: Path | None = None) -> dict[str, Any]:
    a0 = read_json(root / "A0" / "registry_summary.json")
    a1 = read_json(root / "A1" / "a1_summary.json")
    a2 = read_json(root / "A2" / "A2_decision.json")
    holdout = read_json(root / "A2" / "holdout_candidate_manifest.json")
    holdout_activation_path = root / "PhaseD" / "holdout_activation_manifest.json"
    holdout_e1_path = root / "PhaseD" / "holdout_e1_manifest.json"
    holdout_activation = read_json(holdout_activation_path) if holdout_activation_path.is_file() else {}
    holdout_e1 = read_json(holdout_e1_path) if holdout_e1_path.is_file() else {}
    adapter_fields = (
        "input_adapter",
        "feature_selection",
        "normalization",
        "max_features",
        "graph_input",
        "model_input",
    )

    def has_adapter_contract(row: dict[str, Any]) -> bool:
        return all(row.get(field) not in (None, "") for field in adapter_fields)

    activation_rows = holdout_activation.get("datasets", [])
    holdout_jobs = holdout_e1.get("jobs", [])
    required_a0_fields = {
        "dataset",
        "source_artifact",
        "resolved_source_artifact",
        "source_hash",
        "preprocess_hash",
        "readout",
        "k_source",
        "k_hash",
        "labels_used_for_fit",
        "k_used_for_fit",
        "label_k_isolation_status",
        "measurement_timing",
        "causal_status",
        "artifact_status",
        "reused_from",
        "alternative_explanation",
    }
    a0_registry_fields = read_csv_fields(root / "A0" / "mechanism_evidence_registry.csv")
    e1_manifest_path = root / "E1" / "e1_manifest.json"
    e1_manifest = read_json(e1_manifest_path) if e1_manifest_path.is_file() else None
    checks: dict[str, bool] = {
        "a0_row_unit_audit": a0.get("v1_v22_rows") == 2209 and a0.get("v1_v22_paired_rows") == 1637 and a0.get("v1_v22_units") == 431,
        "a0_boundary_isolated": a0.get("v23_v24_boundary_records") == 2 and a0.get("replay_eligible_rows") == 0,
        "a0_registry_schema_complete": required_a0_fields <= a0_registry_fields,
        "a1_seed_aware_atlas": a1.get("paired_rows") == 1637 and a1.get("labels_reloaded_for_atlas") is False,
        "a1_metric_provenance_declared": (
            a1.get("label_free_evaluation") is False
            and str(a1.get("metric_provenance", "")).startswith("paired_delta_ari_and_ari_mean_imported")
        ),
        "a2_veto_capable": a2.get("decision") in {"retain_e1", "cancel_e1", "no_prospective_compute"} and a2.get("no_new_e4") is True,
        "holdout_pre_outcome": holdout.get("selection_policy", {}).get("frozen_before_e1_outcomes") is True and holdout.get("selection_policy", {}).get("selection_uses_labels_or_outcomes") is False,
        "holdout_shortfall_explicit": "candidate_pool_shortfall" in holdout,
        "holdout_adapter_contract_complete": (
            all(has_adapter_contract(row) for row in activation_rows)
            and all(has_adapter_contract(row) for row in holdout_jobs)
            if a2.get("decision") == "retain_e1"
            else True
        ),
        "e1_manifest_gated": (
            (e1_manifest is not None and e1_manifest.get("a2_decision") == "retain_e1" and e1_manifest.get("generated_without_e1_outcomes") is True)
            if a2.get("decision") == "retain_e1"
            else e1_manifest is None
        ),
        "e1_manifest_three_arm_counts": (
            e1_manifest is not None
            and all(
                e1_manifest.get("phases", {}).get(phase, {}).get("expected_panel_jobs") == 9
                and e1_manifest.get("phases", {}).get(phase, {}).get("expected_arm_jobs") == 27
                and len(e1_manifest.get("phases", {}).get(phase, {}).get("jobs", [])) == 27
                for phase in ("pilot", "confirmation")
            )
        ) if a2.get("decision") == "retain_e1" else True,
    }
    e1_report: dict[str, Any] | None = None
    if e1 is not None:
        e1_audit = read_json(e1 / "audit.json")
        e1_summary = read_json(e1 / "summary.json") if (e1 / "summary.json").is_file() else {}
        arm_metrics = []
        for arm in ("N", "R", "T"):
            metrics_path = e1 / arm / "metrics.json"
            if metrics_path.is_file():
                arm_metrics.append(read_json(metrics_path))
        one_step_payload = read_json(e1 / "one_step.json") if (e1 / "one_step.json").is_file() else {}
        one_step_metrics = [value.get("metrics", {}) for value in one_step_payload.values() if isinstance(value, dict)]
        runner = read_json(e1 / "runner_profile.json") if (e1 / "runner_profile.json").is_file() else {}
        independent_panel_audit: dict[str, Any] = {}
        independent_pairs: dict[str, float] = {}
        try:
            from scripts.V25.audit_e1_phase import _panel_audit

            independent_panel_audit, independent_pairs = _panel_audit(e1)
        except (OSError, ValueError, KeyError, TypeError):
            independent_panel_audit = {"audit_ok": False}
        checks.update(
            {
                "e1_allowed_by_a2": a2.get("decision") == "retain_e1",
                "e1_tr_shared_donor": e1_audit.get("TR_shared_schedule_hashes", {}).get("donor") is True,
                "e1_tr_shared_eligible": e1_audit.get("TR_shared_schedule_hashes", {}).get("eligible") is True,
                "e1_tr_shared_budget": e1_audit.get("TR_shared_schedule_hashes", {}).get("budget") is True,
                "e1_tr_shared_noise": e1_audit.get("TR_shared_schedule_hashes", {}).get("selection_noise") is True,
                "e1_none_no_assignment_forward": e1_audit.get("none_contract", {}).get("assignment_forward_calls") == 0,
                "e1_none_no_js_forward": e1_audit.get("none_contract", {}).get("js_forward_calls") == 0,
                "e1_branchpoint_before_assignment": e1_audit.get("branchpoint", {}).get("warmup_branchpoint_before_first_assignment") is True,
                "e1_labels_isolated": runner.get("labels_used_during_fit") is False and e1_audit.get("labels_used_during_fit") is False,
                "e1_known_k_boundary_declared": e1_audit.get("K_used_during_fit") is True and runner.get("K_source") in {"explicit_n_clusters", "benchmark_oracle_from_y"},
                "e1_explicit_or_outer_k": runner.get("K_source") in {"explicit_n_clusters", "benchmark_oracle_from_y"},
                "e1_pairs_complete_after_evaluation": bool(e1_summary.get("pairs")) and all(
                    value is not None for value in e1_summary.get("pairs", {}).values()
                ),
                "e1_metrics_mark_post_fit_labels": len(arm_metrics) == 3 and len(one_step_metrics) == 3 and all(
                    metric.get("labels_used_after_fit_only") is True for metric in arm_metrics + one_step_metrics
                ),
                "e1_independent_panel_audit": independent_panel_audit.get("audit_ok") is True,
                "e1_primary_pairs_recomputed": set(("I_full_ARI", "S_full_ARI")) <= set(independent_pairs),
                "e1_primary_pairs_match_recomputed": all(
                    key in e1_summary.get("pairs", {})
                    and key in independent_pairs
                    and abs(float(e1_summary["pairs"][key]) - float(independent_pairs[key])) <= 1e-10
                    for key in ("I_full_ARI", "S_full_ARI")
                ),
            }
        )
        e1_report = {
            "path": str(e1),
            "audit": e1_audit,
            "runner_profile": runner,
            "independent_panel_audit": independent_panel_audit,
            "recomputed_pairs": independent_pairs,
        }
    result = {
        "protocol_id": "v25_contract_audit_v1",
        "status": "audit_ok" if all(checks.values()) else "invalid_design",
        "checks": checks,
        "a2_decision": a2.get("decision"),
        "e1": e1_report,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--e1", type=Path, default=None, help="optional E1 run directory")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit(args.root, args.e1)
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["status"] == "audit_ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
