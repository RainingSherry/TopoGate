"""Build the frozen RS3 selector-capture failure map.

RS3 is analysis-only.  It reads the completed RS1/RS2 summaries and the
closed project's audited S1 opportunity aggregates; it never fits a selector,
rebuilds a graph, or reads labels.  The output keeps the three primary
development datasets separate from the three pre-registered sentinels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .relation_features import DATASETS, MATERIALITY_DELTA, PRIMARY_DATASETS, RS2_CAPTURE, write_json
from .selectors import SELECTORS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_S1_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S1_oracle_v2"
RS1_ROOT = PROJECT_ROOT / "result/relation_selection_probe/RS1_information"
RS2_ROOT = PROJECT_ROOT / "result/relation_selection_probe/RS2_simple_selectors"
DEFAULT_OUTPUT = Path("result/relation_selection_probe/RS3_decision")

ROLE = {
    "cnae9": "primary_opportunity_development",
    "Campbell": "primary_opportunity_development",
    "sms_spam_collection": "primary_opportunity_and_candidate_boundary",
    "Baron Human": "consumer_sensitive_boundary",
    "Mouse_retina": "low_opportunity_contradiction_sentinel",
    "hate_speech": "candidate_family_sentinel",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _old_opportunity(dataset: str) -> dict[str, Any]:
    summary = _read_json(OLD_S1_ROOT / "s1_summary.json")
    try:
        aggregate = summary["dataset_aggregates"][dataset]
        return {
            "H_pool": float(aggregate["H_pool"]["mean"]),
            "H_full": float(aggregate["H_full"]["mean"]),
            "C_candidate_gap": float(
                aggregate["H_full"]["mean"] - aggregate["H_pool"]["mean"]
            ),
            "H_pool_values": [float(v) for v in aggregate["H_pool"]["values"]],
            "H_full_values": [float(v) for v in aggregate["H_full"]["values"]],
        }
    except KeyError as exc:
        raise ValueError(f"old S1 summary missing audited opportunity field for {dataset}") from exc


def _selector_rows(rs2: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    rows = [row for row in rs2.get("rows", []) if row.get("dataset") == dataset]
    expected = len(SELECTORS) * 3
    if len(rows) != expected:
        raise ValueError(f"RS2 row count for {dataset}: {len(rows)} != {expected}")
    if any(row.get("status") != "completed_valid" for row in rows):
        raise ValueError(f"RS2 contains incomplete rows for {dataset}")
    return rows


def _selector_aggregate(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("selector") == selector]
    if len(selected) != 3:
        raise ValueError(f"expected three paired seeds for {selector}")
    deltas = np.asarray([float(row["Delta_S"]) for row in selected], dtype=np.float64)
    captures = [row["Capture_S"] for row in selected if row.get("Capture_S") is not None]
    h_pool = float(np.mean([float(row["H_pool"]) for row in selected]))
    return {
        "selector": selector,
        "H_pool_mean_from_RS2": h_pool,
        "Delta_S_mean": float(np.mean(deltas)),
        "Delta_S_median": float(np.median(deltas)),
        "Delta_S_all_seeds": [float(value) for value in deltas],
        "Capture_S_median": float(np.median(np.asarray(captures, dtype=np.float64))) if captures else None,
        "Capture_S_all_material_seeds": [float(value) for value in captures],
        "material_opportunity": bool(h_pool >= MATERIALITY_DELTA),
        "material_positive_capture": bool(
            h_pool >= MATERIALITY_DELTA
            and float(np.mean(deltas)) >= MATERIALITY_DELTA
            and captures
            and float(np.median(np.asarray(captures, dtype=np.float64))) >= RS2_CAPTURE
        ),
    }


def _dataset_record(dataset: str, rs2: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opportunity = _old_opportunity(dataset)
    rows = _selector_rows(rs2, dataset)
    selectors = [_selector_aggregate(rows, selector) for selector in SELECTORS]
    material_selectors = [
        row for row in selectors if row["material_opportunity"]
    ]
    best = max(selectors, key=lambda row: row["Delta_S_mean"])
    best_material = max(
        material_selectors,
        key=lambda row: row["Delta_S_mean"],
        default=None,
    )
    record = {
        "dataset": dataset,
        "role": ROLE[dataset],
        **opportunity,
        "candidate_gap_material": bool(opportunity["C_candidate_gap"] >= MATERIALITY_DELTA),
        "material_opportunity": bool(opportunity["H_pool"] >= MATERIALITY_DELTA),
        "selector_count": len(selectors),
        "best_selector": best["selector"],
        "best_Delta_S_mean": best["Delta_S_mean"],
        "best_Delta_S_median": best["Delta_S_median"],
        "best_material_selector": best_material["selector"] if best_material else None,
        "best_material_Delta_S_mean": best_material["Delta_S_mean"] if best_material else None,
        "best_material_Capture_S_median": best_material["Capture_S_median"] if best_material else None,
        "simple_material_positive_selector_count": int(sum(row["material_positive_capture"] for row in selectors)),
        "mouse_contradiction_trigger": bool(
            dataset == "Mouse_retina" and best["Delta_S_mean"] >= MATERIALITY_DELTA
        ),
        "hate_candidate_family_trigger": bool(
            dataset == "hate_speech" and opportunity["C_candidate_gap"] >= MATERIALITY_DELTA
        ),
        "baron_consumer_boundary": bool(
            dataset == "Baron Human" and opportunity["H_pool"] < MATERIALITY_DELTA
        ),
    }
    return record, selectors


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty CSV")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rs1 = _read_json(RS1_ROOT / "rs1_summary.json")
    rs2 = _read_json(RS2_ROOT / "rs2_summary.json")
    if rs1.get("status") != "completed_valid" or rs2.get("status") != "completed_valid":
        raise ValueError("RS3 requires completed-valid RS1 and RS2 summaries")
    if list(rs1.get("primary_datasets", [])) != list(PRIMARY_DATASETS):
        raise ValueError("RS1 primary denominator changed")
    if list(rs2.get("selectors", [])) != list(SELECTORS):
        raise ValueError("RS2 selector set changed")

    dataset_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        record, selectors = _dataset_record(dataset, rs2)
        dataset_rows.append(record)
        for selector in selectors:
            selector_rows.append({"dataset": dataset, "role": ROLE[dataset], **selector})

    primary_rows = [row for row in dataset_rows if row["dataset"] in PRIMARY_DATASETS]
    material_primary = [row for row in primary_rows if row["material_opportunity"]]
    simple_sufficient = bool(rs2.get("simple_rule_sufficient", False))
    rs1_information_passes = bool(rs1.get("information_passes", False))
    mouse_triggered = any(row["mouse_contradiction_trigger"] for row in dataset_rows)
    hate_triggered = any(row["hate_candidate_family_trigger"] for row in dataset_rows)
    candidate_gap_datasets = [
        row["dataset"] for row in dataset_rows if row["candidate_gap_material"]
    ]
    candidate_gap_isolated_to_hate = candidate_gap_datasets == ["hate_speech"]

    if not rs1_information_passes:
        next_decision = "current_relation_evidence_not_sufficient"
    elif simple_sufficient:
        next_decision = "simple_relation_rule_sufficient"
    elif hate_triggered:
        next_decision = "candidate_family_problem_and_learned_rule_only_proposal"
    else:
        next_decision = "learned_decision_rule_justified_proposal_only"

    summary = {
        "project_id": "relation_selection_probe",
        "stage": "RS3_decision",
        "protocol_id": "relation_selection_probe_rs3_v1",
        "status": "completed_valid",
        "rs1_decision": rs1.get("decision"),
        "rs1_information_passes": rs1_information_passes,
        "rs1_semantic_same_class_gate_passes": any(
            value["passes_information_gate"]
            for value in rs1.get("family_gate", {}).get("same_class", {}).values()
        ),
        "rs1_pool_reference_gate_passes": any(
            value["passes_information_gate"]
            for value in rs1.get("family_gate", {}).get("pool_reference_membership", {}).values()
        ),
        "rs2_decision": rs2.get("decision"),
        "simple_rule_sufficient": simple_sufficient,
        "primary_dataset_denominator": list(PRIMARY_DATASETS),
        "primary_datasets_report_only": True,
        "future_learned_selector_requires_separate_holdout": True,
        "primary_material_opportunity_count": len(material_primary),
        "primary_material_opportunity_datasets": [row["dataset"] for row in material_primary],
        "primary_positive_selector_rows": int(
            sum(row["simple_material_positive_selector_count"] for row in primary_rows)
        ),
        "mouse_contradiction_trigger": mouse_triggered,
        "hate_candidate_family_trigger": hate_triggered,
        "candidate_gap_material_datasets": candidate_gap_datasets,
        "candidate_gap_isolated_to_hate": candidate_gap_isolated_to_hate,
        "baron_consumer_boundary_present": any(row["baron_consumer_boundary"] for row in dataset_rows),
        "candidate_family_problem": hate_triggered,
        "learned_decision_rule_justified": bool(rs1_information_passes and not simple_sufficient),
        "learned_decision_rule_execution_authorized": False,
        "learned_rule_only_is_scope_choice": True,
        "new_backbone_execution_authorized": False,
        "holdout_execution_authorized": False,
        "decision": next_decision,
        "interpretation": {
            "information": (
                "Frozen relation features predict pool-reference membership on the primary set; "
                "the semantic same-class target does not pass the full two-threshold gate."
            ),
            "selection": (
                "Every fixed selector fails the material primary capture rule; the failure is not "
                "evidence for a learned selector without a separately frozen protocol."
            ),
            "candidate": (
                "sms_spam_collection and hate_speech have material expanded-minus-pool reference "
                "gaps; hate_speech is the extreme sentinel, so the gap is not isolated to hate and "
                "the candidate-family issue is not rescued by RS2."
            ),
            "sentinel": (
                "Mouse_retina does not trigger the pre-registered material contradiction sentinel; "
                "Baron Human remains a low-opportunity consumer boundary."
            ),
        },
        "labels_used_in_rs3_fit": False,
        "labels_used_only_in_inherited_metrics": True,
    }
    write_json(output_dir / "rs3_summary.json", summary)
    write_json(output_dir / "rs3_dataset_map.json", {"rows": dataset_rows})
    _write_csv(output_dir / "rs3_failure_map.csv", dataset_rows)
    _write_csv(output_dir / "rs3_selector_capture.csv", selector_rows)
    write_json(
        output_dir / "rs3_manifest.json",
        {
            "project_id": "relation_selection_probe",
            "stage": "RS3_decision",
            "protocol_id": "relation_selection_probe_rs3_v1",
            "inputs": {
                "rs1_summary": str(RS1_ROOT / "rs1_summary.json"),
                "rs2_summary": str(RS2_ROOT / "rs2_summary.json"),
                "closed_s1_summary": str(OLD_S1_ROOT / "s1_summary.json"),
            },
            "dataset_roles_frozen": ROLE,
            "primary_dataset_denominator": list(PRIMARY_DATASETS),
            "selectors": list(SELECTORS),
            "materiality_delta": MATERIALITY_DELTA,
            "capture_threshold": RS2_CAPTURE,
            "labels_used_in_rs3_fit": False,
            "status": "completed_valid",
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
