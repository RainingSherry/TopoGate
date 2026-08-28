"""Create the small, weight-free publication bundle for the closed probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .relation_features import DATASETS, PRIMARY_DATASETS, sha256_file, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "reports/relation_selection_probe"
RESULT_ROOT = PROJECT_ROOT / "result/relation_selection_probe"
DEFAULT_OUTPUT = RESULT_ROOT / "FINAL"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_rs1() -> dict[str, Any]:
    value = _read(RESULT_ROOT / "RS1_information/rs1_summary.json")
    primary_rows = []
    for row in value["rows"]:
        if row["dataset"] in PRIMARY_DATASETS:
            primary_rows.append(
                {
                    "dataset": row["dataset"],
                    "target": row["target"],
                    "family": row["family"],
                    "delta_ap": row["delta_ap"],
                    "lift_at_b": row["lift_at_b"],
                }
            )
    return {
        "decision": value["decision"],
        "information_passes": value["information_passes"],
        "primary_datasets": value["primary_datasets"],
        "family_gate": value["family_gate"],
        "primary_rows": primary_rows,
        "labels_used_in_feature_extraction": value["labels_used_in_feature_extraction"],
        "labels_used_in_diagnostic_targets": value["labels_used_in_diagnostic_targets"],
    }


def _compact_rs2() -> dict[str, Any]:
    value = _read(RESULT_ROOT / "RS2_simple_selectors/rs2_summary.json")
    aggregate = {
        selector: {
            dataset: values
            for dataset, values in datasets.items()
            if dataset in PRIMARY_DATASETS
        }
        for selector, datasets in value["aggregate"].items()
    }
    return {
        "decision": value["decision"],
        "status": value["status"],
        "row_count": len(value["rows"]),
        "expected_row_count": 90,
        "selectors": value["selectors"],
        "seeds": value["seeds"],
        "primary_gate": value["primary_gate"],
        "aggregate_primary": aggregate,
        "labels_used_during_fit": value["labels_used_during_fit"],
        "labels_used_for_outer_metrics": value["labels_used_for_outer_metrics"],
    }


def _compact_rs3() -> dict[str, Any]:
    summary = _read(RESULT_ROOT / "RS3_decision/rs3_summary.json")
    dataset_map = _read(RESULT_ROOT / "RS3_decision/rs3_dataset_map.json")
    return {"summary": summary, "dataset_map": dataset_map["rows"]}


def _hashes() -> dict[str, str]:
    paths = [
        "RS0_freeze/rs0_manifest.json",
        "RS1_information/rs1_summary.json",
        "RS1_information/rs1_manifest.json",
        "RS1_information/rs1_metrics.json",
        "RS2_simple_selectors/rs2_summary.json",
        "RS2_simple_selectors/rs2_manifest.json",
        "RS3_decision/rs3_summary.json",
        "RS3_decision/rs3_manifest.json",
        "RS3_decision/rs3_failure_map.csv",
        "RS3_decision/rs3_selector_capture.csv",
    ]
    result: dict[str, str] = {}
    for relative in paths:
        path = RESULT_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        result[relative] = sha256_file(path)
    return result


def build(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = {
        "project_id": "relation_selection_probe",
        "status": "terminal_under_authorized_scope",
        "decision": "candidate_family_problem_and_learned_rule_only_proposal",
        "authorized_stages": ["RS0", "RS1", "RS2", "RS3"],
        "unauthorized_executions": [
            "RS4 learned selector",
            "GNN/Transformer/TopoCut/DCGC transfer",
            "new reconstruction objective",
            "holdout",
            "V-series continuation",
        ],
        "primary_datasets": list(PRIMARY_DATASETS),
        "primary_datasets_report_only": True,
        "future_learned_selector_requires_separate_holdout": True,
        "all_frozen_datasets": list(DATASETS),
        "rs1": _compact_rs1(),
        "rs2": _compact_rs2(),
        "rs3": _compact_rs3(),
        "publication_boundary": {
            "included": [
                "reports/relation_selection_probe/*.md",
                "result/relation_selection_probe/FINAL/*",
                "scripts/relation_selection_probe/*.py",
                "tests/relation_selection_probe/*.py",
            ],
            "excluded": [
                "RS1 feature tables and OOF arrays",
                "RS2 graph NPZ files",
                "RS1/RS2 embeddings and predictions",
                "per-run logs and caches",
                "model weights and input data",
                "__pycache__",
            ],
        },
    }
    write_json(output_dir / "RESULTS_SUMMARY.json", final)
    write_json(output_dir / "EVIDENCE_HASHES.json", {"local_evidence_sha256": _hashes()})
    write_json(
        output_dir / "PUBLICATION_MANIFEST.json",
        {
            "project_id": "relation_selection_probe",
            "status": "weight_free_compact_bundle",
            "source_result_roots": [
                "result/relation_selection_probe/RS0_freeze",
                "result/relation_selection_probe/RS1_information",
                "result/relation_selection_probe/RS2_simple_selectors",
                "result/relation_selection_probe/RS3_decision",
            ],
            "raw_artifacts_published": False,
            "weights_published": False,
            "input_data_published": False,
            "full_evidence_hashes": "EVIDENCE_HASHES.json",
            "summary": "RESULTS_SUMMARY.json",
        },
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
