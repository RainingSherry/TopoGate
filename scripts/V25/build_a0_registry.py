#!/usr/bin/env python3
"""Build the V25 evidence registry from audited historical artifacts.

This command is deliberately read-only with respect to historical result
directories.  It consumes the already audited V1--V22 long table plus the
formal V23/V24 decision artifacts and writes a new V25 registry.  Rows are
kept as records, while dataset/protocol units are emitted separately so the
registry cannot silently treat seeds or variants as independent experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LONG = ROOT / "reports" / "v1_v22_unified_failure_diagnostic_long_20260814.csv"
DEFAULT_COVERAGE = ROOT / "reports" / "v1_v22_unified_failure_diagnostic_coverage_20260814.csv"
DEFAULT_V23 = ROOT / "result" / "V23_cycle_response" / "m0_synthetic_protocol_a_v1" / "m0_decision.json"
DEFAULT_V24 = ROOT / "result" / "V24_conditional_response" / "q1_synthetic_v2" / "calibration.json"
DEFAULT_OUT = ROOT / "result" / "V25_systematic_mechanism_study" / "A0"

COMPLETED = {"completed"}
VALIDATED = {"completed", "reported"}

STRUCTURAL_MAP: dict[str, tuple[str, str, str, str]] = {
    "V09": ("neighbor/topology graph", "variant-specific Gate or control", "sample/neighbor relation", "medium"),
    "V10": ("reliable graph", "learned/static graph control", "sample/neighbor relation", "medium"),
    "V11": ("topological/predictive graph", "learned Gate or fixed control", "sample relation", "medium"),
    "V12": ("latent topology", "stage-3 topology selection", "edge/relationship", "medium"),
    "V13": ("neighbor graph", "hard top-k Gate", "sample relation", "medium"),
    "V14": ("neighbor graph", "advantage Gate", "sample relation", "medium"),
    "V16.1": ("predictive graph", "fixed/learned promotion control", "sample relation", "high"),
    "V18": ("latent relation/affinity", "FISTA/EdgeGate/fixed relation", "affinity/edge", "high"),
    "V19": ("PCA-kNN/reliability graph", "RG adapter or fixed control", "sample mixing/relation", "high"),
    "V20": ("SVD topology statistics", "feature Gate", "feature masking", "medium"),
    "V21": ("SVD-kNN topology statistics", "FeatureGate or assignment control", "feature-coordinate assignment corruption", "high"),
    "V22": ("topology/discriminator statistics", "hard/cooperative Gate", "feature corruption/keep mask", "high"),
}

# The audited long table intentionally does not carry all of the fields needed
# by a replay protocol.  Keep the missingness explicit instead of filling it
# from version names or benchmark labels.
UNAVAILABLE = "unavailable_from_audited_long_table"

TRAINING_TARGET_MAP: dict[str, str] = {
    "V09": "reconstruction_or_neighbor_relation_proxy",
    "V10": "reconstruction_or_reliable_graph_proxy",
    "V11": "reconstruction_or_predictive_graph_proxy",
    "V12": "reconstruction_or_latent_topology_proxy",
    "V13": "reconstruction_or_neighbor_gate_proxy",
    "V14": "reconstruction_or_advantage_gate_proxy",
    "V16.1": "reconstruction_or_predictive_utility_proxy",
    "V18": "reconstruction_or_affinity_self_expression_proxy",
    "V19": "reconstruction_or_reliability_proxy",
    "V20": "reconstruction_or_topology_statistics_proxy",
    "V21": "scMAE_plus_assignment_JS_and_InfoMax",
    "V22": "reconstruction_plus_discriminator_or_adversarial_mask",
}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def number(value: Any) -> float | None:
    if value in (None, "", "NA", "None", "null", "nan", "NaN"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def unique_join(values: Iterable[Any]) -> str:
    return "|".join(sorted({str(value) for value in values if value not in (None, "", "NA")}))


def safe_path(root: Path, relative: str | None) -> Path | None:
    if not relative:
        return None
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def describe(version: str, family: str, variant: str) -> dict[str, str]:
    source, selection, location, confidence = STRUCTURAL_MAP.get(
        version,
        ("not reconstructed from audited artifact", "not reconstructed", "not reconstructed", "low"),
    )
    if family in {"self", "none"}:
        selection = "none/self control"
        location = "no structural intervention"
    elif family == "random":
        selection = "random control"
    elif family in {"fixed", "static"}:
        selection = "fixed/analytic policy"
    elif family in {"hard", "discriminator"}:
        selection = "hard/adversarial policy"
    elif family == "learned":
        selection = "learned policy"
    return {
        "structural_source": source,
        "selection_policy": selection,
        "intervention_location": location,
        "training_target": TRAINING_TARGET_MAP.get(version, UNAVAILABLE),
        "descriptor_confidence": confidence,
        "descriptor_note": f"version-level descriptor; variant={variant}",
    }


def source_status(root: Path, row: dict[str, Any]) -> tuple[str, str, str]:
    source = str(row.get("source_artifact") or "")
    path = safe_path(root / "result", source)
    if path is not None and path.is_file():
        return "source_file_present", str(path.relative_to(root)), "exact source_artifact path exists"
    version = str(row.get("version") or "")
    if version == "V18":
        return "metadata_only_or_unresolved", source, "public V18 snapshot is metadata-only and exact local source path is unresolved"
    return "registry_only", source, "exact source_artifact path is not present at the registry path"


def registry_rows(long_rows: list[dict[str, str]], root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in long_rows:
        desc = describe(str(row.get("version") or ""), str(row.get("variant_family") or ""), str(row.get("variant") or ""))
        artifact_status, resolved_source, artifact_note = source_status(root, row)
        resolved_path = root / "result" / resolved_source if artifact_status == "source_file_present" else None
        out: dict[str, Any] = {
            "record_type": "intervention_record",
            "version": row.get("version"),
            "source_batch": row.get("source_batch"),
            "dataset": row.get("dataset"),
            "dataset_id": row.get("dataset_id"),
            "variant_family": row.get("variant_family"),
            "variant": row.get("variant"),
            "seed_count": row.get("n_runs"),
            "seeds": row.get("seeds"),
            "status": row.get("status"),
            "evidence_level": row.get("evidence_level"),
            "ari_mean": row.get("ari_mean"),
            "ari_std": row.get("ari_std"),
            "paired_control": row.get("paired_control"),
            "paired_delta_ari": row.get("paired_delta_ari"),
            "paired_delta_scope": row.get("paired_delta_scope"),
            "input_protocol": row.get("input_protocol"),
            "readout": row.get("readout"),
            "measurement_timing": "post_intervention" if row.get("paired_delta_ari") else "final_or_unspecified",
            "measurement_timing_source": "paired_delta_present_in_audited_table" if row.get("paired_delta_ari") else UNAVAILABLE,
            "causal_status": "observational",
            "artifact_status": artifact_status,
            "resolved_source_artifact": resolved_source,
            "source_hash": sha256_file(resolved_path) if resolved_path is not None else UNAVAILABLE,
            "preprocess_hash": row.get("preprocess_hash") or UNAVAILABLE,
            "k_source": row.get("K_source") or row.get("k_source") or UNAVAILABLE,
            "k_hash": row.get("K_hash") or row.get("k_hash") or UNAVAILABLE,
            "n_rows": row.get("n_rows") or row.get("N_rows") or UNAVAILABLE,
            "n_rows_source": "audited_long_table" if row.get("n_rows") or row.get("N_rows") else UNAVAILABLE,
            "labels_used_for_fit": row.get("labels_used_for_fit") or row.get("labels_used_during_fit") or UNAVAILABLE,
            "k_used_for_fit": row.get("k_used_for_fit") or row.get("K_used_during_fit") or UNAVAILABLE,
            "label_k_isolation_status": row.get("label_k_isolation_status") or UNAVAILABLE,
            "artifact_note": artifact_note,
            "replay_eligible": False,
            "reused_from": row.get("reused_from") or "",
            "alternative_explanation": row.get("notes") or "",
            "failure_diagnosis": row.get("failure_diagnosis") or "",
            "source_artifact": row.get("source_artifact") or "",
            "statistical_unit": "dataset/protocol/readout with seed aggregate",
            "evidence_scope": "V1-V22 quantitative failure atlas",
        }
        out.update(desc)
        if row.get("evidence_level") == "empirical_not_supported":
            out["causal_status"] = "unidentified"
        if row.get("status") == "incomplete_compute":
            out["causal_status"] = "unidentified"
        result.append(out)
    return result


def boundary_rows(v23: dict[str, Any], v24: dict[str, Any]) -> list[dict[str, Any]]:
    v23_diag = v23.get("diagnostics") or {}
    v23_delta = v23.get("dependency_positive_deltas") or {}
    v24_cal = v24.get("calibration") or {}
    return [
        {
            "record_type": "boundary_evidence",
            "version": "V23",
            "source_batch": "V23_cycle_response/m0_synthetic_protocol_a_v1",
            "dataset": "synthetic_panel",
            "dataset_id": "v23_protocol_a_m0",
            "variant_family": "response_geometry",
            "variant": "canonical_v23_frozen_probe",
            "status": v23.get("decision", "unknown"),
            "evidence_level": "formal_no_go",
            "seed_count": (v23.get("jobs") or {}).get("seeds", []),
            "seeds": unique_join((v23.get("jobs") or {}).get("seeds", [])),
            "measurement_timing": "post_intervention",
            "measurement_timing_source": "formal_boundary_artifact",
            "causal_status": "boundary_evidence",
            "artifact_status": "formal_boundary_artifact",
            "source_hash": sha256_file(DEFAULT_V23),
            "preprocess_hash": UNAVAILABLE,
            "k_source": UNAVAILABLE,
            "k_hash": UNAVAILABLE,
            "n_rows": UNAVAILABLE,
            "labels_used_for_fit": "false_by_protocol_audit",
            "k_used_for_fit": "false_by_protocol_audit",
            "label_k_isolation_status": "labels_outer_evaluation_only_by_protocol_audit",
            "training_target": "perturbation_response_fingerprint",
            "reused_from": "",
            "alternative_explanation": "support/control or decoder shortcut remain viable",
            "statistical_unit": "synthetic world/seed boundary record",
            "evidence_scope": "V23 boundary evidence; excluded from V1-V22 atlas",
            "structural_source": "perturbation-response fingerprint",
            "selection_policy": "fixed mask dictionary",
            "intervention_location": "feature perturbation and repair",
            "descriptor_confidence": "high",
            "primary_result": json.dumps(v23_delta.get("cycle_minus_support", {}), sort_keys=True),
            "boundary_reason": v23.get("unsupported_claim", ""),
            "effective_mask_ratio_mean": v23_diag.get("effective_mask_ratio_mean"),
            "source_sha256": sha256_file(DEFAULT_V23),
        },
        {
            "record_type": "boundary_evidence",
            "version": "V24",
            "source_batch": "V24_conditional_response/q1_synthetic_v2",
            "dataset": "synthetic_panel",
            "dataset_id": "v24_q1_calibration",
            "variant_family": "conditional_response_calibration",
            "variant": "matched_estimator_calibration",
            "status": "calibration_no_go" if not bool(v24_cal.get("calibration", {}).get("calibration_passes")) else "calibration_pass",
            "evidence_level": "formal_calibration",
            "seed_count": v24_cal.get("config", {}).get("primary_seeds", []),
            "seeds": unique_join(v24_cal.get("config", {}).get("primary_seeds", [])),
            "measurement_timing": "post_intervention",
            "measurement_timing_source": "formal_boundary_artifact",
            "causal_status": "boundary_evidence",
            "artifact_status": "formal_boundary_artifact",
            "source_hash": sha256_file(DEFAULT_V24),
            "preprocess_hash": UNAVAILABLE,
            "k_source": UNAVAILABLE,
            "k_hash": UNAVAILABLE,
            "n_rows": UNAVAILABLE,
            "labels_used_for_fit": "false_by_protocol_audit",
            "k_used_for_fit": "false_by_protocol_audit",
            "label_k_isolation_status": "labels_outer_evaluation_only_by_protocol_audit",
            "training_target": "conditional_response_estimator_calibration",
            "reused_from": "",
            "alternative_explanation": "calibration power failure; no efficacy inference",
            "statistical_unit": "synthetic world/replicate boundary record",
            "evidence_scope": "V24 boundary evidence; excluded from V1-V22 atlas",
            "structural_source": "conditional response probe",
            "selection_policy": "fixed synthetic worlds",
            "intervention_location": "response estimator",
            "descriptor_confidence": "high",
            "primary_result": json.dumps(v24_cal.get("calibration", {}), sort_keys=True),
            "boundary_reason": "calibration power is zero; no efficacy conclusion",
            "source_sha256": sha256_file(DEFAULT_V24),
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_units(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("version") or ""),
            str(row.get("source_batch") or ""),
            str(row.get("dataset_id") or ""),
            str(row.get("input_protocol") or ""),
            str(row.get("readout") or ""),
        )
        grouped[key].append(row)
    units: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        version, source_batch, dataset_id, protocol, readout = key
        units.append(
            {
                "version": version,
                "source_batch": source_batch,
                "dataset_id": dataset_id,
                "input_protocol": protocol,
                "readout": readout,
                "row_count": len(group),
                "completed_row_count": sum(str(r.get("status")) == "completed" for r in group),
                "reported_row_count": sum(str(r.get("status")) == "reported" for r in group),
                "incomplete_row_count": sum(str(r.get("status")) == "incomplete_compute" for r in group),
                "paired_row_count": sum(bool(r.get("paired_delta_ari")) for r in group),
                "variant_count": len({str(r.get("variant")) for r in group}),
                "variants": unique_join(r.get("variant") for r in group),
                "seeds": unique_join(r.get("seeds") for r in group),
                "unit_statistical_role": "dataset_protocol_unit; seeds and variants are repeated conditions",
            }
        )
    return units


def build_artifact_availability(rows: list[dict[str, Any]], coverage: list[dict[str, str]], root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("version") or ""), str(row.get("source_batch") or ""))].append(row)
    result: list[dict[str, Any]] = []
    for version, source_batch in sorted(grouped):
        group = grouped[(version, source_batch)]
        statuses = Counter(str(r.get("artifact_status")) for r in group)
        result.append(
            {
                "version": version,
                "source_batch": source_batch,
                "artifact_class": "metadata_or_registry_only",
                "completed_rows": sum(str(r.get("status")) == "completed" for r in group),
                "reported_rows": sum(str(r.get("status")) == "reported" for r in group),
                "incomplete_rows": sum(str(r.get("status")) == "incomplete_compute" for r in group),
                "source_file_present_rows": statuses.get("source_file_present", 0),
                "replay_eligible_rows": 0,
                "artifact_note": "A0 does not promote replay eligibility from a summary table; exact arrays/checkpoints require a separate artifact gate.",
                "source_hash": sha256_file(root / "reports" / "v1_v22_unified_failure_diagnostic_long_20260814.csv"),
            }
        )
    for row in coverage:
        version = str(row.get("version") or "")
        if version in {item["version"] for item in result}:
            continue
        result.append(
            {
                "version": version,
                "source_batch": "",
                "artifact_class": "not_available_in_audited_long_table",
                "completed_rows": row.get("completed_row_count", 0),
                "reported_rows": 0,
                "incomplete_rows": 0,
                "source_file_present_rows": 0,
                "replay_eligible_rows": 0,
                "artifact_note": row.get("description", "no audited artifact in long table"),
                "source_hash": sha256_file(root / "reports" / "v1_v22_unified_failure_diagnostic_coverage_20260814.csv"),
            }
        )
    result.extend(
        [
            {
                "version": "V23",
                "source_batch": "V23_cycle_response/m0_synthetic_protocol_a_v1",
                "artifact_class": "formal_boundary_evidence",
                "completed_rows": 12,
                "reported_rows": 0,
                "incomplete_rows": 0,
                "source_file_present_rows": 1,
                "replay_eligible_rows": 0,
                "artifact_note": "No-Go boundary evidence; not pooled into V1-V22 intervention atlas.",
                "source_hash": sha256_file(DEFAULT_V23),
            },
            {
                "version": "V24",
                "source_batch": "V24_conditional_response/q1_synthetic_v2",
                "artifact_class": "formal_boundary_evidence",
                "completed_rows": 200,
                "reported_rows": 0,
                "incomplete_rows": 0,
                "source_file_present_rows": 1,
                "replay_eligible_rows": 0,
                "artifact_note": "Calibration No-Go; no efficacy conclusion.",
                "source_hash": sha256_file(DEFAULT_V24),
            },
        ]
    )
    return result


def write_markdown(out: Path, rows: list[dict[str, Any]], units: list[dict[str, Any]], boundaries: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_version[str(row.get("version"))].append(row)
    lines = [
        "# V25 A0 Evidence Registry",
        "",
        f"Generated at `{summary['generated_at']}` from the audited V1-V22 long table.",
        "",
        "This is a registry, not a pooled inferential analysis. Rows are repeated records; the primary historical unit is a dataset/protocol/readout unit.",
        "",
        "## Coverage",
        "",
        f"- V1-V22 rows: `{summary['v1_v22_rows']}`; completed: `{summary['v1_v22_completed']}`; reported/unpromoted: `{summary['v1_v22_reported']}`; incomplete: `{summary['v1_v22_incomplete']}`.",
        f"- V1-V22 paired Delta ARI rows: `{summary['v1_v22_paired_rows']}`.",
        f"- V1-V22 dataset/protocol/readout units: `{summary['v1_v22_units']}`; unique dataset IDs: `{summary['v1_v22_unique_datasets']}`.",
        "- V23 and V24 are recorded as boundary evidence and are not included in the quantitative intervention atlas.",
        "",
        "## Statistical boundary",
        "",
        "Seed is a repeated measurement. Variant is an intervention condition. Coordinate, row, and pair counts are never treated as independent experiments.",
        "",
        "## Version summary",
        "",
        "| Version | Rows | Completed | Reported | Incomplete | Units | Unique datasets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for version in sorted(by_version):
        group = by_version[version]
        lines.append(
            f"| {version} | {len(group)} | {sum(r.get('status') == 'completed' for r in group)} | {sum(r.get('status') == 'reported' for r in group)} | {sum(r.get('status') == 'incomplete_compute' for r in group)} | {len({(r.get('source_batch'), r.get('dataset_id'), r.get('input_protocol'), r.get('readout')) for r in group})} | {len({r.get('dataset_id') for r in group})} |"
        )
    lines += [
        "",
        "## Boundary evidence",
        "",
        "| Version | Status | Evidence | Boundary |",
        "|---|---|---|---|",
    ]
    for row in boundaries:
        lines.append(f"| {row['version']} | {row['status']} | {row['evidence_level']} | {row['boundary_reason']} |")
    lines += [
        "",
        "## Replay gate",
        "",
        "A0 deliberately leaves `replay_eligible_rows=0` for the historical summary table. A1 replay must pass a separate artifact-complete gate with exact embeddings/predictions/labels provenance; missing public artifacts remain descriptive-only.",
        "",
    ]
    (out / "A0_REGISTRY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-csv", type=Path, default=DEFAULT_LONG)
    parser.add_argument("--coverage-csv", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--v23-json", type=Path, default=DEFAULT_V23)
    parser.add_argument("--v24-json", type=Path, default=DEFAULT_V24)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.long_csv, args.coverage_csv, args.v23_json, args.v24_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    long_rows = read_csv(args.long_csv)
    coverage = read_csv(args.coverage_csv)
    v23 = read_json(args.v23_json)
    v24 = read_json(args.v24_json)
    records = registry_rows(long_rows, ROOT)
    units = build_units(records)
    boundaries = boundary_rows(v23, v24)
    artifacts = build_artifact_availability(records, coverage, ROOT)

    args.out.mkdir(parents=True, exist_ok=True)
    record_fields = sorted({key for row in records for key in row})
    unit_fields = sorted({key for row in units for key in row})
    artifact_fields = sorted({key for row in artifacts for key in row})
    boundary_fields = sorted({key for row in boundaries for key in row})
    write_csv(args.out / "mechanism_evidence_registry.csv", records, record_fields)
    write_csv(args.out / "dataset_protocol_units.csv", units, unit_fields)
    write_csv(args.out / "artifact_availability.csv", artifacts, artifact_fields)
    write_csv(args.out / "V23_V24_boundary_evidence.csv", boundaries, boundary_fields)

    quantitative = [row for row in records if str(row.get("version", "")).startswith("V") and row.get("version") != "V23" and row.get("version") != "V24"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_id": "v25_a0_evidence_registry_v1",
        "source_long_csv": str(args.long_csv.relative_to(ROOT)),
        "source_long_csv_sha256": sha256_file(args.long_csv),
        "source_coverage_csv_sha256": sha256_file(args.coverage_csv),
        "source_v23_sha256": sha256_file(args.v23_json),
        "source_v24_sha256": sha256_file(args.v24_json),
        "v1_v22_rows": len(quantitative),
        "v1_v22_completed": sum(row.get("status") == "completed" for row in quantitative),
        "v1_v22_reported": sum(row.get("status") == "reported" for row in quantitative),
        "v1_v22_incomplete": sum(row.get("status") == "incomplete_compute" for row in quantitative),
        "v1_v22_paired_rows": sum(bool(row.get("paired_delta_ari")) for row in quantitative),
        "v1_v22_units": len({(row.get("version"), row.get("source_batch"), row.get("dataset_id"), row.get("input_protocol"), row.get("readout")) for row in quantitative}),
        "v1_v22_unique_datasets": len({row.get("dataset_id") for row in quantitative}),
        "v23_v24_boundary_records": len(boundaries),
        "replay_eligible_rows": 0,
        "N_rows": len(quantitative),
        "statistical_unit": "dataset/protocol/readout; seeds are repeated measurements",
        "labels_used_for_registry": False,
        "notes": "V23/V24 are boundary evidence; no rows are pooled with V1-V22 quantitative intervention records. Missing source/preprocess/K/fit-isolation fields remain explicit.",
    }
    (args.out / "registry_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.out, quantitative, units, boundaries, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
