#!/usr/bin/env python3
"""Build a paper-facing evidence bundle from frozen V25 artifacts.

This is an analysis-only exporter. It never reads labels for fitting, never
recomputes a model, and keeps dataset/seed summaries as the inferential unit.
Coordinate-level E2-A values are exported only as descriptive summaries.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "v25_paper_evidence_bundle_v1"
E1_PROTOCOL_ID = "v25_e1_v21_matched_nrt_v1"
E1_METRICS = ("I_full_ARI", "S_full_ARI", "I_1step_ARI", "S_1step_ARI")
E2_METRICS = (
    "model_variance",
    "model_zero_fraction",
    "model_support_frequency",
    "raw_support_frequency",
    "support_mutual_information_posthoc",
    "fisher_separation_posthoc",
    "class_support_enrichment_posthoc",
    "donor_change_magnitude",
    "topology_deviation",
    "topology_dispersion",
)

CLAIM_AUDIT_SCHEMA: dict[str, dict[str, Any]] = {
    "selection": {"primary_endpoint_key": "S_full_ARI", "activation_subset": ["E1_NRT"]},
    "generic_intervention": {"primary_endpoint_key": "I_full_ARI", "activation_subset": ["E1_NRT"]},
    "objective_compatibility": {
        "primary_endpoint_key": "objective_sign_agreement",
        "activation_subset": ["E1_NRT", "E2-B", "E2-C"],
    },
    "local_global": {
        "primary_endpoint_key": "local_positive_and_global_nonpositive",
        "activation_subset": ["E3_frozen_matched_pair"],
    },
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _copy_csv(source: Path, target: Path) -> int:
    rows = _read_csv(source)
    _write_csv(target, rows)
    return len(rows)


def _copy_optional_csv(source: Path, target: Path) -> int | None:
    """Copy an optional analysis artifact without inventing an empty result.

    A1 was hardened after the first formal export.  Older result bundles can
    therefore lack the newer summary files even though their required atlas
    rows are complete.  The source manifest records that absence; the bundle
    must remain exportable while preserving the boundary.
    """
    if not source.is_file():
        return None
    return _copy_csv(source, target)


def _copy_optional_json(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    _copy_json(source, target)
    return True


def _copy_json(source: Path, target: Path) -> dict[str, Any]:
    payload = _read_json(source)
    _write_json(target, payload)
    return payload


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def _expected_e2_coverage(manifest_path: Path) -> dict[str, Any]:
    """Extract the frozen confirmation dataset x seed keys for E2-A."""
    manifest = _read_json(manifest_path)
    phase = manifest.get("phases", {}).get("confirmation")
    if not isinstance(phase, dict) or not isinstance(phase.get("jobs"), list):
        raise ValueError(f"confirmation manifest is missing frozen jobs: {manifest_path}")
    panel_map: dict[str, dict[str, Any]] = {}
    for job in phase["jobs"]:
        if not isinstance(job, dict):
            raise ValueError(f"confirmation manifest contains a non-object job: {manifest_path}")
        key = str(job.get("panel_run_key", ""))
        if not key:
            raise ValueError(f"confirmation manifest job has no panel_run_key: {manifest_path}")
        entry = panel_map.setdefault(key, {"dataset": job.get("dataset"), "seed": job.get("seed"), "arms": []})
        if entry["dataset"] != job.get("dataset") or entry["seed"] != job.get("seed"):
            raise ValueError(f"confirmation panel metadata is inconsistent: {key}")
        entry["arms"].append(str(job.get("arm")))
    if any(sorted(entry["arms"]) != ["N", "R", "T"] for entry in panel_map.values()):
        raise ValueError("confirmation manifest does not contain exact N/R/T panels")
    expected_count = int(phase.get("expected_panel_jobs", len(panel_map)))
    if expected_count != len(panel_map):
        raise ValueError("confirmation manifest panel denominator does not match panel keys")
    return {
        "panel_keys": sorted(panel_map),
        "datasets": sorted({str(entry["dataset"]) for entry in panel_map.values()}),
        "seeds": sorted({int(entry["seed"]) for entry in panel_map.values()}),
        "expected_count": expected_count,
    }


def _effect_rows(phase_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for dataset, payload in sorted(phase_summary.get("datasets", {}).items()):
        for metric in ("I_d", "S_d"):
            effect = payload.get(metric, {})
            values = list(effect.get("seed_values", []))
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "state": effect.get("state"),
                    "mean": effect.get("mean"),
                    "n_seeds": effect.get("n_seeds"),
                    "same_sign_count": effect.get("same_sign_count"),
                    "statistical_unit": payload.get("statistical_unit"),
                    "causal_status": "matched prospective case study",
                }
            )
            for seed_index, value in enumerate(values):
                seed_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "seed_index": seed_index,
                        "effect": value,
                        "statistical_unit": "seed repeated measurement nested in dataset",
                    }
                )
    return dataset_rows, seed_rows


def _e2_rows(
    e2_payload: dict[str, Any],
    expected_coverage: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    audits = list(e2_payload.get("audits", []))
    if not audits:
        return [], []
    if expected_coverage is None:
        raise ValueError("E2-A export requires a frozen expected dataset x seed manifest")
    expected_keys = {
        (str(dataset), int(seed))
        for dataset in expected_coverage["datasets"]
        for seed in expected_coverage["seeds"]
    }
    actual_keys = [(str(audit.get("dataset_id")), int(audit.get("seed"))) for audit in audits]
    actual_key_set = set(actual_keys)
    duplicate_keys = sorted(key for key in actual_key_set if actual_keys.count(key) > 1)
    if (
        len(audits) != int(expected_coverage["expected_count"])
        or actual_key_set != expected_keys
        or duplicate_keys
    ):
        raise ValueError(
            "E2-A coverage mismatch: "
            f"expected={sorted(expected_keys)}, actual={sorted(actual_keys)}, duplicates={duplicate_keys}"
        )
    invalid = [audit.get("dataset_id", "unknown") for audit in audits if audit.get("audit_ok") is not True]
    if invalid:
        raise ValueError(f"E2-A export requires audit_ok=true for every panel; invalid panels: {invalid}")
    for audit in audits:
        dataset = str(audit.get("dataset_id"))
        seed = int(audit.get("seed"))
        for metric in E2_METRICS:
            value = audit.get("metrics", {}).get(metric, {}).get("difference")
            if value is None:
                continue
            value = float(value)
            seed_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "metric": metric,
                    "difference_selected_minus_eligible_not_selected": value,
                    "coordinate_distributions_descriptive_only": True,
                    "measurement_timing": "post_intervention_policy_audit",
                    "causal_status": "observational",
                }
            )
            grouped[(dataset, metric)].append(value)
    summary_rows: list[dict[str, Any]] = []
    for (dataset, metric), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "mean_difference": _mean(values),
                "median_difference": statistics.median(values),
                "positive_seed_count": sum(value > 0 for value in values),
                "negative_seed_count": sum(value < 0 for value in values),
                "n_seeds": len(values),
                "statistical_unit": "dataset x seed",
                "coordinate_distributions_descriptive_only": True,
                "posthoc_label_metrics_not_fit_inputs": True,
            }
        )
    return summary_rows, seed_rows


def _gradient_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    for row in rows:
        row["causal_status"] = "diagnostic"
        row["statistical_unit"] = "dataset x seed x timepoint"
    return rows


def _claim_scope_audit(
    a0: dict[str, Any],
    a1: dict[str, Any],
    a2: dict[str, Any],
    claim: dict[str, Any],
    confirmation: dict[str, Any],
    e2: dict[str, Any],
    holdout: dict[str, Any],
) -> dict[str, Any]:
    if a2.get("decision") != "retain_e1":
        checks = {
            "a0_counts_match_frozen_registry": (
                a0.get("v1_v22_rows") == 2209
                and a0.get("v1_v22_paired_rows") == 1637
                and a0.get("v1_v22_units") == 431
            ),
            "a0_boundary_isolated": a0.get("v23_v24_boundary_records") == 2,
            "a1_no_causal_claim": a1.get("no_causal_claim") is True,
            "a2_veto_recorded": a2.get("decision") in {"cancel_e1", "no_prospective_compute"},
            "no_prospective_endpoint": holdout.get("primary_endpoint_evaluable") is False,
        }
        return {
            "protocol_id": "v25_paper_claim_scope_audit_v1",
            "checks": checks,
            "audit_ok": all(checks.values()),
            "allowed_claim_scope": ["observational V1-V22 failure atlas"],
            "forbidden_claim_scope": [
                "prospective V21 mechanism claim",
                "universal topology superiority",
                "independent holdout replication",
                "pooled causal effect across historical protocols",
                "coordinate-level inferential p-values",
            ],
        }
    claim_family = str(claim.get("claim_family", ""))
    claim_schema = CLAIM_AUDIT_SCHEMA.get(claim_family, {})
    activation_subset = list(claim.get("activation_subset", []))
    requires_e1 = "E1_NRT" in activation_subset
    requires_e2 = any(item.startswith("E2-") for item in activation_subset)
    expected_endpoint = claim_schema.get("primary_endpoint_key")
    expected_activation = claim_schema.get("activation_subset")
    holdout_status = str(holdout.get("status", ""))
    holdout_evaluable = holdout.get("primary_endpoint_evaluable")
    holdout_boundary_ok = (
        (holdout_status == "inconclusive_not_completed" and holdout_evaluable is False)
        or (holdout_status in {"completed", "audit_ok"} and holdout_evaluable is True)
        or (holdout_status == "not_activated_a2_veto" and holdout_evaluable is False)
    )
    checks = {
        "a0_counts_match_frozen_registry": (
            a0.get("v1_v22_rows") == 2209
            and a0.get("v1_v22_paired_rows") == 1637
            and a0.get("v1_v22_units") == 431
        ),
        "a0_boundary_isolated": a0.get("v23_v24_boundary_records") == 2,
        "a1_counts_match_frozen_atlas": (
            a1.get("paired_rows") == 1637
            and a1.get("positive_rows") == 194
            and a1.get("negative_rows") == 680
            and a1.get("small_rows") == 763
        ),
        "a1_no_causal_claim": a1.get("no_causal_claim") is True,
        "a2_veto_and_no_e4_recorded": a2.get("decision") == "retain_e1" and a2.get("no_new_e4") is True,
        "claim_freeze_primary_endpoint_is_frozen": (
            claim_family in CLAIM_AUDIT_SCHEMA
            and claim.get("primary_endpoint_key") == expected_endpoint
            and activation_subset == expected_activation
            and claim.get("holdout_rule") == "activate exactly the listed subset and endpoint; do not substitute a secondary metric"
        ),
        "confirmation_audit_complete": (
            confirmation.get("panel_count") == confirmation.get("audit_ok_count") == 9 if requires_e1 else True
        ),
        "e2_coordinate_unit_declared": (
            (e2.get("statistical_unit") == "dataset_seed_summary" and e2.get("coordinate_distributions_descriptive_only") is True)
            if requires_e2
            else True
        ),
        "e2_all_panel_audits_pass": (
            (
                int(e2.get("panel_count", -1)) == int(e2.get("audit_ok_count", -2))
                and all(audit.get("audit_ok") is True for audit in e2.get("audits", []))
            )
            if requires_e2
            else True
        ),
        "holdout_not_used_as_negative_result": (
            holdout_boundary_ok
        ),
    }
    return {
        "protocol_id": "v25_paper_claim_scope_audit_v1",
        "checks": checks,
        "audit_ok": all(checks.values()),
        "allowed_claim_scope": [
            "observational V1-V22 failure atlas",
            f"predeclared {claim_family} claim with claim-dependent activation subset",
        ],
        "forbidden_claim_scope": [
            "universal topology superiority",
            "independent holdout replication",
            "pooled causal effect across historical protocols",
            "coordinate-level inferential p-values",
            "holdout CUDA OOM as a model performance result",
        ],
    }


def build_bundle(root: Path, output: Path) -> dict[str, Any]:
    a0_dir = root / "A0"
    a1_dir = root / "A1"
    a2_dir = root / "A2"
    confirmation_dir = root / "E1" / "confirmation"
    phase_c = root / "PhaseC"
    phase_d = root / "PhaseD"
    phase_e = root / "PhaseE"
    output.mkdir(parents=True, exist_ok=True)

    a0 = _read_json(a0_dir / "registry_summary.json")
    a1 = _read_json(a1_dir / "a1_summary.json")
    a2 = _read_json(a2_dir / "A2_decision.json")
    claim_path = phase_c / "FROZEN_PAPER_CLAIM.json"
    claim = _read_json(claim_path) if claim_path.is_file() else {}
    if a2.get("decision") == "retain_e1":
        confirmation = _read_json(confirmation_dir / "Audit" / "phase_summary.json")
        e2 = _read_json(confirmation_dir / "Audit" / "e2_feature_audit.json")
        holdout = _read_json(phase_e / "closure.json")["independent_holdout"]
        e2_expected = _expected_e2_coverage(confirmation_dir / "manifest_snapshot.json")
    else:
        confirmation = {"panel_count": 0, "audit_ok_count": 0, "datasets": {}}
        e2 = {"panel_count": 0, "audit_ok_count": 0, "audits": [], "statistical_unit": "not_activated"}
        holdout = {
            "status": "not_activated_a2_veto",
            "expected_panel_count": 0,
            "completed_panel_count": 0,
            "primary_endpoint_evaluable": False,
        }
        e2_expected = None

    dataset_effects, seed_effects = _effect_rows(confirmation)
    e2_summary, e2_seed = _e2_rows(e2, e2_expected)
    gradient_path = confirmation_dir / "Audit" / "gradient_probe.csv"
    gradient = _gradient_rows(gradient_path) if gradient_path.is_file() else []

    source_paths = [
        a0_dir / "registry_summary.json",
        a0_dir / "mechanism_evidence_registry.csv",
        a0_dir / "V23_V24_boundary_evidence.csv",
        a1_dir / "a1_summary.json",
        a1_dir / "failure_atlas.csv",
        a1_dir / "version_family_summary.csv",
        a1_dir / "structural_opportunity_summary.csv",
        a1_dir / "magnitude_gain_summary.csv",
        a1_dir / "failure_localization_taxonomy.csv",
        a1_dir / "e3_replay_summary.json",
        a1_dir / "local_global_boundary.csv",
        a2_dir / "A2_decision.json",
        a2_dir / "CLAIM_EVIDENCE_MATRIX.csv",
        a2_dir / "measurement_schema.json",
        a2_dir / "holdout_candidate_manifest.json",
        phase_c / "FROZEN_PAPER_CLAIM.json",
        phase_c / "FROZEN_PAPER_CLAIM.md",
        confirmation_dir / "Audit" / "phase_summary.json",
        confirmation_dir / "manifest_snapshot.json",
        confirmation_dir / "Audit" / "pair_effects.csv",
        confirmation_dir / "Audit" / "e2_feature_audit.json",
        confirmation_dir / "Audit" / "gradient_probe.csv",
        phase_d / "holdout_activation_manifest.json",
        phase_d / "holdout_e1_manifest.json",
        phase_d / "E1" / "queue_state.json",
        phase_d / "Audit" / "phase_summary.json",
        phase_e / "closure.json",
        phase_e / "closure_audit.json",
    ]
    sources = {
        str(path.relative_to(root)): ({"sha256": _sha256(path), "exists": True} if path.is_file() else {"sha256": None, "exists": False})
        for path in source_paths
    }

    family_rows = _copy_csv(a1_dir / "version_family_summary.csv", output / "atlas_version_family.csv")
    atlas_rows = _copy_csv(a1_dir / "failure_atlas.csv", output / "atlas_rows.csv")
    optional_exports = {
        "structural_opportunity_summary_rows": _copy_optional_csv(
            a1_dir / "structural_opportunity_summary.csv", output / "structural_opportunity_summary.csv"
        ),
        "magnitude_gain_summary_rows": _copy_optional_csv(
            a1_dir / "magnitude_gain_summary.csv", output / "magnitude_gain_summary.csv"
        ),
        "failure_localization_taxonomy_rows": _copy_optional_csv(
            a1_dir / "failure_localization_taxonomy.csv", output / "failure_localization_taxonomy.csv"
        ),
        "e3_replay_summary_present": _copy_optional_json(
            a1_dir / "e3_replay_summary.json", output / "e3_replay_summary.json"
        ),
    }
    boundary_rows = _copy_csv(a1_dir / "local_global_boundary.csv", output / "local_global_boundary.csv")
    pair_source = confirmation_dir / "Audit" / "pair_effects.csv"
    pair_rows = _copy_csv(pair_source, output / "e1_pair_effects.csv") if pair_source.is_file() else 0
    _copy_json(a2_dir / "A2_decision.json", output / "a2_decision.json")
    _copy_csv(a2_dir / "CLAIM_EVIDENCE_MATRIX.csv", output / "a2_claim_evidence_matrix.csv")
    _copy_json(a2_dir / "measurement_schema.json", output / "measurement_schema.json")
    _copy_json(a2_dir / "holdout_candidate_manifest.json", output / "holdout_candidate_manifest.json")
    if claim_path.is_file():
        _copy_json(claim_path, output / "frozen_claim.json")
    for source, target in (
        (phase_d / "holdout_activation_manifest.json", output / "holdout_activation_manifest.json"),
        (phase_d / "holdout_e1_manifest.json", output / "holdout_e1_manifest.json"),
        (phase_e / "closure.json", output / "closure.json"),
        (phase_e / "closure_audit.json", output / "closure_audit.json"),
    ):
        if source.is_file():
            _copy_json(source, target)

    _write_csv(output / "e1_dataset_effects.csv", dataset_effects)
    _write_csv(output / "e1_seed_effects.csv", seed_effects)
    _write_csv(output / "e2_semantic_dataset_seed.csv", e2_seed)
    _write_csv(output / "e2_semantic_dataset_summary.csv", e2_summary)
    _write_csv(output / "e2_gradient_geometry.csv", gradient)

    claim_audit = _claim_scope_audit(a0, a1, a2, claim, confirmation, e2, holdout)
    _write_json(output / "claim_scope_audit.json", claim_audit)
    _write_json(output / "source_manifest.json", {"protocol_id": PROTOCOL_ID, "sources": sources})

    summary = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": _now(),
        "statistical_unit": "dataset/protocol/readout for atlas; dataset x seed for E1/E2 summaries",
        "retrospective": {
            "a0": {key: a0.get(key) for key in ("v1_v22_rows", "v1_v22_paired_rows", "v1_v22_units", "v1_v22_unique_datasets", "v23_v24_boundary_records")},
            "a1": {key: a1.get(key) for key in ("paired_rows", "unit_count", "unique_datasets", "positive_rows", "negative_rows", "small_rows")},
            "causal_status": "observational",
        },
        "governance": {
            "a2_decision": a2.get("decision"),
            "a2_no_new_e4": a2.get("no_new_e4"),
            "claim_family": claim.get("claim_family"),
            "primary_endpoint": claim.get("primary_endpoint"),
            "activation_subset": claim.get("activation_subset"),
        },
        "confirmation": {
            "protocol_id": E1_PROTOCOL_ID,
            "panel_count": confirmation.get("panel_count"),
            "audit_ok_count": confirmation.get("audit_ok_count"),
            "dataset_effect_rows": len(dataset_effects),
            "seed_effect_rows": len(seed_effects),
            "primary_metric": "S_full_ARI",
            "interpretation": "conditional heterogeneous V21 case study" if a2.get("decision") == "retain_e1" else "not activated after A2 veto",
        },
        "e2": {
            "dataset_seed_rows": len(e2_seed),
            "dataset_summary_rows": len(e2_summary),
            "expected_dataset_count": len(e2_expected["datasets"]) if e2_expected else 0,
            "expected_seed_count": len(e2_expected["seeds"]) if e2_expected else 0,
            "expected_panel_count": e2_expected["expected_count"] if e2_expected else 0,
            "coverage_complete": len(e2_seed) > 0 and e2_expected is not None,
            "coordinate_distributions_descriptive_only": True,
            "posthoc_label_metrics_not_fit_inputs": True,
            "gradient_rows": len(gradient),
        },
        "boundary": {"local_global_rows": boundary_rows},
        "holdout": {
            "status": holdout.get("status"),
            "expected_panel_count": holdout.get("expected_panel_count"),
            "completed_panel_count": holdout.get("completed_panel_count"),
            "primary_endpoint_evaluable": holdout.get("primary_endpoint_evaluable"),
        },
        "claim_scope_audit": claim_audit,
        "export_counts": {
            "atlas_version_family_rows": family_rows,
            "atlas_rows": atlas_rows,
            "pair_rows": pair_rows,
            "boundary_rows": boundary_rows,
            **optional_exports,
        },
        "missing_source_files": sorted(
            relative for relative, metadata in sources.items() if metadata.get("exists") is not True
        ),
    }
    _write_json(output / "paper_evidence_summary.json", summary)

    lines = [
        "# V25 Paper Evidence Bundle",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "This bundle is an analysis-only export from frozen V25 artifacts. It does not retrain",
        "a model and does not treat rows, coordinates, or seeds as independent population units.",
        "",
        "## Primary facts",
        "",
        f"- A0: `{a0.get('v1_v22_rows')}` registry rows, `{a0.get('v1_v22_paired_rows')}` paired rows, `{a0.get('v1_v22_units')}` units.",
        f"- A1: `{a1.get('positive_rows')}` material positive, `{a1.get('negative_rows')}` material negative, `{a1.get('small_rows')}` observed-small.",
        f"- E1 confirmation: `{confirmation.get('panel_count')}/{confirmation.get('audit_ok_count')}` panels audited successfully.",
        "- E1 primary interpretation: conditional/heterogeneous V21 case study; not universal topology superiority." if a2.get("decision") == "retain_e1" else "- E1 was vetoed by A2; no prospective mechanism endpoint is claimed.",
        "- E1 evaluation boundary: real dataset ground truth is used after fitting, while benchmark-known K can size the cluster head during fitting; this is not fully label-free fitting.",
        f"- Independent holdout: `{holdout.get('completed_panel_count')}/{holdout.get('expected_panel_count')}` panels completed; status `{holdout.get('status')}`.",
        "",
        "## Scope firewall",
        "",
        "- Atlas rows are observational and stratified by protocol/readout.",
        "- E2-A coordinate distributions are descriptive; inference is dataset x seed.",
        "- Post-hoc Fisher/MI/class-support metrics were not fit inputs.",
        "- Holdout CUDA OOM is incomplete compute, not a performance result.",
        "",
        "## Files",
        "",
        "- `atlas_version_family.csv`, `atlas_rows.csv`, `structural_opportunity_summary.csv`, `magnitude_gain_summary.csv`, `failure_localization_taxonomy.csv`, `local_global_boundary.csv`",
        "- `e1_dataset_effects.csv`, `e1_seed_effects.csv`, `e1_pair_effects.csv`",
        "- `e2_semantic_dataset_seed.csv`, `e2_semantic_dataset_summary.csv`, `e2_gradient_geometry.csv`",
        "- `a2_decision.json`, `a2_claim_evidence_matrix.csv`, `measurement_schema.json`, `frozen_claim.json`",
        "- `holdout_activation_manifest.json`, `holdout_e1_manifest.json`, `closure.json`, `closure_audit.json`",
        "- `figures/` and `figure_manifest.json` (generated separately by `build_paper_figures.py`)",
        "- `claim_scope_audit.json`, `source_manifest.json`, `paper_evidence_summary.json`",
    ]
    if summary["missing_source_files"]:
        lines.extend(
            [
                "",
                "## Missing optional inputs",
                "",
                "The following source artifacts were absent in the frozen result bundle and were not reconstructed:",
                *[f"- `{path}`" for path in summary["missing_source_files"]],
            ]
        )
    (output / "PAPER_EVIDENCE_BUNDLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("result/V25_systematic_mechanism_study"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "PaperEvidence"
    summary = build_bundle(args.root, output)
    print(json.dumps({"protocol_id": summary["protocol_id"], "output": str(output), "claim_audit_ok": summary["claim_scope_audit"]["audit_ok"]}, ensure_ascii=False))
    return 0 if summary["claim_scope_audit"]["audit_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
