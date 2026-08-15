#!/usr/bin/env python3
"""Freeze the V25 mechanism triage and holdout contract.

The triage is intentionally veto-capable.  It reads retrospective artifacts and
historical protocol metadata, but it never reads a prospective outcome to
select a holdout dataset.  The default decision is ``retain_e1`` only when the
historical V21 heterogeneity and the missing matched counterfactual are both
auditable.  Otherwise the protocol closes without new training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
V25_ROOT = ROOT / "result" / "V25_systematic_mechanism_study"
DEFAULT_A0 = V25_ROOT / "A0"
DEFAULT_A1 = V25_ROOT / "A1"
DEFAULT_V21 = ROOT / "result" / "V21" / "v21_ari_confirm_aw0.1_glr0.00025_ep80_20260811" / "confirm_summary.json"
DEFAULT_MANIFESTS = (
    ROOT / "result" / "V21" / "v21_extended13_readoutfix_manifest_20260811.json",
    ROOT / "datasets" / "external" / "v22_dataset_extension_round2_20260812" / "manifest.json",
    ROOT / "datasets" / "external" / "v22_dataset_extension_20260812" / "manifest.json",
)
DEFAULT_OUT = V25_ROOT / "A2"

DELTA = 0.03
PILOT_DATASETS = ("cnae9", "Mouse_retina", "sms_spam_collection")
CONFIRMATION_DATASETS = ("Baron Human", "Campbell", "hate_speech")
SEEDS = (42, 123, 7)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: Any) -> float | None:
    if value in (None, "", "NA", "None", "null", "nan", "NaN"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_manifest_entries(payload: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    entries = payload.get("datasets") or payload.get("records") or []
    if not isinstance(entries, list):
        return []
    output: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        entry = dict(raw)
        entry["manifest_source"] = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
        output.append(entry)
    return output


def adapter_contract(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    """Validate only input compatibility; never inspect labels or outcomes."""
    path = Path(str(entry.get("source_path", "")))
    if not path.is_absolute():
        path = root / path
    protocol = str(entry.get("input_protocol", ""))
    family = str(entry.get("family", ""))
    supported = protocol in {"clubench_bridge", "shared_text", "scRNA_count"}
    exists = path.is_file()
    adapter = "prepare_dual_input" if supported else "unsupported_protocol"
    status = "valid" if supported and exists else ("missing_source" if supported else "incompatible_adapter")
    family_lower = family.lower()
    name_lower = str(entry.get("name", "")).lower()
    is_scrna = "scrna" in family_lower or "scrna" in protocol.lower() or "pbmc" in name_lower or "smartseq" in name_lower
    is_sparse_text = any(token in family_lower for token in ("text", "sparse_text", "web", "libsvm_sparse"))
    return {
        "domain": "scRNA" if is_scrna else ("sparse_text" if is_sparse_text else "other"),
        "dataset_id": str(entry.get("dataset_id", entry.get("name", ""))),
        "name": str(entry.get("name", "")),
        "input_adapter": adapter,
        "feature_selection": "adapter_default_label_free",
        "normalization": "prepare_dual_input_frozen",
        "max_features": "adapter_default",
        "graph_input": "X_graph_from_prepare_dual_input",
        "model_input": "X_model_from_prepare_dual_input",
        "input_protocol": protocol,
        "source_path": str(path),
        "source_hash": entry.get("source_hash") or entry.get("processed_sha256") or sha256_file(path),
        "source_manifest": entry.get("manifest_source", ""),
        "labels_available_outer_only": bool(entry.get("labels_available_outer_only", False)),
        "adapter_status": status,
        "outcome_selection_declared": bool(entry.get("selection_uses_labels_or_outcomes", False)),
        "holdout_eligible": bool(status == "valid" and not entry.get("selection_uses_labels_or_outcomes", False)),
        "exclusion_reason": "" if status == "valid" else ("input_adapter_not_frozen_for_protocol" if status == "incompatible_adapter" else "source_path_missing"),
    }


def build_holdout_manifest(manifest_paths: tuple[Path, ...], root: Path) -> dict[str, Any]:
    seen: dict[str, dict[str, Any]] = {}
    for source in manifest_paths:
        if not source.is_file():
            continue
        for entry in flatten_manifest_entries(read_json(source), source):
            item = adapter_contract(entry, root)
            key = item["dataset_id"]
            previous = seen.get(key)
            if previous is None or item["adapter_status"] == "valid":
                seen[key] = item

    # Development datasets are listed for audit but can never become external
    # holdout rows.  The V21 six-dataset names are fixed before any E1 result.
    for item in seen.values():
        if item["name"] in set(PILOT_DATASETS + CONFIRMATION_DATASETS):
            item["holdout_eligible"] = False
            item["exclusion_reason"] = "overlaps_v21_development_panel"

    eligible = [item for item in seen.values() if item["holdout_eligible"]]
    # Freeze a deterministic candidate pool without claiming that its target
    # domain counts are already satisfied.
    selected = sorted(eligible, key=lambda item: (item["domain"], item["dataset_id"]))
    scRNA = [item for item in selected if item["domain"] == "scRNA"]
    text = [item for item in selected if item["domain"] == "sparse_text"]
    return {
        "manifest_id": "v25_holdout_candidate_manifest_v1",
        "selection_policy": {
            "frozen_before_e1_outcomes": True,
            "selection_uses_labels_or_outcomes": False,
            "selection_basis": "predeclared source manifests plus input-adapter validity only",
            "development_overlap_excluded": True,
        },
        "target_counts": {"scRNA": 4, "sparse_text": 2},
        "eligible_counts": {"scRNA": len(scRNA), "sparse_text": len(text)},
        "candidate_pool_shortfall": {
            "scRNA": max(0, 4 - len(scRNA)),
            "sparse_text": max(0, 2 - len(text)),
        },
        "adapter_contract": {
            "input_adapter": "prepare_dual_input",
            "feature_selection": "adapter_default_label_free",
            "normalization": "prepare_dual_input_frozen",
            "max_features": "adapter_default",
            "graph_input": "X_graph_from_prepare_dual_input",
            "model_input": "X_model_from_prepare_dual_input",
            "label_boundary": "labels and K are outer-evaluation only; no labels in fit/graph/gate",
        },
        "candidates": selected,
        "excluded": sorted((item for item in seen.values() if not item["holdout_eligible"]), key=lambda item: item["dataset_id"]),
    }


def v21_evidence(v21: dict[str, Any]) -> dict[str, Any]:
    rows = v21.get("per_dataset") or []
    delta_values = [number(row.get("delta_vs_scmae_only")) for row in rows]
    delta_values = [value for value in delta_values if value is not None]
    positive = sum(value > DELTA for value in delta_values)
    negative = sum(value < -DELTA for value in delta_values)
    return {
        "dataset_count": len(delta_values),
        "positive_material_count": positive,
        "negative_material_count": negative,
        "delta_mean": mean(delta_values),
        "heterogeneous_signs": positive > 0 and negative > 0,
        "historical_selection_bias": bool(v21.get("selection_uses_labels")),
        "formal_random_counterfactual_present": False,
        "formal_none_arm_present": False,
        "source_protocol_id": v21.get("protocol_id"),
    }


def build_claim_matrix(a0: dict[str, Any], a1: dict[str, Any], v21: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = v21_evidence(v21)
    atlas_negative = int(a1.get("negative_rows", 0))
    atlas_positive = int(a1.get("positive_rows", 0))
    return [
        {
            "hypothesis_id": "H1_structural_quality_not_utility",
            "hypothesis": "structural quality does not imply intervention utility",
            "historical_support": f"A1 has {atlas_positive} material positive and {atlas_negative} material negative paired rows",
            "counterexample": "observational rows span multiple protocols and do not identify a common causal mechanism",
            "identifiability": "high_retrospective_only",
            "generality_required": "multiple intervention families and dataset/protocol units",
            "fatal_falsifier": "audited atlas loses the sign heterogeneity after unit/protocol stratification",
            "alternative_explanation": "protocol, readout, baseline and dose confounding",
            "novelty": "failure localization rather than a new architecture",
            "required_new_compute": "none",
            "status": "retain_as_retrospective_claim",
        },
        {
            "hypothesis_id": "H2_topology_selectivity_conditional",
            "hypothesis": "topology-dependent selection has conditional incremental utility beyond matched intervention",
            "historical_support": f"V21 shows {evidence['positive_material_count']} positive and {evidence['negative_material_count']} negative material dataset means",
            "counterexample": "historical control is scMAE-only; selection used labels at outer development and random policy is not fully matched",
            "identifiability": "high_with_new_NRT_protocol",
            "generality_required": "V21 case study plus independent holdout; not universal population claim",
            "fatal_falsifier": "no material, seed-stable I or S effect in all pilot datasets or branchpoint matching fails",
            "alternative_explanation": "generic intervention dose, optimizer state, stochastic selection or readout mismatch",
            "novelty": "matched selection-policy localization",
            "required_new_compute": "E1 only after A2 retain_e1",
            "status": "provisionally_reserved",
        },
        {
            "hypothesis_id": "H3_generic_intervention_effect",
            "hypothesis": "assignment intervention itself changes utility independently of topology selection",
            "historical_support": "not identified by the V1-V22 summary table",
            "counterexample": "no matched none/random three-arm historical control",
            "identifiability": "high_with_E1_only",
            "generality_required": "V21 N/R arm and claim-dependent holdout",
            "fatal_falsifier": "N/R effect is observed-small in all predeclared cases",
            "alternative_explanation": "training budget or InfoMax/head effects",
            "novelty": "control decomposition, not a standalone architecture",
            "required_new_compute": "E1 shared with H2",
            "status": "secondary_E1_quantity",
        },
        {
            "hypothesis_id": "H4_local_global_conversion",
            "hypothesis": "local geometry improvement can fail to convert to global clustering gain",
            "historical_support": "V23 boundary evidence contains positive local/non-positive ARI cases",
            "counterexample": "V23 is a separate response protocol and not pooled with V1-V22",
            "identifiability": "moderate_offline_replay; prospective only if artifacts exist",
            "generality_required": "artifact-complete cases and frozen local/global metrics",
            "fatal_falsifier": "all label-free local improvements co-occur with positive global gain",
            "alternative_explanation": "readout mismatch or label-aligned post-hoc metric",
            "novelty": "conversion failure localization",
            "required_new_compute": "none unless frozen claim requires holdout replay",
            "status": "retain_as_candidate_claim",
        },
    ]


def triage_decision(a0: dict[str, Any], a1: dict[str, Any], v21: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    evidence = v21_evidence(v21)
    checks = {
        "a0_rows_present": int(a0.get("v1_v22_rows", 0)) > 0,
        "a1_paired_rows_present": int(a1.get("paired_rows", 0)) > 0,
        "v21_heterogeneity": bool(evidence["heterogeneous_signs"]),
        "matched_counterfactual_missing": not evidence["formal_random_counterfactual_present"] and not evidence["formal_none_arm_present"],
        "historical_artifact_audit_ok": bool(v21.get("audit_ok")),
    }
    # The historical V21 protocol is explicitly development-selected.  That is
    # not a reason to cancel E1; it is the reason E1 must be prospectively
    # matched.  Keep this check descriptive rather than treating it as causal.
    checks["historical_v21_artifact_complete"] = bool(v21.get("audit_ok")) and int(v21.get("completed_valid_jobs", 0)) == int(v21.get("expected_jobs", -1))
    if all(checks[key] for key in ("a0_rows_present", "a1_paired_rows_present", "v21_heterogeneity", "matched_counterfactual_missing", "historical_v21_artifact_complete")):
        decision = "retain_e1"
        reason = "V21 has auditable sign heterogeneity and the missing matched N/R/T counterfactual is identifiable; preserve the historical label-selected status as a limitation."
    elif checks["a0_rows_present"] and checks["a1_paired_rows_present"]:
        decision = "cancel_e1"
        reason = "Retrospective evidence is present but the predeclared V21 identifiability gate is not satisfied; do not invent E4."
    else:
        decision = "no_prospective_compute"
        reason = "The historical evidence registry or atlas is unavailable/incomplete; close V25 without new training."
    return decision, {"checks": checks, "reason": reason, "v21_evidence": evidence}


def write_markdown(out: Path, decision: str, details: dict[str, Any], claims: list[dict[str, Any]], holdout: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# V25 A2 Mechanism Triage",
        "",
        f"Decision: **`{decision}`**",
        "",
        details["reason"],
        "",
        "A2 has veto authority. If the decision is not `retain_e1`, no E4 or replacement prospective experiment is permitted.",
        "",
        "## Gate checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, result in details["checks"].items():
        lines.append(f"| `{name}` | `{result}` |")
    lines += [
        "",
        "## Claim-evidence matrix",
        "",
        "| Hypothesis | Identifiability | Required compute | Status | Fatal falsifier |",
        "|---|---|---|---|---|",
    ]
    for row in claims:
        lines.append(f"| `{row['hypothesis_id']}` | {row['identifiability']} | {row['required_new_compute']} | {row['status']} | {row['fatal_falsifier']} |")
    lines += [
        "",
        "## Frozen measurement schema",
        "",
        "Primary readout is clean embedding plus known-K KMeans; Student-t is secondary. Holdout activation is claim-dependent and may not add a new endpoint after Claim Freeze.",
        "",
        "- `delta_threshold`: `0.03` (primary; sensitivity values `0.02` and `0.05` are descriptive only).",
        "- E1 selection endpoint: `S_full_ARI = ARI_T - ARI_R`; generic intervention endpoint: `I_full_ARI = ARI_R - ARI_N`.",
        "- E3 endpoint: `local_positive_and_global_nonpositive`; kNN label purity is post-hoc supervised geometry.",
        "- E2-C endpoint: sign agreement between `S_1step` and `S_full`.",
        "",
        "## Holdout contract",
        "",
        f"- Candidate pool is frozen before E1 outcomes: `{holdout['selection_policy']['frozen_before_e1_outcomes']}`.",
        f"- Eligible candidates by domain: `{json.dumps(holdout['eligible_counts'], sort_keys=True)}`; target counts: `{json.dumps(holdout['target_counts'], sort_keys=True)}`.",
        f"- Pool shortfall is recorded, not silently filled: `{json.dumps(holdout['candidate_pool_shortfall'], sort_keys=True)}`.",
        "- Adapter validity checks inspect source existence and frozen input protocol only; they never inspect ARI or other outcomes.",
        "",
        "## Scope boundary",
        "",
        "V1-V22 support a retrospective observational claim. V23/V24 remain boundary evidence. Any E1 result is a V21 case study and cannot be promoted to a universal population claim without a separate predeclared replication.",
        "",
    ]
    (out / "A2_DECISION.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0", type=Path, default=DEFAULT_A0)
    parser.add_argument("--a1", type=Path, default=DEFAULT_A1)
    parser.add_argument("--v21", type=Path, default=DEFAULT_V21)
    parser.add_argument("--manifest", type=Path, action="append", dest="manifests")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    a0 = read_json(args.a0 / "registry_summary.json")
    a1 = read_json(args.a1 / "a1_summary.json")
    v21 = read_json(args.v21)
    manifests = tuple(args.manifests or DEFAULT_MANIFESTS)
    claims = build_claim_matrix(a0, a1, v21)
    decision, details = triage_decision(a0, a1, v21)
    holdout = build_holdout_manifest(manifests, ROOT)
    measurement = {
        "schema_id": "v25_measurement_schema_v1",
        "delta_threshold": DELTA,
        "threshold_sensitivity": [0.02, 0.03, 0.05],
        "primary_readout": "clean_embedding_known_k_kmeans",
        "secondary_readout": "student_t_training_head",
        "claim_activation": {
            "selection": ["E1_NRT"],
            "generic_intervention": ["E1_NRT"],
            "feature_semantics": ["E1_NRT", "E2_A"],
            "objective_compatibility": ["E1_NRT", "E2_B", "E2_C"],
            "local_global": ["E3_frozen_matched_pair"],
        },
        "post_treatment_metrics": ["effective_corruption", "embedding_drift", "final_neighborhood"],
        "coordinate_inference_unit": "dataset_seed_summary",
        "equivalence_claim_allowed": False,
    }
    summary = {
        "protocol_id": "v25_a2_mechanism_triage_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "a0_protocol_id": a0.get("protocol_id"),
        "a1_protocol_id": a1.get("protocol_id"),
        "v21_source_sha256": sha256_file(args.v21),
        "holdout_manifest_id": holdout["manifest_id"],
        "no_new_e4": True,
        "labels_used_for_triage": False,
        "details": details,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "CLAIM_EVIDENCE_MATRIX.csv", claims)
    (args.out / "holdout_candidate_manifest.json").write_text(json.dumps(holdout, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "measurement_schema.json").write_text(json.dumps(measurement, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "A2_decision.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.out, decision, details, claims, holdout, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
