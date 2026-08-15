#!/usr/bin/env python3
"""Build the final, weight-free V25 closure artifacts.

The V25 result directory contains large training artifacts that are useful for
local auditing but are not suitable for publication in the source repository.
This module consumes only the audited CSV/JSON summaries and emits compact,
deterministic tables and Markdown decisions:

* ``V25_GAP_MAP.csv/.md``
* ``failure_localization_taxonomy.csv``
* ``E1_MECHANISM_SUMMARY.csv``
* ``V25_NEXT_SERIES_DECISION.md``

No model is loaded, no label is used for fitting, and no metric is estimated
from a checkpoint.  The statistical unit remains the unit declared by the
source artifact (dataset/protocol for A1 and dataset with seeds as repeated
measurements for E1/E2).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "v25_closure_artifacts_v1"
E1_DATASETS = (
    "cnae9",
    "Mouse_retina",
    "sms_spam_collection",
    "Baron Human",
    "Campbell",
    "hate_speech",
)
SEEDS = (7, 42, 123)
ALLOWED_STAGES = {
    "Opportunity",
    "Selection",
    "Intervention",
    "Representation",
    "Readout",
    "Unidentified",
    "Boundary",
}
TAXONOMY_COLUMNS = [
    "version",
    "primary_stage",
    "secondary_stage",
    "classification_status",
    "evidence_scope",
    "causal_status",
    "confidence",
    "failure_pattern",
    "existing_evidence",
    "v25_established",
    "remaining_unknown",
    "alternative_explanation",
    "evidence_source",
    "do_new_experiment",
    "suggested_next_v",
    "priority",
]
GAP_COLUMNS = [
    "gap_id",
    "Gap",
    "mechanism_stage",
    "evidence_scope",
    "causal_status",
    "confidence",
    "Existing evidence",
    "What V25 established",
    "What remains unknown",
    "Why it matters",
    "Do we need new experiment?",
    "Suggested next V",
    "Priority",
    "closure_status",
    "alternative_explanation",
]
E1_COLUMNS = [
    "dataset",
    "phase",
    "i_d",
    "i_state",
    "i_seed_values",
    "i_same_sign_count",
    "s_d",
    "s_state",
    "s_seed_values",
    "s_same_sign_count",
    "seed_count",
    "statistical_unit",
    "audit_ok",
    "e2_status",
    "e2_panel_count",
    "e2_gradient_rows",
    "e2_topology_deviation_mean_difference",
    "e2_topology_dispersion_mean_difference",
    "e2_donor_change_magnitude_mean_difference",
    "e2_fisher_separation_mean_difference",
    "e2_class_support_enrichment_mean_difference",
    "e2_raw_support_frequency_mean_difference",
    "e2_model_variance_mean_difference",
    "e2_model_support_frequency_mean_difference",
    "e2_posthoc_label_metrics_not_fit_inputs",
    "one_step_available",
    "causal_status",
    "notes",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def _number(value: float | None) -> str | float:
    return "" if value is None else value


def _normalise_version(value: str) -> str:
    """Map source labels such as V09 to the compact taxonomy label V9."""
    if not value.startswith("V"):
        return value
    suffix = value[1:]
    if "." in suffix:
        major, minor = suffix.split(".", 1)
        return f"V{int(major)}.{minor}"
    try:
        return f"V{int(suffix)}"
    except ValueError:
        return value


def _version_order() -> list[str]:
    versions = [f"V{i}" for i in range(1, 25)]
    versions.insert(16, "V16.1")
    return versions


def _mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def _phase_summaries(v25_root: Path) -> list[tuple[str, dict[str, Any]]]:
    summaries: list[tuple[str, dict[str, Any]]] = []
    for phase in ("pilot", "confirmation"):
        path = v25_root / "E1" / phase / "Audit" / "phase_summary.json"
        if path.is_file():
            summaries.append((phase, _read_json(path)))
    if not summaries:
        raise FileNotFoundError("no audited E1 phase_summary.json found")
    return summaries


def _load_e2(v25_root: Path) -> tuple[dict[tuple[str, str], float], Counter[str], dict[str, int]]:
    """Load confirmation E2-A summaries at dataset x seed unit.

    The raw E2 audit is the source of the dataset-level means.  Coordinate
    counts are intentionally not returned as inferential sample sizes.
    """
    audit_path = v25_root / "E1" / "confirmation" / "Audit" / "e2_feature_audit.json"
    gradient_path = v25_root / "E1" / "confirmation" / "Audit" / "gradient_probe.csv"
    differences: dict[tuple[str, str], list[float]] = defaultdict(list)
    panel_counts: Counter[str] = Counter()
    if audit_path.is_file():
        payload = _read_json(audit_path)
        for audit in payload.get("audits", []):
            if audit.get("audit_ok") is not True:
                continue
            dataset = str(audit.get("dataset_id"))
            panel_counts[dataset] += 1
            for metric, metric_payload in audit.get("metrics", {}).items():
                difference = metric_payload.get("difference")
                if difference is not None:
                    differences[(dataset, metric)].append(float(difference))
    gradient_counts: Counter[str] = Counter()
    if gradient_path.is_file():
        for row in _read_csv(gradient_path):
            gradient_counts[row.get("dataset", "")] += 1
    means = {
        key: statistics.mean(values)
        for key, values in differences.items()
        if values
    }
    return means, panel_counts, dict(gradient_counts)


def _load_one_step_availability(v25_root: Path) -> dict[tuple[str, str], bool]:
    result: dict[tuple[str, str], bool] = {}
    for phase in ("pilot", "confirmation"):
        path = v25_root / "E1" / phase / "Audit" / "pair_effects.csv"
        if not path.is_file():
            continue
        for row in _read_csv(path):
            dataset = row.get("dataset", "")
            values = (_float(row.get("I_1step_ARI")), _float(row.get("S_1step_ARI")))
            result[(phase, dataset)] = all(value is not None for value in values)
    return result


def build_e1_summary(v25_root: Path) -> list[dict[str, Any]]:
    e2_means, e2_panels, gradient_counts = _load_e2(v25_root)
    one_step = _load_one_step_availability(v25_root)
    rows: list[dict[str, Any]] = []
    e2_fields = {
        "topology_deviation": "e2_topology_deviation_mean_difference",
        "topology_dispersion": "e2_topology_dispersion_mean_difference",
        "donor_change_magnitude": "e2_donor_change_magnitude_mean_difference",
        "fisher_separation_posthoc": "e2_fisher_separation_mean_difference",
        "class_support_enrichment_posthoc": "e2_class_support_enrichment_mean_difference",
        "raw_support_frequency": "e2_raw_support_frequency_mean_difference",
        "model_variance": "e2_model_variance_mean_difference",
        "model_support_frequency": "e2_model_support_frequency_mean_difference",
    }
    seen: set[str] = set()
    for phase, summary in _phase_summaries(v25_root):
        for dataset in sorted(summary.get("datasets", {})):
            if dataset in seen:
                raise ValueError(f"dataset appears in multiple E1 phases: {dataset}")
            seen.add(dataset)
            payload = summary["datasets"][dataset]
            i_effect = payload.get("I_d", {})
            s_effect = payload.get("S_d", {})
            confirmation_e2 = phase == "confirmation" and e2_panels.get(dataset, 0) > 0
            row: dict[str, Any] = {
                "dataset": dataset,
                "phase": phase,
                "i_d": _number(_float(i_effect.get("mean"))),
                "i_state": i_effect.get("state", ""),
                "i_seed_values": _json_value(i_effect.get("seed_values", [])),
                "i_same_sign_count": i_effect.get("same_sign_count", ""),
                "s_d": _number(_float(s_effect.get("mean"))),
                "s_state": s_effect.get("state", ""),
                "s_seed_values": _json_value(s_effect.get("seed_values", [])),
                "s_same_sign_count": s_effect.get("same_sign_count", ""),
                "seed_count": payload.get("seeds", []).__len__(),
                "statistical_unit": payload.get("statistical_unit", "dataset; seeds are repeated measurements"),
                "audit_ok": True,
                "e2_status": "confirmation_only" if confirmation_e2 else "deferred",
                "e2_panel_count": e2_panels.get(dataset, 0) if confirmation_e2 else 0,
                "e2_gradient_rows": gradient_counts.get(dataset, 0) if confirmation_e2 else 0,
                "e2_posthoc_label_metrics_not_fit_inputs": confirmation_e2,
                "one_step_available": one_step.get((phase, dataset), False),
                "causal_status": "matched prospective case study",
                "notes": (
                    "E2-A was not replayed for pilot panels because historical pilot artifacts did not save "
                    "feature-selection counts."
                    if not confirmation_e2
                    else "E2-A coordinate distributions are descriptive; dataset x seed is the inferential unit."
                ),
            }
            for metric, field in e2_fields.items():
                row[field] = _number(e2_means.get((dataset, metric))) if confirmation_e2 else ""
            rows.append(row)
    if set(seen) != set(E1_DATASETS):
        raise ValueError(f"expected exactly six E1 datasets, got {sorted(seen)}")
    return sorted(rows, key=lambda row: (E1_DATASETS.index(row["dataset"]), row["phase"]))


def _a1_version_rows(v25_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    a0 = _read_csv(v25_root / "A0" / "mechanism_evidence_registry.csv")
    a1 = _read_csv(v25_root / "A1" / "failure_atlas.csv")
    return a0, a1


def _taxonomy_row(
    version: str,
    a0_rows: list[dict[str, str]],
    a1_rows: list[dict[str, str]],
    e1_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if version == "V23":
        primary = [r for r in boundary_rows if r.get("version") == "V23"]
        return {
            "version": version,
            "primary_stage": "Boundary",
            "secondary_stage": "Representation",
            "classification_status": "closed_no_go",
            "evidence_scope": "V23 boundary evidence; isolated from V1-V22 atlas",
            "causal_status": "boundary evidence",
            "confidence": "moderate",
            "failure_pattern": "dependency-specific response explanation did not pass the conditional null",
            "existing_evidence": f"{len(primary)} isolated V23 boundary record(s); local kNN/non-positive global rows retained separately",
            "v25_established": "Cycle is a No-Go as a dependency-specific mechanism; this does not test all response utility.",
            "remaining_unknown": "Whether another response estimator or representation could identify a useful signal.",
            "alternative_explanation": "support statistics or generic perturbation response may explain the observed local changes",
            "evidence_source": "A0/V23_V24_boundary_evidence.csv; A1/local_global_boundary.csv",
            "do_new_experiment": "No under V25 closure",
            "suggested_next_v": "none; permanently closed for this study",
            "priority": "P1",
        }
    if version == "V24":
        primary = [r for r in boundary_rows if r.get("version") == "V24"]
        return {
            "version": version,
            "primary_stage": "Boundary",
            "secondary_stage": "Readout",
            "classification_status": "closed_no_go",
            "evidence_scope": "V24 calibration boundary evidence; not an intervention efficacy result",
            "causal_status": "boundary evidence",
            "confidence": "high",
            "failure_pattern": "conditional-response estimator calibration No-Go",
            "existing_evidence": f"{len(primary)} isolated V24 calibration record(s); weak-alternative power was zero",
            "v25_established": "The frozen V24 estimator/alternative pair was not sufficiently calibrated or powered.",
            "remaining_unknown": "No claim about Cycle efficacy or other response estimators follows from this calibration failure.",
            "alternative_explanation": "estimator/alternative mismatch and weak-signal power, not necessarily absent utility",
            "evidence_source": "A0/V23_V24_boundary_evidence.csv",
            "do_new_experiment": "No under V25 closure",
            "suggested_next_v": "none; calibration rescue is permanently closed",
            "priority": "P1",
        }

    source_rows = [r for r in a0_rows if _normalise_version(r.get("version", "")) == version]
    atlas_rows = [r for r in a1_rows if _normalise_version(r.get("version", "")) == version]
    if not source_rows:
        return {
            "version": version,
            "primary_stage": "Unidentified",
            "secondary_stage": "Unidentified",
            "classification_status": "unresolved",
            "evidence_scope": "No auditable A0 registry rows for this version in the current V25 source set",
            "causal_status": "unidentified",
            "confidence": "low",
            "failure_pattern": "no current quantitative evidence",
            "existing_evidence": "Historical notes may exist, but no quantitative row is in the audited V25 registry.",
            "v25_established": "V25 does not retroactively reconstruct missing version evidence.",
            "remaining_unknown": "Opportunity, selection, intervention, representation, and readout behavior remain unknown.",
            "alternative_explanation": "missing source/artifact coverage; absence of a registry row is not a null result",
            "evidence_source": "A0/mechanism_evidence_registry.csv (no matching rows); historical notes not used as quantitative proof",
            "do_new_experiment": "No under V25 closure",
            "suggested_next_v": "none; archival gap only",
            "priority": "P3",
        }

    structural = Counter(r.get("structural_source", "") for r in source_rows).most_common(1)[0][0]
    locations = Counter(r.get("intervention_location", "") for r in source_rows if r.get("intervention_location")).most_common()
    location = locations[0][0] if locations else "unknown"
    effects = Counter(r.get("effect_state", "") for r in atlas_rows)
    paired = len(atlas_rows)
    if version == "V21":
        e1_effects = [r for r in e1_rows if r.get("dataset") in E1_DATASETS]
        s_states = Counter(r.get("s_state", "") for r in e1_effects)
        primary_stage, secondary_stage = "Selection", "Intervention"
        classification = "partially_resolved"
        causal_status = "matched prospective"
        confidence = "moderate"
        established = (
            "The matched V21 case study identifies conditional, sign-heterogeneous topology-selection utility "
            f"(S states: {dict(s_states)}); it does not establish universal topology superiority."
        )
        remaining = "Independent holdout replication and a label-free explanation of the sign remain unknown."
        alternative = "generic intervention, optimizer/readout effects, and dataset compatibility may contribute to sign heterogeneity"
        next_v = "gated external replication only; not authorized by V25"
        priority = "P1"
    elif version in {"V20", "V22"}:
        primary_stage, secondary_stage = "Selection", "Intervention"
        classification = "partially_resolved" if paired else "unresolved"
        causal_status, confidence = "observational", "low"
        established = "V25 records feature-coordinate selection/corruption outcomes in the observational atlas; the causal failure stage is not identified."
        remaining = "Whether the selector damages cluster-defining coordinates, and whether representation or readout converts that damage, remains unknown."
        alternative = "dose, baseline ceiling, objective compatibility, and readout confounding"
        next_v = "none; future replication would require a separately frozen claim"
        priority = "P2"
    else:
        primary_stage, secondary_stage = "Intervention", "Selection"
        classification = "partially_resolved" if paired else "unresolved"
        causal_status, confidence = "observational", "low"
        established = "V25 places this version in the retrospective atlas and preserves its paired outcome pattern without assigning a causal mechanism."
        remaining = "The relative contributions of opportunity, policy, intervention dose, representation, and readout remain unknown."
        alternative = "protocol, preprocessing, baseline, budget, optimizer, and readout confounding"
        next_v = "none; archival evidence is closed unless a future claim is independently predeclared"
        priority = "P2"
    return {
        "version": version,
        "primary_stage": primary_stage,
        "secondary_stage": secondary_stage,
        "classification_status": classification,
        "evidence_scope": "V1-V22 quantitative Failure Atlas",
        "causal_status": causal_status,
        "confidence": confidence,
        "failure_pattern": f"paired outcome mix: positive={effects.get('positive', 0)}, negative={effects.get('negative', 0)}, observed-small={effects.get('observed-small', 0)}, rows={paired}",
        "existing_evidence": f"A0 rows={len(source_rows)}; A1 paired atlas rows={paired}; dominant structural source={structural}; dominant intervention={location}",
        "v25_established": established,
        "remaining_unknown": remaining,
        "alternative_explanation": alternative,
        "evidence_source": "A0/mechanism_evidence_registry.csv; A1/failure_atlas.csv",
        "do_new_experiment": "No under V25 closure",
        "suggested_next_v": next_v,
        "priority": priority,
    }


def build_taxonomy(v25_root: Path, e1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a0_rows, a1_rows = _a1_version_rows(v25_root)
    boundary_rows = _read_csv(v25_root / "A0" / "V23_V24_boundary_evidence.csv")
    rows = [
        _taxonomy_row(version, a0_rows, a1_rows, e1_rows, boundary_rows)
        for version in _version_order()
    ]
    for row in rows:
        if row["primary_stage"] not in ALLOWED_STAGES or row["secondary_stage"] not in ALLOWED_STAGES:
            raise ValueError(f"invalid taxonomy stage: {row}")
    return rows


def _atlas_facts(v25_root: Path) -> dict[str, Any]:
    a0 = _read_json(v25_root / "A0" / "registry_summary.json")
    a1 = _read_json(v25_root / "A1" / "a1_summary.json")
    a2 = _read_json(v25_root / "A2" / "A2_decision.json")
    closure = _read_json(v25_root / "PhaseE" / "closure.json")
    holdout = closure.get("independent_holdout", {})
    pilot = _read_json(v25_root / "E1" / "pilot" / "Audit" / "phase_summary.json")
    confirmation = _read_json(v25_root / "E1" / "confirmation" / "Audit" / "phase_summary.json")
    return {
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "closure": closure,
        "holdout": holdout,
        "pilot": pilot,
        "confirmation": confirmation,
    }


def build_gap_map(v25_root: Path, e1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts = _atlas_facts(v25_root)
    a0, a1 = facts["a0"], facts["a1"]
    e1_s = {row["dataset"]: row for row in e1_rows}
    s_values = [float(row["s_d"]) for row in e1_rows]
    i_values = [float(row["i_d"]) for row in e1_rows]
    holdout = facts["holdout"]
    e2_rows = [row for row in e1_rows if row["e2_status"] == "confirmation_only"]
    e2_gradient_rows = sum(int(row["e2_gradient_rows"] or 0) for row in e2_rows)
    rows: list[dict[str, Any]] = [
        {
            "gap_id": "G01",
            "Gap": "Structural opportunity versus intervention utility",
            "mechanism_stage": "Opportunity -> Intervention",
            "evidence_scope": "A0/A1 V1-V22 retrospective atlas",
            "causal_status": "observational",
            "confidence": "moderate",
            "Existing evidence": f"A0 has {a0['v1_v22_paired_rows']} paired rows across {a0['v1_v22_units']} units; A1 has {a1['positive_rows']} material positive, {a1['negative_rows']} material negative, and {a1['small_rows']} observed-small rows.",
            "What V25 established": "Structural quality/opportunity descriptors do not reliably imply a positive downstream intervention gain in the audited historical records.",
            "What remains unknown": "Whether the heterogeneity is caused by structure quality, protocol compatibility, intervention policy, or readout differences.",
            "Why it matters": "This is the retrospective paper-level problem statement, but it cannot by itself support a pooled causal mechanism.",
            "Do we need new experiment?": "No for the retrospective claim; only a separately authorized replication can test causality.",
            "Suggested next V": "Gated external replication only",
            "Priority": "P0",
            "closure_status": "partially_resolved",
            "alternative_explanation": "protocol, preprocessing, baseline, budget, optimizer, and readout confounding",
        },
        {
            "gap_id": "G02",
            "Gap": "Topology-dependent selection increment",
            "mechanism_stage": "Selection",
            "evidence_scope": "E1 audited V21 pilot plus confirmation; six datasets x three seeds",
            "causal_status": "matched prospective case study",
            "confidence": "moderate",
            "Existing evidence": "The N/R/T protocol passed 18/18 panel audits. S_d is positive for Baron Human (+0.044617) and negative for Campbell (-0.065332) and hate_speech (-0.033410); pilot effects are also heterogeneous.",
            "What V25 established": "Topology-dependent selection has conditional, sign-heterogeneous incremental utility in this audited V21 case study, not universal superiority.",
            "What remains unknown": "Whether the sign pattern survives independent datasets under the frozen endpoint; the holdout produced no evaluable panel.",
            "Why it matters": "It separates topology policy value from generic intervention value through S_d = Q(T)-Q(R).",
            "Do we need new experiment?": "Only a separately authorized frozen replication; V25 itself is closed.",
            "Suggested next V": "Gated external replication of the frozen S_full_ARI endpoint",
            "Priority": "P0",
            "closure_status": "partially_resolved",
            "alternative_explanation": "generic intervention, optimizer/head state, dataset compatibility, or readout may contribute to sign heterogeneity",
        },
        {
            "gap_id": "G03",
            "Gap": "Generic intervention effect versus topology selection",
            "mechanism_stage": "Intervention",
            "evidence_scope": "E1 N/R/T decomposition",
            "causal_status": "matched prospective case study",
            "confidence": "moderate",
            "Existing evidence": f"I_d ranges from {min(i_values):+.6f} to {max(i_values):+.6f} across the six audited datasets; Campbell and SMS show positive material I_d while other datasets are small/inconclusive.",
            "What V25 established": "Generic intervention and topology selection are empirically separable quantities: Q(T)-Q(N) = I_d + S_d.",
            "What remains unknown": "Which training or dose components explain I_d sign and whether I_d generalizes beyond V21.",
            "Why it matters": "Without this decomposition, historical TopoGate gains cannot be attributed to topology-aware selection.",
            "Do we need new experiment?": "No additional V25 experiment; future work would need a predeclared control claim.",
            "Suggested next V": "None under V25; reuse I_d only in a separately frozen replication",
            "Priority": "P1",
            "closure_status": "partially_resolved",
            "alternative_explanation": "InfoMax, optimizer state, budget, and readout can alter I_d independently of topology",
        },
        {
            "gap_id": "G04",
            "Gap": "Feature semantics of selected coordinates",
            "mechanism_stage": "Selection -> Representation",
            "evidence_scope": "E2-A confirmation diagnostics only; 30 metric-summary rows / 90 dataset-seed metric rows",
            "causal_status": "observational",
            "confidence": "low",
            "Existing evidence": f"E2-A covers {len(e2_rows)} confirmation datasets, 9 dataset-seed panels, 30 metric-summary rows, and {e2_gradient_rows} gradient rows; coordinate distributions are descriptive and post-hoc label metrics are not fit inputs.",
            "What V25 established": "Selection semantics can be summarized at dataset x seed level without treating millions of coordinates as independent observations.",
            "What remains unknown": "Whether selected coordinates are nuisance-sensitive or cluster-defining in a way that explains full-training sign.",
            "Why it matters": "This is the most direct candidate explanation for positive versus harmful topology interventions.",
            "Do we need new experiment?": "No new model under V25; a future claim would require a frozen, label-free diagnostic and replication.",
            "Suggested next V": "Gated diagnostic replication only",
            "Priority": "P1",
            "closure_status": "partially_resolved",
            "alternative_explanation": "selection scores may correlate with generic magnitude or sparse support rather than semantic utility",
        },
        {
            "gap_id": "G05",
            "Gap": "Objective compatibility and gradient conflict",
            "mechanism_stage": "Intervention -> Representation",
            "evidence_scope": "E2-B/C confirmation diagnostics",
            "causal_status": "diagnostic only",
            "confidence": "low",
            "Existing evidence": f"Gradient geometry and actual Adam N/R/T one-step artifacts are available for the confirmation panels ({e2_gradient_rows} gradient rows).",
            "What V25 established": "The protocol can measure objective geometry and actual optimizer counterfactuals at the shared branchpoint.",
            "What remains unknown": "Whether objective compatibility predicts the full-training sign of S_d or I_d.",
            "Why it matters": "A plausible gradient conflict is not a mechanism until it predicts the observed sign under the real Adam state.",
            "Do we need new experiment?": "No under V25 closure; do not promote diagnostics into a causal law.",
            "Suggested next V": "None; only a predeclared objective-compatibility replication could reopen this gap",
            "Priority": "P2",
            "closure_status": "unresolved",
            "alternative_explanation": "one-step geometry may not capture multi-epoch schedule, optimizer, or representation effects",
        },
        {
            "gap_id": "G06",
            "Gap": "Representation change and local geometry conversion",
            "mechanism_stage": "Representation",
            "evidence_scope": "A1 replay gate plus isolated V23 boundary rows",
            "causal_status": "observational boundary evidence",
            "confidence": "low",
            "Existing evidence": f"A1 has {a1['artifact_complete_replay_candidate_rows']} artifact-complete replay candidates; V23 retains six local/global boundary comparisons outside the atlas.",
            "What V25 established": "No artifact-complete V1-V22 replay was available; a local metric increase can coexist with a non-positive global metric in a bounded V23 example.",
            "What remains unknown": "Whether the same disconnect recurs under label-free geometry and matched structural interventions.",
            "Why it matters": "Local improvement is not sufficient evidence of global cluster recovery.",
            "Do we need new experiment?": "No V25 replay is possible without artifacts; future replication must freeze both local and global endpoints.",
            "Suggested next V": "Gated local-to-global replication only",
            "Priority": "P1",
            "closure_status": "partially_resolved",
            "alternative_explanation": "post-hoc metric choice and readout mismatch may create the apparent disconnect",
        },
        {
            "gap_id": "G07",
            "Gap": "Readout conversion and clean clustering endpoint",
            "mechanism_stage": "Readout",
            "evidence_scope": "E1 primary clean-embedding known-K KMeans; Student-t secondary",
            "causal_status": "matched prospective case study",
            "confidence": "moderate",
            "Existing evidence": "E1 fixes clean embedding + known-K KMeans as the primary readout and keeps Student-t head metrics secondary.",
            "What V25 established": "Readout semantics are explicit and shared across N/R/T, preventing the head from silently replacing the primary endpoint.",
            "What remains unknown": "How much historical sign heterogeneity is attributable to readout mismatch rather than intervention or representation.",
            "Why it matters": "A changing readout can make an apparent structural gain or failure non-comparable.",
            "Do we need new experiment?": "No under V25 closure; do not add readout sweeps after seeing endpoints.",
            "Suggested next V": "None; preserve the frozen readout contract",
            "Priority": "P2",
            "closure_status": "unresolved",
            "alternative_explanation": "readout may discard local geometry or react differently to latent scale",
        },
        {
            "gap_id": "G08",
            "Gap": "Independent external validation",
            "mechanism_stage": "All stages",
            "evidence_scope": "Phase D frozen claim-dependent holdout",
            "causal_status": "inconclusive_not_completed",
            "confidence": "high",
            "Existing evidence": f"The frozen holdout expected {holdout.get('expected_panel_count', 6)} panels and completed {holdout.get('completed_panel_count', 0)}; status={holdout.get('status', 'inconclusive_not_completed')}.",
            "What V25 established": "No independent replication result was produced; this is not a negative performance result.",
            "What remains unknown": "Whether the conditional V21 selection claim transfers to new domains under the frozen adapter and endpoint.",
            "Why it matters": "The six audited E1 datasets are an internal case study, not a population validation set.",
            "Do we need new experiment?": "Only if a separate project authorizes the frozen adapter/resource contract; V25 does not reopen it.",
            "Suggested next V": "Gated external replication, not an automatic V26",
            "Priority": "P0",
            "closure_status": "inconclusive_not_completed",
            "alternative_explanation": "resource boundary prevented evaluation; no model conclusion follows",
        },
        {
            "gap_id": "G09",
            "Gap": "Label-free do-no-harm diagnostic",
            "mechanism_stage": "Opportunity -> Selection",
            "evidence_scope": "A2 triage and E1/E2 diagnostics",
            "causal_status": "unidentified",
            "confidence": "low",
            "Existing evidence": "A2 retained E1 but did not identify a validated label-free certificate; E2 diagnostics remain explanatory rather than predictive.",
            "What V25 established": "No current diagnostic is authorized to decide automatically whether structural intervention should abstain.",
            "What remains unknown": "Whether any label-free quantity predicts harm robustly across domains and protocols.",
            "Why it matters": "A do-no-harm certificate would be more actionable than another unconstrained Gate architecture.",
            "Do we need new experiment?": "No under V25 closure; a future attempt must be separately identifiable and pre-registered.",
            "Suggested next V": "Only if a label-free diagnostic becomes identifiable before any new training",
            "Priority": "P2",
            "closure_status": "unresolved",
            "alternative_explanation": "available diagnostics may be post-treatment or label-dependent",
        },
        {
            "gap_id": "G10",
            "Gap": "Universal response surrogate and rescue architectures",
            "mechanism_stage": "Opportunity -> Readout",
            "evidence_scope": "V23/V24 boundary evidence and V25 closure governance",
            "causal_status": "boundary evidence",
            "confidence": "high",
            "Existing evidence": "V23 dependency-specific Cycle explanation is No-Go; V24 calibration is No-Go; V25 closure forbids new Gate/loss/selector/DCBoost/V18/V22/V24 rescue routes.",
            "What V25 established": "Continuing to search for a universal utility surrogate or rescue module is not justified by the current evidence.",
            "What remains unknown": "A different future project could formulate a new, independently identifiable question, but it is outside V25.",
            "Why it matters": "This closes the open-ended V9-V24 iteration loop and protects the paper claim boundary.",
            "Do we need new experiment?": "No; permanently closed within V25.",
            "Suggested next V": "None; any future work requires a new project and preregistered question",
            "Priority": "P0",
            "closure_status": "closed_no_go",
            "alternative_explanation": "No-Go applies to the frozen estimator/mechanism, not every possible structural signal",
        },
    ]
    return rows


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _gap_markdown(rows: list[dict[str, Any]], facts: dict[str, Any]) -> str:
    lines = [
        "# V25 Gap Map",
        "",
        "V25 is `V25_systematic_mechanism_study`, a systematic Failure Atlas and mechanism-localization study, not a new TopoGate architecture.",
        "The map distinguishes retrospective observation, matched prospective case-study evidence, boundary evidence, and incomplete holdout evaluation.",
        "",
        "## Frozen status",
        "",
        f"- A0: {facts['a0']['v1_v22_paired_rows']} paired rows, {facts['a0']['v1_v22_units']} dataset/protocol/readout units; repeated rows are not independent datasets.",
        f"- A1: {facts['a1']['positive_rows']} material positive, {facts['a1']['negative_rows']} material negative, {facts['a1']['small_rows']} observed-small; no artifact-complete replay candidate.",
        "- E1: audited V21 N/R/T case study with conditional, sign-heterogeneous `S_d`; no universal population claim.",
        f"- Holdout: {facts['holdout'].get('completed_panel_count', 0)}/{facts['holdout'].get('expected_panel_count', 6)} evaluable, `inconclusive_not_completed`.",
        "",
        "## Gap table",
        "",
        "| ID | Gap | Stage | Existing evidence | V25 established | Remains unknown | New experiment? | Next V | Status | Priority |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _md_escape(row[key])
                for key in (
                    "gap_id",
                    "Gap",
                    "mechanism_stage",
                    "Existing evidence",
                    "What V25 established",
                    "What remains unknown",
                    "Do we need new experiment?",
                    "Suggested next V",
                    "closure_status",
                    "Priority",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The atlas supports the observational statement `structural quality != intervention utility`; it does not identify a pooled causal effect across historical protocols. The only mechanistic prospective result is the audited V21 N/R/T case study. The Phase D resource failure produces no endpoint and must not be used as a negative result.",
            "",
            "The complete machine-readable fields, including causal status, confidence, alternative explanations, and closure status, are in `V25_GAP_MAP.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def _next_series_markdown(v25_root: Path, facts: dict[str, Any], e1_rows: list[dict[str, Any]]) -> str:
    closure = facts["closure"]
    confirmation = facts["confirmation"]
    lines = [
        "# V25 Next-Series Decision",
        "",
        "## Decision",
        "",
        "V25 is closed as `V25_systematic_mechanism_study`. `close_without_v26` means that this study does not authorize V26, a new Gate, a new loss, a selector, DCBoost, or rescue training for V18, V22, V23, or V24.",
        "",
        "**V25 closure is not authorization to run V26.** Any future work requires a new project, a new predeclared question, and a new resource/adapter contract.",
        "",
        "## What V25 established",
        "",
        f"- Retrospective: {facts['a0']['v1_v22_paired_rows']} paired V1-V22 rows across {facts['a0']['v1_v22_units']} dataset/protocol/readout units show heterogeneous intervention outcomes ({facts['a1']['positive_rows']} material positive, {facts['a1']['negative_rows']} material negative, {facts['a1']['small_rows']} observed-small). This is observational evidence.",
        f"- E1: pilot and confirmation each passed {facts['pilot'].get('audit_ok_count', 0)}/{facts['pilot'].get('panel_count', 0)} and {confirmation.get('audit_ok_count', 0)}/{confirmation.get('panel_count', 0)} panel audits. Confirmation S_d is +0.044617 (Baron Human), -0.065332 (Campbell), and -0.033410 (hate_speech).",
        "- E2/E3: feature and objective diagnostics are localization evidence; no coordinate-level inferential sample size or universal objective-conflict law is claimed. E3 replay had zero artifact-complete candidates.",
        f"- Holdout: {closure['independent_holdout'].get('completed_panel_count', 0)}/{closure['independent_holdout'].get('expected_panel_count', 6)} panels completed; status is `inconclusive_not_completed`, not a negative result.",
        "",
        "## Permanently closed in V25",
        "",
        "- automatic V26 or V25-as-a-new-model reinterpretation",
        "- new TopoGate architecture, Gate, loss, selector, or open-ended utility sweep",
        "- V18 waterfall rescue and V22 discriminator rescue",
        "- V23 Cycle response rescue and V24 calibration rescue",
        "- DCBoost or any new method introduced solely to improve an unattractive V25 result",
        "",
        "## Gated future questions (not authorization)",
        "",
        "1. External replication is allowed only if the frozen adapter, preprocessing, primary endpoint, and resource contract are feasible before outcome inspection.",
        "2. A future do-no-harm diagnostic is allowed only if a label-free, pre-treatment quantity becomes identifiable and its falsifier is frozen before training.",
        "3. A new V26/V27 would be a separate paper/project only after an independently replicated claim, not a continuation of result hunting.",
        "",
        "## Publication boundary",
        "",
        "The defensible paper claims are: (i) an observational V1-V22 Failure Atlas, and (ii) conditional, sign-heterogeneous topology-selection utility in the audited V21 case study. The paper must not claim universal topology superiority, pooled historical causality, fully label-free E1 fitting, or independent holdout replication.",
        "",
        "## Closure artifacts",
        "",
        "- `V25_GAP_MAP.md` / `V25_GAP_MAP.csv`",
        "- `failure_localization_taxonomy.csv`",
        "- `E1_MECHANISM_SUMMARY.csv`",
        "- `V25_NEXT_SERIES_DECISION.md`",
        "- `V25_CLOSURE_ARTIFACTS.json` (source hashes and audit metadata)",
        "",
    ]
    return "\n".join(lines)


def build_closure_artifacts(v25_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    v25_root = Path(v25_root)
    output_dir = Path(output_dir) if output_dir is not None else v25_root
    output_dir.mkdir(parents=True, exist_ok=True)
    e1_rows = build_e1_summary(v25_root)
    taxonomy_rows = build_taxonomy(v25_root, e1_rows)
    gap_rows = build_gap_map(v25_root, e1_rows)
    facts = _atlas_facts(v25_root)

    _write_csv(output_dir / "E1_MECHANISM_SUMMARY.csv", e1_rows, E1_COLUMNS)
    _write_csv(output_dir / "failure_localization_taxonomy.csv", taxonomy_rows, TAXONOMY_COLUMNS)
    _write_csv(output_dir / "V25_GAP_MAP.csv", gap_rows, GAP_COLUMNS)
    (output_dir / "V25_GAP_MAP.md").write_text(_gap_markdown(gap_rows, facts), encoding="utf-8")
    (output_dir / "V25_NEXT_SERIES_DECISION.md").write_text(
        _next_series_markdown(v25_root, facts, e1_rows), encoding="utf-8"
    )

    source_paths = [
        v25_root / "A0" / "registry_summary.json",
        v25_root / "A0" / "mechanism_evidence_registry.csv",
        v25_root / "A0" / "V23_V24_boundary_evidence.csv",
        v25_root / "A1" / "a1_summary.json",
        v25_root / "A1" / "failure_atlas.csv",
        v25_root / "A2" / "A2_decision.json",
        v25_root / "E1" / "pilot" / "Audit" / "phase_summary.json",
        v25_root / "E1" / "confirmation" / "Audit" / "phase_summary.json",
        v25_root / "E1" / "confirmation" / "Audit" / "e2_feature_audit.json",
        v25_root / "PhaseE" / "closure.json",
    ]
    source_manifest = {
        str(path.relative_to(v25_root)): {"exists": path.is_file(), "sha256": _sha256(path) if path.is_file() else None}
        for path in source_paths
    }
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "study": "V25_systematic_mechanism_study",
        "source_root": str(v25_root),
        "output_root": str(output_dir),
        "outputs": [
            "V25_GAP_MAP.md",
            "V25_GAP_MAP.csv",
            "failure_localization_taxonomy.csv",
            "E1_MECHANISM_SUMMARY.csv",
            "V25_NEXT_SERIES_DECISION.md",
        ],
        "e1_dataset_count": len(e1_rows),
        "taxonomy_row_count": len(taxonomy_rows),
        "gap_row_count": len(gap_rows),
        "holdout_status": facts["holdout"].get("status", "inconclusive_not_completed"),
        "closure_decision": facts["closure"].get("closure_decision"),
        "a2_decision": facts["a2"].get("decision"),
        "weight_free": True,
        "source_manifest": source_manifest,
        "audit": {
            "e1_exactly_six_datasets": len(e1_rows) == 6,
            "taxonomy_covers_v1_v24": {f"V{i}": any(row["version"] == f"V{i}" for row in taxonomy_rows) for i in range(1, 25)},
            "taxonomy_stages_allowed": all(row["primary_stage"] in ALLOWED_STAGES and row["secondary_stage"] in ALLOWED_STAGES for row in taxonomy_rows),
            "holdout_not_negative": facts["holdout"].get("status") == "inconclusive_not_completed",
            "closure_without_v26": facts["closure"].get("closure_decision") == "close_without_v26",
        },
    }
    (output_dir / "V25_CLOSURE_ARTIFACTS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v25-root",
        type=Path,
        default=Path("result/V25_systematic_mechanism_study"),
        help="audited V25 result root (read-only inputs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory; defaults to --v25-root",
    )
    args = parser.parse_args()
    manifest = build_closure_artifacts(args.v25_root, args.output_dir)
    print(json.dumps({"output_root": manifest["output_root"], "outputs": manifest["outputs"], "audit": manifest["audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
