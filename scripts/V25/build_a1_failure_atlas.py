#!/usr/bin/env python3
"""Build the V25 retrospective Failure Atlas from the A0 registry.

The atlas is descriptive.  It uses dataset/protocol/readout units as the
historical comparison boundary, keeps seed/variant rows as repeated records,
and labels post-treatment quantities without turning them into causal claims.
V23 local/global diagnostics are emitted as boundary evidence and are never
pooled with the V1--V22 intervention rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A0 = ROOT / "result" / "V25_systematic_mechanism_study" / "A0"
DEFAULT_V23 = ROOT / "result" / "V23_cycle_response" / "m0_synthetic_protocol_a_v1" / "m0_decision.json"
DEFAULT_OUT = ROOT / "result" / "V25_systematic_mechanism_study" / "A1"

PAIR_DELTA = 0.03
NUMBER_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def number(value: Any) -> float | None:
    if value in (None, "", "NA", "None", "null", "nan", "NaN"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def unique_join(values: list[Any]) -> str:
    return "|".join(sorted({str(value) for value in values if value not in (None, "", "NA")}))


def parse_proxy_values(raw: Any) -> dict[str, float]:
    if not raw:
        return {}
    result: dict[str, float] = {}
    for key, value in NUMBER_RE.findall(str(raw)):
        parsed = number(value)
        if parsed is not None:
            result[key] = parsed
    return result


def effect_state(delta: float | None) -> str:
    if delta is None:
        return "unavailable"
    if delta > PAIR_DELTA:
        return "positive"
    if delta < -PAIR_DELTA:
        return "negative"
    return "observed-small"


def failure_pattern(delta: float | None) -> str:
    """Name an observational pattern without assigning a mechanism."""
    state = effect_state(delta)
    return {
        "positive": "observed_positive_intervention_gain",
        "negative": "observed_intervention_harm",
        "observed-small": "observed_small_effect",
        "unavailable": "mechanism_unidentified",
    }[state]


def baseline_bin(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value < 0.2:
        return "<0.2"
    if value < 0.4:
        return "0.2-0.4"
    if value < 0.6:
        return "0.4-0.6"
    if value < 0.8:
        return "0.6-0.8"
    return ">=0.8"


def key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row.get("version") or ""),
        str(row.get("source_batch") or ""),
        str(row.get("dataset_id") or ""),
        str(row.get("input_protocol") or ""),
        str(row.get("readout") or ""),
        str(row.get("seeds") or ""),
        str(row.get("variant") or ""),
    )


def paired_atlas_rows(registry: list[dict[str, str]]) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {key(row): row for row in registry}
    result: list[dict[str, Any]] = []
    for row in registry:
        if str(row.get("record_type")) != "intervention_record":
            continue
        delta = number(row.get("paired_delta_ari"))
        control = str(row.get("paired_control") or "")
        if delta is None or not control:
            continue
        control_key = (
            str(row.get("version") or ""),
            str(row.get("source_batch") or ""),
            str(row.get("dataset_id") or ""),
            str(row.get("input_protocol") or ""),
            str(row.get("readout") or ""),
            str(row.get("seeds") or ""),
            control,
        )
        control_row = lookup.get(control_key)
        control_ari = number(control_row.get("ari_mean")) if control_row else None
        current_ari = number(row.get("ari_mean"))
        proxies = parse_proxy_values(row.get("gate_usage"))
        result.append(
            {
                "version": row.get("version"),
                "source_batch": row.get("source_batch"),
                "dataset_id": row.get("dataset_id"),
                "variant_family": row.get("variant_family"),
                "variant": row.get("variant"),
                "control_variant": control,
                "input_protocol": row.get("input_protocol"),
                "readout": row.get("readout"),
                "status": row.get("status"),
                "evidence_level": row.get("evidence_level"),
                "seed_count": row.get("seed_count"),
                "seeds": row.get("seeds"),
                "current_ari": current_ari,
                "control_ari": control_ari,
                "delta_ari": delta,
                "effect_state": effect_state(delta),
                "failure_pattern": failure_pattern(delta),
                "confidence": "low_observational_summary",
                "evidence_source": "A0_audited_long_table",
                "baseline_bin": baseline_bin(control_ari),
                "positive_headroom": (1.0 - control_ari) if control_ari is not None else None,
                "negative_headroom": control_ari if control_ari is not None else None,
                "paired_delta_scope": row.get("paired_delta_scope"),
                "structural_source": row.get("structural_source"),
                "selection_policy": row.get("selection_policy"),
                "intervention_location": row.get("intervention_location"),
                "measurement_timing": "post_intervention",
                "causal_status": "observational",
                "artifact_status": row.get("artifact_status"),
                "replay_eligible": False,
                "gate_usage_raw": row.get("gate_usage"),
                "magnitude_proxy_json": json.dumps(proxies, sort_keys=True),
                "magnitude_proxy_keys": unique_join(list(proxies)),
                "alternative_explanation": row.get("alternative_explanation"),
                "post_treatment_descriptor": "paired/final ARI is downstream of the intervention; no causal direction inferred",
            }
        )
    return result


def artifact_complete_replay_rows(registry: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return only rows that explicitly pass the replay artifact gate.

    A source CSV or a completed summary is not enough.  Historical rows are
    therefore normally absent here, which is an auditable no-replay result,
    not an invitation to reconstruct embeddings from ARI.
    """
    eligible: list[dict[str, Any]] = []
    for row in registry:
        artifact_status = str(row.get("artifact_status") or "")
        replay_flag = str(row.get("replay_eligible") or "").lower() == "true"
        if artifact_status == "artifact_complete" and replay_flag:
            eligible.append(dict(row))
    return eligible


def group_summary(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field) or "") for field in fields)].append(row)
    result: list[dict[str, Any]] = []
    for group_key, items in sorted(groups.items()):
        deltas = [number(item.get("delta_ari")) for item in items]
        clean = [value for value in deltas if value is not None]
        result.append(
            {
                **{field: value for field, value in zip(fields, group_key)},
                "row_count": len(items),
                "dataset_count": len({item.get("dataset_id") for item in items}),
                "delta_mean": mean(deltas),
                "delta_std": statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None),
                "positive_count": sum(value is not None and value > PAIR_DELTA for value in deltas),
                "negative_count": sum(value is not None and value < -PAIR_DELTA for value in deltas),
                "observed_small_count": sum(value is not None and abs(value) <= PAIR_DELTA for value in deltas),
                "incomplete_count": sum(item.get("status") == "incomplete_compute" for item in items),
                "statistical_unit": "dataset/protocol/readout; seed and variant rows are repeated records",
            }
        )
    return result


def structural_opportunity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the opportunity field without pretending it is observed.

    The historical long table has no uniform fixed-graph-versus-null measure.
    Keep the stratification visible and mark the missing opportunity endpoint
    explicitly so a future artifact-complete replay cannot be confused with it.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("structural_source") or "unavailable")].append(row)
    result: list[dict[str, Any]] = []
    for source, items in sorted(groups.items()):
        deltas = [number(item.get("delta_ari")) for item in items]
        result.append(
            {
                "structural_source": source,
                "row_count": len(items),
                "dataset_count": len({item.get("dataset_id") for item in items}),
                "delta_mean": mean(deltas),
                "positive_count": sum(value is not None and value > PAIR_DELTA for value in deltas),
                "negative_count": sum(value is not None and value < -PAIR_DELTA for value in deltas),
                "opportunity_metric": "unavailable_from_audited_long_table",
                "opportunity_status": "not_identifiable_without_artifact_complete_control",
                "causal_status": "observational",
                "confidence": "low_observational_summary",
            }
        )
    return result


def magnitude_gain_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe available post-treatment magnitude strings by their raw keys."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        keys = str(row.get("magnitude_proxy_keys") or "unavailable")
        groups[(str(row.get("version") or ""), keys)].append(row)
    result: list[dict[str, Any]] = []
    for (version, keys), items in sorted(groups.items()):
        deltas = [number(item.get("delta_ari")) for item in items]
        result.append(
            {
                "version": version,
                "magnitude_proxy_keys": keys,
                "row_count": len(items),
                "dataset_count": len({item.get("dataset_id") for item in items}),
                "delta_mean": mean(deltas),
                "measurement_timing": "post_intervention_descriptor",
                "causal_status": "observational",
                "confidence": "low_observational_summary",
                "interpretation": "descriptor association only; magnitude is post-treatment and not a cause estimate",
            }
        )
    return result


def baseline_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("baseline_bin"))].append(row)
    result: list[dict[str, Any]] = []
    for bin_name, items in sorted(groups.items()):
        deltas = [number(item.get("delta_ari")) for item in items]
        clean = [value for value in deltas if value is not None]
        result.append(
            {
                "baseline_bin": bin_name,
                "row_count": len(items),
                "dataset_count": len({item.get("dataset_id") for item in items}),
                "baseline_mean": mean([number(item.get("control_ari")) for item in items]),
                "delta_mean": mean(deltas),
                "delta_std": statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None),
                "harm_below_minus_delta": sum(value is not None and value < -PAIR_DELTA for value in deltas),
                "positive_above_delta": sum(value is not None and value > PAIR_DELTA for value in deltas),
                "interpretation": "descriptive headroom-stratified association; not a causal effect of baseline strength",
            }
        )
    return result


def local_global_boundary(v23: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[tuple[str, dict[str, Any]]] = []
    positive = v23.get("dependency_positive_deltas") or {}
    groups.extend((name, value) for name, value in positive.items() if isinstance(value, dict))
    conditional = v23.get("conditional_null_cycle_minus_precycle")
    global_null = v23.get("global_null_cycle_minus_precycle")
    if isinstance(conditional, dict):
        groups.append(("conditional_null_cycle_minus_precycle", conditional))
    if isinstance(global_null, dict):
        groups.append(("global_null_cycle_minus_precycle", global_null))
    result: list[dict[str, Any]] = []
    for condition, values in groups:
        local = number(values.get("knn_purity_at_10_mean"))
        global_delta = number(values.get("ari_mean"))
        result.append(
            {
                "version": "V23",
                "condition": condition,
                "local_metric": "knn_purity_at_10",
                "local_delta": local,
                "global_metric": "ARI",
                "global_delta": global_delta,
                "local_positive": local is not None and local > 0.0,
                "global_nonpositive": global_delta is not None and global_delta <= 0.0,
                "local_global_disconnect": local is not None and global_delta is not None and local > 0.0 and global_delta <= 0.0,
                "local_metric_label_use": "posthoc_supervised",
                "label_free_local_metric_available": False,
                "measurement_timing": "post_intervention",
                "causal_status": "boundary_evidence",
                "confidence": "moderate_boundary_example",
                "evidence_source": "V23_M0_decision_artifact",
                "interpretation": "V23 boundary diagnostic; not pooled with V1-V22 atlas rows",
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    out: Path,
    paired: list[dict[str, Any]],
    family: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    opportunity: list[dict[str, Any]],
    magnitude: list[dict[str, Any]],
    local: list[dict[str, Any]],
    replay: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# V25 A1 Failure Atlas",
        "",
        "This report is descriptive and provenance-aware. It does not treat rows, seeds, variants, coordinates, or pair counts as independent population samples.",
        "",
        "The atlas imports `paired_delta_ari` and `ari_mean` from the audited historical table; it does not reload labels. This is an evidence-ingestion boundary, not a label-free evaluation claim: the original benchmark metrics may have used dataset labels, and no A1 row is re-evaluated here.",
        "",
        "## Scope",
        "",
        f"- V1-V22 paired records: `{summary['paired_rows']}`.",
        f"- V1-V22 dataset/protocol/readout units represented: `{summary['unit_count']}`.",
        f"- Positive (`Delta ARI > {PAIR_DELTA}`): `{summary['positive_rows']}`; negative (`Delta ARI < -{PAIR_DELTA}`): `{summary['negative_rows']}`; observed-small: `{summary['small_rows']}`.",
        "- V23 local/global rows are boundary evidence only and are not pooled into the intervention atlas.",
        "",
        "## Version/family summary",
        "",
        "| Version | Family | Rows | Datasets | Mean Delta ARI | Positive | Negative | Small |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in family:
        lines.append(
            f"| {row.get('version')} | {row.get('variant_family')} | {row.get('row_count')} | {row.get('dataset_count')} | {row.get('delta_mean') if row.get('delta_mean') is not None else 'NA'} | {row.get('positive_count')} | {row.get('negative_count')} | {row.get('observed_small_count')} |"
        )
    lines += [
        "",
        "## Baseline/headroom analysis",
        "",
        "The baseline table is a fixed-bin descriptive sensitivity. It cannot establish that a strong baseline causes intervention harm; ceiling/headroom remains a competing explanation.",
        "",
        "| Baseline bin | Rows | Datasets | Mean baseline | Mean Delta ARI | Harm count | Positive count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in baseline:
        lines.append(
            f"| {row.get('baseline_bin')} | {row.get('row_count')} | {row.get('dataset_count')} | {row.get('baseline_mean') if row.get('baseline_mean') is not None else 'NA'} | {row.get('delta_mean') if row.get('delta_mean') is not None else 'NA'} | {row.get('harm_below_minus_delta')} | {row.get('positive_above_delta')} |"
        )
    lines += [
        "",
        "## Structural opportunity and intervention magnitude",
        "",
        "The audited long table has no common fixed-graph/null opportunity endpoint. The opportunity table is therefore a stratified missingness record. Magnitude strings are post-treatment descriptors and cannot be interpreted as causes.",
        "",
        f"- Structural opportunity groups: `{len(opportunity)}`; uniform opportunity endpoint available: `False`.",
        f"- Magnitude descriptor groups: `{len(magnitude)}`; causal magnitude inference: `False`.",
        "",
        "## Artifact-complete replay gate",
        "",
        f"- Rows admitted to offline E3 replay: `{len(replay)}`.",
        "- Metadata-only rows are excluded. An empty replay set is an explicit boundary, not a reconstructed result.",
    ]
    lines += [
        "",
        "## Local/global boundary evidence",
        "",
        "These rows come from V23 M0 and are retained as a separate boundary branch.",
        "",
        "| Condition | Local delta (kNN purity@10) | Global delta (ARI) | Disconnect |",
        "|---|---:|---:|---|",
    ]
    for row in local:
        lines.append(f"| {row.get('condition')} | {row.get('local_delta')} | {row.get('global_delta')} | {row.get('local_global_disconnect')} |")
    lines += [
        "",
        "## Missing evidence boundary",
        "",
        "The long registry does not contain a uniform graph-quality, embedding-drift, or artifact-complete replay field across V1-V22. Those quantities remain unavailable rather than being reconstructed from ARI or gate strings. Gate strings are retained only as post-treatment magnitude proxies. V23 kNN purity is explicitly post-hoc supervised geometry; no label-free local metric is claimed.",
        "",
    ]
    (out / "FAILURE_ATLAS.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0", type=Path, default=DEFAULT_A0)
    parser.add_argument("--v23-json", type=Path, default=DEFAULT_V23)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_path = args.a0 / "mechanism_evidence_registry.csv"
    summary_path = args.a0 / "registry_summary.json"
    if not registry_path.is_file() or not summary_path.is_file() or not args.v23_json.is_file():
        raise FileNotFoundError("A0 registry, registry_summary.json, and V23 decision are required")
    registry = read_csv(registry_path)
    a0_summary = read_json(summary_path)
    v23 = read_json(args.v23_json)
    paired = paired_atlas_rows(registry)
    family = group_summary(paired, ("version", "variant_family"))
    baseline = baseline_summary(paired)
    opportunity = structural_opportunity_summary(paired)
    magnitude = magnitude_gain_summary(paired)
    local = local_global_boundary(v23)
    replay = artifact_complete_replay_rows(registry)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "failure_atlas.csv", paired)
    write_csv(args.out / "version_family_summary.csv", family)
    write_csv(args.out / "baseline_headroom_summary.csv", baseline)
    write_csv(args.out / "structural_opportunity_summary.csv", opportunity)
    write_csv(args.out / "magnitude_gain_summary.csv", magnitude)
    write_csv(args.out / "local_global_boundary.csv", local)
    write_csv(args.out / "failure_localization_taxonomy.csv", paired + local)
    (args.out / "e3_replay_summary.json").write_text(
        json.dumps(
            {
                "protocol_id": "v25_e3_offline_replay_gate_v1",
                "artifact_complete_only": True,
                "candidate_rows": len(replay),
                "status": "not_run_no_artifact_complete_rows" if not replay else "candidate_rows_available",
                "causal_status": "observational",
                "measurement_timing": "post_intervention",
                "note": "No metadata-only row is replayed; local/global boundary rows from V23 remain separate evidence.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "protocol_id": "v25_a1_failure_atlas_v1",
        "a0_protocol_id": a0_summary.get("protocol_id"),
        "paired_rows": len(paired),
        "unit_count": len({(row.get("version"), row.get("source_batch"), row.get("dataset_id"), row.get("input_protocol"), row.get("readout")) for row in paired}),
        "unique_datasets": len({row.get("dataset_id") for row in paired}),
        "positive_rows": sum(row.get("effect_state") == "positive" for row in paired),
        "negative_rows": sum(row.get("effect_state") == "negative" for row in paired),
        "small_rows": sum(row.get("effect_state") == "observed-small" for row in paired),
        "unavailable_rows": sum(row.get("effect_state") == "unavailable" for row in paired),
        "local_global_boundary_rows": len(local),
        "artifact_complete_replay_candidate_rows": len(replay),
        "structural_opportunity_groups": len(opportunity),
        "magnitude_descriptor_groups": len(magnitude),
        "labels_reloaded_for_atlas": False,
        "metric_provenance": "paired_delta_ari_and_ari_mean_imported_from_audited_historical_table; labels may have been used by the original benchmark",
        "label_free_evaluation": False,
        "primary_statistical_unit": "dataset/protocol/readout; seeds and variants repeated",
        "post_treatment_fields": ["measurement_timing", "gate_usage_raw", "magnitude_proxy_json", "positive_headroom", "negative_headroom", "post_treatment_descriptor"],
        "local_metric_label_use": "posthoc_supervised_only_for_V23_boundary_rows",
        "replay_gate": "artifact_complete_only",
        "no_causal_claim": True,
    }
    (args.out / "a1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.out, paired, family, baseline, opportunity, magnitude, local, replay, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
