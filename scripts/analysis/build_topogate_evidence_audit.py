#!/usr/bin/env python3
"""Build a provenance-aware cross-version evidence table.

This is a read-only audit of existing result artifacts.  It never selects a
variant using labels and never rewrites historical run outputs.  The generated
CSV/Markdown files are written below result/analysis, which is the repository's
result symlink target.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"
OUT = RESULT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def number(value: Any) -> float | None:
    if value is None or value == "" or value == "None":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    result = number(value)
    return int(result) if result is not None else None


def mean(values: list[float | None]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def std(values: list[float | None]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def first_nonempty(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value not in (None, "", "unknown"):
            return value
    return None


def unique_join(values: list[Any]) -> str:
    return "|".join(sorted({str(value) for value in values if value not in (None, "")}))


def summary_for(root: Path, dataset: str, variant: str, seed: Any) -> dict[str, Any] | None:
    seed_text = str(integer(seed) if integer(seed) is not None else seed)
    exact = root / f"{dataset}__{variant}__seed{seed_text}" / "summary.json"
    candidates = [exact] if exact.exists() else []
    if not candidates:
        candidates = [
            Path(path)
            for path in glob.glob(
                str(root / "**" / f"{dataset}__{variant}__seed{seed_text}" / "summary.json"),
                recursive=True,
            )
        ]
    return read_json(candidates[0]) if candidates else None


def history_features(summary: dict[str, Any] | None) -> dict[str, float | None]:
    if not summary:
        return {}
    history = summary.get("history") or []
    active = [row for row in history if number(row.get("ramp")) not in (None, 0.0)]
    if not active:
        active = history
    result: dict[str, float | None] = {}
    for key in (
        "gate",
        "target_gate",
        "gate_evidence",
        "risk_help",
        "reconstruction_help",
        "cluster_help",
        "graph",
        "edge_consistency",
        "temporal_recurrence",
        "topology_cls",
        "risk_improvement",
    ):
        result[f"{key}_mean_active"] = mean([number(row.get(key)) for row in active])
        result[f"{key}_last_active"] = number(active[-1].get(key)) if active else None
    return result


def metric_dict(summary: dict[str, Any] | None) -> dict[str, float | None]:
    if not summary:
        return {}
    metrics = summary.get("metrics") or {}
    head = metrics.get("head") or metrics
    kmeans = metrics.get("kmeans") or {}
    return {
        "ari": number(metrics.get("ari")),
        "nmi": number(metrics.get("nmi")),
        "acc": number(metrics.get("acc")),
        "head_ari": number(head.get("ari")),
        "kmeans_ari": number(kmeans.get("ari")),
    }


def base_row(**kwargs: Any) -> dict[str, Any]:
    row = {
        "batch": None,
        "version": None,
        "variant": None,
        "control_variant": None,
        "dataset": None,
        "protocol": None,
        "source_catalog": None,
        "n_runs": None,
        "seeds": None,
        "n_samples": None,
        "n_features": None,
        "n_clusters": None,
        "ari_mean": None,
        "ari_std": None,
        "nmi_mean": None,
        "nmi_std": None,
        "acc_mean": None,
        "acc_std": None,
        "head_ari_mean": None,
        "head_ari_std": None,
        "kmeans_ari_mean": None,
        "kmeans_ari_std": None,
        "ari_vs_control": None,
        "head_ari_vs_control": None,
        "kmeans_ari_vs_control": None,
        "ari_vs_ahdpc": None,
        "ari_vs_hdpc": None,
        "gate_mean_active": None,
        "target_gate_mean_active": None,
        "risk_help_mean_active": None,
        "reconstruction_help_mean_active": None,
        "cluster_help_mean_active": None,
        "edge_consistency_mean_active": None,
        "temporal_recurrence_mean_active": None,
        "graph_mean_active": None,
        "topology_trust_mean": None,
        "effective_neighbor_count": None,
        "source_sha256_count": None,
        "k_source": None,
        "labels_used_during_fit": None,
        "provenance_status": None,
        "evidence_tier": None,
        "notes": None,
    }
    row.update(kwargs)
    return row


def aggregate_run_rows(
    rows: list[dict[str, Any]],
    *,
    batch: str,
    version: str,
    protocol: str,
    source_catalog: str,
    control_variant: str | None,
    provenance_status: str,
    evidence_tier: str,
    notes: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("status", "completed")) not in ("completed", "ok", ""):
            continue
        grouped[(str(row["dataset"]), str(row["variant"]))].append(row)

    result: list[dict[str, Any]] = []
    for (dataset, variant), group in sorted(grouped.items()):
        numeric_keys = (
            "ari",
            "nmi",
            "acc",
            "head_ari",
            "kmeans_ari",
            "gate_mean_active",
            "target_gate_mean_active",
            "risk_help_mean_active",
            "reconstruction_help_mean_active",
            "cluster_help_mean_active",
            "edge_consistency_mean_active",
            "temporal_recurrence_mean_active",
            "graph_mean_active",
            "topology_trust_mean",
            "effective_neighbor_count",
        )
        values = {key: [number(row.get(key)) for row in group] for key in numeric_keys}
        row = base_row(
            batch=batch,
            version=version,
            variant=variant,
            control_variant=control_variant,
            dataset=dataset,
            protocol=protocol,
            source_catalog=source_catalog,
            n_runs=len(group),
            seeds=unique_join([integer(item.get("seed")) for item in group]),
            n_samples=first_nonempty(group, "n_samples"),
            n_features=first_nonempty(group, "n_features"),
            n_clusters=first_nonempty(group, "n_clusters"),
            provenance_status=provenance_status,
            evidence_tier=evidence_tier,
            notes=notes,
            source_sha256_count=len({item.get("source_sha256") for item in group if item.get("source_sha256")}),
            k_source=first_nonempty(group, "k_source"),
            labels_used_during_fit=first_nonempty(group, "labels_used_during_fit"),
        )
        for key in numeric_keys:
            row[f"{key}_mean"] = mean(values[key])
            if key in {"ari", "nmi", "acc", "head_ari", "kmeans_ari"}:
                row[f"{key}_std"] = std(values[key])
        result.append(row)

    by_key = {(row["dataset"], row["variant"]): row for row in result}
    for row in result:
        if control_variant:
            control = by_key.get((row["dataset"], control_variant))
            if control:
                row["ari_vs_control"] = (
                    number(row["ari_mean"]) - number(control["ari_mean"])
                    if row["ari_mean"] is not None and control["ari_mean"] is not None
                    else None
                )
                row["head_ari_vs_control"] = (
                    number(row["head_ari_mean"]) - number(control["head_ari_mean"])
                    if row["head_ari_mean"] is not None and control["head_ari_mean"] is not None
                    else None
                )
                row["kmeans_ari_vs_control"] = (
                    number(row["kmeans_ari_mean"]) - number(control["kmeans_ari_mean"])
                    if row["kmeans_ari_mean"] is not None and control["kmeans_ari_mean"] is not None
                    else None
                )
    return result


def add_summary_features(row: dict[str, Any], summary: dict[str, Any] | None) -> None:
    row.update(metric_dict(summary))
    row.update(history_features(summary))
    if summary:
        gate = summary.get("gate_summary") or {}
        risk = summary.get("risk_summary") or {}
        edge = summary.get("edge_reliability_summary") or {}
        row["gate_mean_active"] = number(gate.get("mean_node_gate", row.get("gate_mean_active")))
        row["target_gate_mean_active"] = number(gate.get("mean_target_gate", row.get("target_gate_mean_active")))
        row["topology_trust_mean"] = number(risk.get("mean_topology_trust"))
        row["effective_neighbor_count"] = number(edge.get("effective_neighbor_count"))


def v9_paper_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = RESULT / "v9_results_2026-08-02_paper_preprocess"
    comparison = read_csv(root / "comparison_by_dataset.csv")
    runs: list[dict[str, Any]] = []
    for item in comparison:
        dataset = item["dataset"]
        summaries = [
            read_json(Path(path))
            for path in glob.glob(str(root / f"{dataset}__v9_adaptive__seed*" / "summary.json"))
        ]
        summary = summaries[0] if summaries else {}
        row = base_row(
            batch="v9_paper_preprocess",
            version="V9",
            variant="v9_adaptive",
            control_variant="AHDPC_reference",
            dataset=dataset,
            protocol="paper_preprocess",
            source_catalog="comparison_by_dataset.csv + per-run summary.json",
            n_runs=integer(item.get("v9_n")),
            seeds=unique_join([summary.get("seed") for summary in summaries]),
            n_samples=summary.get("n_samples"),
            n_features=summary.get("n_features"),
            n_clusters=summary.get("n_clusters"),
            ari_mean=number(item.get("v9_ari_mean")),
            ari_std=number(item.get("v9_ari_std")),
            nmi_mean=number(item.get("v9_nmi_mean")),
            nmi_std=number(item.get("v9_nmi_std")),
            acc_mean=number(item.get("v9_acc_mean")),
            acc_std=number(item.get("v9_acc_std")),
            ari_vs_ahdpc=number(item.get("v9_minus_ahdpc_ari")),
            ari_vs_hdpc=number(item.get("v9_minus_hdpc_ari")),
            source_sha256_count=len({s.get("source_sha256") for s in summaries if s.get("source_sha256")}),
            k_source=summary.get("k_source"),
            labels_used_during_fit=summary.get("labels_used_during_fit"),
            provenance_status="complete_per_run_summary",
            evidence_tier="3-seed paired V9; persisted single-run baselines",
            notes="AHDPC/HDPC are persisted references, not symmetric multi-seed reruns.",
        )
        result = summary.get("metrics") or {}
        row["head_ari_mean"] = row["ari_mean"]
        row["kmeans_ari_mean"] = None
        row["source_catalog"] = "comparison_by_dataset.csv + per-run summary.json"
        runs.append(row)
    return runs, {
        "batch": "v9_paper_preprocess",
        "version": "V9",
        "root": root,
        "primary_csvs": [root / "comparison_by_dataset.csv"],
    }


def v9_ablation_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = RESULT / "v9_results_2026-08-02_advantage_ablation"
    raw = read_csv(root / "ablation_runs.csv")
    allowed = {"v9_full", "v9_nomix", "v9_static", "v9_random"}
    rows: list[dict[str, Any]] = []
    for item in raw:
        if item.get("variant") not in allowed:
            continue
        record = dict(item)
        record["ari"] = number(item.get("ari"))
        record["nmi"] = number(item.get("nmi"))
        record["acc"] = number(item.get("acc"))
        record["k_source"] = "runner_code:np.unique(y)"
        record["labels_used_during_fit"] = "code_audit_false"
        record["status"] = item.get("status", "completed")
        rows.append(record)
    aggregated = aggregate_run_rows(
        rows,
        batch="v9_advantage_ablation",
        version="V9",
        protocol="raw_related_advantage",
        source_catalog="ablation_runs.csv + run_record.json",
        control_variant="v9_nomix",
        provenance_status="partial_summary_dataset_adhoc; CSV/run_record identity retained",
        evidence_tier="3-seed paired ablation",
        notes="The underlying run_record/CSV has the real dataset; generated summary.json says dataset=adhoc and omits source metadata.",
    )
    return aggregated, {
        "batch": "v9_advantage_ablation",
        "version": "V9",
        "root": root,
        "primary_csvs": [root / "ablation_runs.csv"],
    }


def legacy_table_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = RESULT / "ablation"
    raw = read_csv(root / "merged_summary.csv")
    rows: list[dict[str, Any]] = []
    for item in raw:
        row = dict(item)
        row["status"] = "completed"
        row["ari"] = number(item.get("ari"))
        row["nmi"] = number(item.get("nmi"))
        row["acc"] = number(item.get("acc"))
        row["k_source"] = "table_only"
        row["labels_used_during_fit"] = "unknown"
        rows.append(row)
    aggregated = aggregate_run_rows(
        rows,
        batch="static_gate_legacy_table",
        version="StaticGate",
        protocol="historical_merged_summary",
        source_catalog="ablation/merged_summary.csv",
        control_variant="static_gate_nomix",
        provenance_status="metrics_table_without_source_hash_or_label_flag",
        evidence_tier="multi-seed legacy ablation table",
        notes="Useful for mechanism direction only; do not merge with V9 paper-preprocess rows.",
    )
    return aggregated, {
        "batch": "static_gate_legacy_table",
        "version": "StaticGate",
        "root": root,
        "primary_csvs": [root / "merged_summary.csv"],
    }


def v12_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = RESULT / "v12_results_2026-08-03_advantage"
    raw = read_csv(root / "runs.csv")
    rows: list[dict[str, Any]] = []
    for item in raw:
        if item.get("status") != "completed":
            continue
        # The V12 directory intentionally contains V9 controls as well.  The
        # controls are useful for the V12-vs-V9 comparison, but they must not
        # be labelled as V12 rows or enter the V12 Full-vs-NoMix pairing.
        if not str(item.get("variant", "")).startswith("v12_"):
            continue
        dataset = item.get("dataset") or "unknown"
        variant = item.get("variant") or "unknown"
        summary = summary_for(root, dataset, variant, item.get("seed"))
        row = dict(item)
        row["ari"] = number(item.get("ari"))
        row["nmi"] = number(item.get("nmi"))
        row["acc"] = number(item.get("acc"))
        row["k_source"] = "runner_code:np.unique(y)"
        row["labels_used_during_fit"] = "code_audit_false"
        add_summary_features(row, summary)
        rows.append(row)
    aggregated = aggregate_run_rows(
        rows,
        batch="v12_advantage",
        version="V12",
        protocol="raw_related_advantage",
        source_catalog="runs.csv + per-run summary.json",
        control_variant="v12_nomix",
        provenance_status="partial_summary_dataset_adhoc; CSV contains source hash",
        evidence_tier="3-seed paired mechanism comparison",
        notes="V12 risk summary is available, but summary.json lacks source_path, K protocol, and explicit label flag.",
    )
    return aggregated, {
        "batch": "v12_advantage",
        "version": "V12",
        "root": root,
        "primary_csvs": [root / "runs.csv"],
    }


def v13_v14_rows(root: Path, version: str, batch: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = read_csv(root / "runs.csv")
    rows: list[dict[str, Any]] = []
    for item in raw:
        if item.get("status") != "completed":
            continue
        dataset = item.get("dataset") or "unknown"
        variant = item.get("variant") or "unknown"
        summary = summary_for(root, dataset, variant, item.get("seed"))
        row = dict(item)
        row["ari"] = number(item.get("ari", item.get("head_ari")))
        row["nmi"] = number(item.get("nmi"))
        row["acc"] = number(item.get("acc"))
        row["head_ari"] = number(item.get("head_ari", item.get("ari")))
        row["kmeans_ari"] = number(item.get("kmeans_ari"))
        row["k_source"] = "runner_code:np.unique(y)"
        row["labels_used_during_fit"] = "code_audit_false"
        add_summary_features(row, summary)
        rows.append(row)
    aggregated = aggregate_run_rows(
        rows,
        batch=batch,
        version=version,
        protocol="raw_related_advantage",
        source_catalog="runs.csv + per-run summary.json",
        control_variant=f"{version.lower()}_nomix",
        provenance_status="source_hash_and_k_in_runs_csv; explicit label flag absent",
        evidence_tier="3-seed paired mechanism comparison",
        notes="Full and NoMix are paired within this batch; do not compare dataset names across different source manifests.",
    )
    return aggregated, {
        "batch": batch,
        "version": version,
        "root": root,
        "primary_csvs": [root / "runs.csv"],
    }


def v11_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = RESULT / "V11" / "topogate_v11_minimum_5x3"
    raw: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/comparison.csv")):
        raw.extend(read_csv(path))
    rows: list[dict[str, Any]] = []
    for item in raw:
        dataset = item.get("dataset") or "unknown"
        variant = item.get("variant") or "unknown"
        summary = summary_for(root, dataset, variant, item.get("seed"))
        row = dict(item)
        row["ari"] = number(item.get("ari"))
        row["nmi"] = number(item.get("nmi"))
        row["acc"] = number(item.get("acc"))
        row["head_ari"] = number(item.get("ari"))
        row["kmeans_ari"] = number((summary or {}).get("metrics", {}).get("kmeans", {}).get("ari"))
        if summary:
            row["n_samples"] = summary.get("n_samples")
            row["n_features"] = summary.get("n_features")
            row["n_clusters"] = summary.get("n_clusters")
            row["source_sha256"] = summary.get("source_sha256")
            row["k_source"] = summary.get("k_protocol")
            row["labels_used_during_fit"] = "code_audit_false; field absent"
        add_summary_features(row, summary)
        rows.append(row)
    aggregated = aggregate_run_rows(
        rows,
        batch="v11_minimum_5x3",
        version="V11",
        protocol="extended_5_dataset_multiseed",
        source_catalog="gpu*/comparison.csv + per-run summary.json",
        control_variant="V11_nomix",
        provenance_status="source_hash_and_k_protocol_present; explicit label flag absent",
        evidence_tier="5-dataset x 3-seed paired candidate",
        notes="Candidate minimum combiner; not a universal V11 claim and not the same data manifest as V14.",
    )
    return aggregated, {
        "batch": "v11_minimum_5x3",
        "version": "V11",
        "root": root,
        "primary_csvs": sorted(root.glob("**/comparison.csv")),
    }


def provenance_audit(batch_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in batch_meta:
        root = Path(meta["root"])
        summary_paths = [Path(path) for path in glob.glob(str(root / "**" / "summary.json"), recursive=True)]
        summaries = []
        for path in summary_paths:
            try:
                summaries.append(read_json(path))
            except (OSError, json.JSONDecodeError):
                continue
        # Audit only the source tables declared by each batch loader.  The
        # result tree also contains this audit's derived CSVs and other
        # diagnostics, which must not be counted as raw provenance records.
        csv_paths = [
            Path(path)
            for path in meta.get("primary_csvs", [])
            if Path(path).is_file()
        ]
        completed = 0
        raw_rows = 0
        source_hash = 0
        k_protocol = 0
        label_flag = 0
        identity_adhoc = 0
        datasets: set[str] = set()
        variants: set[str] = set()
        seeds: set[str] = set()
        for path in csv_paths:
            try:
                csv_rows = read_csv(path)
            except (OSError, csv.Error):
                continue
            for item in csv_rows:
                if item.get("status") not in (None, "", "completed", "ok"):
                    continue
                raw_rows += 1
                if item.get("source_sha256") or item.get("source_hash"):
                    source_hash += 1
                if item.get("k_source") or item.get("k_protocol"):
                    k_protocol += 1
                if item.get("labels_used_during_fit"):
                    label_flag += 1
                if item.get("dataset"):
                    datasets.add(item["dataset"])
                if item.get("variant"):
                    variants.add(item["variant"])
                if item.get("seed"):
                    seeds.add(str(item["seed"]))
        for summary in summaries:
            if summary.get("run_status", "completed") == "completed":
                completed += 1
            if summary.get("source_sha256"):
                source_hash += 1
            if summary.get("k_source") or summary.get("k_protocol"):
                k_protocol += 1
            if "labels_used_during_fit" in summary:
                label_flag += 1
            if summary.get("dataset") == "adhoc":
                identity_adhoc += 1
            if summary.get("dataset"):
                datasets.add(str(summary["dataset"]))
            if summary.get("variant"):
                variants.add(str(summary["variant"]))
            if summary.get("seed") is not None:
                seeds.add(str(summary["seed"]))
        rows.append({
            "batch": meta["batch"],
            "version": meta["version"],
            "result_root": str(root.relative_to(ROOT)),
            "csv_file_count": len(csv_paths),
            "summary_file_count": len(summary_paths),
            "completed_summary_count": completed,
            "raw_completed_row_count": raw_rows,
            "unique_datasets_seen": len(datasets),
            "unique_variants_seen": len(variants),
            "unique_seeds_seen": len(seeds),
            "source_hash_records_seen": source_hash,
            "k_protocol_records_seen": k_protocol,
            "label_flag_records_seen": label_flag,
            "summary_dataset_adhoc_count": identity_adhoc,
            "status": "metadata_gap" if identity_adhoc or not label_flag else "usable_with_review",
            "notes": (
                "Use CSV/run_record identity when summary.json says adhoc."
                if identity_adhoc
                else "Check label flag before promoting to a paper table."
            ),
        })
    return rows


def paired_summary(rows: list[dict[str, Any]], version: str, variant_a: str, variant_b: str) -> dict[str, Any]:
    by_key = {(row["dataset"], row["variant"]): row for row in rows if row["version"] == version}
    deltas = []
    for dataset in sorted({row["dataset"] for row in rows if row["version"] == version}):
        left = by_key.get((dataset, variant_a))
        right = by_key.get((dataset, variant_b))
        if left and right and left.get("ari_mean") is not None and right.get("ari_mean") is not None:
            deltas.append(number(left["ari_mean"]) - number(right["ari_mean"]))
    return {
        "version": version,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "datasets": len(deltas),
        "mean_delta_ari": mean(deltas),
        "std_delta_ari": std(deltas),
        "positive_dataset_count": sum(delta > 0 for delta in deltas),
        "negative_dataset_count": sum(delta < 0 for delta in deltas),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for builder in (v9_paper_rows, v9_ablation_rows, v12_rows, v11_rows, legacy_table_rows):
        rows, batch_meta = builder()
        all_rows.extend(rows)
        meta.append(batch_meta)
    for root, version, batch in (
        (RESULT / "v13_results_2026-08-03_advantage", "V13", "v13_advantage"),
        (RESULT / "v14_results_2026-08-03_advantage_5ds", "V14", "v14_advantage_5ds"),
    ):
        rows, batch_meta = v13_v14_rows(root, version, batch)
        all_rows.extend(rows)
        meta.append(batch_meta)

    ordered_fields = [
        "batch", "version", "variant", "control_variant", "dataset", "protocol", "source_catalog",
        "n_runs", "seeds", "n_samples", "n_features", "n_clusters", "ari_mean", "ari_std",
        "nmi_mean", "nmi_std", "acc_mean", "acc_std", "head_ari_mean", "head_ari_std",
        "kmeans_ari_mean", "kmeans_ari_std", "ari_vs_control", "head_ari_vs_control",
        "kmeans_ari_vs_control", "ari_vs_ahdpc", "ari_vs_hdpc", "gate_mean_active",
        "target_gate_mean_active", "risk_help_mean_active", "reconstruction_help_mean_active",
        "cluster_help_mean_active", "edge_consistency_mean_active", "temporal_recurrence_mean_active",
        "graph_mean_active", "topology_trust_mean", "effective_neighbor_count", "source_sha256_count",
        "k_source", "labels_used_during_fit", "provenance_status", "evidence_tier", "notes",
    ]
    write_csv(OUT / "cross_version_evidence_2026-08-03.csv", all_rows, ordered_fields)
    audit = provenance_audit(meta)
    write_csv(OUT / "provenance_audit_2026-08-03.csv", audit)

    pair_rows = [
        paired_summary(all_rows, "V9", "v9_full", "v9_nomix"),
        paired_summary(all_rows, "V11", "V11_full", "V11_nomix"),
        paired_summary(all_rows, "V12", "v12_full", "v12_nomix"),
        paired_summary(all_rows, "V13", "v13_full", "v13_nomix"),
        paired_summary(all_rows, "V14", "v14_full", "v14_nomix"),
        paired_summary(all_rows, "StaticGate", "static_gate_full", "static_gate_nomix"),
    ]
    write_csv(OUT / "paired_version_deltas_2026-08-03.csv", pair_rows)

    lines = [
        "# TopoGate 跨版本证据与 provenance 审计",
        "",
        "生成时间：2026-08-03。该报告只读取当前 `result/` 软链接目标中的 CSV/JSON；不重新训练、不读取标签做选择，也不改写历史产物。",
        "",
        "## 输出",
        "",
        "- `cross_version_evidence_2026-08-03.csv`：按数据集和 variant 聚合的多种子指标、Full/NoMix 差值和 gate/risk 诊断。",
        "- `paired_version_deltas_2026-08-03.csv`：同一批次内的配对 ARI 差值，不能跨协议合并。",
        "- `provenance_audit_2026-08-03.csv`：summary/CSV 中 source hash、K 来源、标签字段和数据集身份的覆盖情况。",
        "",
        "## 解释规则",
        "",
        "1. `V9 paper_preprocess` 的 AHDPC/HDPC 是持久化单次参考；它的 `ari_vs_ahdpc`/`ari_vs_hdpc` 不是对称多种子差值。",
        "2. V9、V12、V13、V14 和 V11 的 Full/NoMix 差值只在同一结果批次内配对；同名数据集如果 source hash 或预处理不同，不能拼成跨版本纵向结论。",
        "3. `provenance_status` 不是性能评价。`partial_*` 表示指标仍可由 CSV/run_record 和 source hash 追溯，但不能把缺失字段写成已记录。",
        "",
        "## 关键发现",
        "",
        "- V9 论文匹配批次必须保留 `3/1/20` 的 AHDPC 胜/平/负边界；三个正差值为 `spect_heart`、`balance_scale`、`landsat`。",
        "- V9 优势消融的 `summary.json` 有 `dataset=adhoc`，真实身份在 `ablation_runs.csv`/`run_record.json` 中；该批次被标为 `partial_summary_dataset_adhoc`。",
        "- V12 的 summary 同样保留 `adhoc`，且 source path、K protocol、显式 label flag 不在 summary 中；使用 runs.csv 的 source/hash 和 runner 源码审计作为补充。",
        "- V11 minimum 5x3 具备 source hash、`benchmark_oracle_from_y` 和语义分离的 prediction/labels_true 输出，但 summary 未显式写 `labels_used_during_fit=false`；这是文档契约缺口，不是标签泄漏证据。",
        "- V14 的 gate/risk 诊断可以证明路径被调用，但 paired ARI 增益仍不足以晋级主方法；应继续报告 gate coverage、target gate 和 readout 分歧。",
        "",
        "## 书籍与数学边界",
        "",
        "当前源码中的 kNN、mutual/SNN、动态图刷新和边可靠性属于依赖度量的有限图结构；它们没有 filtration、simplicial complex、boundary operator、homology 或 persistence diagram。因此本表不会把这些量标成 persistent homology。真正 TDA 的第一版应作为 detached edge prior/诊断，并保留 NoMix、原 V11、random prior 和 fixed filtration 控制。",
        "",
        "参考书映射见 `TopoGate_whole_project_math_TDA_audit_2026-08-03.md`：拓扑学用于区分邻域图与同调不变量，数学分析用于约束 kNN 离散跳变和 EMA 稳定性表述，Bishop/PRML 用于解释 mixture responsibility、目标错配和无监督 K 协议。",
    ]
    (OUT / "cross_version_evidence_audit_2026-08-03.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(all_rows)} aggregate rows and {len(audit)} provenance rows")


if __name__ == "__main__":
    main()
