#!/usr/bin/env python3
"""Build a provenance-aware V1-V22 failure diagnostic table.

The public ``final_results`` snapshot is metadata-only and is currently kept
in the publication worktree.  This script accepts that snapshot explicitly,
normalizes only completed/audited artifacts, and writes a long table plus a
dataset-level wide view under ``reports/``.  It never selects a result using
labels and never pairs rows across protocols, readouts, or source batches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FAMILIES = (
    "self",
    "fixed",
    "random",
    "static",
    "learned",
    "hard",
    "assignment-adversarial",
    "discriminator",
)

VERSION_COVERAGE = {
    "V01": ("none", "No non-smoke final artifact in the audited snapshot."),
    "V02": ("none", "No non-smoke final artifact in the audited snapshot."),
    "V03": ("none", "No non-smoke final artifact in the audited snapshot."),
    "V04": ("none", "No non-smoke final artifact in the audited snapshot."),
    "V05": ("none", "No non-smoke final artifact in the audited snapshot."),
    "V06": ("none", "No non-smoke final artifact in the audited snapshot."),
    "V07": ("none", "No non-smoke final artifact in the audited snapshot."),
    "V08": ("none", "No non-smoke final artifact in the audited snapshot."),
    "V09": ("available", "Advantage, legacy multiseed, CLUBench, and paper-preprocess artifacts."),
    "V10": ("available", "Legacy nomix/nomix-init comparison only."),
    "V11": ("available", "Five-dataset, three-seed sparse H0 pilot."),
    "V12": ("available", "Stage-3 topology grid with stage-2 paired CSV."),
    "V13": ("available", "Five-dataset, three-seed hard-gate batch."),
    "V14": ("available", "Five-dataset, three-seed advantage batch."),
    "V15": ("none", "No promotable non-smoke final table."),
    "V16": ("none", "No promotable final table; V16.1 is separate."),
    "V16.1": ("available", "Promotion summaries only; all rows empirical_not_supported."),
    "V17": ("none", "Reference implementation only; no performance evidence."),
    "V18": ("available", "149-dataset, ten-variant, three-seed matrix."),
    "V19": ("available", "Post-freeze, PlantNet transfer, and sparse extension panels."),
    "V20": ("available", "Eight-dataset single-seed coarse screen without matched control."),
    "V21": ("available", "Six-dataset graph-fix assignment-adversarial matrix."),
    "V22": ("available", "Hard-gate recovery plus cooperative Keep-Gate audit."),
}


def num(value: Any) -> float | None:
    if value in (None, "", "None", "null", "nan", "NaN"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def integer(value: Any) -> int | None:
    out = num(value)
    return int(out) if out is not None else None


def mean(values: list[float | None]) -> float | None:
    clean = [x for x in values if x is not None]
    return statistics.fmean(clean) if clean else None


def fmt(value: Any, digits: int = 4) -> str:
    parsed = num(value)
    return "NA" if parsed is None else f"{parsed:+.{digits}f}"


def fmt_plain(value: Any, digits: int = 4) -> str:
    parsed = num(value)
    return "NA" if parsed is None else f"{parsed:.{digits}f}"


def stat_text(values: list[Any], *, signed: bool = False, digits: int = 4) -> str:
    clean = [parsed for parsed in (num(value) for value in values) if parsed is not None]
    if not clean:
        return "NA"
    center = statistics.fmean(clean)
    spread = statistics.stdev(clean) if len(clean) > 1 else 0.0
    sign = "+" if signed else ""
    return f"{center:{sign}.{digits}f}±{spread:.{digits}f}"


def row_stat_text(
    rows: list[dict[str, Any]],
    value_field: str,
    *,
    std_field: str | None = None,
    signed: bool = False,
    digits: int = 4,
) -> str:
    """Summarize per-seed rows without discarding a published aggregate std."""
    clean = [parsed for parsed in (num(row.get(value_field)) for row in rows) if parsed is not None]
    if not clean:
        return "NA"
    center = statistics.fmean(clean)
    provided = num(rows[0].get(std_field)) if len(rows) == 1 and std_field else None
    spread = provided if provided is not None else (statistics.stdev(clean) if len(clean) > 1 else 0.0)
    sign = "+" if signed else ""
    return f"{center:{sign}.{digits}f}±{spread:.{digits}f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def unique_join(values: list[Any]) -> str:
    return "|".join(sorted({str(x) for x in values if x not in (None, "", "NA")}))


def protocol_from_dataset_id(dataset_id: Any, fallback: str = "protocol not recorded") -> str:
    """Recover the explicit V19 layer suffix when the aggregate omits it."""
    value = str(dataset_id or "")
    if value.endswith("__rg_native"):
        return "rg_native"
    if value.endswith("__clubench_bridge"):
        return "clubench_bridge"
    if value.endswith("__shared_text"):
        return "shared_text"
    return fallback


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def base_row(**kwargs: Any) -> dict[str, Any]:
    row = {
        "version": None,
        "source_batch": None,
        "dataset": None,
        "dataset_id": None,
        "variant_family": None,
        "variant": None,
        "ari_mean": None,
        "ari_std": None,
        "n_runs": None,
        "seeds": None,
        "paired_control": None,
        "paired_delta_ari": None,
        "paired_delta_scope": None,
        "gate_usage": None,
        "gate_usage_semantics": None,
        "input_protocol": None,
        "readout": None,
        "status": None,
        "evidence_level": None,
        "failure_diagnosis": None,
        "source_artifact": None,
        "notes": None,
    }
    row.update(kwargs)
    return row


def diagnosis(delta: float | None, status: str, evidence: str, note: str = "") -> str:
    if status == "incomplete_compute":
        return "incomplete_compute; no performance conclusion"
    if "empirical_not_supported" in evidence:
        return "restricted_evidence_not_promoted"
    if status not in {"completed", "ok", "reported"}:
        return status or "not_available"
    if delta is None:
        if "single_seed" in evidence or "no paired" in note.lower():
            return "no_paired_control"
        return "paired_delta_not_available"
    if delta < -0.03:
        return "paired_regression"
    if delta > 0.03:
        return "paired_positive_scope_limited"
    return "paired_neutral_or_small"


def add_pair_deltas(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    control_variant: str,
    metric_field: str = "ari",
    scope: str = "seed_matched_exact",
) -> None:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("status") not in {"completed", "ok", "reported"}:
            continue
        key = tuple(row.get(field) for field in key_fields)
        value = num(row.get(metric_field))
        if value is not None:
            grouped[key][str(row.get("variant"))].append(value)
    for row in rows:
        # The control is the reference, not an intervention.  Leaving it in
        # the loop would incorrectly encode its own Delta ARI as zero.
        if str(row.get("variant")) == control_variant:
            continue
        key = tuple(row.get(field) for field in key_fields)
        control = grouped.get(key, {}).get(control_variant, [])
        current = num(row.get(metric_field))
        if not control or current is None:
            continue
        # Exact per-seed rows are preferred. If the artifact is already a
        # matched mean summary, this still computes a matched group delta but
        # the caller should provide a corresponding scope label.
        row["paired_control"] = control_variant
        row["paired_delta_ari"] = current - statistics.fmean(control)
        row["paired_delta_scope"] = scope


def v09_advantage(root: Path) -> list[dict[str, Any]]:
    batch = "V09/advantage_ablation"
    path = root / "V09/advantage_ablation/ablation_runs.csv"
    raw = [r for r in read_csv(path) if r.get("status") == "completed"]
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in raw:
        by_dataset[r["dataset"]].append(r)
    rows: list[dict[str, Any]] = []
    family_map = {
        "v9_nomix": "self",
        "v9_static": "static",
        "v9_random": "random",
        "v9_full": "learned",
    }
    for dataset, group in sorted(by_dataset.items()):
        for variant, family in family_map.items():
            selected = [r for r in group if r["variant"] == variant]
            if not selected:
                continue
            rows.append(
                base_row(
                    version="V09",
                    source_batch=batch,
                    dataset=dataset,
                    dataset_id=dataset,
                    variant_family=family,
                    variant=variant,
                    ari_mean=mean([num(r.get("ari")) for r in selected]),
                    ari_std=statistics.stdev([num(r.get("ari")) for r in selected]) if len(selected) > 1 else 0.0,
                    n_runs=len(selected),
                    seeds=unique_join([r.get("seed") for r in selected]),
                    input_protocol="raw_related_advantage (protocol detail absent from final_results CSV)",
                    readout="KMeans/legacy V9 readout (exact mode not recorded in snapshot)",
                    status="completed",
                    evidence_level="3-seed paired ablation",
                    gate_usage="NA",
                    gate_usage_semantics="Gate usage not recorded in final_results/ablation_runs.csv",
                    source_artifact=str(path.relative_to(root)),
                    notes="Self control is v9_nomix; deltas are recomputed from the same dataset-seed rows.",
                )
            )
    add_pair_deltas(
        rows,
        key_fields=("version", "source_batch", "dataset"),
        control_variant="v9_nomix",
        metric_field="ari_mean",
        scope="seed_matched_exact_from_ablation_runs",
    )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v09_legacy(root: Path) -> list[dict[str, Any]]:
    batch = "V09/legacy_multiseed"
    path = root / "V09/legacy_multiseed/ablation_table.csv"
    raw = [r for r in read_csv(path) if r.get("n_err", "0") == "0"]
    family_map = {
        "v9_nomix": "self",
        "v9_random_neighbors": "random",
        "v9_static_gate": "static",
        "v9_static_random": "random",
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in raw:
        grouped[r["dataset"]].append(r)
    rows: list[dict[str, Any]] = []
    for dataset, group in sorted(grouped.items()):
        control = next((r for r in group if r["variant"] == "v9_nomix"), None)
        control_ari = num(control.get("ari_mean")) if control else None
        for r in group:
            variant = r["variant"]
            family = family_map.get(variant)
            if family is None:
                continue
            current = num(r.get("ari_mean"))
            delta = current - control_ari if current is not None and control_ari is not None else None
            rows.append(
                base_row(
                    version="V09",
                    source_batch=batch,
                    dataset=dataset,
                    dataset_id=dataset,
                    variant_family=family,
                    variant=variant,
                    ari_mean=current,
                    ari_std=num(r.get("ari_std")),
                    n_runs=integer(r.get("n_seeds")),
                    seeds="7|42|123",
                    paired_control="v9_nomix" if variant != "v9_nomix" else None,
                    paired_delta_ari=delta if variant != "v9_nomix" else None,
                    paired_delta_scope="same-seed-count aggregate; seed-wise rows not published" if variant != "v9_nomix" else "reference",
                    input_protocol="historical_merged_summary (input protocol not recorded in final_results)",
                    readout="V9 legacy readout",
                    status="completed",
                    evidence_level="3-seed aggregate ablation",
                    gate_usage="NA",
                    gate_usage_semantics="Gate usage not recorded in final_results/ablation_table.csv",
                    source_artifact=str(path.relative_to(root)),
                    notes="The published aggregate table contains same-seed counts but not per-seed ARI rows.",
                )
            )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v09_single_seed(root: Path) -> list[dict[str, Any]]:
    path = root / "V09/clubench_131_single_seed/comparison_long.csv"
    manifest = read_json(root / "V09/clubench_131_single_seed/MANIFEST.json")
    rows: list[dict[str, Any]] = []
    for r in read_csv(path):
        if r.get("method") != "V9" or r.get("status") != "completed":
            continue
        rows.append(
            base_row(
                version="V09",
                source_batch="V09/clubench_131_single_seed",
                dataset=r["dataset"],
                dataset_id=r["dataset"],
                variant_family="learned",
                variant="V9",
                ari_mean=num(r.get("ARI")),
                ari_std=0.0,
                n_runs=1,
                seeds=r.get("seed"),
                paired_control=None,
                paired_delta_ari=None,
                paired_delta_scope="no paired control in this final artifact",
                gate_usage="NA",
                gate_usage_semantics="Gate usage not included in comparison_long.csv",
                input_protocol=f"clubench_bridge: {manifest.get('input_preprocessing', 'CLUBench.load_data')}",
                readout="V9 benchmark readout",
                status="completed",
                evidence_level="single_seed_control comparison",
                source_artifact=str(path.relative_to(root)),
                notes="AHDPC/HDPC are external references, not self/no-gate controls for this row.",
            )
        )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(None, row["status"], row["evidence_level"], row["notes"])
    return rows


def v09_paper(root: Path) -> list[dict[str, Any]]:
    path = root / "V09/paper_preprocess/comparison_per_run.csv"
    rows: list[dict[str, Any]] = []
    for r in read_csv(path):
        if r.get("method") != "V9" or r.get("status") != "completed":
            continue
        rows.append(
            base_row(
                version="V09",
                source_batch="V09/paper_preprocess",
                dataset=r["dataset"],
                dataset_id=r["dataset"],
                variant_family="learned",
                variant="V9",
                ari_mean=num(r.get("ari")),
                ari_std=0.0,
                n_runs=1,
                seeds=r.get("seed"),
                paired_control=None,
                paired_delta_ari=None,
                paired_delta_scope="no paired control in this final artifact",
                gate_usage="NA",
                gate_usage_semantics="Gate usage not included in comparison_per_run.csv",
                input_protocol=r.get("protocol"),
                readout="V9 paper-preprocess readout",
                status="completed",
                evidence_level="3-seed single-variant comparison",
                source_artifact=str(path.relative_to(root)),
                notes="No self/no-gate row is present in this artifact.",
            )
        )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(None, row["status"], row["evidence_level"], row["notes"])
    return rows


def v10(root: Path) -> list[dict[str, Any]]:
    batch = "V10/comparison"
    paths = [root / "V10/comparison_ablation.csv", root / "V10/comparison_multiseed.csv"]
    raw: list[dict[str, str]] = []
    for path in paths:
        raw.extend(read_csv(path))
    rows: list[dict[str, Any]] = []
    for r in raw:
        if r.get("error"):
            continue
        variant = r["variant"]
        family = "self" if variant == "v10_nomix" else "learned"
        rows.append(
            base_row(
                version="V10",
                source_batch=batch,
                dataset=r["dataset"],
                dataset_id=r["dataset"],
                variant_family=family,
                variant=variant,
                ari_mean=num(r.get("ari")),
                ari_std=0.0,
                n_runs=1,
                seeds=r.get("seed"),
                input_protocol="AHDPC processed legacy; protocol field absent from final_results CSV",
                readout="V10 legacy readout",
                status="completed",
                evidence_level="3-seed legacy comparison" if r["dataset"] != "Campbell" else "single variant multiseed record",
                gate_usage=f"effective_gate_max={fmt_plain(r.get('effective_gate_max'), 3)} (capacity only)",
                gate_usage_semantics="Configured maximum, not realized gate usage; no realized usage field published",
                source_artifact=str((root / "V10/comparison_ablation.csv").relative_to(root)) if r["dataset"] != "Campbell" else str((root / "V10/comparison_multiseed.csv").relative_to(root)),
                notes="v10_nomix_init is treated as the learned-init candidate; pairing is only valid where v10_nomix is present in the same CSV batch.",
            )
        )
    # Per-seed pairing is exact within the ablation file for the three shared datasets.
    add_pair_deltas(
        rows,
        key_fields=("version", "source_batch", "dataset", "seeds"),
        control_variant="v10_nomix",
        metric_field="ari_mean",
        scope="seed_matched_exact_within_ablation_csv",
    )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v11(root: Path) -> list[dict[str, Any]]:
    root_dir = root / "V11/tda_h0_pilot"
    path = root_dir / "run_diagnostics.csv"
    raw = [r for r in read_csv(path) if not r.get("error")]
    family_map = {
        "V11_nomix": "self",
        "V11_tda_fixed_filtration": "fixed",
        "V11_tda_random": "random",
        "V11_full": "learned",
        "V11_tda_h0_mst": "fixed",
    }
    rows: list[dict[str, Any]] = []
    for r in raw:
        variant = r["variant"]
        if variant not in family_map:
            continue
        usage = (
            f"mean_gate={fmt_plain(r.get('mean_gate'), 5)}; "
            f"final_gate={fmt_plain(r.get('final_gate'), 5)}; "
            f"edge_change={fmt_plain(r.get('graph_edge_change_fraction'), 5)}"
        )
        rows.append(
            base_row(
                version="V11",
                source_batch="V11/tda_h0_pilot",
                dataset=r["dataset"],
                dataset_id=r["dataset"],
                variant_family=family_map[variant],
                variant=variant,
                ari_mean=num(r.get("head_ari")),
                ari_std=0.0,
                n_runs=1,
                seeds=r.get("seed"),
                gate_usage=usage,
                gate_usage_semantics="Gate scalar plus graph edge-change fraction; not a single mask percentage",
                input_protocol="AHDPC processed NPZ; raw-PCA kNN sparse 1-skeleton; TDA-H0 pilot",
                readout="head_ari (paired table also publishes KMeans ARI)",
                status="completed",
                evidence_level="5-dataset x 3-seed paired pilot",
                source_artifact=str(path.relative_to(root)),
                notes="The H0/fixed/random rows are prior controls; all use the same V11 protocol and seed set.",
            )
        )
    add_pair_deltas(
        rows,
        key_fields=("version", "source_batch", "dataset", "seeds"),
        control_variant="V11_nomix",
        metric_field="ari_mean",
        scope="seed_matched_exact_run_diagnostics",
    )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v12(root: Path) -> list[dict[str, Any]]:
    root_dir = root / "V12/stage3_topology_search"
    path = root_dir / "summary_by_dataset_config.csv"
    paired = read_csv(root_dir / "paired_deltas_vs_stage2.csv")
    paired_lookup: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in paired:
        delta = num(r.get("ari_delta"))
        if delta is not None:
            paired_lookup[(r["dataset"], r["config"], str(r["seed"]))].append(delta)
    rows: list[dict[str, Any]] = []
    for r in read_csv(path):
        config = r["config"]
        family = "self" if config.startswith("self_null") else "learned"
        deltas = [v for (dataset, cfg, _seed), values in paired_lookup.items() if dataset == r["dataset"] and cfg == config for v in values]
        usage = (
            f"self_mass={fmt_plain(r.get('self_mass_mean'), 5)}; "
            f"edge_entropy={fmt_plain(r.get('edge_entropy_mean'), 5)}; "
            f"effective_neighbors={fmt_plain(r.get('effective_neighbor_count_mean'), 5)}"
        )
        rows.append(
            base_row(
                version="V12",
                source_batch="V12/stage3_topology_search",
                dataset=r["dataset"],
                dataset_id=r["dataset"],
                variant_family=family,
                variant=config,
                ari_mean=num(r.get("ari_mean")),
                ari_std=num(r.get("ari_std")),
                n_runs=integer(r.get("n_completed")),
                seeds="7|42|123",
                paired_control="self_null_lambda01" if family != "self" else None,
                paired_delta_ari=mean(deltas) if family != "self" else None,
                paired_delta_scope=(
                    "reported paired_deltas_vs_stage2.csv; config mean over three seeds"
                    if family != "self" else "reference/config row; no intervention delta"
                ),
                gate_usage=usage,
                gate_usage_semantics="Topology self mass/edge entropy/effective neighbor count; not a node-mask rate",
                input_protocol="AHDPC processed NPZ; raw-PCA kNN sparse 1-skeleton; V12 stage-3 topology search",
                readout="V12 legacy ARI readout",
                status="completed",
                evidence_level="144-run stage-3 grid; no-go report",
                source_artifact=str(path.relative_to(root)),
                notes="The paired control is the stage-2 self_null_lambda01 reference, not a scMAE-only control; do not merge across V12 configs.",
            )
        )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v13(root: Path) -> list[dict[str, Any]]:
    root_dir = root / "V13/hard_gate"
    path = root_dir / "runs.csv"
    raw = [r for r in read_csv(path) if r.get("status") == "completed"]
    rows: list[dict[str, Any]] = []
    for r in raw:
        variant = r["variant"]
        family = "self" if variant == "nomix" else "hard"
        usage = (
            f"selected_neighbors={fmt_plain(r.get('selected_neighbor_count'), 4)}; "
            f"effective_neighbors={fmt_plain(r.get('effective_neighbor_count'), 4)}"
        )
        rows.append(
            base_row(
                version="V13",
                source_batch="V13/hard_gate",
                dataset=r["dataset"],
                dataset_id=r["dataset"],
                variant_family=family,
                variant=variant,
                ari_mean=num(r.get("ari")),
                ari_std=0.0,
                n_runs=1,
                seeds=r.get("seed"),
                gate_usage=usage,
                gate_usage_semantics="Selected/effective neighbor count; topk2 should be near 2",
                input_protocol="AHDPC processed NPZ; V13 Gumbel-Top-k hard-gate batch",
                readout="ARI column in runs.csv",
                status="completed",
                evidence_level="5-dataset x 3-seed paired hard-gate batch",
                source_artifact=str(path.relative_to(root)),
                notes="nomix is the matched self control; topk2 is the hard-gate intervention.",
            )
        )
    add_pair_deltas(
        rows,
        key_fields=("version", "source_batch", "dataset", "seeds"),
        control_variant="nomix",
        metric_field="ari_mean",
        scope="seed_matched_exact_runs_csv",
    )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v14(root: Path) -> list[dict[str, Any]]:
    path = root / "V14/advantage_5ds/runs.csv"
    raw = [r for r in read_csv(path) if r.get("status") == "completed"]
    rows: list[dict[str, Any]] = []
    for r in raw:
        variant = r["variant"]
        family = "self" if variant == "v14_nomix" else "learned"
        usage = (
            f"gate_mean={fmt_plain(r.get('gate_mean'), 5)}; "
            f"gate_last={fmt_plain(r.get('gate_last'), 5)}; "
            f"target_mean={fmt_plain(r.get('target_gate_mean'), 5)}"
        )
        rows.append(
            base_row(
                version="V14",
                source_batch="V14/advantage_5ds",
                dataset=r["dataset"],
                dataset_id=r["dataset"],
                variant_family=family,
                variant=variant,
                ari_mean=num(r.get("ari")),
                ari_std=0.0,
                n_runs=1,
                seeds=r.get("seed"),
                gate_usage=usage,
                gate_usage_semantics="Gate and target-gate scalars; not a binary mask rate",
                input_protocol="AHDPC processed NPZ; V14 advantage batch",
                readout="ARI column in runs.csv (head_ari also present)",
                status="completed",
                evidence_level="5-dataset x 3-seed paired advantage batch",
                source_artifact=str(path.relative_to(root)),
                notes="v14_nomix is the matched self control; full is the learned topology path.",
            )
        )
    add_pair_deltas(
        rows,
        key_fields=("version", "source_batch", "dataset", "seeds"),
        control_variant="v14_nomix",
        metric_field="ari_mean",
        scope="seed_matched_exact_runs_csv",
    )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v16_1(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "V16_1").rglob("*promotion_summary.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, list):
            continue
        for r in payload:
            dataset = r.get("dataset")
            if not dataset:
                continue
            values = {
                "self": ("clean_self", r.get("clean_self_ari_mean"), None),
                "fixed": ("clean_fixed", r.get("clean_fixed_ari_mean"), None),
                "random": ("clean_shuffled", r.get("clean_shuffled_ari_mean"), None),
                "learned": ("clean_v16_1", r.get("clean_v16_1_ari_mean"), r.get("clean_delta_self_mean")),
            }
            self_ari = num(r.get("clean_self_ari_mean"))
            fixed_ari = num(r.get("clean_fixed_ari_mean"))
            random_ari = num(r.get("clean_shuffled_ari_mean"))
            if fixed_ari is not None and self_ari is not None:
                values["fixed"] = ("clean_fixed", fixed_ari, fixed_ari - self_ari)
            if random_ari is not None and self_ari is not None:
                values["random"] = ("clean_shuffled", random_ari, random_ari - self_ari)
            for family, (variant, ari, delta) in values.items():
                rows.append(
                    base_row(
                        version="V16.1",
                        source_batch=f"V16_1/{path.parent.name}",
                        dataset=dataset,
                        dataset_id=dataset,
                        variant_family=family,
                        variant=variant,
                        ari_mean=num(ari),
                        ari_std=None,
                        n_runs=3,
                        seeds="7|42|123",
                        paired_control="clean_self" if family != "self" else None,
                        paired_delta_ari=num(delta) if family != "self" else None,
                        paired_delta_scope="reported promotion summary mean; seed-level rows absent",
                        gate_usage="NA",
                        gate_usage_semantics="Gate usage not recorded in promotion summary",
                        input_protocol="scRNA count / V16.1 clean-compound promotion protocol",
                        readout="clean ARI summary",
                        status="reported",
                        evidence_level="empirical_not_supported",
                        source_artifact=str(path.relative_to(root)),
                        notes="Rows are retained as restricted evidence; none is promoted by the preregistered rule.",
                    )
                )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v18(root: Path) -> list[dict[str, Any]]:
    root_dir = root / "V18/scmae_mainline_v2_2"
    summary = read_json(root_dir / "v18_summary_v2_2.json")
    manifest_path = root_dir / "v18_dataset_manifest_v2_2.json"
    if not manifest_path.is_file():
        matches = sorted(root_dir.glob("v18_dataset_manifest_v2_2_*.json"))
        if not matches:
            raise FileNotFoundError(f"V18 dataset manifest not found under {root_dir}")
        manifest_path = matches[-1]
    manifest = read_json(manifest_path)
    manifest_by_name = {str(r.get("name")): r for r in manifest.get("datasets", [])}
    family_map = {
        "scmae_only": "self",
        "latent_GW_frozen": "fixed",
        "v18_shuffled_E0": "random",
        "latent_C_exactzero": "static",
        "v18_full": "learned",
    }
    # v18_full is a learned HardConcrete gate. It is emitted once as learned
    # and once as hard in the wide alias view, but remains one source row here.
    rows: list[dict[str, Any]] = []
    for group in summary.get("groups", []):
        variant = str(group.get("variant"))
        family = family_map.get(variant, "auxiliary")
        dataset = str(group.get("dataset"))
        manifest_row = manifest_by_name.get(dataset, {})
        protocol = str(manifest_row.get("preprocessing") or "v18 preprocessing not recorded")
        usage_values = []
        for key in ("hard_open_rate_mean", "edge_retention_rate_mean", "abstention_rate_mean"):
            value = num(group.get(key))
            if value is not None:
                usage_values.append(f"{key}={fmt_plain(value, 5)}")
        usage = "; ".join(usage_values) if usage_values else "0/no gate field"
        usage_semantics = "HardConcrete open/edge-retention/abstention diagnostics" if usage_values else "No gate usage for scMAE-only or field absent"
        rows.append(
            base_row(
                version="V18",
                source_batch="V18/scmae_mainline_v2_2",
                dataset=dataset,
                dataset_id=str(manifest_row.get("dataset_id") or dataset),
                variant_family=family,
                variant=variant,
                ari_mean=num(group.get("ari_active_mean")),
                ari_std=num(group.get("ari_active_std")),
                n_runs=integer(group.get("runs_seen")),
                seeds=unique_join(group.get("seeds_completed", [])),
                gate_usage=usage,
                gate_usage_semantics=usage_semantics,
                input_protocol=f"v18_scmae_mainline_v2_2; {protocol}",
                readout="active ARI; V18 also publishes all-row ARI and Leiden readout",
                status="completed" if integer(group.get("completed", group.get("runs_seen"))) != 0 else "reported",
                evidence_level="149-dataset x 3-seed matrix" if family != "auxiliary" else "auxiliary V18 variant",
                source_artifact=str((root_dir / "v18_summary_v2_2.json").relative_to(root)),
                notes="Auxiliary variants are retained in the long table; requested family columns use only the explicit mapping.",
            )
        )
    # Pair all requested mapped variants to scmae_only using the same dataset
    # and matched three-run group means. This is an aggregate pairing because
    # the publication snapshot contains group summaries rather than raw rows.
    controls = {(r["dataset"], r["source_batch"]): r for r in rows if r["variant"] == "scmae_only"}
    for row in rows:
        control = controls.get((row["dataset"], row["source_batch"]))
        if control and row["variant"] != "scmae_only" and num(row["ari_mean"]) is not None and num(control["ari_mean"]) is not None:
            row["paired_control"] = "scmae_only"
            row["paired_delta_ari"] = num(row["ari_mean"]) - num(control["ari_mean"])
            row["paired_delta_scope"] = "matched three-run group means; raw per-seed rows not in snapshot"
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v19(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Sparse extension: exact run table plus dataset-level aggregate.
    ext_root = root / "V19/extended_sparse"
    ext_summary = read_json(ext_root / "extension_summary.json")
    for r in read_csv(ext_root / "extension_dataset_table.csv"):
        for family, variant, ari, std in (
            ("self", "scmae_only", r.get("scmae_ari_mean"), None),
            ("learned", "rg_full", r.get("rg_ari_mean"), None),
        ):
            delta = None if family == "self" else num(r.get("delta_ari_mean"))
            rows.append(
                base_row(
                    version="V19",
                    source_batch="V19/extended_sparse",
                    dataset=r.get("dataset"),
                    dataset_id=r.get("dataset_id"),
                    variant_family=family,
                    variant=variant,
                    ari_mean=num(ari),
                    ari_std=num(std),
                    n_runs=3,
                    seeds="7|42|123",
                    paired_control="scmae_only" if family != "self" else None,
                    paired_delta_ari=delta,
                    paired_delta_scope="seed-matched aggregate reported in extension_dataset_table.csv" if family != "self" else "reference",
                    gate_usage="NA",
                    gate_usage_semantics="Gate usage not recorded in sparse extension final table",
                    input_protocol=r.get("input_protocol"),
                    readout="mean ARI over three seeds",
                    status="completed",
                    evidence_level="13-dataset x 3-seed sparse extension",
                    source_artifact=str((ext_root / "extension_dataset_table.csv").relative_to(root)),
                    notes="RG wins scMAE on 6/13 by mean ARI; macro delta is negative in the published summary.",
                )
            )
    # PlantNet transfer panel: aggregate_metrics is metric-long; keep only ARI.
    transfer_root = root / "V19/plantnet_transfer"
    transfer_rows = read_csv(transfer_root / "aggregate_metrics.csv")
    by_dataset: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in transfer_rows:
        if r.get("metric") == "ari":
            by_dataset[r["dataset_id"]][r["variant"]] = r
    for dataset_id, variants in sorted(by_dataset.items()):
        control = num(variants.get("scmae_only", {}).get("mean"))
        for variant, family in (("scmae_only", "self"), ("rg_full", "learned")):
            r = variants.get(variant)
            if not r:
                continue
            current = num(r.get("mean"))
            rows.append(
                base_row(
                    version="V19",
                    source_batch="V19/plantnet_transfer",
                    dataset=dataset_id,
                    dataset_id=dataset_id,
                    variant_family=family,
                    variant=variant,
                    ari_mean=current,
                    ari_std=num(r.get("std")),
                    n_runs=integer(r.get("n")),
                    seeds="7|42|123",
                    paired_control="scmae_only" if family != "self" else None,
                    paired_delta_ari=current - control if family != "self" and current is not None and control is not None else None,
                    paired_delta_scope="matched aggregate_metrics means" if family != "self" else "reference",
                    gate_usage="NA",
                    gate_usage_semantics="Gate usage not recorded in transfer aggregate",
                    input_protocol=protocol_from_dataset_id(
                        dataset_id,
                        "v19_rg_plantnet_transfer; layer suffix not recorded",
                    ),
                    readout="ARI aggregate",
                    status="completed",
                    evidence_level="8-layer x 3-seed transfer aggregate",
                    source_artifact=str((transfer_root / "aggregate_metrics.csv").relative_to(root)),
                    notes="Transfer rows are a separate panel; they are not paired with sparse-extension rows of the same name.",
                )
            )
    # Post-freeze V19 is the canonical 11-stratum matched matrix.  Keep it as
    # its own source batch because its bridge/native layers must not be mixed
    # with the sparse-extension or PlantNet-transfer panels above.
    post_root = root / "V19/postfreeze"
    post_path = post_root / "comparison.csv"
    post_protocol = read_json(post_root / "stage_spec.json") if (post_root / "stage_spec.json").is_file() else {}
    post_variants = {
        "V19_scmae_only": "self",
        "V19_rg_constant_gate": "static",
        "V19_rg_default": "learned",
        "V19_rg_nomix": "learned",
        "V19_rg_reliability_off": "learned",
        "V19_rg_full": "learned",
    }
    post_raw = [r for r in read_csv(post_path) if r.get("metric") == "ari" and r.get("method") in post_variants]
    protocol_by_dataset = {}
    for dataset_id in post_protocol.get("dataset_ids", []):
        protocol_by_dataset[str(dataset_id)] = protocol_from_dataset_id(dataset_id)
    post_rows: list[dict[str, Any]] = []
    for r in post_raw:
        method = str(r["method"])
        family = post_variants[method]
        dataset_id = str(r["dataset_id"])
        usage = "0/no gate" if family == "self" else "NA (post-freeze aggregate does not publish realized gate usage)"
        usage_semantics = "No gate in scMAE-only control" if family == "self" else "Gate usage not recorded in V19 post-freeze comparison.csv"
        post_rows.append(base_row(
            version="V19",
            source_batch="V19/postfreeze",
            dataset=dataset_id,
            dataset_id=dataset_id,
            variant_family=family,
            variant=method,
            ari_mean=num(r.get("mean")),
            ari_std=num(r.get("std")),
            n_runs=integer(r.get("n")),
            seeds="7|42|123",
            gate_usage=usage,
            gate_usage_semantics=usage_semantics,
            input_protocol=protocol_by_dataset.get(dataset_id, "V19 post-freeze protocol (layer not recorded)"),
            readout="V19 post-freeze ARI aggregate",
            status="completed",
            evidence_level="11-stratum x 6-variant x 3-seed post-freeze matrix",
            source_artifact=str(post_path.relative_to(root)),
            notes="Canonical V19 post-freeze matrix; paired deltas are matched aggregate means from the same dataset/protocol/seed count.",
        ))
    controls = {r["dataset_id"]: r for r in post_rows if r["variant"] == "V19_scmae_only"}
    for row in post_rows:
        control = controls.get(row["dataset_id"])
        if control and row["variant"] != "V19_scmae_only" and num(row["ari_mean"]) is not None and num(control["ari_mean"]) is not None:
            row["paired_control"] = "V19_scmae_only"
            row["paired_delta_ari"] = num(row["ari_mean"]) - num(control["ari_mean"])
            row["paired_delta_scope"] = "matched three-seed aggregate means in V19 postfreeze comparison.csv"
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    rows.extend(post_rows)
    for row in rows:
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v20(root: Path) -> list[dict[str, Any]]:
    root_dir = root / "V20/full8_seed42"
    rows: list[dict[str, Any]] = []
    for r in read_csv(root_dir / "summary_by_dataset.csv"):
        path = root_dir / "run_summaries" / r["source_summary"]
        summary = read_json(path) if path.is_file() else {}
        diagnostics = summary.get("diagnostics") or {}
        history = diagnostics.get("history") or []
        last = history[-1] if history else {}
        usage = (
            f"requested_mask={fmt_plain(diagnostics.get('requested_mask_rate'), 5)}; "
            f"effective_mask={fmt_plain(last.get('effective_mask_rate'), 5)}; "
            f"gate_updates={diagnostics.get('gate_updates', 'NA')}; "
            f"nonzero_update_rate={fmt_plain(diagnostics.get('gate_nonzero_update_rate'), 5)}"
        )
        rows.append(
            base_row(
                version="V20",
                source_batch="V20/full8_seed42",
                dataset=r["dataset"],
                dataset_id=r["dataset"],
                variant_family="learned",
                variant=r["variant"],
                ari_mean=num(r.get("ari")),
                ari_std=0.0,
                n_runs=1,
                seeds=r.get("seed"),
                paired_control=None,
                paired_delta_ari=None,
                paired_delta_scope="NA - no paired control in final_results",
                gate_usage=usage,
                gate_usage_semantics="Requested training mask versus effective value-change mask; update rate is separate",
                input_protocol=r.get("input_protocol"),
                readout="clean embedding KMeans",
                status=r.get("status"),
                evidence_level="single-seed coarse screen",
                source_artifact=str((root_dir / "summary_by_dataset.csv").relative_to(root)),
                notes="No matched scMAE-only/random/control aggregate; do not infer a delta from V19 or V21.",
            )
        )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(None, row["status"], row["evidence_level"], row["notes"])
    return rows


def v21(root: Path) -> list[dict[str, Any]]:
    root_dir = root / "V21/formal6_graphfix"
    path = root_dir / "aggregate_metrics.csv"
    stage = read_json(root_dir / "stage_spec.json")
    protocol_by_dataset = {r["dataset"]: r.get("input_protocol") for r in stage.get("datasets", [])}
    raw = [r for r in read_csv(path)]
    rows: list[dict[str, Any]] = []
    for r in raw:
        variant = r["variant"]
        family = "self" if variant == "scmae_only" else "assignment-adversarial"
        usage = (
            f"eligible_rate={fmt_plain(r.get('final_assignment_eligible_rate'), 5)}; "
            f"effective_rate={fmt_plain(r.get('final_assignment_effective_rate'), 5)}; "
            f"gate_updates={r.get('gate_updates', 'NA')}"
        )
        rows.append(
            base_row(
                version="V21",
                source_batch="V21/formal6_graphfix",
                dataset=r["dataset"],
                dataset_id=r["dataset"],
                variant_family=family,
                variant=variant,
                ari_mean=num(r.get("ari")),
                ari_std=0.0,
                n_runs=1,
                seeds=r.get("seed"),
                gate_usage=usage,
                gate_usage_semantics="Assignment-eligible versus effective value-change rate; not requested mask rate",
                input_protocol=protocol_by_dataset.get(r["dataset"], r.get("input_protocol")),
                readout="v2 Student-t head ARI (formal graph-fix primary)",
                status="completed",
                evidence_level="36-run, six-dataset, three-seed formal matrix",
                source_artifact=str(path.relative_to(root)),
                notes="The published formal primary uses the v2 Student-t head; v3 clean-KMeans is not in this snapshot.",
            )
        )
    add_pair_deltas(
        rows,
        key_fields=("version", "source_batch", "dataset", "seeds"),
        control_variant="scmae_only",
        metric_field="ari_mean",
        scope="seed_matched_exact_aggregate_metrics",
    )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(num(row["paired_delta_ari"]), row["status"], row["evidence_level"], row["notes"])
    return rows


def v22(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("V22/hard_gate_resource_recovery/aggregate_summary.json", "hard", "hard"),
        ("V22/hard_gate_resource_recovery/aggregate_summary.json", "discriminator", "hard"),
        ("V22/cooperative_full_single_seed/aggregate_summary.json", "discriminator", "cooperative"),
    ]
    manifest_protocols: dict[tuple[str, str], str] = {}
    for manifest_rel in (
        "V22/hard_gate_resource_recovery/manifest.json",
        "V22/cooperative_full_single_seed/manifest.json",
    ):
        manifest_path = root / manifest_rel
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        for record in manifest.get("records", []):
            dataset_id = str(record.get("dataset_id") or "")
            protocol = record.get("input_protocol")
            if dataset_id and protocol:
                manifest_protocols[(manifest_rel.split("/", 2)[1], dataset_id)] = str(protocol)
    for rel, family, branch in specs:
        path = root / rel
        payload = read_json(path)
        for r in payload.get("rows", []):
            dataset = r.get("dataset") or r.get("name") or r.get("dataset_id")
            if not dataset:
                dataset = r.get("dataset_id")
            run_key = str(r.get("run_key", ""))
            variant = run_key.split("::")[1] if "::" in run_key else (
                "v22_topology_discriminator_hard_gate" if branch == "hard" else "v22_topology_discriminator_cooperative_keep_gate"
            )
            status = str(r.get("status", "unknown"))
            usage_parts = []
            for key in ("effective_mask_rate_last", "gate_nonzero_update_rate", "gate_updates", "d_real_accuracy_last", "d_gate_fake_accuracy_last", "d_scmae_fake_accuracy_last"):
                if r.get(key) is not None:
                    value = fmt_plain(r.get(key), 5) if "rate" in key or "accuracy" in key else str(r.get(key))
                    usage_parts.append(f"{key}={value}")
            rows.append(
                base_row(
                    version="V22",
                    source_batch="V22/hard_gate_resource_recovery" if branch == "hard" else "V22/cooperative_full_single_seed",
                    dataset=dataset,
                    dataset_id=r.get("dataset_id") or dataset,
                    variant_family=family,
                    variant=variant,
                    ari_mean=num(r.get("ari")),
                    ari_std=None,
                    n_runs=1,
                    seeds="42",
                    paired_control=None,
                    paired_delta_ari=None,
                    paired_delta_scope="NA - no matched self/scMAE control in V22 final artifact",
                    gate_usage="; ".join(usage_parts) if usage_parts else "NA",
                    gate_usage_semantics="Effective mask/update and discriminator accuracies; hard and discriminator aliases intentionally overlap for hard branch",
                    input_protocol=(
                        r.get("input_protocol")
                        or manifest_protocols.get((rel.split("/", 2)[1], str(r.get("dataset_id") or "")))
                        or str(r.get("stratum", "protocol not recorded"))
                    ),
                    readout="clean embedding KMeans",
                    status=status,
                    evidence_level="single-seed full-component audit" if status == "completed" else "incomplete_compute",
                    source_artifact=str(path.relative_to(root)),
                    notes="V22 has no matched scMAE-only/random baseline; two cooperative jobs remain incomplete_compute.",
                )
            )
    for row in rows:
        row["failure_diagnosis"] = diagnosis(None, row["status"], row["evidence_level"], row["notes"])
    return rows


def normalize_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loader in (v09_advantage, v09_legacy, v09_single_seed, v09_paper, v10, v11, v12, v13, v14, v16_1, v18, v19, v20, v21, v22):
        rows.extend(loader(root))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wide_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["version"], row["source_batch"], row["dataset"])].append(row)
    result: list[dict[str, Any]] = []
    for (version, batch, dataset), group in sorted(grouped.items()):
        first = group[0]
        out: dict[str, Any] = {
            "version": version,
            "source_batch": batch,
            "dataset": dataset,
            "dataset_id": first.get("dataset_id"),
            "input_protocol": unique_join([r.get("input_protocol") for r in group]),
            "readout": unique_join([r.get("readout") for r in group]),
            "status": unique_join([r.get("status") for r in group]),
            "evidence_level": unique_join([r.get("evidence_level") for r in group]),
        }
        for family in FAMILIES:
            candidates = [r for r in group if r.get("variant_family") == family]
            if not candidates:
                out[f"{family}__variant"] = "NA"
                out[f"{family}__ari"] = "NA"
                out[f"{family}__delta_ari"] = "NA"
                out[f"{family}__gate_usage"] = "NA"
                out[f"{family}__diagnosis"] = "NA"
                continue
            # Aggregate repeated seeds for the same variant, while retaining
            # distinct configs/aliases as separate ``||`` pieces.  The long
            # table remains the source of exact per-seed values.
            by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for candidate in candidates:
                by_variant[str(candidate.get("variant"))].append(candidate)
            pieces = []
            for variant, variant_rows in sorted(by_variant.items()):
                pieces.append(
                    (
                        variant,
                        row_stat_text(variant_rows, "ari_mean", std_field="ari_std"),
                        row_stat_text(variant_rows, "paired_delta_ari", signed=True),
                        unique_join([r.get("gate_usage") for r in variant_rows]),
                        unique_join([r.get("failure_diagnosis") for r in variant_rows]),
                    )
                )
            out[f"{family}__variant"] = " || ".join(piece[0] for piece in pieces)
            out[f"{family}__ari"] = " || ".join(piece[1] for piece in pieces)
            out[f"{family}__delta_ari"] = " || ".join(piece[2] for piece in pieces)
            out[f"{family}__gate_usage"] = " || ".join(piece[3] for piece in pieces)
            out[f"{family}__diagnosis"] = " || ".join(piece[4] for piece in pieces)
        result.append(out)
    return result


def version_rows(rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_version[row["version"]].append(row)
    out: list[dict[str, Any]] = []
    for version, (status, description) in VERSION_COVERAGE.items():
        group = by_version.get(version, [])
        out.append(
            {
                "version": version,
                "coverage_status": "available" if group else status,
                "dataset_count": len({r["dataset"] for r in group}),
                "row_count": len(group),
                "completed_row_count": sum(r.get("status") == "completed" for r in group),
                "paired_row_count": sum(num(r.get("paired_delta_ari")) is not None for r in group),
                "description": description,
            }
        )
    return out


def write_markdown(path: Path, rows: list[dict[str, Any]], wide: list[dict[str, Any]], coverage: list[dict[str, Any]], source: str, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = sum(r.get("status") == "completed" for r in rows)
    paired = sum(num(r.get("paired_delta_ari")) is not None for r in rows)
    negative = sum((num(r.get("paired_delta_ari")) or 0.0) < -0.03 for r in rows if num(r.get("paired_delta_ari")) is not None)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# V1-V22 unified failure diagnostic\n\n")
        handle.write("This report is generated from the latest metadata-only `final_results` snapshot. It keeps source batches separate. Exact paired Delta ARI requires the same dataset, seed, readout, input protocol, and budget; published aggregate pairs are retained only when the source artifact explicitly supplies the match, with their scope recorded in the long table.\n\n")
        handle.write(f"- Source snapshot: `{source}`\n- Snapshot SHA256 (recursive path+content): `{digest}`\n- Generated: `{datetime.now(timezone.utc).isoformat()}`\n- Long rows: `{len(rows)}`; completed rows: `{completed}`; rows with paired Delta ARI: `{paired}`; paired regressions below -0.03: `{negative}`\n\n")
        handle.write("## Family semantics\n\n")
        handle.write("`self` is the same-batch no-gate/no-mix reference where available. `fixed`, `random`, `static`, and `learned` retain source semantics; V19 no-mix/reliability ablations remain under `learned` because they still use the learned RG path, while V19 constant-gate is `static`. `hard` is the explicit hard-selection family (V13 topk2 and the V22 hard branch). `assignment-adversarial` is the V21 assignment-objective intervention. `discriminator` is the V22 discriminator-backed branch; the V22 hard branch is intentionally aliased into both `hard` and `discriminator` views. `NA` means the final snapshot did not publish a matched control or a usage denominator.\n\n")
        handle.write("## Version coverage\n\n")
        handle.write("| Version | Coverage | Datasets | Long rows | Paired rows | Boundary |\n|---|---|---:|---:|---:|---|\n")
        for r in coverage:
            handle.write(f"| {r['version']} | {r['coverage_status']} | {r['dataset_count']} | {r['row_count']} | {r['paired_row_count']} | {r['description']} |\n")
        handle.write("\n## Per-dataset table\n\n")
        handle.write("Each family cell is `variant; ARI; paired Delta ARI; Gate usage`. The CSV is the canonical machine-readable form; this wide table preserves multiple configs with `||` rather than selecting one by ARI.\n\n")
        headers = ["Version", "Batch", "Dataset", "Protocol", "self", "fixed", "random", "static", "learned", "hard", "assignment-adversarial", "discriminator"]
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "---|" * len(headers) + "\n")
        for r in wide:
            cells = [r["version"], r["source_batch"], r["dataset"], r["input_protocol"]]
            for family in FAMILIES:
                cell = (
                    f"{r[f'{family}__variant']}; ARI={r[f'{family}__ari']}; "
                    f"d={r[f'{family}__delta_ari']}; gate={r[f'{family}__gate_usage']}"
                )
                cells.append(cell.replace("|", "/"))
            handle.write("| " + " | ".join(str(c).replace("\n", " ") for c in cells) + " |\n")
        handle.write("\n## Interpretation boundaries\n\n")
        handle.write("- `NA - no paired control` is not a zero effect. It means the snapshot cannot support a paired claim.\n")
        handle.write("- V20 is single-seed and has no matched control; its effective mask is not a paired performance result.\n")
        handle.write("- V21 is the formal v2 Student-t-head graph-fix readout. A clean-embedding KMeans audit is a separate readout and is not merged here.\n")
        handle.write("- V22 hard and cooperative branches are single-seed; the cooperative panel retains `incomplete_compute` rows and has no matched scMAE-only baseline.\n")
        handle.write("- V18/V19 group summaries are matched aggregate means where raw per-seed rows are absent from the publication snapshot; their scope is explicitly recorded in `paired_delta_scope`.\n")
        handle.write("\n## Reproduction\n\n")
        handle.write("```bash\npython scripts/analysis/build_v1_v22_failure_diagnostic.py --final-results-root <path-to-final_results>\n```\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-results-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.final_results_root
    if root is None:
        root = Path(__file__).resolve().parents[2] / "result/final_results"
    root = root.resolve()
    if not root.is_dir():
        raise SystemExit(f"final_results root not found: {root}")
    output = args.output_dir or (Path(__file__).resolve().parents[2] / "reports")
    output = output.resolve()
    rows = normalize_rows(root)
    wide = wide_rows(rows)
    coverage = version_rows(rows, root)
    fields = list(base_row().keys())
    write_csv(output / "v1_v22_unified_failure_diagnostic_long_20260814.csv", rows, fields)
    wide_fields = list(wide[0].keys()) if wide else ["version", "source_batch", "dataset"]
    write_csv(output / "v1_v22_unified_failure_diagnostic_wide_20260814.csv", wide, wide_fields)
    write_csv(output / "v1_v22_unified_failure_diagnostic_coverage_20260814.csv", coverage, list(coverage[0].keys()))
    digest = source_hash(root)
    provenance = {
        "source_root": "result/final_results",
        "source_recursive_sha256": digest,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count_long": len(rows),
        "row_count_wide": len(wide),
        "requested_families": list(FAMILIES),
        "pairing_rule": "exact same source batch/dataset/seed/readout/input protocol/budget where raw rows exist; source-reported aggregate pairs are retained only with an explicit paired_delta_scope; otherwise NA",
    }
    (output / "v1_v22_unified_failure_diagnostic_provenance_20260814.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    write_markdown(
        output / "v1_v22_unified_failure_diagnostic_20260814.md",
        rows,
        wide,
        coverage,
        "result/final_results",
        digest,
    )
    print(json.dumps({"source": str(root), "long_rows": len(rows), "wide_rows": len(wide), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
