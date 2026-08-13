#!/usr/bin/env python
"""Summarize the frozen V19 ARI development final matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.V19.summarize_ari_dev import (  # noqa: E402
    METRICS,
    _load_baseline,
    _normalise_name,
    _read_json,
)
from scripts.V19.tune_ari_dev import FINAL_PROTOCOL_ID, PROTOCOL_ID, SELECTION_EVIDENCE, _write_json


def _read_final(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = _read_json(root / "stage_spec.json")
    if spec.get("protocol_id") != FINAL_PROTOCOL_ID:
        raise ValueError("not a V19 ARI final root")
    expected = set(str(value) for value in spec["expected_run_keys"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(root.glob("**/summary.json")):
        if "attempts" in path.parts:
            continue
        payload = _read_json(path)
        if payload.get("status") != "completed":
            continue
        key = str(payload.get("run_key", ""))
        if key in seen or key not in expected:
            raise ValueError(f"duplicate or unexpected final run key: {key}")
        seen.add(key)
        audit = {
            "labels_used_during_fit": False,
            "labels_used_during_preprocessing": False,
            "labels_used_for_graph": False,
            "labels_used_for_gate": False,
            "labels_used_for_loss": False,
            "labels_used_for_selection": True,
            "selection_evidence_type": SELECTION_EVIDENCE,
        }
        if any(payload.get(name) != value for name, value in audit.items()):
            raise ValueError(f"final label audit failed: {path}")
        if payload.get("final_protocol_id") != FINAL_PROTOCOL_ID:
            raise ValueError(f"wrong final protocol: {path}")
        if payload.get("metrics", {}).get("labels_available") is not True:
            raise ValueError(f"missing final metrics: {path}")
        for metric in METRICS:
            if not math.isfinite(float(payload["metrics"][metric])):
                raise ValueError(f"non-finite {metric}: {path}")
        rows.append(payload)
    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(f"final matrix incomplete: {len(seen)}/{len(expected)}, missing={missing[:5]}")
    return spec, rows


def _dataset_baseline_names(dataset_id: str, dataset: str) -> set[str]:
    base = str(dataset_id).split("__", 1)[0]
    values = {str(dataset), base, str(dataset_id)}
    if base == "mouse_retina":
        values.update({"Mouse_retina", "mouse_retina"})
    if base == "baron_human":
        values.update({"Baron Human", "baron_human"})
    return {_normalise_name(value) for value in values if value}


def summarize(final_root: Path, output_dir: Path, baseline_csv: Path | None) -> dict[str, Any]:
    spec, rows = _read_final(final_root)
    selected = _read_json(final_root.parent / "refine" / "selected_config.json")
    baseline = _load_baseline(baseline_csv)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset_id"]), str(row["evaluation_variant"]))].append(row)
    table: list[dict[str, Any]] = []
    for (dataset_id, variant), group in sorted(grouped.items()):
        record = group[0]
        summary = {
            "dataset_id": dataset_id,
            "dataset": record.get("dataset"),
            "input_protocol": record.get("input_protocol"),
            "variant": variant,
            "n": len(group),
        }
        for metric in METRICS:
            values = [float(row["metrics"][metric]) for row in group]
            summary[f"{metric}_mean"] = float(mean(values))
            summary[f"{metric}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        names = _dataset_baseline_names(dataset_id, str(record.get("dataset", "")))
        sota_values = [baseline[name] for name in names if name in baseline]
        summary["archived_sota_ari"] = max(sota_values) if sota_values else None
        summary["above_archived_sota"] = bool(
            summary["archived_sota_ari"] is not None and summary["ari_mean"] > summary["archived_sota_ari"]
        )
        table.append(summary)

    by_key = {(row["dataset_id"], row["variant"]): row for row in table}
    selected_variant = by_key
    deltas: list[dict[str, Any]] = []
    for dataset_id in sorted({str(row["dataset_id"]) for row in rows}):
        rg = selected_variant[(dataset_id, "rg_full")]
        scmae = selected_variant[(dataset_id, "scmae_only")]
        deltas.append(
            {
                "dataset_id": dataset_id,
                "dataset": rg["dataset"],
                "rg_ari_mean": rg["ari_mean"],
                "scmae_ari_mean": scmae["ari_mean"],
                "delta_ari": rg["ari_mean"] - scmae["ari_mean"],
                "rg_above_archived_sota": rg["above_archived_sota"],
                "archived_sota_ari": rg["archived_sota_ari"],
            }
        )
    rg = [row for row in deltas if row["rg_ari_mean"] is not None]
    result = {
        "status": "completed",
        "protocol_id": FINAL_PROTOCOL_ID,
        "tuning_protocol_id": PROTOCOL_ID,
        "selection_evidence_type": SELECTION_EVIDENCE,
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_overrides": selected.get("overrides", {}),
        "expected_runs": int(spec["expected_runs"]),
        "completed_runs": len(rows),
        "datasets": sorted({str(row["dataset_id"]) for row in rows}),
        "variants": list(spec["variants"]),
        "rg_above_scmae_datasets": int(sum(row["delta_ari"] > 0.0 for row in deltas)),
        "rg_above_archived_sota_datasets": int(sum(bool(row["rg_above_archived_sota"]) for row in deltas)),
        "rg_mean_macro_ari": float(mean(row["rg_ari_mean"] for row in rg)),
        "scmae_mean_macro_ari": float(mean(row["scmae_ari_mean"] for row in deltas)),
        "outputs": {},
        "labels_used_during_fit": False,
        "labels_used_for_selection": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "final_dataset_variant_table.csv"
    delta_path = output_dir / "rg_vs_scmae.csv"
    _write_csv(table_path, table)
    _write_csv(delta_path, deltas)
    result["outputs"] = {"dataset_variant_table": str(table_path), "rg_vs_scmae": str(delta_path)}
    _write_json(output_dir / "final_summary.json", result)
    report = [
        "# V19 ARI-selected development final comparison",
        "",
        f"- Selected RG candidate: `{selected.get('candidate_id')}`",
        f"- Selection evidence: `{SELECTION_EVIDENCE}`",
        "- All 8 datasets were used for selection; these are development results, not held-out generalization evidence.",
        "- Labels were used only for benchmark K, post-fit metrics, and candidate selection.",
        "",
        "| Dataset | RG-full ARI | scMAE ARI | Delta | RG > archived SOTA |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in deltas:
        report.append(
            f"| {row['dataset_id']} | {row['rg_ari_mean']:.4f} | {row['scmae_ari_mean']:.4f} | "
            f"{row['delta_ari']:+.4f} | {'yes' if row['rg_above_archived_sota'] else 'no/NA'} |"
        )
    report.extend([
        "",
        "Variant-level means and standard deviations are in `final_dataset_variant_table.csv`; archived baseline rows retain their source provenance and are not fresh matched reruns.",
    ])
    report_path = output_dir / "final_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    result["outputs"]["report"] = str(report_path)
    _write_json(output_dir / "final_summary.json", result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize V19 ARI final matrix")
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, default=ROOT / "result" / "baseline_comparison" / "summary.csv")
    args = parser.parse_args()
    result = summarize(args.final_dir, args.output_dir, args.baseline_csv)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
