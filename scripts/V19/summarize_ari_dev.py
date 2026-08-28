#!/usr/bin/env python
"""Summarize and select the V19 ARI-selected development matrix."""

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

from scripts.V19.tune_ari_dev import (  # noqa: E402
    DEFAULT_OUTPUT,
    FORMAL_SEEDS,
    PROTOCOL_ID,
    SELECTION_EVIDENCE,
    TARGET_DATASET_IDS,
    _read_json,
    _write_json,
    catalog,
    load_manifest,
)


METRICS = ("ari", "nmi", "acc")


def _normalise_name(value: str) -> str:
    return str(value).strip().casefold().replace(" ", "_").replace("-", "_")


def _record_names(record: dict[str, Any]) -> set[str]:
    dataset_id = str(record["dataset_id"])
    base = dataset_id.split("__", 1)[0]
    values = {str(record.get("name", "")), base, dataset_id}
    if base == "mouse_retina":
        values.update({"Mouse_retina", "mouse_retina"})
    if base == "baron_human":
        values.update({"Baron Human", "baron_human"})
    if base == "campbell":
        values.update({"Campbell", "campbell"})
    return {_normalise_name(value) for value in values if value}


def _load_baseline(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    best: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("ARI")
            if value in (None, ""):
                continue
            try:
                ari = float(value)
            except ValueError:
                continue
            key = _normalise_name(str(row.get("dataset", "")))
            if key and math.isfinite(ari):
                best[key] = max(best.get(key, float("-inf")), ari)
    return best


def _read_summaries(root: Path, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/summary.json")):
        if "attempts" in path.parts:
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if payload.get("ari_dev_protocol_id") != PROTOCOL_ID or payload.get("stage") != stage:
            continue
        if payload.get("status") != "completed":
            continue
        required_audit = {
            "labels_used_during_fit": False,
            "labels_used_during_preprocessing": False,
            "labels_used_for_graph": False,
            "labels_used_for_gate": False,
            "labels_used_for_loss": False,
            "labels_used_for_selection": True,
            "selection_evidence_type": SELECTION_EVIDENCE,
        }
        if any(payload.get(key) != value for key, value in required_audit.items()):
            raise ValueError(f"ARI label-boundary audit failed: {path}")
        metrics = payload.get("metrics", {})
        if metrics.get("labels_available") is not True:
            raise ValueError(f"ARI development run has no post-fit labels: {path}")
        for metric in METRICS:
            if metric not in metrics or not math.isfinite(float(metrics[metric])):
                raise ValueError(f"non-finite {metric} in {path}")
        rows.append(payload)
    return rows


def _stage_spec(root: Path) -> dict[str, Any]:
    spec = _read_json(root / "stage_spec.json")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"invalid ARI stage spec: {root}")
    return spec


def _validate_complete(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    expected = {
        (str(dataset_id), str(candidate_id), int(seed))
        for dataset_id in spec["dataset_ids"]
        for candidate_id in spec["candidate_ids"]
        for seed in spec["seed_order"]
    }
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["dataset_id"]), str(row["candidate_id"]), int(row["seed"]))
        if key in index:
            raise ValueError(f"duplicate ARI run key: {key}")
        index[key] = row
    if set(index) != expected:
        missing = sorted(expected - set(index))
        extra = sorted(set(index) - expected)
        raise RuntimeError(f"incomplete ARI stage: got {len(index)}/{len(expected)}, missing={missing[:5]}, extra={extra[:5]}")
    return index


def _reference_index(reference_root: Path, spec: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    ref_spec = _stage_spec(reference_root)
    if ref_spec.get("stage") != "reference":
        raise ValueError("reference root does not contain a reference stage")
    rows = _read_summaries(reference_root, "reference")
    expected = {(str(dataset_id), int(seed)) for dataset_id in spec["dataset_ids"] for seed in FORMAL_SEEDS}
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["dataset_id"]), int(row["seed"]))
        if key in index:
            raise ValueError(f"duplicate scMAE reference: {key}")
        index[key] = row
    if set(index) != expected:
        missing = sorted(expected - set(index))
        raise RuntimeError(f"incomplete scMAE reference: got {len(index)}/{len(expected)}, missing={missing[:5]}")
    return index


def _score_candidate(
    candidate_id: str,
    index: dict[tuple[str, str, int], dict[str, Any]],
    spec: dict[str, Any],
    records: dict[str, dict[str, Any]],
    reference: dict[tuple[str, int], dict[str, Any]] | None,
    sota: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_rows: list[dict[str, Any]] = []
    dataset_means: list[float] = []
    dataset_std: list[float] = []
    above_scmae = 0
    above_sota = 0
    sota_covered = 0
    for dataset_id in spec["dataset_ids"]:
        values = [
            float(index[(str(dataset_id), candidate_id, int(seed))]["metrics"]["ari"])
            for seed in spec["seed_order"]
        ]
        mean_ari = float(mean(values))
        std_ari = float(pstdev(values)) if len(values) > 1 else 0.0
        dataset_means.append(mean_ari)
        dataset_std.append(std_ari)
        reference_mean = None
        if reference is not None:
            reference_values = [
                float(reference[(str(dataset_id), int(seed))]["metrics"]["ari"])
                for seed in spec["seed_order"]
            ]
            reference_mean = float(mean(reference_values))
            if mean_ari > reference_mean:
                above_scmae += 1
        sota_value = None
        for name in _record_names(records[str(dataset_id)]):
            if name in sota:
                sota_value = float(sota[name])
                break
        if sota_value is not None:
            sota_covered += 1
            if mean_ari > sota_value:
                above_sota += 1
        dataset_rows.append(
            {
                "candidate_id": candidate_id,
                "dataset_id": str(dataset_id),
                "dataset": str(records[str(dataset_id)].get("name", dataset_id)),
                "ari_mean": mean_ari,
                "ari_std": std_ari,
                "ari_values": values,
                "scmae_ari_mean": reference_mean,
                "delta_vs_scmae": None if reference_mean is None else mean_ari - reference_mean,
                "archived_sota_ari": sota_value,
                "above_scmae": reference_mean is not None and mean_ari > reference_mean,
                "above_archived_sota": sota_value is not None and mean_ari > sota_value,
            }
        )
    score = {
        "candidate_id": candidate_id,
        "macro_ari": float(mean(dataset_means)),
        "above_scmae_datasets": int(above_scmae),
        "above_archived_sota_datasets": int(above_sota),
        "archived_sota_coverage": int(sota_covered),
        "worst_dataset_ari": float(min(dataset_means)),
        "mean_seed_std": float(mean(dataset_std)),
        "max_seed_std": float(max(dataset_std)),
        "n_datasets": len(dataset_means),
        "dataset_means": {row["dataset_id"]: row["ari_mean"] for row in dataset_rows},
    }
    return score, dataset_rows


def summarize_screen(root: Path, manifest_path: Path, *, top_k: int) -> dict[str, Any]:
    spec = _stage_spec(root)
    if spec.get("stage") != "screen" or int(spec.get("expected_runs", 0)) != 384:
        raise ValueError("screen contract must be 384 runs")
    rows = _read_summaries(root, "screen")
    index = _validate_complete(rows, spec)
    manifest = load_manifest(manifest_path)
    records = {str(row["dataset_id"]): row for row in manifest["datasets"]}
    scores: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    for candidate_id in spec["candidate_ids"]:
        score, dataset_rows = _score_candidate(candidate_id, index, spec, records, None, {})
        scores.append(score)
        for row in dataset_rows:
            table.append({**row, "seed": 42})
    scores.sort(key=lambda row: (-row["macro_ari"], -row["worst_dataset_ari"], row["candidate_id"]))
    top = scores[: int(top_k)]
    catalog_by_id = {str(row["candidate_id"]): row for row in catalog()}
    selection = {
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "stage": "screen",
        "selection_evidence_type": SELECTION_EVIDENCE,
        "selection_metric": "8-dataset macro ARI at seed 42",
        "labels_used_for_selection": True,
        "candidate_id": top[0]["candidate_id"],
        "top_candidate_ids": [row["candidate_id"] for row in top],
        "top_candidate_definitions": [catalog_by_id[row["candidate_id"]] for row in top],
        "candidate_scores": scores,
        "expected_runs": 384,
        "completed_runs": len(index),
        "manifest_id": manifest.get("manifest_id"),
    }
    _write_json(root / "screen_selection.json", selection)
    _write_json(root / "top12_config.json", selection)
    _write_json(root / "screen_summary.json", selection)
    _write_csv(root / "screen_candidate_scores.csv", scores)
    _write_csv(root / "screen_dataset_ari.csv", table)
    return selection


def summarize_refine(
    root: Path,
    manifest_path: Path,
    reference_root: Path,
    baseline_csv: Path | None,
) -> dict[str, Any]:
    spec = _stage_spec(root)
    if spec.get("stage") != "refine" or int(spec.get("expected_runs", 0)) != 288:
        raise ValueError("refine contract must be 288 runs")
    rows = _read_summaries(root, "refine")
    index = _validate_complete(rows, spec)
    manifest = load_manifest(manifest_path)
    records = {str(row["dataset_id"]): row for row in manifest["datasets"]}
    ref_spec = _stage_spec(reference_root)
    reference_rows = _read_summaries(reference_root, "reference")
    reference_index = {(str(row["dataset_id"]), int(row["seed"])): row for row in reference_rows}
    expected_ref = {(str(dataset_id), int(seed)) for dataset_id in spec["dataset_ids"] for seed in spec["seed_order"]}
    if set(reference_index) != expected_ref:
        raise RuntimeError(f"reference is incomplete: {len(reference_index)}/{len(expected_ref)}")
    sota = _load_baseline(baseline_csv)
    scores: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for candidate_id in spec["candidate_ids"]:
        score, rows_for_candidate = _score_candidate(candidate_id, index, spec, records, reference_index, sota)
        scores.append(score)
        detail.extend(rows_for_candidate)
    scores.sort(
        key=lambda row: (
            -row["macro_ari"],
            -row["above_scmae_datasets"],
            -row["above_archived_sota_datasets"],
            -row["worst_dataset_ari"],
            row["mean_seed_std"],
            row["candidate_id"],
        )
    )
    best = scores[0]
    definitions = {str(row["candidate_id"]): row for row in catalog(list(spec["candidate_ids"]))}
    selected = {
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "stage": "refine",
        "selection_method": "global 8-dataset ARI selection with fixed scMAE and archived SOTA secondary counts",
        "selection_evidence_type": SELECTION_EVIDENCE,
        "selection_target": "maximize one shared RG configuration across biological and sparse-text data",
        "selection_status": "selected",
        "no_go": False,
        "candidate_id": best["candidate_id"],
        "candidate_family": definitions[best["candidate_id"]]["family"],
        "overrides": definitions[best["candidate_id"]]["overrides"],
        "top_candidate_ids": [row["candidate_id"] for row in scores[:12]],
        "best_summary": best,
        "candidate_scores": scores,
        "manifest_id": manifest.get("manifest_id"),
        "reference_root": str(reference_root.resolve()),
        "base_config": str(Path(spec["base_config"]).resolve()),
        "labels_used_during_fit": False,
        "labels_used_for_graph": False,
        "labels_used_for_gate": False,
        "labels_used_for_loss": False,
        "labels_used_for_selection": True,
        "K_source": "benchmark_oracle_from_y",
        "development_scope": "all 8 target datasets were used for ARI selection; this is development evidence, not held-out generalization",
    }
    summary = {
        "status": "completed",
        "protocol_id": PROTOCOL_ID,
        "stage": "refine",
        "selection_evidence_type": SELECTION_EVIDENCE,
        "expected_runs": 288,
        "completed_runs": len(index),
        "reference_completed_runs": len(reference_index),
        "candidate_scores": scores,
        "selected_config": selected,
        "manifest_id": manifest.get("manifest_id"),
        "labels_used_during_fit": False,
        "labels_used_for_selection": True,
    }
    _write_json(root / "ari_selection.json", summary)
    _write_json(root / "selected_config.json", selected)
    _write_csv(root / "candidate_scores.csv", scores)
    _write_csv(root / "dataset_ari_table.csv", detail)
    return summary


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
            serial = dict(row)
            for key, value in list(serial.items()):
                if isinstance(value, (dict, list)):
                    serial[key] = json.dumps(value, ensure_ascii=True, sort_keys=True)
            writer.writerow(serial)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize V19 ARI development evidence")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "refine"), required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--baseline-csv", type=Path, default=ROOT / "result" / "baseline_comparison" / "summary.csv")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    if args.stage == "screen":
        if int(args.top_k) != 12:
            raise ValueError("screen contract requires top-k=12")
        result = summarize_screen(args.output_dir, args.manifest, top_k=12)
    else:
        if args.reference_dir is None:
            raise ValueError("refine summary requires --reference-dir")
        result = summarize_refine(args.output_dir, args.manifest, args.reference_dir, args.baseline_csv)
    print(json.dumps(result.get("selected_config", result), ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
