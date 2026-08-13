#!/usr/bin/env python
"""Audit and summarize a fixed-parameter V19 RG/matched-scMAE matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = ("ari", "nmi", "acc")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _aggregate(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(mean(clean)),
        "std": float(stdev(clean)) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def _load_rows(root: Path, dataset_ids: set[str]) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(root.glob("**/summary.json")):
        if "attempts" in path.parts:
            continue
        payload = _read_json(path)
        dataset_id = str(payload.get("dataset_id", ""))
        variant = str(payload.get("evaluation_variant") or payload.get("variant") or "")
        seed = int(payload.get("seed", -1))
        if dataset_id not in dataset_ids or not variant or seed < 0:
            continue
        key = (dataset_id, variant, seed)
        if key in rows:
            raise ValueError(f"duplicate completed run key: {key}")
        rows[key] = {"path": path, "payload": payload}
    return rows


def _load_external_best(path: Path, dataset_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    underlying = {dataset_id.split("__", 1)[0] for dataset_id in dataset_ids}
    candidates: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset = str(row.get("dataset", "")).casefold().replace(" ", "_")
            value = row.get("ARI")
            if dataset not in underlying or value in (None, ""):
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            if math.isfinite(numeric):
                candidates[dataset].append((str(row.get("model", "unknown")), numeric))
    output = {}
    for dataset, values in candidates.items():
        method, value = max(values, key=lambda item: item[1])
        output[dataset] = {"method": method, "ari": float(value)}
    return output


def summarize(root: Path, reference_root: Path | None, baseline_csv: Path | None) -> dict[str, Any]:
    spec = _read_json(root / "experiment_spec.json")
    dataset_ids = [str(value) for value in spec["dataset_ids"]]
    variants = [str(value) for value in spec["variants"]]
    seeds = [int(value) for value in spec["seeds"]]
    expected = {(dataset, variant, seed) for dataset in dataset_ids for variant in variants for seed in seeds}
    rows = _load_rows(root, set(dataset_ids))
    actual = set(rows)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    audit_errors: list[str] = []
    if missing:
        audit_errors.append(f"missing {len(missing)} expected runs")
    if unexpected:
        audit_errors.append(f"found {len(unexpected)} unexpected runs")

    for key in sorted(expected.intersection(actual)):
        path = rows[key]["path"]
        payload = rows[key]["payload"]
        if payload.get("status") != "completed":
            audit_errors.append(f"non-completed summary: {path}")
        if payload.get("labels_used_during_fit") is not False:
            audit_errors.append(f"label-fit audit failed: {path}")
        if payload.get("labels_used_during_preprocessing") is not False:
            audit_errors.append(f"label-preprocess audit failed: {path}")
        if payload.get("K_source") != "benchmark_oracle_from_y":
            audit_errors.append(f"unexpected K source: {path}")
        for metric in METRICS:
            value = payload.get("metrics", {}).get(metric)
            if value is None or not math.isfinite(float(value)):
                audit_errors.append(f"invalid {metric}: {path}")
        run_dir = path.parent
        for required in (
            "status.json",
            "run_record.json",
            "resolved_config.json",
            "metrics.json",
            "predictions.npy",
            "labels_true.npy",
            "embedding_final.npy",
        ):
            if not (run_dir / required).is_file():
                audit_errors.append(f"missing {required}: {run_dir}")
        resolved = _read_json(run_dir / "resolved_config.json")
        if int(resolved.get("knn_pca_dim", -1)) != int(spec["pca_requested"]):
            audit_errors.append(f"requested PCA mismatch: {run_dir}")
        if key[1] == "rg_full":
            graph = _read_json(run_dir / "neighbor_graph_profile.json")
            actual_dim = int(graph.get("knn_pca_dim", 0))
            if actual_dim <= 0 or actual_dim > int(spec["pca_requested"]):
                audit_errors.append(f"realized PCA mismatch: {run_dir}")

    reference_rows = (
        _load_rows(reference_root, set(dataset_ids)) if reference_root is not None else {}
    )
    external = _load_external_best(baseline_csv, dataset_ids) if baseline_csv is not None else {}
    aggregate_rows: list[dict[str, Any]] = []
    dataset_summary: dict[str, Any] = {}
    for dataset_id in dataset_ids:
        dataset_entry: dict[str, Any] = {}
        for variant in variants:
            variant_rows = [rows[(dataset_id, variant, seed)]["payload"] for seed in seeds]
            metrics = {
                metric: _aggregate([float(row["metrics"][metric]) for row in variant_rows])
                for metric in METRICS
            }
            dataset_entry[variant] = metrics
            for metric, values in metrics.items():
                aggregate_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "variant": variant,
                        "metric": metric,
                        **values,
                    }
                )
        paired = {}
        for metric in METRICS:
            values = [
                float(rows[(dataset_id, "rg_full", seed)]["payload"]["metrics"][metric])
                - float(rows[(dataset_id, "scmae_only", seed)]["payload"]["metrics"][metric])
                for seed in seeds
            ]
            paired[metric] = _aggregate(values)
        dataset_entry["paired_rg_minus_scmae"] = paired
        if reference_rows:
            old_rg = [
                float(reference_rows[(dataset_id, "rg_full", seed)]["payload"]["metrics"]["ari"])
                for seed in seeds
                if (dataset_id, "rg_full", seed) in reference_rows
            ]
            if old_rg:
                dataset_entry["reference_v19_rg_full_ari"] = _aggregate(old_rg)
                dataset_entry["new_rg_minus_reference_rg"] = float(
                    dataset_entry["rg_full"]["ari"]["mean"] - mean(old_rg)
                )
        underlying = dataset_id.split("__", 1)[0]
        if underlying in external:
            dataset_entry["archived_external_best"] = external[underlying]
            dataset_entry["new_rg_minus_archived_best"] = float(
                dataset_entry["rg_full"]["ari"]["mean"] - external[underlying]["ari"]
            )
        dataset_summary[dataset_id] = dataset_entry

    overall = {}
    for variant in variants:
        overall[variant] = {
            metric: _aggregate(
                [
                    float(rows[(dataset_id, variant, seed)]["payload"]["metrics"][metric])
                    for dataset_id in dataset_ids
                    for seed in seeds
                ]
            )
            for metric in METRICS
        }
    overall["paired_rg_minus_scmae"] = {
        metric: _aggregate(
            [
                float(rows[(dataset_id, "rg_full", seed)]["payload"]["metrics"][metric])
                - float(rows[(dataset_id, "scmae_only", seed)]["payload"]["metrics"][metric])
                for dataset_id in dataset_ids
                for seed in seeds
            ]
        )
        for metric in METRICS
    }
    overall["dataset_wins_rg_over_scmae_ari"] = sum(
        float(dataset_summary[dataset]["paired_rg_minus_scmae"]["ari"]["mean"]) > 0.0
        for dataset in dataset_ids
    )

    csv_path = root / "aggregate_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset_id", "variant", "metric", "mean", "std", "n"),
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    report = [
        "# V19 PlantNet-ARI fixed-parameter PCA200 evaluation",
        "",
        f"- Completed: {len(actual)}/{len(expected)}",
        f"- Audit: {'passed' if not audit_errors else 'failed'}",
        "- Labels were excluded from preprocessing and fitting; benchmark labels supplied K and post-fit metrics only.",
        "- The fixed configuration was selected with ARI in PlantNet, so this is benchmark-transfer evidence, not label-free tuning.",
        "",
        "| Dataset | RG ARI mean+-std | matched scMAE ARI mean+-std | paired delta | old V19 RG | archived best |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_id in dataset_ids:
        row = dataset_summary[dataset_id]
        rg = row["rg_full"]["ari"]
        sc = row["scmae_only"]["ari"]
        delta = row["paired_rg_minus_scmae"]["ari"]
        old = row.get("reference_v19_rg_full_ari", {}).get("mean")
        ext = row.get("archived_external_best")
        old_text = "NA" if old is None else f"{float(old):.4f}"
        ext_text = "NA" if not ext else f"{ext['method']} {float(ext['ari']):.4f}"
        report.append(
            f"| {dataset_id} | {float(rg['mean']):.4f}+-{float(rg['std']):.4f} | "
            f"{float(sc['mean']):.4f}+-{float(sc['std']):.4f} | "
            f"{float(delta['mean']):+.4f} | {old_text} | {ext_text} |"
        )
    report.extend(
        [
            "",
            f"Overall ARI: RG {overall['rg_full']['ari']['mean']:.6f}; matched scMAE {overall['scmae_only']['ari']['mean']:.6f}; paired delta {overall['paired_rg_minus_scmae']['ari']['mean']:+.6f}.",
            f"RG wins on {overall['dataset_wins_rg_over_scmae_ari']}/{len(dataset_ids)} datasets by mean ARI.",
            "",
        ]
    )
    (root / "aggregate_report.md").write_text("\n".join(report), encoding="utf-8")
    result = {
        "status": "completed" if not audit_errors else "audit_failed",
        "root": str(root),
        "expected_runs": len(expected),
        "completed_runs": len(actual),
        "audit_ok": not audit_errors,
        "audit_errors": audit_errors,
        "missing": missing,
        "unexpected": unexpected,
        "labels_used_during_fit": False,
        "selection_labels_used": bool(spec.get("selection_labels_used", False)),
        "overall": overall,
        "datasets": dataset_summary,
        "outputs": {
            "aggregate_metrics_csv": str(csv_path),
            "aggregate_report_md": str(root / "aggregate_report.md"),
        },
    }
    _write_json(root / "aggregate_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--baseline-csv", type=Path, default=None)
    args = parser.parse_args()
    result = summarize(args.root, args.reference_root, args.baseline_csv)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["audit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
