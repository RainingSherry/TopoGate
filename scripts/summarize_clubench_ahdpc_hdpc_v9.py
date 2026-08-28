#!/usr/bin/env python3
"""Summarize the CLUBench AHDPC/HDPC/V9 benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = ("AHDPC", "HDPC", "V9")
METRICS = ("ACC", "NMI", "ARI", "AMI", "RI", "FMI")


def _float(value: str | None) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    return float(value)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(output_dir: Path) -> None:
    long_path = output_dir / "comparison_long.csv"
    if not long_path.exists():
        raise FileNotFoundError(long_path)
    rows = _read_rows(long_path)
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        for key in ("n_samples", "n_features", "n_clusters", "seed"):
            if row.get(key) not in (None, ""):
                row[key] = int(float(row[key]))
        for metric in METRICS:
            row[metric] = _float(row.get(metric))
        grouped[row["dataset"]][row["method"]] = row

    summary_rows: list[dict] = []
    for method in METHODS:
        completed = [r for r in rows if r["method"] == method and r["status"] == "completed"]
        errors = [r for r in rows if r["method"] == method and r["status"] == "error"]
        out = {"method": method, "completed": len(completed), "errors": len(errors)}
        for metric in METRICS:
            vals = [r[metric] for r in completed if r[metric] is not None]
            out[f"{metric}_mean"] = statistics.mean(vals) if vals else None
            out[f"{metric}_median"] = statistics.median(vals) if vals else None
        summary_rows.append(out)
    summary_fields = ["method", "completed", "errors"]
    summary_fields.extend(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "median"))
    _write_csv(output_dir / "method_summary.csv", summary_rows, summary_fields)

    pairwise_rows: list[dict] = []
    report: list[str] = []
    report.append("# CLUBench: AHDPC vs HDPC vs V9\n")
    report.append("## Protocol\n")
    report.append(
        "- Input: CLUBench `load_data` column-wise z-score.\n"
        "- `K = int(np.unique(y).size)` is used only for benchmark K and post-fit metrics.\n"
        "- AHDPC/HDPC: fixed epsilon=1.0, `paper_semantic` normalization, "
        "`table_reproduction` adaptive-distance rule.\n"
        "- V9: `learnable_gate_v9_adaptive`, seed=42, 80 epochs, already-standardized "
        "input with `scale_input=false`.\n"
    )
    total = len(grouped)
    complete_triplets = sum(
        all(grouped[dataset].get(method, {}).get("status") == "completed" for method in METHODS)
        for dataset in grouped
    )
    report.append(f"- Dataset records present: **{total}**; complete three-method datasets: **{complete_triplets}**.\n")

    report.append("## Method-level aggregate metrics\n")
    report.append("| Method | Completed | Errors | Mean ARI | Median ARI | Mean NMI | Mean ACC |")
    report.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        report.append(
            f"| {row['method']} | {row['completed']} | {row['errors']} | "
            f"{_fmt(row['ARI_mean'])} | {_fmt(row['ARI_median'])} | "
            f"{_fmt(row['NMI_mean'])} | {_fmt(row['ACC_mean'])} |"
        )
    report.append("")

    for opponent in ("AHDPC", "HDPC"):
        report.append(f"## V9 vs {opponent}\n")
        valid = []
        for dataset, entries in grouped.items():
            v9 = entries.get("V9", {})
            other = entries.get(opponent, {})
            if v9.get("status") != "completed" or other.get("status") != "completed":
                continue
            deltas = {}
            for metric in METRICS:
                if v9.get(metric) is not None and other.get(metric) is not None:
                    deltas[metric] = v9[metric] - other[metric]
            if "ARI" in deltas:
                valid.append((dataset, deltas, v9, other))
                pairwise_rows.append(
                    {
                        "dataset": dataset,
                        "opponent": opponent,
                        "v9_ARI": v9["ARI"],
                        f"{opponent}_ARI": other["ARI"],
                        "delta_ARI": deltas["ARI"],
                        "v9_NMI": v9.get("NMI"),
                        f"{opponent}_NMI": other.get("NMI"),
                        "delta_NMI": deltas.get("NMI"),
                        "v9_ACC": v9.get("ACC"),
                        f"{opponent}_ACC": other.get("ACC"),
                        "delta_ACC": deltas.get("ACC"),
                    }
                )
        wins = sum(d["ARI"] > 1e-12 for _, d, _, _ in valid)
        ties = sum(abs(d["ARI"]) <= 1e-12 for _, d, _, _ in valid)
        losses = sum(d["ARI"] < -1e-12 for _, d, _, _ in valid)
        mean_delta = statistics.mean(d["ARI"] for _, d, _, _ in valid) if valid else None
        report.append(
            f"- Valid paired datasets: **{len(valid)}**; ARI wins/ties/losses: "
            f"**{wins}/{ties}/{losses}**; mean ΔARI: **{_fmt(mean_delta)}**.\n"
        )
        if valid:
            report.append("| Dataset | V9 ARI | Opponent ARI | ΔARI | ΔNMI | ΔACC |")
            report.append("|---|---:|---:|---:|---:|---:|")
            ranked = sorted(valid, key=lambda item: item[1]["ARI"], reverse=True)
            selected = ranked[:10] + (ranked[-10:] if len(ranked) > 10 else [])
            seen: set[str] = set()
            for dataset, delta, v9, other in selected:
                if dataset in seen:
                    continue
                seen.add(dataset)
                report.append(
                    f"| {dataset} | {_fmt(v9.get('ARI'))} | {_fmt(other.get('ARI'))} | "
                    f"{_fmt(delta.get('ARI'))} | {_fmt(delta.get('NMI'))} | {_fmt(delta.get('ACC'))} |"
                )
            report.append("")

    report.append("## Per-dataset status\n")
    report.append("| Dataset | AHDPC | HDPC | V9 |")
    report.append("|---|---|---|---|")
    for dataset in sorted(grouped):
        statuses = [grouped[dataset].get(method, {}).get("status", "missing") for method in METHODS]
        report.append(f"| {dataset} | {statuses[0]} | {statuses[1]} | {statuses[2]} |")
    report.append("")
    (output_dir / "comparison_report.md").write_text("\n".join(report), encoding="utf-8")

    pairwise_fields = [
        "dataset",
        "opponent",
        "v9_ARI",
        "AHDPC_ARI",
        "HDPC_ARI",
        "delta_ARI",
        "v9_NMI",
        "AHDPC_NMI",
        "HDPC_NMI",
        "delta_NMI",
        "v9_ACC",
        "AHDPC_ACC",
        "HDPC_ACC",
        "delta_ACC",
    ]
    # Normalize missing opponent columns so one file can contain both comparisons.
    normalized = []
    for row in pairwise_rows:
        normalized.append({field: row.get(field) for field in pairwise_fields})
    _write_csv(output_dir / "pairwise_comparison.csv", normalized, pairwise_fields)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    _summarize(args.output_dir.resolve())
    print(f"Wrote {args.output_dir.resolve() / 'comparison_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
