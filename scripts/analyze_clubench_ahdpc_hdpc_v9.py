#!/usr/bin/env python3
"""Analyse the paired CLUBench AHDPC/HDPC/V9 results.

This is deliberately a post-fit analysis utility.  It never reads labels to
choose a model or hyperparameter; labels are present only through the metrics
already materialised by the benchmark runner.  ARI is the primary ranking
metric, while all six stored metrics are reported for auditability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METHODS = ("AHDPC", "HDPC", "V9")
METRICS = ("ACC", "NMI", "ARI", "AMI", "RI", "FMI")
EPS = 1e-12


def _num(value):
    if value in (None, "", "None", "nan", "NaN"):
        return None
    return float(value)


def _fmt(value, digits=4):
    return "—" if value is None else f"{float(value):.{digits}f}"


def _read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(values) if values else None


def _median(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(values) if values else None


def _paired(rows):
    grouped = {}
    for raw in rows:
        dataset = raw.get("dataset")
        method = raw.get("method")
        if not dataset or method not in METHODS or raw.get("status") != "completed":
            continue
        item = dict(raw)
        for key in ("n_samples", "n_features", "n_clusters", "seed"):
            try:
                item[key] = int(float(item[key]))
            except (TypeError, ValueError):
                item[key] = None
        for metric in METRICS:
            item[metric] = _num(item.get(metric))
        grouped.setdefault(dataset, {})[method] = item
    out = []
    for dataset, entries in sorted(grouped.items()):
        if not all(method in entries for method in METHODS):
            continue
        row = {
            "dataset": dataset,
            "n_samples": entries["V9"].get("n_samples"),
            "n_features": entries["V9"].get("n_features"),
            "n_clusters": entries["V9"].get("n_clusters"),
        }
        for method in METHODS:
            for metric in METRICS:
                row[f"{method}_{metric}"] = entries[method].get(metric)
        for opponent in ("AHDPC", "HDPC"):
            for metric in METRICS:
                row[f"delta_{metric}_V9_minus_{opponent}"] = (
                    row[f"V9_{metric}"] - row[f"{opponent}_{metric}"]
                    if row[f"V9_{metric}"] is not None and row[f"{opponent}_{metric}"] is not None
                    else None
                )
        out.append(row)
    return out


def _counts(rows, opponent, metric="ARI"):
    delta_key = f"delta_{metric}_V9_minus_{opponent}"
    vals = [row[delta_key] for row in rows if row[delta_key] is not None]
    return {
        "n": len(vals),
        "wins": sum(value > EPS for value in vals),
        "ties": sum(abs(value) <= EPS for value in vals),
        "losses": sum(value < -EPS for value in vals),
        "mean_delta": _mean(vals),
        "median_delta": _median(vals),
    }


def _write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rank(rows, opponent, reverse, limit=20):
    key = f"delta_ARI_V9_minus_{opponent}"
    return sorted((row for row in rows if row[key] is not None), key=lambda row: row[key], reverse=reverse)[:limit]


def _table(rows, opponent):
    lines = [
        "| Dataset | n | d | K | V9 ARI | AHDPC ARI | HDPC ARI | ΔARI | ΔNMI | ΔACC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row.get('n_samples', '—')} | {row.get('n_features', '—')} | {row.get('n_clusters', '—')} | "
            f"{_fmt(row.get('V9_ARI'))} | {_fmt(row.get('AHDPC_ARI'))} | {_fmt(row.get('HDPC_ARI'))} | "
            f"{_fmt(row.get(f'delta_ARI_V9_minus_{opponent}'))} | "
            f"{_fmt(row.get(f'delta_NMI_V9_minus_{opponent}'))} | {_fmt(row.get(f'delta_ACC_V9_minus_{opponent}'))} |"
        )
    return lines


def analyse(output_dir: Path) -> None:
    long_path = output_dir / "comparison_long.csv"
    if not long_path.exists():
        raise FileNotFoundError(long_path)
    rows = _paired(_read_rows(long_path))
    if not rows:
        raise RuntimeError("no complete AHDPC/HDPC/V9 triplets found")

    aggregate = {}
    for method in METHODS:
        aggregate[method] = {
            "completed_triplets": len(rows),
            **{
                f"{metric}_{stat}": value
                for metric in METRICS
                for stat, value in (("mean", _mean([row[f"{method}_{metric}"] for row in rows])), ("median", _median([row[f"{method}_{metric}"] for row in rows])))
            },
        }

    pairwise = {opponent: _counts(rows, opponent) for opponent in ("AHDPC", "HDPC")}
    # Thresholds are descriptive strata, not model-selection rules.
    positive = [row for row in rows if row["delta_ARI_V9_minus_AHDPC"] is not None and row["delta_ARI_V9_minus_AHDPC"] >= 0.10]
    negative = [row for row in rows if row["delta_ARI_V9_minus_AHDPC"] is not None and row["delta_ARI_V9_minus_AHDPC"] <= -0.10]
    strong_baseline_regression = [row for row in negative if row["AHDPC_ARI"] is not None and row["AHDPC_ARI"] >= 0.50]
    shared_difficulty = [
        row for row in rows
        if all(row.get(f"{method}_ARI") is not None and row[f"{method}_ARI"] <= 0.10 for method in METHODS)
    ]
    v9_strong = [row for row in positive if row["V9_ARI"] is not None and row["V9_ARI"] >= 0.50]
    payload = {
        "protocol": {
            "input": "CLUBench.load_data column-wise z-score",
            "K": "int(np.unique(y).size), used only for benchmark K and post-fit metrics",
            "AHDPC_HDPC": "epsilon=1.0, paper_semantic, table_reproduction, block_size=256",
            "V9": "learnable_gate_v9_adaptive, seed=42, epochs=80, batch_size=256, scale_input=false",
            "primary_metric": "ARI",
            "scope": "single seed; descriptive comparison, not multi-seed statistical evidence",
        },
        "complete_triplets": len(rows),
        "aggregate": aggregate,
        "pairwise_ARI": pairwise,
        "strata": {
            "positive_delta_ARI_ge_0.10": len(positive),
            "negative_delta_ARI_le_-0.10": len(negative),
            "strong_AHDPC_regression_AHDPC_ARI_ge_0.50": len(strong_baseline_regression),
            "shared_difficulty_all_ARI_le_0.10": len(shared_difficulty),
            "strong_V9_positive_V9_ARI_ge_0.50": len(v9_strong),
        },
    }
    (output_dir / "analysis_full.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_dir / "analysis_by_dataset.csv", rows)
    _write_csv(output_dir / "advantage_over_ahdpc.csv", _rank(positive, "AHDPC", True, limit=len(positive)))
    _write_csv(output_dir / "regression_vs_ahdpc.csv", _rank(negative, "AHDPC", False, limit=len(negative)))
    _write_csv(output_dir / "shared_difficulty.csv", sorted(shared_difficulty, key=lambda row: (row.get("V9_ARI") or -math.inf), reverse=True))

    report = [
        "# CLUBench AHDPC vs HDPC vs V9: paired analysis",
        "",
        "## Scope and protocol",
        "",
        "This report uses only complete three-method records in `comparison_long.csv`. ARI is the primary comparison metric; ACC, NMI, AMI, RI and FMI are retained in `analysis_by_dataset.csv`. The run is single-seed (42), so the tables are engineering evidence and require multi-seed confirmation before a paper-level claim.",
        "",
        f"Complete triplets: **{len(rows)}**.",
        "",
        "## Aggregate metrics",
        "",
        "| Method | Mean ARI | Median ARI | Mean NMI | Mean ACC |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        report.append(f"| {method} | {_fmt(aggregate[method]['ARI_mean'])} | {_fmt(aggregate[method]['ARI_median'])} | {_fmt(aggregate[method]['NMI_mean'])} | {_fmt(aggregate[method]['ACC_mean'])} |")
    report += ["", "## Paired V9 comparison by ARI", "", "| Opponent | n | V9 wins | ties | V9 losses | Mean ΔARI | Median ΔARI |", "|---|---:|---:|---:|---:|---:|---:|"]
    for opponent in ("AHDPC", "HDPC"):
        item = pairwise[opponent]
        report.append(f"| {opponent} | {item['n']} | {item['wins']} | {item['ties']} | {item['losses']} | {_fmt(item['mean_delta'])} | {_fmt(item['median_delta'])} |")

    report += ["", "## V9 advantages over AHDPC", "", f"Descriptive threshold: ΔARI ≥ 0.10 ({len(positive)} datasets); strong V9 outcome additionally requires V9 ARI ≥ 0.50 ({len(v9_strong)} datasets).", ""]
    report += _table(_rank(positive, "AHDPC", True), "AHDPC")
    report += ["", "## Negative datasets relative to AHDPC", "", f"Descriptive threshold: ΔARI ≤ −0.10 ({len(negative)} datasets). The first table isolates cases where AHDPC itself is strong (ARI ≥ 0.50; {len(strong_baseline_regression)} datasets); the second contains all substantial regressions.", "", "### AHDPC-strong regressions", ""]
    report += _table(_rank(strong_baseline_regression, "AHDPC", False), "AHDPC")
    report += ["", "### All substantial regressions", ""]
    report += _table(_rank(negative, "AHDPC", False), "AHDPC")
    report += ["", "## Shared-difficulty datasets", "", f"All three methods have ARI ≤ 0.10 ({len(shared_difficulty)} datasets). These are not V9-specific failures and should not be counted as evidence against the gate.", ""]
    report += _table(sorted(shared_difficulty, key=lambda row: (row.get("V9_ARI") or -math.inf), reverse=True), "AHDPC")
    report += ["", "## Interpretation boundary", "", "The deltas describe one fixed protocol and one seed. They identify where V9 helps or regresses relative to the frozen AHDPC/HDPC implementations; they do not establish significance, robustness, or a universally superior method.", ""]
    (output_dir / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"complete_triplets={len(rows)}")
    print(f"wrote {output_dir / 'analysis_report.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    analyse(args.output_dir.resolve())


if __name__ == "__main__":
    main()
