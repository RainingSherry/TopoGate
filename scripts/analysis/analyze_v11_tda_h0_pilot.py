#!/usr/bin/env python3
"""Summarise the pre-registered V11 sparse-H0 pilot.

The script reads only persisted run summaries and emits machine-readable
paired statistics next to the experiment. It never uses labels to select a
variant; labels appear only in the benchmark metrics already written by the
runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "result" / "V11" / "tda_h0_pilot_2026-08-03"
DATASETS = ["balance_scale", "spect_heart", "banknote", "flame", "vehicle"]
SEEDS = [42, 123, 7]
VARIANTS = [
    "V11_full",
    "V11_nomix",
    "V11_tda_h0_mst",
    "V11_tda_fixed_filtration",
    "V11_tda_random",
]
METRICS = ["head_ari", "kmeans_ari", "nmi", "silhouette"]
DIAGNOSTICS = [
    "final_gate",
    "final_target_gate",
    "mean_gate",
    "mean_target_gate",
    "mean_graph_loss",
    "final_graph_loss",
    "mean_tda_prior",
    "final_tda_prior",
    "h0_merge_count",
    "tda_scale",
    "tda_nonzero_fraction",
    "graph_edge_change_fraction",
    "train_seconds",
]


def _float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    return float(value)


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else math.nan


def _std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _fmt(value: float, spread: float | None = None) -> str:
    if not math.isfinite(value):
        return "NA"
    if spread is None:
        return f"{value:.4f}"
    return f"{value:.4f} +/- {spread:.4f}"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paired_stats(left: list[float], right: list[float]) -> dict[str, object]:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    finite = np.isfinite(delta)
    delta = delta[finite]
    if delta.size == 0:
        return {
            "n": 0,
            "mean_delta": math.nan,
            "median_delta": math.nan,
            "std_delta": math.nan,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "wilcoxon_p": math.nan,
            "paired_t_p": math.nan,
        }
    wins = int(np.sum(delta > 1e-12))
    losses = int(np.sum(delta < -1e-12))
    ties = int(delta.size - wins - losses)
    try:
        if np.all(np.abs(delta) <= 1e-12):
            p_w = 1.0
        else:
            p_w = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        p_w = math.nan
    try:
        p_t = float(ttest_rel(np.asarray(left)[finite], np.asarray(right)[finite]).pvalue)
    except (ValueError, FloatingPointError):
        p_t = math.nan
    return {
        "n": int(delta.size),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "std_delta": float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wilcoxon_p": p_w,
        "paired_t_p": p_t,
    }


def load_rows() -> list[dict[str, object]]:
    paths = sorted(OUTPUT_DIR.glob("*__*__seed*/summary.json"))
    expected = len(DATASETS) * len(VARIANTS) * len(SEEDS)
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} summary files, found {len(paths)}")
    rows: list[dict[str, object]] = []
    for path in paths:
        parts = path.parent.name.split("__")
        if len(parts) != 3 or not parts[2].startswith("seed"):
            raise RuntimeError(f"unexpected run directory: {path.parent}")
        dataset, variant, seed_text = parts
        seed = int(seed_text.removeprefix("seed"))
        if dataset not in DATASETS or variant not in VARIANTS or seed not in SEEDS:
            raise RuntimeError(f"run outside pre-registered suite: {path.parent}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        history = summary["history"]
        graph_history = summary["graph_history"]
        final_history = history[-1]
        final_graph = graph_history[-1] if graph_history else {}
        tda_graph_rows = [row for row in graph_history if row.get("tda_prior_mode") != "none"]
        tda_graph = tda_graph_rows[-1] if tda_graph_rows else final_graph
        if summary.get("labels_used_during_fit") is not False:
            raise RuntimeError(f"label boundary is not explicit in {path}")
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "n_samples": summary["n_samples"],
                "n_features": summary["n_features"],
                "n_clusters": summary["n_clusters"],
                "k_protocol": summary["k_protocol"],
                "labels_used_during_fit": summary["labels_used_during_fit"],
                "source_sha256": summary["source_sha256"],
                "head_ari": metrics["head"]["ari"],
                "kmeans_ari": metrics["kmeans"]["ari"],
                "nmi": metrics["nmi"],
                "silhouette": _float(metrics.get("silhouette")),
                "final_gate": _float(final_history.get("gate")),
                "final_target_gate": _float(final_history.get("target_gate")),
                "mean_gate": _mean([_float(row.get("gate")) for row in history]),
                "mean_target_gate": _mean([_float(row.get("target_gate")) for row in history]),
                "mean_graph_loss": _mean([_float(row.get("graph")) for row in history]),
                "final_graph_loss": _float(final_history.get("graph")),
                "mean_tda_prior": _mean([_float(row.get("tda_prior")) for row in history]),
                "final_tda_prior": _float(final_history.get("tda_prior")),
                "h0_merge_count": _float(tda_graph.get("tda_h0_merge_count")),
                "tda_scale": _float(tda_graph.get("tda_scale")),
                "tda_nonzero_fraction": _float(tda_graph.get("tda_prior_nonzero_fraction")),
                "graph_edge_change_fraction": _float(final_graph.get("edge_change_fraction")),
                "train_seconds": summary["train_seconds"],
                "summary_path": str(path),
            }
        )
    return sorted(rows, key=lambda row: (str(row["dataset"]), VARIANTS.index(str(row["variant"])), int(row["seed"])))


def main() -> None:
    global OUTPUT_DIR, DATASETS, SEEDS, VARIANTS
    parser = argparse.ArgumentParser(description="Summarise a persisted V11 sparse-H0 pilot")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    if args.datasets is not None:
        DATASETS = list(args.datasets)
    if args.seeds is not None:
        SEEDS = list(args.seeds)
    if args.variants is not None:
        VARIANTS = list(args.variants)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    run_fields = [
        "dataset", "variant", "seed", "n_samples", "n_features", "n_clusters",
        "k_protocol", "labels_used_during_fit", "source_sha256", *METRICS,
        *DIAGNOSTICS, "summary_path",
    ]
    _write_csv(OUTPUT_DIR / "run_diagnostics.csv", rows, run_fields)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["variant"]))].append(row)
    aggregate_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = grouped[(dataset, variant)]
            output: dict[str, object] = {"dataset": dataset, "variant": variant, "n": len(group)}
            for field in [*METRICS, *DIAGNOSTICS]:
                values = [float(row[field]) for row in group if math.isfinite(float(row[field]))]
                output[f"{field}_mean"] = _mean(values)
                output[f"{field}_std"] = _std(values)
            output["source_sha256"] = group[0]["source_sha256"]
            output["n_clusters"] = group[0]["n_clusters"]
            output["k_protocol"] = group[0]["k_protocol"]
            aggregate_rows.append(output)
    aggregate_fields = ["dataset", "variant", "n", "n_clusters", "k_protocol", "source_sha256"]
    aggregate_fields += [f"{field}_{stat}" for field in [*METRICS, *DIAGNOSTICS] for stat in ["mean", "std"]]
    _write_csv(OUTPUT_DIR / "summary_by_dataset_variant.csv", aggregate_rows, aggregate_fields)

    by_key = {(str(row["dataset"]), int(row["seed"]), str(row["variant"])): row for row in rows}
    comparison_candidates = [
        ("V11_full", "V11_nomix"),
        ("V11_tda_h0_mst", "V11_full"),
        ("V11_tda_fixed_filtration", "V11_full"),
        ("V11_tda_random", "V11_full"),
        ("V11_tda_h0_early_mst", "V11_full"),
        ("V11_tda_h0_mst", "V11_tda_fixed_filtration"),
        ("V11_tda_h0_mst", "V11_tda_random"),
    ]
    comparisons = [
        pair for pair in comparison_candidates
        if pair[0] in VARIANTS and pair[1] in VARIANTS
    ]
    paired_rows: list[dict[str, object]] = []
    for left_variant, right_variant in comparisons:
        for dataset in ["ALL", *DATASETS]:
            keys = [(dataset_name, seed) for dataset_name in DATASETS for seed in SEEDS]
            if dataset != "ALL":
                keys = [(dataset, seed) for seed in SEEDS]
            for metric in [*METRICS, "final_gate", "mean_gate", "mean_graph_loss"]:
                left = [float(by_key[(name, seed, left_variant)][metric]) for name, seed in keys]
                right = [float(by_key[(name, seed, right_variant)][metric]) for name, seed in keys]
                stats = _paired_stats(left, right)
                paired_rows.append(
                    {
                        "scope": dataset,
                        "left": left_variant,
                        "right": right_variant,
                        "metric": metric,
                        **stats,
                    }
                )
    paired_fields = [
        "scope", "left", "right", "metric", "n", "mean_delta", "median_delta", "std_delta",
        "wins", "ties", "losses", "wilcoxon_p", "paired_t_p",
    ]
    _write_csv(OUTPUT_DIR / "paired_deltas.csv", paired_rows, paired_fields)

    overall = {(row["left"], row["right"], row["metric"]): row for row in paired_rows if row["scope"] == "ALL"}
    table_lines = [
        "# V11 sparse H0 TDA pilot: formal comparison",
        "",
        f"**Status**: {len(rows)}/{len(DATASETS) * len(VARIANTS) * len(SEEDS)} completed; this is a fixed-protocol performance comparison, not a universal claim.",
        "",
        "## Protocol",
        "",
        f"- Datasets: `{', '.join(DATASETS)}`; seeds: `{SEEDS}`; variants: `{', '.join(VARIANTS)}`.",
        "- Input: AHDPC processed `x/y` NPZ files; `K=int(np.unique(y).size)` only for benchmark K and post-fit metrics.",
        "- Training: V11 default YAML, 80 epochs, CPU `--no-cuda`, one thread per numerical backend; no per-dataset tuning.",
        "- TDA object: fixed raw-PCA kNN sparse 1-skeleton, unit-row Euclidean chord filtration, exact H0 union-find; H1/dense VR are not computed.",
        "- Controls: `fixed_filtration` is distance-only; `random` is deterministic edge-shared random prior; `h0_early_mst` reverses the H0 merge-distance ordering; all prior values are detached.",
        "- Evidence: every run has `summary.json`, resolved config, source hash, predictions, labels_true, and `labels_used_during_fit=false`.",
        "",
        "## Aggregate metrics",
        "",
        "Values are mean +/- sample standard deviation over 5 datasets x 3 seeds.",
        "",
        "| Variant | Head ARI | KMeans ARI | NMI | Silhouette | Final gate | Mean graph loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        group = [row for row in rows if row["variant"] == variant]
        def cell(field: str) -> str:
            values = [float(row[field]) for row in group if math.isfinite(float(row[field]))]
            return _fmt(_mean(values), _std(values))
        table_lines.append(
            f"| `{variant}` | {cell('head_ari')} | {cell('kmeans_ari')} | {cell('nmi')} | "
            f"{cell('silhouette')} | {cell('final_gate')} | {cell('mean_graph_loss')} |"
        )

    table_lines += [
        "",
        "## Paired tests",
        "",
        "`mean_delta` is left minus right over the 15 paired dataset-seed runs. P-values are descriptive checks, not a license to select a method after seeing labels.",
        "",
        "| Comparison | Metric | Delta | Wins/Ties/Losses | Wilcoxon p | Paired t p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for left_variant, right_variant in comparisons[:4]:
        for metric in ["head_ari", "kmeans_ari", "nmi", "silhouette"]:
            item = overall[(left_variant, right_variant, metric)]
            table_lines.append(
                f"| `{left_variant}` - `{right_variant}` | `{metric}` | {float(item['mean_delta']):.6f} | "
                f"{item['wins']}/{item['ties']}/{item['losses']} | {float(item['wilcoxon_p']):.4f} | {float(item['paired_t_p']):.4f} |"
            )

    table_lines += ["", "## Dataset-level head ARI", ""]
    table_lines.append("| Dataset | " + " | ".join(VARIANTS) + " |")
    table_lines.append("|---|" + "|".join("---:" for _ in VARIANTS) + "|")
    for dataset in DATASETS:
        values = []
        for variant in VARIANTS:
            agg = next(row for row in aggregate_rows if row["dataset"] == dataset and row["variant"] == variant)
            values.append(_fmt(float(agg["head_ari_mean"]), float(agg["head_ari_std"])))
        table_lines.append(f"| `{dataset}` | " + " | ".join(values) + " |")

    table_lines += [
        "",
        "## TDA diagnostics",
        "",
        "- The H0 prior is structurally active: its nonzero edge fraction and merge count are recorded in `run_diagnostics.csv`; `fixed_filtration` and `random` have different score distributions by construction.",
        "- Compare `mean_graph_loss` and `final_gate` jointly with clustering metrics. A lower graph loss or larger gate is not a clustering improvement by itself.",
        "- If H0, fixed-filtration, and random produce similar ARI while graph diagnostics differ, the result supports a no-go for this prior as a validated clustering mechanism, not a claim that TDA is generally ineffective.",
        "",
        "## Reproducibility inputs",
        "",
        "- Raw run outputs: this directory's `*__*__seed*/summary.json` and arrays.",
        "- Aggregation script: `scripts/analysis/analyze_v11_tda_h0_pilot.py`.",
        "- Source manifest: `datasets/AHDPC/MANIFEST.json`.",
        "- Existing mathematical boundary audit: `result/analysis/TopoGate_whole_project_math_TDA_audit_2026-08-03.md`.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    protocol = {
        "datasets": DATASETS,
        "seeds": SEEDS,
        "variants": VARIANTS,
        "expected_runs": len(rows),
        "labels_used_during_fit": False,
        "k_protocol": "benchmark_oracle_from_y",
        "output_dir": str(OUTPUT_DIR),
        "runner": "scripts/V11/run_v11_multiseed.py",
        "analysis_script": str(Path(__file__).resolve()),
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote diagnostics for {len(rows)} runs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
