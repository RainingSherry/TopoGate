#!/usr/bin/env python
"""Aggregate auditable V12 stage-3 topology-signal grid results.

Stage 3 amplifies the topology branch and the rank signal; the summarizer
emphasis is on (a) whether edge entropy drops below ``log(5)`` and whether
``effective_neighbor_count`` falls below 3, and (b) paired ARI comparisons
against the nominal stage-2 baseline `(lambda_topology=0.1, rank_margin=0.1,
self_init_weight=0.8)` when both runs share the same seed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import math


METRICS = ("ari", "nmi", "acc", "fmi")
DIAGNOSTICS = (
    "self_mass",
    "edge_entropy",
    "effective_neighbor_count",
    "topology_loss",
    "rank_loss",
    "rank_active_fraction",
    "reconstruction_loss",
    "mask_loss",
)
LOG_K = math.log(5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--stage2-dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "result"
            / "V12"
            / "v12_edge_rank_stage2_2026-08-04"
        ),
        help="Stage-2 baseline directory used for paired comparisons.",
    )
    return parser.parse_args()


def _float(value: str) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    return mean(clean), (stdev(clean) if len(clean) > 1 else 0.0)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_baseline_lookup(stage2_dir: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Return {(dataset, seed): stage-2 self_null_lambda01 row}.

    The stage-2 grid is the nominal fixed control: rank_loss_weight=0.1,
    rank_margin=0.1, self_init_weight=0.8, lambda_topology=0.1. We use the
    self_null row as the local baseline for paired ARI deltas.
    """

    csv_path = stage2_dir / "runs.csv"
    if not csv_path.exists():
        return {}
    out: dict[tuple[str, int], dict[str, str]] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == "self_null_lambda01" and row.get("status") == "completed":
                key = (row["dataset"], int(row["seed"]))
                out[key] = row
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir).resolve()
    with (root / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            groups[(row["dataset"], row["config"])].append(row)

    # 1) per (dataset, config) summary
    config_rows: list[dict[str, object]] = []
    for (dataset, config), group in sorted(groups.items()):
        result: dict[str, object] = {
            "dataset": dataset,
            "config": config,
            "n_completed": len(group),
        }
        for field in (*METRICS, *DIAGNOSTICS):
            values = [_float(row.get(field, "")) for row in group]
            avg, std = _mean_std([value for value in values if value is not None])
            result[f"{field}_mean"] = avg
            result[f"{field}_std"] = std
        cols = {row["config"] for row in group}
        result["source_sha256_values"] = sorted(
            set(row.get("source_sha256", "") for row in group)
        )
        result["runner_source_sha256_values"] = sorted(
            set(row.get("runner_source_sha256", "") for row in group)
        )
        result["labels_used_during_fit_values"] = sorted(
            set(row.get("labels_used_during_fit", "") for row in group)
        )
        config_rows.append(result)
    config_fields = ["dataset", "config", "n_completed"]
    config_fields += [
        f"{field}_{suffix}"
        for field in (*METRICS, *DIAGNOSTICS)
        for suffix in ("mean", "std")
    ]
    config_fields += [
        "labels_used_during_fit_values",
        "source_sha256_values",
        "runner_source_sha256_values",
    ]
    _write_csv(
        root / "summary_by_dataset_config.csv",
        config_rows,
        config_fields,
    )

    # 2) per (config) summary across datasets
    present_configs = sorted(
        {
            row.get("config", "")
            for row in rows
            if row.get("status") == "completed" and row.get("config", "")
        }
    )
    by_config: list[dict[str, object]] = []
    for config in present_configs:
        group = [
            row
            for row in rows
            if row.get("status") == "completed" and row.get("config") == config
        ]
        result = {"config": config, "n_completed": len(group)}
        for field in (*METRICS, *DIAGNOSTICS):
            values = [_float(row.get(field, "")) for row in group]
            avg, std = _mean_std([value for value in values if value is not None])
            result[f"{field}_mean"] = avg
            result[f"{field}_std"] = std
        by_config.append(result)
    _write_csv(
        root / "summary_by_config.csv",
        by_config,
        ["config", "n_completed"]
        + [
            f"{field}_{suffix}"
            for field in (*METRICS, *DIAGNOSTICS)
            for suffix in ("mean", "std")
        ],
    )

    # 3) per dataset summary across configs
    present_datasets = sorted(
        {
            row.get("dataset", "")
            for row in rows
            if row.get("status") == "completed" and row.get("dataset", "")
        }
    )
    by_dataset: list[dict[str, object]] = []
    for dataset in present_datasets:
        group = [
            row
            for row in rows
            if row.get("status") == "completed" and row.get("dataset") == dataset
        ]
        result = {"dataset": dataset, "n_completed": len(group)}
        for field in (*METRICS, *DIAGNOSTICS):
            values = [_float(row.get(field, "")) for row in group]
            avg, std = _mean_std([value for value in values if value is not None])
            result[f"{field}_mean"] = avg
            result[f"{field}_std"] = std
        by_dataset.append(result)
    _write_csv(
        root / "summary_by_dataset.csv",
        by_dataset,
        ["dataset", "n_completed"]
        + [
            f"{field}_{suffix}"
            for field in (*METRICS, *DIAGNOSTICS)
            for suffix in ("mean", "std")
        ],
    )

    # 4) edge-entropy diagnostic table — the headline metric for stage-3.
    entropy_rows: list[dict[str, object]] = []
    for row in config_rows:
        entropy = row.get("edge_entropy_mean")
        eff = row.get("effective_neighbor_count_mean")
        rank_loss = row.get("rank_loss_mean")
        entropy_rows.append(
            {
                "dataset": row["dataset"],
                "config": row["config"],
                "edge_entropy_mean": entropy,
                "effective_neighbor_count_mean": eff,
                "rank_loss_mean": rank_loss,
                "below_log5": (
                    "yes"
                    if entropy is not None and entropy < LOG_K
                    else "no"
                ),
                "below_target_1.0": (
                    "yes" if entropy is not None and entropy < 1.0 else "no"
                ),
                "effective_neighbors_below_3": (
                    "yes" if eff is not None and eff < 3.0 else "no"
                ),
            }
        )
    _write_csv(
        root / "entropy_diagnostic.csv",
        entropy_rows,
        [
            "dataset",
            "config",
            "edge_entropy_mean",
            "effective_neighbor_count_mean",
            "rank_loss_mean",
            "below_log5",
            "below_target_1.0",
            "effective_neighbors_below_3",
        ],
    )

    # 5) paired ARI delta against stage-2 self_null baseline.
    stage2_dir = Path(args.stage2_dir).resolve()
    stage2_lookup = _load_baseline_lookup(stage2_dir)
    paired: list[dict[str, object]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        baseline = stage2_lookup.get((row["dataset"], int(row["seed"])))
        if baseline is None:
            continue
        result: dict[str, object] = {
            "dataset": row["dataset"],
            "seed": row["seed"],
            "config": row["config"],
            "stage2_baseline": "self_null_lambda01",
        }
        for metric in METRICS:
            current = _float(row.get(metric, ""))
            reference = _float(baseline.get(metric, ""))
            result[f"{metric}_current"] = current
            result[f"{metric}_stage2"] = reference
            result[f"{metric}_delta"] = (
                None if current is None or reference is None else current - reference
            )
        for diag in ("edge_entropy", "effective_neighbor_count", "rank_loss"):
            current = _float(row.get(diag, ""))
            reference = _float(baseline.get(diag, ""))
            result[f"{diag}_current"] = current
            result[f"{diag}_stage2"] = reference
            result[f"{diag}_delta"] = (
                None if current is None or reference is None else current - reference
            )
        paired.append(result)
    paired_fields = ["dataset", "seed", "config", "stage2_baseline"]
    for key in METRICS + ("edge_entropy", "effective_neighbor_count", "rank_loss"):
        for suffix in ("current", "stage2", "delta"):
            paired_fields.append(f"{key}_{suffix}")
    _write_csv(root / "paired_deltas_vs_stage2.csv", paired, paired_fields)

    # 6) mark-down report.
    completed = sum(row.get("status") == "completed" for row in rows)
    failed = [row for row in rows if row.get("status") != "completed"]
    below_1 = [row for row in entropy_rows if row["below_target_1.0"] == "yes"]
    below_log5 = [row for row in entropy_rows if row["below_log5"] == "yes"]
    report_lines = [
        "# V12 stage-3 topology-signal grid report",
        "",
        "This report summarizes the V12 stage-3 grid search. The grid amplifies",
        "the topology signal by sweeping `lambda_topology`, `rank_margin`, and",
        "`self_init_weight` (self_null only). Edge entropy is the headline metric:",
        "the goal is to push conditional edge entropy below 1.0 (effective",
        "neighbors < 3) and below `log(5) ≈ 1.6094` (effective neighbors < 5).",
        "",
        f"- Expected runs: {len(rows)}",
        f"- Completed runs: {completed}",
        f"- Failed/incomplete records: {len(failed)}",
        f"- Configs present: {', '.join(present_configs) if present_configs else 'none'}",
        f"- Datasets present: {', '.join(present_datasets) if present_datasets else 'none'}",
        f"- (dataset, config) cells with edge entropy < log(5): {len(below_log5)}",
        f"- (dataset, config) cells with edge entropy < 1.0: {len(below_1)}",
        "",
        "## Configuration count",
        "",
        f"- self_null: 2 (lambda) × 2 (rank_margin) × 2 (self_init) = 8 configs",
        f"- edge_only: 2 (lambda) × 2 (rank_margin) = 4 configs",
        f"- Total configs: 12",
        "",
        "## Edge-entropy diagnostic (headline metric)",
        "",
        "| dataset | config | edge_entropy | effective_neighbors | rank_loss | < log(5) | < 1.0 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in entropy_rows:
        ent = row["edge_entropy_mean"]
        eff = row["effective_neighbor_count_mean"]
        rl = row["rank_loss_mean"]
        ent_text = "NA" if ent is None else f"{ent:.4f}"
        eff_text = "NA" if eff is None else f"{eff:.4f}"
        rl_text = "NA" if rl is None else f"{rl:.4f}"
        report_lines.append(
            f"| {row['dataset']} | {row['config']} | {ent_text} | {eff_text} | {rl_text} | "
            f"{row['below_log5']} | {row['below_target_1.0']} |"
        )
    report_lines.extend(
        [
            "",
            "## Per-config mean ARI (across all 4 datasets × 3 seeds)",
            "",
            "| config | ARI mean ± std | edge_entropy mean | effective_neighbors mean | rank_loss mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in by_config:
        ari = row["ari_mean"]
        ari_std = row["ari_std"]
        ari_text = "NA" if ari is None else f"{ari:.4f} ± {ari_std:.4f}"
        ent = row["edge_entropy_mean"]
        ent_text = "NA" if ent is None else f"{ent:.4f}"
        eff = row["effective_neighbor_count_mean"]
        eff_text = "NA" if eff is None else f"{eff:.4f}"
        rl = row["rank_loss_mean"]
        rl_text = "NA" if rl is None else f"{rl:.4f}"
        report_lines.append(
            f"| {row['config']} | {ari_text} | {ent_text} | {eff_text} | {rl_text} |"
        )
    report_lines.extend(
        [
            "",
            "## Paired interpretation",
            "",
            "Use `paired_deltas_vs_stage2.csv` for seed-matched ARI comparisons against the",
            "stage-2 self_null_lambda01 baseline (lambda=0.1, rank_margin=0.1, self_init=0.8).",
            "A positive delta > 0.03 ARI is evidence for a real improvement; values in",
            "[-0.03, 0.03] are within the documented noise band.",
            "",
        ]
    )
    if failed:
        report_lines.extend(["## Failures", ""])
        for row in failed:
            report_lines.append(
                f"- {row.get('dataset')}/{row.get('config')}/seed_{row.get('seed')}: "
                f"{row.get('status')} (log: {row.get('log_path', '')})"
            )
    (root / "report.md").write_text("\n".join(report_lines) + "\n")
    (root / "coverage.json").write_text(
        json.dumps(
            {
                "completed_runs": completed,
                "failed_records": len(failed),
                "configs_present": present_configs,
                "datasets_present": present_datasets,
                "stage2_baseline_dir": str(stage2_dir),
                "stage2_baseline_lookup_rows": len(stage2_lookup),
                "below_log5_configurations": len(below_log5),
                "below_target_1.0_configurations": len(below_1),
                "labels_used_during_fit_values": sorted(
                    set(row.get("labels_used_during_fit", "") for row in rows)
                ),
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
