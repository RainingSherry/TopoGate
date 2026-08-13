#!/usr/bin/env python
"""Aggregate auditable V12 stage-2 edge-rank run artifacts into comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


VARIANTS = (
    "nomix",
    "edge_only",
    "self_null_lambda001",
    "self_null_lambda003",
    "self_null_lambda01",
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
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


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir).resolve()
    with (root / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            groups[(row["dataset"], row["variant"])].append(row)

    variant_rows: list[dict[str, object]] = []
    for (dataset, variant), group in sorted(groups.items()):
        result: dict[str, object] = {
            "dataset": dataset,
            "variant": variant,
            "n_completed": len(group),
        }
        for field in (*METRICS, *DIAGNOSTICS):
            values = [_float(row.get(field, "")) for row in group]
            avg, std = _mean_std([value for value in values if value is not None])
            result[f"{field}_mean"] = avg
            result[f"{field}_std"] = std
        result["labels_used_during_fit_values"] = sorted(
            set(row.get("labels_used_during_fit", "") for row in group)
        )
        result["source_sha256_values"] = sorted(
            set(row.get("source_sha256", "") for row in group)
        )
        variant_rows.append(result)
    variant_fields = ["dataset", "variant", "n_completed"]
    variant_fields += [
        f"{field}_{suffix}"
        for field in (*METRICS, *DIAGNOSTICS)
        for suffix in ("mean", "std")
    ]
    variant_fields += ["labels_used_during_fit_values", "source_sha256_values"]
    _write_csv(root / "summary_by_dataset_variant.csv", variant_rows, variant_fields)
    _write_csv(root / "summary_by_dataset.csv", variant_rows, variant_fields)

    present_variants = sorted(
        {
            row.get("variant", "")
            for row in rows
            if row.get("status") == "completed" and row.get("variant", "")
        }
    )
    by_variant: list[dict[str, object]] = []
    for variant in present_variants:
        group = [
            row
            for row in rows
            if row.get("status") == "completed" and row.get("variant") == variant
        ]
        result = {"variant": variant, "n_completed": len(group)}
        for field in (*METRICS, *DIAGNOSTICS):
            values = [_float(row.get(field, "")) for row in group]
            avg, std = _mean_std([value for value in values if value is not None])
            result[f"{field}_mean"] = avg
            result[f"{field}_std"] = std
        by_variant.append(result)
    _write_csv(
        root / "summary_by_variant.csv",
        by_variant,
        ["variant", "n_completed"]
        + [
            f"{field}_{suffix}"
            for field in (*METRICS, *DIAGNOSTICS)
            for suffix in ("mean", "std")
        ],
    )

    lookup = {(row["dataset"], row["seed"], row["variant"]): row for row in rows}
    paired: list[dict[str, object]] = []
    for row in rows:
        if row.get("status") != "completed" or row.get("variant") == "nomix":
            continue
        baseline = lookup.get((row["dataset"], row["seed"], "nomix"))
        if baseline is None or baseline.get("status") != "completed":
            continue
        result: dict[str, object] = {
            "dataset": row["dataset"],
            "seed": row["seed"],
            "variant": row["variant"],
        }
        for metric in METRICS:
            current = _float(row.get(metric, ""))
            reference = _float(baseline.get(metric, ""))
            result[f"{metric}_current"] = current
            result[f"{metric}_nomix"] = reference
            result[f"{metric}_delta"] = (
                None if current is None or reference is None else current - reference
            )
        paired.append(result)
    _write_csv(
        root / "paired_deltas.csv",
        paired,
        ["dataset", "seed", "variant"]
        + [
            f"{metric}_{suffix}"
            for metric in METRICS
            for suffix in ("current", "nomix", "delta")
        ],
    )

    completed = sum(row.get("status") == "completed" for row in rows)
    failed = [row for row in rows if row.get("status") != "completed"]
    report_lines = [
        "# V12 stage-2 edge-rank report",
        "",
        "This report summarizes the registered paired runs across the selected datasets and variants. Labels are used only for benchmark K and metrics; they are not passed to graph construction, gate, loss, or variant selection. The per-edge rank signal is registered in `rank_alignment_loss` and pushed the gate toward higher weight on reliable neighbours.",
        "",
        f"- Expected runs: {len(rows)}",
        f"- Completed runs: {completed}",
        f"- Failed/incomplete records: {len(failed)}",
        f"- Variants present: {', '.join(present_variants) if present_variants else 'none'}",
        "",
        "## Coverage",
        "",
        "| dataset | variant | completed | ARI mean +/- std | self mass | edge entropy | effective neighbors | rank_loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_rows:
        ari_mean = row["ari_mean"]
        ari_std = row["ari_std"]
        ari_text = "NA" if ari_mean is None else f"{ari_mean:.4f} +/- {ari_std:.4f}"
        def _fmt(key):
            value = row.get(f"{key}_mean")
            if value is None:
                return "NA"
            return f"{value:.4f}"

        report_lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['n_completed']} | "
            f"{ari_text} | {_fmt('self_mass')} | {_fmt('edge_entropy')} | "
            f"{_fmt('effective_neighbor_count')} | {_fmt('rank_loss')} |"
        )
    report_lines.extend(
        [
            "",
            "## Paired interpretation",
            "",
            "Use paired_deltas.csv for seed-matched Full-NoMix comparisons. Do not select lambda from one dataset or one seed. A positive delta is evidence for that pair only; it is not a universal topology claim.",
            "",
        ]
    )
    if failed:
        report_lines.extend(["## Failures", ""])
        for row in failed:
            report_lines.append(
                f"- {row.get('dataset')}/{row.get('variant')}/seed_{row.get('seed')}: "
                f"{row.get('status')} (log: {row.get('log_path', '')})"
            )
    (root / "report.md").write_text("\n".join(report_lines) + "\n")
    (root / "coverage.json").write_text(
        json.dumps(
            {
                "completed_runs": completed,
                "failed_records": len(failed),
                "labels_used_during_fit_values": sorted(
                    set(row.get("labels_used_during_fit", "") for row in rows)
                ),
                "present_variants": present_variants,
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
