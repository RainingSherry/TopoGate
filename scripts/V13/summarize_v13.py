#!/usr/bin/env python
"""Aggregate auditable V13 Gumbel-Top-k experiment results.

Stage 1 smoke: flame/enron × nomix/topk2 × 3 seeds.
Stage 2 formal: 5 AHDPC × nomix/topk2 × 3 seeds.

The headline diagnostic for V13 is:
    effective_neighbor_count == top_k_neighbors (exactly 2 at inference)
If this holds, the hard gate is working; if not, the gate is collapsing or
saturating to a different behaviour.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

METRICS = ("ari", "nmi", "acc", "fmi")
DIAGNOSTICS = (
    "selected_neighbor_count",
    "effective_neighbor_count",
    "topology_loss",
    "reconstruction_loss",
    "mask_loss",
)
TOP_K = 2  # fixed for all topk2 runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--v12-stage2-dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "result" / "V12" / "v12_edge_rank_stage2_2026-08-04"
        ),
    )
    return parser.parse_args()


def _float(value: str) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    return mean(clean), (stdev(clean) if len(clean) > 1 else 0.0)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir).resolve()
    with (root / "runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            groups[(row["dataset"], row["variant"])].append(row)

    # 1) (dataset, variant) summary
    by_dv: list[dict] = []
    for (dataset, variant), group in sorted(groups.items()):
        result: dict = {
            "dataset": dataset,
            "variant": variant,
            "n_completed": len(group),
        }
        for field in (*METRICS, *DIAGNOSTICS):
            vals = [_float(row.get(field, "")) for row in group]
            avg, std = _mean_std([v for v in vals if v is not None])
            result[f"{field}_mean"] = avg
            result[f"{field}_std"] = std
        by_dv.append(result)
    _write_csv(
        root / "summary_by_dataset_variant.csv",
        by_dv,
        ["dataset", "variant", "n_completed"]
        + [f"{f}_{s}" for f in (*METRICS, *DIAGNOSTICS) for s in ("mean", "std")],
    )

    # 2) per variant across all datasets
    by_variant: list[dict] = []
    present_variants = sorted({row.get("variant", "") for row in rows if row.get("status") == "completed"})
    for variant in present_variants:
        group = [r for r in rows if r.get("status") == "completed" and r.get("variant") == variant]
        result: dict = {"variant": variant, "n_completed": len(group)}
        for field in (*METRICS, *DIAGNOSTICS):
            vals = [_float(row.get(field, "")) for row in group]
            avg, std = _mean_std([v for v in vals if v is not None])
            result[f"{field}_mean"] = avg
            result[f"{field}_std"] = std
        by_variant.append(result)
    _write_csv(
        root / "summary_by_variant.csv",
        by_variant,
        ["variant", "n_completed"]
        + [f"{f}_{s}" for f in (*METRICS, *DIAGNOSTICS) for s in ("mean", "std")],
    )

    # 3) paired deltas vs nomix (within same dataset+seed)
    paired: list[dict] = []
    nomix_lookup: dict[tuple[str, int], dict] = {}
    for row in rows:
        if row.get("status") == "completed" and row.get("variant") == "nomix":
            nomix_lookup[(row["dataset"], int(row["seed"]))] = row
    for row in rows:
        if row.get("status") != "completed" or row.get("variant") == "nomix":
            continue
        baseline = nomix_lookup.get((row["dataset"], int(row["seed"])))
        if baseline is None:
            continue
        result: dict = {
            "dataset": row["dataset"],
            "seed": row["seed"],
            "variant": row["variant"],
        }
        for key in METRICS + DIAGNOSTICS:
            cur = _float(row.get(key, ""))
            ref = _float(baseline.get(key, ""))
            result[f"{key}_topk2"] = cur
            result[f"{key}_nomix"] = ref
            result[f"{key}_delta"] = (cur - ref) if cur is not None and ref is not None else None
        paired.append(result)
    paired_fields = ["dataset", "seed", "variant"]
    for key in METRICS + DIAGNOSTICS:
        paired_fields += [f"{key}_topk2", f"{key}_nomix", f"{key}_delta"]
    _write_csv(root / "paired_deltas_vs_nomix.csv", paired, paired_fields)

    # 4) markdown report
    completed = sum(r.get("status") == "completed" for r in rows)
    failed = [r for r in rows if r.get("status") != "completed"]
    present_datasets = sorted({r.get("dataset", "") for r in rows if r.get("status") == "completed"})
    lines = [
        "# V13 Gumbel-Top-k experiment report",
        "",
        f"- Expected runs: {len(rows)}",
        f"- Completed: {completed}",
        f"- Failed: {len(failed)}",
        f"- Datasets: {', '.join(present_datasets)}",
        f"- Variants: {', '.join(present_variants)}",
        "",
        "## Headline diagnostic: effective_neighbor_count vs top_k=2",
        "",
        "The hard gate should produce `effective_neighbor_count ≈ 2.0` at inference.",
        "Values significantly below 2.0 indicate the gate is not selecting;",
        "values above 2.0 indicate the gate is using soft relaxation at eval time.",
        "",
        "| dataset | variant | eff_neigh mean | eff_neigh std | topology_loss |",
        "|---|---|---:|---:|---:|",
    ]
    for row in by_dv:
        eff = row.get("effective_neighbor_count_mean")
        eff_std = row.get("effective_neighbor_count_std")
        topo = row.get("topology_loss_mean")
        eff_t = "NA" if eff is None else f"{eff:.4f} ± {eff_std:.4f}" if eff_std else f"{eff:.4f}"
        topo_t = "NA" if topo is None else f"{topo:.5f}"
        lines.append(f"| {row['dataset']} | {row['variant']} | {eff_t} | {topo_t} |")
    lines.extend(
        [
            "",
            "## ARI mean ± std (across all seeds)",
            "",
            "| dataset | nomix ARI | topk2 ARI | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    nomix_by_ds = {r["dataset"]: r for r in by_dv if r["variant"] == "nomix"}
    topk2_by_ds = {r["dataset"]: r for r in by_dv if r["variant"] == "topk2"}
    for ds in present_datasets:
        nm = nomix_by_ds.get(ds, {})
        tk = topk2_by_ds.get(ds, {})
        nm_ari = nm.get("ari_mean")
        tk_ari = tk.get("ari_mean")
        nm_std = nm.get("ari_std", 0.0) or 0.0
        tk_std = tk.get("ari_std", 0.0) or 0.0
        nm_t = "NA" if nm_ari is None else f"{nm_ari:.4f} ± {nm_std:.4f}"
        tk_t = "NA" if tk_ari is None else f"{tk_ari:.4f} ± {tk_std:.4f}"
        delta = (tk_ari - nm_ari) if tk_ari is not None and nm_ari is not None else None
        delta_t = "NA" if delta is None else f"{delta:+.4f}"
        lines.append(f"| {ds} | {nm_t} | {tk_t} | {delta_t} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use `paired_deltas_vs_nomix.csv` for seed-matched ARI comparisons.",
            "Positive delta > 0.03 ARI is evidence for a real improvement;",
            "values in [-0.03, 0.03] are within the documented noise band.",
            "effective_neighbor_count far from 2.0 indicates the gate has not",
            "learned to make hard selections.",
        ]
    )
    if failed:
        lines.extend(["", "## Failures", ""])
        for row in failed:
            lines.append(
                f"- {row.get('dataset')}/{row.get('variant')}/seed_{row.get('seed')}: "
                f"{row.get('status')}"
            )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    (root / "coverage.json").write_text(
        json.dumps(
            {
                "completed_runs": completed,
                "failed_runs": len(failed),
                "datasets": present_datasets,
                "variants": present_variants,
                "v12_stage2_dir": str(Path(args.v12_stage2_dir).resolve()),
                "labels_used_during_fit_values": sorted(
                    {r.get("labels_used_during_fit", "") for r in rows}
                ),
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
