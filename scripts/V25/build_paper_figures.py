#!/usr/bin/env python3
"""Render deterministic V25 diagnostic figures from the frozen evidence bundle.

The figures are descriptive paper assets.  They do not refit a model, select a
dataset, or treat historical rows, seeds, or coordinates as independent
population samples.  Every output records the input CSV hashes and the
claim-scope boundary used for the plot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/v25-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROTOCOL_ID = "v25_paper_figures_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _save(fig: Any, path: Path) -> list[str]:
    """Save one figure in display and archival formats.

    The PNG remains convenient for review, while PDF/SVG preserve text and
    geometry for a paper source.  All formats are rendered from the same
    figure object and therefore do not represent separate analyses.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    outputs = [path, stem.with_suffix(".pdf"), stem.with_suffix(".svg")]
    for output in outputs:
        fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(output.relative_to(path.parents[1])) for output in outputs]


def _plot_atlas(rows: list[dict[str, str]], output: Path) -> None:
    rows = sorted(rows, key=lambda row: (row["version"], row["variant_family"]))
    labels = [f"{row['version']} / {row['variant_family']}" for row in rows]
    means = [_float(row, "delta_mean") for row in rows]
    stds = [_float(row, "delta_std") if row["delta_std"] else 0.0 for row in rows]
    colors = ["#1b6ca8" if value >= 0 else "#b33a3a" for value in means]
    fig, ax = plt.subplots(figsize=(9.0, max(4.8, 0.31 * len(rows) + 1.6)))
    positions = list(range(len(rows)))
    ax.errorbar(means, positions, xerr=stds, fmt="none", ecolor="#9aa4ad", alpha=0.7, capsize=2, zorder=1)
    ax.scatter(means, positions, c=colors, s=38, zorder=2)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.axvline(0.03, color="#777777", linewidth=0.7, linestyle="--")
    ax.axvline(-0.03, color="#777777", linewidth=0.7, linestyle="--")
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Mean paired Delta ARI (observational; error bars are row SD)")
    ax.set_title("V25 Failure Atlas: V1-V22 structural interventions")
    ax.grid(axis="x", alpha=0.2)
    fig.text(0.01, 0.005, "Rows are repeated records within dataset/protocol/readout units; no pooled causal inference.", fontsize=8)
    _save(fig, output)


def _plot_e1(rows: list[dict[str, str]], output: Path) -> None:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], {})[row["metric"]] = row
    datasets = sorted(grouped)
    x = [_float(grouped[name]["I_d"], "mean") for name in datasets]
    y = [_float(grouped[name]["S_d"], "mean") for name in datasets]
    colors = ["#1b6ca8" if value > 0.03 else "#b33a3a" if value < -0.03 else "#6b7280" for value in y]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.axhline(0.03, color="#777777", linewidth=0.7, linestyle="--")
    ax.axhline(-0.03, color="#777777", linewidth=0.7, linestyle="--")
    ax.axvline(0.03, color="#777777", linewidth=0.7, linestyle="--")
    ax.axvline(-0.03, color="#777777", linewidth=0.7, linestyle="--")
    ax.scatter(x, y, c=colors, s=70, edgecolor="white", linewidth=0.8, zorder=2)
    for name, x_value, y_value in zip(datasets, x, y):
        ax.annotate(name, (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"Generic intervention effect $I_d = Q(R)-Q(N)$")
    ax.set_ylabel(r"Topology selectivity effect $S_d = Q(T)-Q(R)$")
    ax.set_title("E1 V21 case study: matched three-arm decomposition")
    ax.grid(alpha=0.2)
    fig.text(0.01, 0.005, "Each point is a dataset mean over three seeds; signs are conditional, not universal.", fontsize=8)
    _save(fig, output)


def _plot_local_global(rows: list[dict[str, str]], output: Path) -> None:
    x = [_float(row, "local_delta") for row in rows]
    y = [_float(row, "global_delta") for row in rows]
    disconnect = [row["local_global_disconnect"] == "True" for row in rows]
    colors = ["#b33a3a" if value else "#6b7280" for value in disconnect]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.scatter(x, y, c=colors, s=62, edgecolor="white", linewidth=0.8, zorder=2)
    for row, x_value, y_value in zip(rows, x, y):
        ax.annotate(row["condition"], (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set_xlabel(r"Local delta: kNN purity@10")
    ax.set_ylabel(r"Global delta: ARI")
    ax.set_title("V23 boundary evidence: local improvement need not convert globally")
    ax.grid(alpha=0.2)
    fig.text(0.01, 0.005, "Boundary evidence only; V23 rows are not pooled into the V1-V22 atlas.", fontsize=8)
    _save(fig, output)


def _plot_mechanism_chain(output: Path) -> None:
    """Render the frozen localization chain without implying a fitted model."""
    labels = ["Opportunity", "Selection", "Intervention", "Representation", "Readout"]
    colors = ["#e5eef7", "#e8f3ec", "#fff2d9", "#f2e8f5", "#f8e1e1"]
    fig, ax = plt.subplots(figsize=(12.0, 2.6))
    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    ax.axis("off")
    width = 0.82
    y = 0.43
    for index, (label, color) in enumerate(zip(labels, colors)):
        patch = FancyBboxPatch(
            (index + (1.0 - width) / 2.0, y - 0.12),
            width,
            0.24,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.0,
            edgecolor="#4b5563",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(index + 0.5, y, label, ha="center", va="center", fontsize=11)
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(index + 1.0 + (1.0 - width) / 2.0 - 0.03, y),
                xytext=(index + width + (1.0 - width) / 2.0 + 0.03, y),
                arrowprops={"arrowstyle": "->", "color": "#6b7280", "linewidth": 1.2},
            )
    ax.text(
        len(labels) / 2.0,
        0.10,
        "V25 localizes where structural information is lost; it does not introduce a new architecture.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4b5563",
    )
    ax.set_title("Mechanism localization chain", pad=12)
    _save(fig, output)


def _plot_diagnostics(
    gradient_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    output: Path,
) -> None:
    """Plot diagnostic geometry and one-step/full effects without causal fitting."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    ax = axes[0]
    timepoints = ["T0", "T1", "T2"]
    means_assignment = []
    means_infomax = []
    for timepoint in timepoints:
        values_assignment = [
            float(row["cos_base_assignment"])
            for row in gradient_rows
            if row["timepoint"] == timepoint
        ]
        values_infomax = [
            float(row["cos_base_infomax"])
            for row in gradient_rows
            if row["timepoint"] == timepoint
        ]
        means_assignment.append(sum(values_assignment) / len(values_assignment) if values_assignment else 0.0)
        means_infomax.append(sum(values_infomax) / len(values_infomax) if values_infomax else 0.0)
    ax.plot(timepoints, means_assignment, marker="o", label="base vs assignment")
    ax.plot(timepoints, means_infomax, marker="o", label="base vs InfoMax")
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Mean gradient cosine")
    ax.set_title("Gradient geometry (diagnostic)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    datasets = sorted({row["dataset"] for row in pair_rows})
    palette = {dataset: plt.cm.tab10(index % 10) for index, dataset in enumerate(datasets)}
    for row in pair_rows:
        ax.scatter(
            float(row["S_1step_ARI"]),
            float(row["S_full_ARI"]),
            s=34,
            color=palette[row["dataset"]],
            alpha=0.85,
            label=row["dataset"],
        )
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlabel("S one-step ARI")
    ax.set_ylabel("S full ARI")
    ax.set_title("One-step vs full effect (diagnostic)")
    handles, labels_seen = ax.get_legend_handles_labels()
    unique = dict(zip(labels_seen, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=7, loc="best")
    ax.grid(alpha=0.2)
    fig.text(0.01, 0.005, "Diagnostics are descriptive and do not establish a universal objective law.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, output)


def build_figures(evidence_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    atlas_path = evidence_dir / "atlas_version_family.csv"
    e1_path = evidence_dir / "e1_dataset_effects.csv"
    pair_path = evidence_dir / "e1_pair_effects.csv"
    gradient_path = evidence_dir / "e2_gradient_geometry.csv"
    boundary_path = evidence_dir / "local_global_boundary.csv"
    atlas = _read_csv(atlas_path)
    e1 = _read_csv(e1_path)
    pair = _read_csv(pair_path)
    gradient = _read_csv(gradient_path)
    boundary = _read_csv(boundary_path)
    _plot_atlas(atlas, figure_dir / "V25_Figure1_failure_atlas.png")
    _plot_mechanism_chain(figure_dir / "V25_Figure2_mechanism_chain.png")
    _plot_e1(e1, figure_dir / "V25_Figure3_e1_selectivity.png")
    _plot_diagnostics(gradient, pair, figure_dir / "V25_Figure4_diagnostics.png")
    _plot_local_global(boundary, figure_dir / "V25_Figure5_local_global_boundary.png")
    figure_pngs = [
        "figures/V25_Figure1_failure_atlas.png",
        "figures/V25_Figure2_mechanism_chain.png",
        "figures/V25_Figure3_e1_selectivity.png",
        "figures/V25_Figure4_diagnostics.png",
        "figures/V25_Figure5_local_global_boundary.png",
    ]
    figure_assets = [
        str(Path(path).with_suffix(f".{extension}"))
        for path in figure_pngs
        for extension in ("png", "pdf", "svg")
    ]
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_dir": str(evidence_dir.resolve()),
        "sources": {
            str(path.name): {"sha256": _sha256(path), "rows": len(_read_csv(path))}
            for path in (atlas_path, e1_path, pair_path, gradient_path, boundary_path)
        },
        "figures": figure_pngs,
        "figure_assets": figure_assets,
        "figure_formats": ["png", "pdf", "svg"],
        "claim_scope": {
            "atlas": "observational V1-V22; rows are repeated records within dataset/protocol/readout units",
            "e1": "conditional heterogeneous audited V21 case study; dataset means over three seeds",
            "local_global": "V23 boundary evidence; not pooled with the V1-V22 intervention atlas",
            "holdout": "not represented; Phase D is inconclusive_not_completed",
        },
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path("result/V25_systematic_mechanism_study/PaperEvidence"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.evidence_dir
    manifest = build_figures(args.evidence_dir, output_dir)
    print(json.dumps({"protocol_id": manifest["protocol_id"], "output_dir": str(output_dir), "figure_count": len(manifest["figures"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
