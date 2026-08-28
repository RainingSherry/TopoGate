#!/usr/bin/env python
"""Generate ablation visualisation figures.

Reads:
  result/ablation/merged_summary.csv

Outputs:
  papers/figures/ablation_core_5datasets.png  (5 datasets × 8 variants grouped bars)
  papers/figures/ablation_ext_10datasets.png  (10 datasets × 4 variants grouped bars)
  papers/figures/ablation_variants_overall.png  (mean ACC across datasets per variant)
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

CORE_DATASETS = [
    "Mouse_retina",
    "sms_spam_collection",
    "enron",
    "har",
    "breast_cancer_wisconsin_original",
]
EXT_DATASETS = [
    "reuters", "ISOLET", "spambase", "cnae9", "Campbell",
    "hrvatin_filtered", "Quake_Smart-seq2_Lung", "mammographic_mass",
    "first-order-theorem-proving", "iris",
]
ALL_VARIANTS = [
    "topogate_full",
    "topogate_nomix",
    "topogate_random_neighbors",
    "topogate_far_neighbors",
    "topogate_constant_gate",
    "topogate_gate_only",
    "topogate_edge_only",
    "topogate_no_topology_features",
]
KEY_VARIANTS = [
    "topogate_full",
    "topogate_nomix",
    "topogate_random_neighbors",
    "topogate_constant_gate",
]

VARIANT_LABELS = {
    "topogate_full": "Full",
    "topogate_nomix": "NoMix",
    "topogate_random_neighbors": "RandomNb",
    "topogate_far_neighbors": "FarNb",
    "topogate_constant_gate": "ConstGate",
    "topogate_gate_only": "GateOnly",
    "topogate_edge_only": "EdgeOnly",
    "topogate_no_topology_features": "NoTopoFeat",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged_csv",
                   default="/home/luolie/ToPoGate/result/ablation/merged_summary.csv")
    p.add_argument("--figures_dir",
                   default="/home/luolie/ToPoGate/papers/figures")
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.merged_csv)
    print(f"Loaded {len(df)} rows from {args.merged_csv}")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Core layer (5 datasets × 8 variants) ──────────────
    core_df = df[df["dataset"].isin(CORE_DATASETS) & df["variant"].isin(ALL_VARIANTS)]
    make_grouped_bar(core_df, CORE_DATASETS, ALL_VARIANTS,
                     figures_dir / "ablation_core_5datasets.png",
                     title="TopoGate Ablation (Core 5 datasets, 8 variants)")

    # ── Figure 2: Extended layer (10 datasets × 4 key variants) ─────
    ext_df = df[df["dataset"].isin(EXT_DATASETS) & df["variant"].isin(KEY_VARIANTS)]
    make_grouped_bar(ext_df, EXT_DATASETS, KEY_VARIANTS,
                     figures_dir / "ablation_ext_10datasets.png",
                     title="TopoGate Ablation (Extended 10 datasets, 4 key variants)")

    # ── Figure 3: Overall mean ACC per variant ──────────────────────
    make_variant_overall(df, ALL_VARIANTS,
                         figures_dir / "ablation_variants_overall.png")


def make_grouped_bar(df, datasets, variants, out_path, title):
    import matplotlib.pyplot as plt
    import numpy as np

    n_var = len(variants)
    n_ds = len(datasets)
    x = np.arange(n_ds)
    width = 0.8 / n_var

    fig, ax = plt.subplots(figsize=(max(10, n_ds * 1.5), 5))
    for i, var in enumerate(variants):
        accs = []
        for ds in datasets:
            sub = df[(df["dataset"] == ds) & (df["variant"] == var)]
            if len(sub) == 0:
                accs.append(0.0)
            else:
                accs.append(sub["acc"].mean())
        ax.bar(x + i * width - 0.4 + width / 2, accs, width,
               label=VARIANT_LABELS.get(var, var))

    ax.set_xlabel("Dataset")
    ax.set_ylabel("ACC")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def make_variant_overall(df, variants, out_path):
    import matplotlib.pyplot as plt

    means = []
    stds = []
    for var in variants:
        sub = df[df["variant"] == var]["acc"]
        if len(sub) == 0:
            means.append(0.0)
            stds.append(0.0)
        else:
            means.append(sub.mean())
            stds.append(sub.std())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(variants))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], rotation=20, ha="right")
    ax.set_ylabel("Mean ACC (across all 15 datasets)")
    ax.set_title("TopoGate Ablation — Mean ACC per variant")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
