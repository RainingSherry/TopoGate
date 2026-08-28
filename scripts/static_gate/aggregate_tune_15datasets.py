#!/usr/bin/env python
"""Determine the best (epochs, mask_ratio, neighbor_k) per dataset from Phase 0 tuning.

Reads:
  result/tune_15datasets/<dataset>/<dataset>__ep<ep>_mr<mr>_k<k>.json

Outputs:
  result/tune_15datasets/best_per_dataset.csv  (15 rows)
  result/tune_15datasets/dominant_hparams.json   (single dict for ablation)
  result/tune_15datasets/transfer_analysis.md    (vs 131-dataset result)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

DATASETS = [
    "Mouse_retina",
    "sms_spam_collection",
    "enron",
    "ISOLET",
    "har",
    "spambase",
    "breast_cancer_wisconsin_original",
    "reuters",
    "cnae9",
    "Campbell",
    "mammographic_mass",
    "first-order-theorem-proving",
    "hrvatin_filtered",
    "Quake_Smart-seq2_Lung",
    "iris",
]

GRID_EPOCHS = [40, 80, 150]
GRID_MASK_RATIO = [0.3, 0.4, 0.5]
GRID_NEIGHBOR_K = [5, 10, 20]

# 131-dataset optimal (from previous rounds)
OPT_131 = dict(epochs=40, mask_ratio=0.3, neighbor_k=10)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tune_dir", default="/home/luolie/ToPoGate/result/tune_15datasets")
    return p.parse_args()


def main():
    args = parse_args()
    tune_dir = Path(args.tune_dir)

    # Per-dataset: best config by ACC
    best_rows = []
    dominant_votes = {"epochs": Counter(), "mask_ratio": Counter(), "neighbor_k": Counter()}
    # Also track mean of bests for ablation
    all_bests = []

    for ds in DATASETS:
        ds_dir = tune_dir / ds
        if not ds_dir.exists():
            print(f"[warn] no dir for {ds}; skipping")
            continue
        candidates = []
        for ep in GRID_EPOCHS:
            for mr in GRID_MASK_RATIO:
                for k in GRID_NEIGHBOR_K:
                    fname = f"{ds}__ep{ep}_mr{mr}_k{k}.json"
                    fp = ds_dir / fname
                    if not fp.exists():
                        continue
                    with open(fp) as f:
                        row = json.load(f)
                    candidates.append((row["acc"], row))
        if not candidates:
            print(f"[warn] no results for {ds}")
            continue
        candidates.sort(key=lambda x: -x[0])
        best = candidates[0][1]
        all_bests.append(best)
        best_rows.append({
            "dataset": ds,
            "best_acc": best["acc"],
            "best_nmi": best["nmi"],
            "best_ari": best["ari"],
            "best_epochs": best["epochs"],
            "best_mask_ratio": best["mask_ratio"],
            "best_neighbor_k": best["neighbor_k"],
            "best_runtime": best["runtime_seconds"],
            "n_configs_tested": len(candidates),
        })
        dominant_votes["epochs"][best["epochs"]] += 1
        dominant_votes["mask_ratio"][best["mask_ratio"]] += 1
        dominant_votes["neighbor_k"][best["neighbor_k"]] += 1

    # Write best_per_dataset.csv
    out_csv = tune_dir / "best_per_dataset.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        for row in best_rows:
            writer.writerow(row)
    print(f"Wrote {out_csv}")

    # Find dominant hyperparameters (most common best)
    dominant = {
        "epochs": dominant_votes["epochs"].most_common(1)[0][0],
        "mask_ratio": dominant_votes["mask_ratio"].most_common(1)[0][0],
        "neighbor_k": dominant_votes["neighbor_k"].most_common(1)[0][0],
        "votes_epochs": dict(dominant_votes["epochs"]),
        "votes_mask_ratio": dict(dominant_votes["mask_ratio"]),
        "votes_neighbor_k": dict(dominant_votes["neighbor_k"]),
    }
    out_json = tune_dir / "dominant_hparams.json"
    with open(out_json, "w") as f:
        json.dump(dominant, f, indent=2)
    print(f"Wrote {out_json}")
    print(f"Dominant: epochs={dominant['epochs']}  mask_ratio={dominant['mask_ratio']}  neighbor_k={dominant['neighbor_k']}")

    # Write transfer analysis
    out_md = tune_dir / "transfer_analysis.md"
    with open(out_md, "w") as f:
        f.write("# 15-dataset vs 131-dataset Hyperparameter Transfer Analysis\n\n")
        f.write(f"## 131-dataset optimal (Round 2)\n\n")
        f.write(f"```\n{OPT_131}\n```\n\n")
        f.write(f"## 15-dataset optimal (Phase 0)\n\n")
        f.write(f"```\n")
        f.write(f"epochs      = {dominant['epochs']}  (votes: {dominant['votes_epochs']})\n")
        f.write(f"mask_ratio  = {dominant['mask_ratio']}  (votes: {dominant['votes_mask_ratio']})\n")
        f.write(f"neighbor_k  = {dominant['neighbor_k']}  (votes: {dominant['votes_neighbor_k']})\n")
        f.write(f"```\n\n")
        f.write(f"## Per-dataset best configs\n\n")
        f.write("| Dataset | best ACC | epochs | mask_ratio | neighbor_k |\n")
        f.write("|---------|----------|--------|------------|------------|\n")
        for row in best_rows:
            f.write(f"| {row['dataset']} | {row['best_acc']:.4f} | "
                    f"{row['best_epochs']} | {row['best_mask_ratio']} | "
                    f"{row['best_neighbor_k']} |\n")
        f.write(f"\n## Recommendation for ablation\n\n")
        f.write(f"Use dominant hyperparameters: "
                f"`epochs={dominant['epochs']}`, "
                f"`mask_ratio={dominant['mask_ratio']}`, "
                f"`neighbor_k={dominant['neighbor_k']}`.\n")
        diff_count = sum(1 for r in best_rows
                         if (r["best_epochs"], r["best_mask_ratio"], r["best_neighbor_k"])
                         != (OPT_131["epochs"], OPT_131["mask_ratio"], OPT_131["neighbor_k"]))
        f.write(f"\n{diff_count}/{len(best_rows)} datasets have a different best from 131-dataset tune.\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
