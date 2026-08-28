#!/usr/bin/env python
"""Aggregate ablation results into layer summaries and merged summary.

Reads:
  result/ablation/<dataset>/<dataset>__<variant>__ep<ep>_mr<mr>_k<k>.json

Outputs:
  result/ablation/core/summary.csv    (5 datasets × 8 variants = 40 rows)
  result/ablation/ext/summary.csv     (10 datasets × 4 variants = 40 rows)
  result/ablation/merged_summary.csv  (80 rows)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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
ALL_DATASETS = CORE_DATASETS + EXT_DATASETS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", default="/home/luolie/ToPoGate/result/ablation")
    return p.parse_args()


def collect_rows(result_dir: Path):
    rows = []
    for ds in ALL_DATASETS:
        ds_dir = result_dir / ds
        if not ds_dir.exists():
            continue
        for fp in sorted(ds_dir.glob(f"{ds}__topogate_*.json")):
            if fp.name.endswith(".error.json"):
                continue
            with open(fp) as f:
                row = json.load(f)
            rows.append(row)
    return rows


def write_csv(rows, out_path: Path, fieldnames):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    rows = collect_rows(result_dir)
    if not rows:
        print(f"No rows found in {result_dir}")
        return
    print(f"Collected {len(rows)} rows")

    fieldnames = [
        "dataset", "layer", "variant", "seed", "gpu",
        "n_samples", "n_features", "n_clusters",
        "epochs", "mask_ratio", "neighbor_k", "hidden_size",
        "acc", "nmi", "ari", "f1_macro", "runtime_seconds",
    ]

    # Core: 5 core datasets, all variants
    core_rows = [r for r in rows if r["dataset"] in CORE_DATASETS]
    core_path = result_dir / "core" / "summary.csv"
    write_csv(core_rows, core_path, fieldnames)
    print(f"Wrote {core_path}: {len(core_rows)} rows")

    # Ext: 10 ext datasets, restricted variants
    ext_rows = [r for r in rows if r["dataset"] in EXT_DATASETS]
    ext_path = result_dir / "ext" / "summary.csv"
    write_csv(ext_rows, ext_path, fieldnames)
    print(f"Wrote {ext_path}: {len(ext_rows)} rows")

    # Merged: 80 rows
    merged_path = result_dir / "merged_summary.csv"
    write_csv(rows, merged_path, fieldnames)
    print(f"Wrote {merged_path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
