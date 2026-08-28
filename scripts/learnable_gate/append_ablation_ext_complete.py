#!/usr/bin/env python3
"""Append completed ext ablation rows (from run_ablation_ext_complete.sh) to merged_summary.csv.

Reads result/ablation/<ds>/<ds>__static_gate_<variant>__ep30*.json files,
extracts metrics, appends to merged_summary.csv.
"""
import csv
import json
from pathlib import Path

ABL_DIR = Path("/home/luolie/ToPoGate/result/ablation")
CSV_PATH = ABL_DIR / "merged_summary.csv"

EXT_DATASETS = [
    "reuters", "ISOLET", "spambase", "cnae9", "Campbell",
    "hrvatin_filtered", "Quake_Smart-seq2_Lung", "mammographic_mass",
    "first-order-theorem-proving", "iris",
]
MISSING_VARIANTS = [
    "static_gate_far_neighbors",
    "static_gate_edge_only",
    "static_gate_gate_only",
    "static_gate_no_topology_features",
]

# Load existing CSV
existing_keys = set()
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    existing_rows = list(reader)
    fieldnames = reader.fieldnames
    for r in existing_rows:
        existing_keys.add((r["dataset"], r["variant"], r["epochs"]))

# Find new rows
new_rows = []
for ds in EXT_DATASETS:
    for variant in MISSING_VARIANTS:
        # Match ep30 files (the latest run)
        candidates = list(ABL_DIR.glob(f"{ds}/{ds}__{variant}__ep30*.json"))
        # Filter out error files
        candidates = [c for c in candidates if "error" not in c.name]
        if not candidates:
            print(f"[MISS] {ds} {variant}: no file")
            continue
        for c in candidates:
            with open(c) as f:
                row = json.load(f)
            key = (row["dataset"], row["variant"], str(row["epochs"]))
            if key in existing_keys:
                continue
            # Ensure layer=ext for these
            if "layer" not in row:
                row["layer"] = "ext"
            # Drop any extra fields not in CSV fieldnames
            for k in list(row.keys()):
                if k not in fieldnames:
                    row.pop(k)
            new_rows.append(row)
            existing_keys.add(key)
            print(f"[OK] {ds} {variant}: ACC={row['acc']:.4f}")

# Append
if new_rows:
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(new_rows)
    print(f"\nAppended {len(new_rows)} rows to {CSV_PATH}")
else:
    print("No new rows to append.")