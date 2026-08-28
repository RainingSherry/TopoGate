#!/usr/bin/env python
"""Aggregate v5 results and compare to v4_static baseline."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path("/home/luolie/ToPoGate")
V5_DIR = ROOT / "result" / "learnable_gate_smoke" / "v5" / "v5_main"
V4_DIR = ROOT / "result" / "learnable_gate_smoke" / "v4_baseline"

V5_VARIANTS = ["v5_1g_ste", "v5_1g_fixed", "v5_4f_fixed"]
DATASETS_8 = [
    "Mouse_retina", "breast_cancer_wisconsin_original", "enron",
    "har", "iris", "mammographic_mass", "spambase", "hrvatin_filtered",
]


def load_metric(d: Path) -> float | None:
    mf = d / "metrics.json"
    if not mf.exists():
        return None
    with open(mf) as f:
        return json.load(f).get("ari")


def main():
    # v4_static baseline (avg over seeds)
    print("Per-dataset v4_static baseline (note: seed 1 only used if available):")
    v4_static = {}
    for ds in DATASETS_8:
        vals = []
        for seed_dir in V4_DIR.glob(f"{ds}__v4_static__seed*"):
            v = load_metric(seed_dir)
            if v is not None:
                vals.append(v)
        if vals:
            v4_static[ds] = sum(vals) / len(vals)
            print(f"  {ds}: {v4_static[ds]:.4f} (n={len(vals)})")

    print()
    print("Per-dataset v5 results (seed 1):")
    for variant in V5_VARIANTS:
        print(f"\n  --- {variant} ---")
        for ds in DATASETS_8:
            d = V5_DIR / f"{ds}__{variant}__seed1"
            if d.exists():
                v = load_metric(d)
                if v is not None:
                    delta = v - v4_static.get(ds, 0)
                    print(f"  {ds}: {v:.4f}  (Δ={delta:+.4f})")
                else:
                    print(f"  {ds}: FAIL")
            else:
                print(f"  {ds}: NOT RUN")

    # Average per variant
    print("\nAverage per variant:")
    for variant in V5_VARIANTS:
        deltas = []
        for ds in DATASETS_8:
            d = V5_DIR / f"{ds}__{variant}__seed1"
            v = load_metric(d) if d.exists() else None
            if v is not None and ds in v4_static:
                deltas.append(v - v4_static[ds])
        if deltas:
            avg = sum(deltas) / len(deltas)
            print(f"  {variant}: avg Δ={avg:+.4f} (n={len(deltas)})")
            wins = sum(1 for d in deltas if d > 0)
            print(f"    wins vs v4_static: {wins}/{len(deltas)}")


if __name__ == "__main__":
    main()
