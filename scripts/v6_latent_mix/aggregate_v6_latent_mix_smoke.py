#!/usr/bin/env python
"""Aggregate v6 smoke results and compare to LearnableGate@sched baseline.

Reads:
  - result/v6_latent_mix/smoke/<dataset>__v6_latent_mix_smoke__seed<seed>/metrics.json
  - result/learnable_gate_smoke/multiseed/<dataset>__learnable_gate_sched__seed<seed>/metrics.json

Writes a tabular comparison (printed) showing per-dataset ARI for both variants
plus the delta.  This is the Phase-5.1 acceptance test for the v6 plan.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/luolie/ToPoGate")
V6_DIR = ROOT / "result" / "v6_latent_mix" / "smoke"
LG_DIR = ROOT / "result" / "learnable_gate_smoke" / "multiseed"

DATASETS = [
    "Mouse_retina",
    "enron",
    "har",
    "Campbell",
    "breast_cancer_wisconsin_original",
]

V6_VARIANT = "v6_latent_mix_smoke"
LG_VARIANT = "learnable_gate_sched"


def load_metric(d: Path) -> float | None:
    """Read metrics.json from a directory. Fallback: read a flat JSON file."""
    # Try directory layout first
    mf = d / "metrics.json"
    if mf.exists():
        with open(mf) as f:
            return json.load(f).get("ari")
    # Try flat-file layout (LearnableGate@sched stores ARI directly in a JSON
    # sibling at <dataset>__<variant>__seed<seed>.json)
    flat = d.with_suffix(".json")
    if flat.exists():
        with open(flat) as f:
            return json.load(f).get("ari")
    return None


def main():
    print("Per-dataset v6 vs LearnableGate@sched (ARI; seed 42 only for smoke):")
    print()
    deltas = []
    for ds in DATASETS:
        v6_path = V6_DIR / f"{ds}__{V6_VARIANT}__seed42"
        lg_path = LG_DIR / f"{ds}__{LG_VARIANT}__seed42"
        v6_ari = load_metric(v6_path)
        lg_ari = load_metric(lg_path)
        if v6_ari is None or lg_ari is None:
            print(f"  {ds:40s}  v6={'N/A':>8}  lg={'N/A':>8}  delta={'N/A':>8}  ({'v6 missing' if v6_ari is None else 'lg missing'})")
            continue
        delta = v6_ari - lg_ari
        deltas.append(delta)
        sign = "+" if delta >= 0 else ""
        print(f"  {ds:40s}  v6={v6_ari:.4f}  lg={lg_ari:.4f}  delta={sign}{delta:+.4f}")

    print()
    if deltas:
        avg = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        ties = len(deltas) - wins - losses
        print(f"Average ΔARI (v6 - LearnableGate@sched): {avg:+.4f} (n={len(deltas)})")
        print(f"Wins / Losses / Ties: {wins} / {losses} / {ties}")
        if avg >= -0.01 and losses <= 1:
            print("Phase 5.1 acceptance criterion PASSED.")
        else:
            print("Phase 5.1 acceptance criterion NOT MET.")


if __name__ == "__main__":
    main()