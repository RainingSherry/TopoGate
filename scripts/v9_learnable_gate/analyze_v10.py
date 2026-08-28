#!/usr/bin/env python3
"""Analyze v10 nomix_init results vs v9_adaptive.

Usage: python scripts/v9_learnable_gate/analyze_v10.py
"""
import json, glob, statistics, os
from pathlib import Path

REPO = Path("/home/luolie/ToPoGate")
V10_MS = REPO / "result/v10_learnable_gate/multiseed"
V10_AB = REPO / "result/v10_learnable_gate/ablation"
V9_MS = Path("/home/luolie/ToPoGate/result/v9_learnable_gate/multiseed")
V9_AB = REPO / "result/v9_learnable_gate/ablation"

# NOTE: Campbell is excluded (OOm on this machine). v9 data from v9_multiseed
# directory; v10 data from v10_multiseed directory.

def load_grouped(dir, variant_pattern):
    """Load results grouped by dataset."""
    by_ds = {}
    for j in glob.glob(f"{dir}/*{variant_pattern}*.json"):
        fname = os.path.basename(j)
        parts = fname.split("__")
        ds = parts[0]
        with open(j) as f:
            d = json.load(f)
        if ds not in by_ds:
            by_ds[ds] = []
        by_ds[ds].append(d)
    return by_ds

def stats(lst):
    if not lst:
        return None, None
    return statistics.mean(lst), statistics.stdev(lst) if len(lst) > 1 else 0.0

print("=" * 100)
print("v10 nomix_init vs v9_adaptive — 3-seed mean ARI comparison")
print("=" * 100)

v10 = load_grouped(V10_MS, "v10_nomix_init")
v9  = load_grouped(V10_MS, "v9_adaptive")

all_ds = sorted(set(v10.keys()) & set(v9.keys()))

print(f"\n{'Dataset':<40} {'v10_nomix':<14} {'v9_adaptive':<14} {'Delta':<8} {'β direction':<15} {'Verdict'}")
print("-" * 105)

for ds in all_ds:
    v10_aris = [d["ari"] for d in v10[ds] if d.get("ari") is not None]
    v9_aris  = [d["ari"]  for d in v9[ds]  if d.get("ari") is not None]
    v10_bM   = [d["beta"]["beta_mutual"]    for d in v10[ds] if d.get("beta")]
    v10_bS   = [d["beta"]["beta_snn"]       for d in v10[ds] if d.get("beta")]
    v10_bP   = [d["beta"]["beta_perturb"]   for d in v10[ds] if d.get("beta")]

    if not v10_aris or not v9_aris:
        continue

    m10, s10 = stats(v10_aris)
    m9,  s9  = stats(v9_aris)
    delta = m10 - m9

    bm = statistics.mean(v10_bM)
    bs = statistics.mean(v10_bS)
    bp = statistics.mean(v10_bP)

    # β direction: positive = introducing topology, negative = suppressing
    direction = "→Topo" if (bm + bs + bp) / 3 > 0 else "→NoMix"

    if delta > 0.01:
        verdict = "BETTER"
    elif delta < -0.01:
        verdict = "WORSE"
    else:
        verdict = "SAME"

    print(f"{ds:<40} {m10:.4f}±{s10:.3f}  {m9:.4f}±{s9:.3f}  {delta:+.4f}  {direction:<15} {verdict}")

print("\n" + "=" * 100)
print("β final values (v10 nomix_init, mean across seeds)")
print("=" * 100)
print(f"{'Dataset':<40} {'β_mutual':>10} {'β_snn':>10} {'β_perturb':>10} {'init=-1.5 →'}")
print("-" * 85)
for ds in sorted(v10.keys()):
    bM = [d["beta"]["beta_mutual"]    for d in v10[ds] if d.get("beta")]
    bS = [d["beta"]["beta_snn"]       for d in v10[ds] if d.get("beta")]
    bP = [d["beta"]["beta_perturb"]   for d in v10[ds] if d.get("beta")]
    if bM:
        print(f"{ds:<40} {statistics.mean(bM):>+10.3f} {statistics.mean(bS):>+10.3f} {statistics.mean(bP):>+10.3f}")
