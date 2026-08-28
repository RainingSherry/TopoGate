#!/usr/bin/env python3
"""Aggregate v9 main + ablation results into analysis CSVs.

Outputs:
  result/v9_learnable_gate/v9_summary.csv - v9 vs v2 multiseed (15 ds × 3 seeds)
  result/v9_learnable_gate/ablation_table.csv - 4 variant × 15 ds × 3 seeds
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

MS_DIR = Path("/home/luolie/ToPoGate/result/v9_learnable_gate/multiseed")
ABL_DIR = Path("/home/luolie/ToPoGate/result/v9_learnable_gate/ablation")
V2_DIR = Path("/home/luolie/ToPoGate/result/learnable_gate_smoke/multiseed")
OUT_DIR = Path("/home/luolie/ToPoGate/result/v9_learnable_gate")


def load_results(directory: Path, pattern: str = "*.json"):
    """Load all JSON results, keyed by (dataset, variant, seed)."""
    out = defaultdict(dict)
    for f in directory.glob(pattern):
        try:
            with open(f) as fp:
                d = json.load(fp)
        except Exception:
            continue
        ds = d.get("dataset")
        var = d.get("variant")
        seed = d.get("seed")
        if ds and var and seed is not None:
            out[(ds, var)][seed] = d
    return out


def mean_std(values):
    if not values:
        return None, None, 0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, var ** 0.5, n


def main():
    ms = load_results(MS_DIR)
    abl = load_results(ABL_DIR)
    v2 = load_results(V2_DIR, "*learnable_gate_sched*.json")

    # Collect datasets present in v9 main
    v9_main_datasets = sorted({ds for ds, var in ms.keys()})
    # Ablation datasets (incl. hrvatin)
    abl_variants = sorted({var for ds, var in abl.keys()})
    abl_datasets = sorted({ds for ds, var in abl.keys()})

    # === v9_summary.csv: v9 main vs v2 multiseed ===
    summary_path = OUT_DIR / "v9_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "v9_ari_mean", "v9_ari_std", "v9_n_seeds",
            "v2_ari_mean", "v2_ari_std", "v2_n_seeds",
            "delta_ari", "v9_acc_mean", "v9_nmi_mean",
        ])
        for ds in v9_main_datasets:
            # hrvatin runs use variant v9_adaptive_hrvatin
            v9_main_variants = ("v9_adaptive_hrvatin",) if ds == "hrvatin_filtered" else ("v9_adaptive",)
            v9_runs = {}
            for v in v9_main_variants:
                v9_runs.update(ms.get((ds, v), {}))
            v9_ari = [d["ari"] for d in v9_runs.values() if d.get("ari") is not None and d.get("error") is None]
            v9_acc = [d["acc"] for d in v9_runs.values() if d.get("acc") is not None]
            v9_nmi = [d["nmi"] for d in v9_runs.values() if d.get("nmi") is not None]
            v2_runs = v2.get((ds, "learnable_gate_sched"), {})
            v2_ari = [d["ari"] for d in v2_runs.values() if d.get("ari") is not None and d.get("error") is None]
            m9, s9, n9 = mean_std(v9_ari)
            m2, s2, n2 = mean_std(v2_ari)
            m_acc, _, _ = mean_std(v9_acc)
            m_nmi, _, _ = mean_std(v9_nmi)
            delta = (m9 - m2) if (m9 is not None and m2 is not None) else None
            w.writerow([
                ds,
                f"{m9:.4f}" if m9 is not None else "",
                f"{s9:.4f}" if s9 is not None else "",
                n9,
                f"{m2:.4f}" if m2 is not None else "",
                f"{s2:.4f}" if s2 is not None else "",
                n2,
                f"{delta:+.4f}" if delta is not None else "",
                f"{m_acc:.4f}" if m_acc is not None else "",
                f"{m_nmi:.4f}" if m_nmi is not None else "",
            ])
    print(f"Wrote {summary_path}")

    # === ablation_table.csv: 4 variant × ds × 3 seeds ===
    abl_path = OUT_DIR / "ablation_table.csv"
    with open(abl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "variant", "gate_mode", "mix_mode", "neighbor_source",
            "ari_mean", "ari_std", "n_seeds", "n_ok", "n_err",
        ])
        # Variant → (gate_mode, mix_mode, neighbor_source)
        variant_meta = {
            "v9_static_gate":      ("topology", "reliability", "topo"),
            "v9_random_neighbors": ("learned",  "random",      "random"),
            "v9_static_random":    ("topology", "random",      "random"),
            "v9_nomix":            ("learned",  "none",        "—"),
        }
        for ds in abl_datasets:
            for var in abl_variants:
                runs = abl.get((ds, var), {})
                aris = [d["ari"] for d in runs.values()
                        if d.get("ari") is not None and d.get("error") is None]
                n_err = sum(1 for d in runs.values() if d.get("error") is not None)
                n_ok = len(aris)
                m, s, _ = mean_std(aris)
                gm, mm, ns = variant_meta[var]
                w.writerow([
                    ds, var, gm, mm, ns,
                    f"{m:.4f}" if m is not None else "",
                    f"{s:.4f}" if s is not None else "",
                    len(runs),
                    n_ok,
                    n_err,
                ])
    print(f"Wrote {abl_path}")

    # === ablation_summary.csv: 4 variant aggregate (mean Δ vs v9 main) ===
    summary2 = OUT_DIR / "ablation_summary.csv"
    with open(summary2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset",
            "v9_adaptive_ari",
            "v9_static_gate_delta",
            "v9_random_neighbors_delta",
            "v9_static_random_delta",
            "v9_nomix_delta",
        ])
        all_ds = sorted(set(v9_main_datasets) | set(abl_datasets))
        for ds in all_ds:
            v9_main_variants = ("v9_adaptive_hrvatin",) if ds == "hrvatin_filtered" else ("v9_adaptive",)
            v9_main_runs = {}
            for v in v9_main_variants:
                v9_main_runs.update(ms.get((ds, v), {}))
            v9_main_ari = [d["ari"] for d in v9_main_runs.values()
                           if d.get("ari") is not None and d.get("error") is None]
            m_main, _, n_main = mean_std(v9_main_ari)
            row = [ds, f"{m_main:.4f}" if m_main is not None else ""]
            for var in ["v9_static_gate", "v9_random_neighbors",
                        "v9_static_random", "v9_nomix"]:
                v_runs = abl.get((ds, var), {})
                v_ari = [d["ari"] for d in v_runs.values()
                         if d.get("ari") is not None and d.get("error") is None]
                m_v, _, _ = mean_std(v_ari)
                if m_main is not None and m_v is not None:
                    row.append(f"{m_v - m_main:+.4f}")
                else:
                    row.append("")
            w.writerow(row)
    print(f"Wrote {summary2}")


if __name__ == "__main__":
    main()