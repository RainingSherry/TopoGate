#!/usr/bin/env python3
"""
Aggregate LearnableGate 134-dataset sweep results.

Reads:
  result/learnable_gate_134_sweep/stage1.csv
  result/learnable_gate_134_sweep/stage2.csv
  result/learnable_gate_134_sweep/stage3.csv
  result/learnable_gate_134_sweep/stage{1,2,3}/   (JSON files)

Outputs:
  result/learnable_gate_134_sweep/merged_summary.csv       — all runs
  result/learnable_gate_134_sweep/best_per_dataset.csv      — best LG config per dataset
  result/learnable_gate_134_sweep/comparison_lg_vs_sg.csv  — LG vs SG per dataset
  result/learnable_gate_134_sweep/summary_by_type.csv       — grouped by data type
  result/learnable_gate_134_sweep/paper_tables/
      main_table.csv    — per-dataset ARI with LG vs SG + Δ + p-value
      appendix_table.csv — full metrics (ACC, NMI, ARI) for all 134 datasets
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULT_ROOT = REPO_ROOT / "result" / "learnable_gate_134_sweep"
DATA_DIR = Path("/data/luolie/ToPoGate/datasets")

# ── Dataset metadata ────────────────────────────────────────────────────────────
def load_dataset_info() -> Dict[str, dict]:
    csv_path = REPO_ROOT / "result" / "dataset_npz_info.csv"
    if not csv_path.exists():
        return {}
    info = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            name = row["dataset_name"]
            info[name] = {
                "n_samples": int(row["n_samples"]),
                "n_features": int(row["n_features"]),
                "n_clusters": int(row["n_clusters"]),
                "data_type": row.get("data_type", "unknown"),
            }
    return info


# ── Load all JSON results ──────────────────────────────────────────────────────
def load_all_results() -> List[dict]:
    """Load every JSON result file from stage{1,2,3}/ directories."""
    rows = []
    for stage in [1, 2, 3]:
        stage_dir = RESULT_ROOT / f"stage{stage}"
        if not stage_dir.exists():
            continue
        for json_path in stage_dir.glob("*.json"):
            if json_path.stem.endswith(".error"):
                continue
            try:
                with open(json_path) as f:
                    r = json.load(f)
                rows.append(r)
            except Exception:
                pass

    # Also parse CSV as fallback
    for stage in [1, 2, 3]:
        csv_path = RESULT_ROOT / f"stage{stage}.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "acc" in row and row["acc"]:
                        rows.append({k: _try_float(v) for k, v in row.items()})
    return rows


def _try_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


# ── Best per dataset ───────────────────────────────────────────────────────────
def best_per_dataset(results: List[dict]) -> Dict[str, dict]:
    """Per dataset, pick the LG config with highest ARI."""
    by_ds = defaultdict(list)
    for r in results:
        if r.get("variant") == "static_gate_full":
            continue
        by_ds[r["dataset"]].append(r)

    best = {}
    for ds, runs in by_ds.items():
        best_run = max(runs, key=lambda r: r.get("ari", -999))
        best[ds] = best_run
    return best


# ── LG vs SG comparison ────────────────────────────────────────────────────────
def lg_vs_sg(results: List[dict], ds_info: Dict) -> List[dict]:
    """Build per-dataset comparison rows: LG (mean over seeds) vs SG (mean over seeds)."""
    by_ds = defaultdict(lambda: defaultdict(list))
    for r in results:
        ds = r["dataset"]
        variant = r.get("variant", "unknown")
        seed = int(r.get("seed", 42))
        key = (ds, variant, seed)
        by_ds[ds][variant].append(r)

    rows = []
    for ds in sorted(by_ds):
        lg_runs = by_ds[ds].get("learnable_gate_sched", [])
        sg_runs = by_ds[ds].get("static_gate_full", [])

        if not lg_runs and not sg_runs:
            continue

        # LG stats
        if lg_runs:
            lg_ari = [r.get("ari", 0) for r in lg_runs]
            lg_acc = [r.get("acc", 0) for r in lg_runs]
            lg_nmi = [r.get("nmi", 0) for r in lg_runs]
            lg_ari_mean = float(np.mean(lg_ari))
            lg_ari_std = float(np.std(lg_ari))
            lg_acc_mean = float(np.mean(lg_acc))
            lg_nmi_mean = float(np.mean(lg_nmi))
        else:
            lg_ari_mean = lg_ari_std = lg_acc_mean = lg_nmi_mean = np.nan

        # SG stats
        if sg_runs:
            sg_ari = [r.get("ari", 0) for r in sg_runs]
            sg_acc = [r.get("acc", 0) for r in sg_runs]
            sg_nmi = [r.get("nmi", 0) for r in sg_runs]
            sg_ari_mean = float(np.mean(sg_ari))
            sg_ari_std = float(np.std(sg_ari))
            sg_acc_mean = float(np.mean(sg_acc))
            sg_nmi_mean = float(np.mean(sg_nmi))
        else:
            sg_ari_mean = sg_ari_std = sg_acc_mean = sg_nmi_mean = np.nan

        delta_ari = lg_ari_mean - sg_ari_mean

        # Wilcoxon test (paired per seed)
        if len(lg_runs) >= 3 and len(sg_runs) >= 3:
            lg_ari_s = sorted([r.get("ari", 0) for r in lg_runs])
            sg_ari_s = sorted([r.get("ari", 0) for r in sg_runs])
            min_len = min(len(lg_ari_s), len(sg_ari_s))
            try:
                _, p_value = stats.wilcoxon(lg_ari_s[:min_len], sg_ari_s[:min_len])
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan

        verdict = ("LG wins" if delta_ari > 0.01
                   else "SG wins" if delta_ari < -0.01
                   else "tie")

        info = ds_info.get(ds, {})
        rows.append({
            "dataset": ds,
            "data_type": info.get("data_type", "unknown"),
            "n_samples": info.get("n_samples", 0),
            "n_clusters": info.get("n_clusters", 0),
            "lg_ari_mean": lg_ari_mean,
            "lg_ari_std": lg_ari_std,
            "lg_acc_mean": lg_acc_mean,
            "lg_nmi_mean": lg_nmi_mean,
            "sg_ari_mean": sg_ari_mean,
            "sg_ari_std": sg_ari_std,
            "sg_acc_mean": sg_acc_mean,
            "sg_nmi_mean": sg_nmi_mean,
            "delta_ari": delta_ari,
            "p_value": p_value,
            "verdict": verdict,
            "n_lg_runs": len(lg_runs),
            "n_sg_runs": len(sg_runs),
        })
    return rows


# ── Summary by data type ───────────────────────────────────────────────────────
def summary_by_type(comparison: List[dict]) -> List[dict]:
    """Aggregate win/tie/loss counts and mean ARI by data type."""
    by_type = defaultdict(lambda: {"datasets": [], "lg_ari": [], "sg_ari": [], "delta": []})
    for row in comparison:
        t = row["data_type"]
        by_type[t]["datasets"].append(row["dataset"])
        by_type[t]["lg_ari"].append(row["lg_ari_mean"])
        by_type[t]["sg_ari"].append(row["sg_ari_mean"])
        by_type[t]["delta"].append(row["delta_ari"])

    rows = []
    for t, d in sorted(by_type.items()):
        wins = sum(1 for delta in d["delta"] if delta > 0.01)
        losses = sum(1 for delta in d["delta"] if delta < -0.01)
        ties = len(d["delta"]) - wins - losses
        lg_ari_valid = [v for v in d["lg_ari"] if not np.isnan(v)]
        sg_ari_valid = [v for v in d["sg_ari"] if not np.isnan(v)]
        rows.append({
            "data_type": t,
            "n_datasets": len(d["datasets"]),
            "lg_ari_mean": float(np.mean(lg_ari_valid)) if lg_ari_valid else np.nan,
            "lg_ari_std": float(np.std(lg_ari_valid)) if lg_ari_valid else np.nan,
            "sg_ari_mean": float(np.mean(sg_ari_valid)) if sg_ari_valid else np.nan,
            "sg_ari_std": float(np.std(sg_ari_valid)) if sg_ari_valid else np.nan,
            "delta_mean": float(np.mean(d["delta"])) if d["delta"] else np.nan,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        })
    return rows


# ── Write CSV helpers ─────────────────────────────────────────────────────────
def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote {path}  ({len(rows)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading results…")
    results = load_all_results()
    print(f"  {len(results)} total run results loaded")
    if not results:
        print("No results found. Run Stage 1/2/3 first.")
        return

    ds_info = load_dataset_info()

    # 1. merged_summary.csv
    print("\n[1/5] Building merged_summary.csv …")
    merged_rows = []
    for r in results:
        merged_rows.append({
            "dataset": r.get("dataset", ""),
            "variant": r.get("variant", ""),
            "seed": r.get("seed", ""),
            "n_samples": r.get("n_samples", ""),
            "n_clusters": r.get("n_clusters", ""),
            "epochs": r.get("epochs", ""),
            "mask_ratio": r.get("mask_ratio", ""),
            "neighbor_k": r.get("neighbor_k", ""),
            "gate_max": r.get("gate_max", ""),
            "runtime_seconds": r.get("runtime_seconds", ""),
            "acc": r.get("acc", ""),
            "nmi": r.get("nmi", ""),
            "ari": r.get("ari", ""),
        })
    merged_fields = ["dataset", "variant", "seed", "n_samples", "n_clusters",
                     "epochs", "mask_ratio", "neighbor_k", "gate_max",
                     "runtime_seconds", "acc", "nmi", "ari"]
    write_csv(RESULT_ROOT / "merged_summary.csv", merged_rows, merged_fields)

    # 2. best_per_dataset.csv
    print("[2/5] Building best_per_dataset.csv …")
    best = best_per_dataset(results)
    best_rows = []
    for ds, r in sorted(best.items()):
        best_rows.append({
            "dataset": ds,
            "variant": r.get("variant", ""),
            "seed": r.get("seed", ""),
            "n_samples": r.get("n_samples", ""),
            "n_clusters": r.get("n_clusters", ""),
            "epochs": r.get("epochs", ""),
            "mask_ratio": r.get("mask_ratio", ""),
            "neighbor_k": r.get("neighbor_k", ""),
            "gate_max": r.get("gate_max", ""),
            "runtime_seconds": r.get("runtime_seconds", ""),
            "acc": r.get("acc", ""),
            "nmi": r.get("nmi", ""),
            "ari": r.get("ari", ""),
        })
    best_fields = ["dataset", "variant", "seed", "n_samples", "n_clusters",
                   "epochs", "mask_ratio", "neighbor_k", "gate_max",
                   "runtime_seconds", "acc", "nmi", "ari"]
    write_csv(RESULT_ROOT / "best_per_dataset.csv", best_rows, best_fields)

    # 3. comparison_lg_vs_sg.csv
    print("[3/5] Building comparison_lg_vs_sg.csv …")
    comp = lg_vs_sg(results, ds_info)
    comp_fields = ["dataset", "data_type", "n_samples", "n_clusters",
                   "lg_ari_mean", "lg_ari_std", "lg_acc_mean", "lg_nmi_mean",
                   "sg_ari_mean", "sg_ari_std", "sg_acc_mean", "sg_nmi_mean",
                   "delta_ari", "p_value", "verdict", "n_lg_runs", "n_sg_runs"]
    write_csv(RESULT_ROOT / "comparison_lg_vs_sg.csv", comp, comp_fields)

    # 4. summary_by_type.csv
    print("[4/5] Building summary_by_type.csv …")
    by_type = summary_by_type(comp)
    type_fields = ["data_type", "n_datasets",
                   "lg_ari_mean", "lg_ari_std", "sg_ari_mean", "sg_ari_std",
                   "delta_mean", "wins", "ties", "losses"]
    write_csv(RESULT_ROOT / "summary_by_type.csv", by_type, type_fields)

    # 5. paper tables
    print("[5/5] Building paper_tables/ …")
    paper_dir = RESULT_ROOT / "paper_tables"
    paper_dir.mkdir(parents=True, exist_ok=True)

    # main_table.csv — minimal for paper (ARI comparison + Δ + p-value)
    main_fields = ["dataset", "data_type", "n_samples", "n_clusters",
                   "lg_ari_mean", "lg_ari_std", "sg_ari_mean", "sg_ari_std",
                   "delta_ari", "p_value", "verdict"]
    main_rows = [{k: row.get(k, "") for k in main_fields} for row in comp]
    write_csv(paper_dir / "main_table.csv", main_rows, main_fields)

    # appendix_table.csv — full metrics
    app_rows = []
    for row in comp:
        app_rows.append({
            "dataset": row["dataset"],
            "data_type": row["data_type"],
            "n_samples": row["n_samples"],
            "n_clusters": row["n_clusters"],
            "lg_acc": row["lg_acc_mean"],
            "lg_nmi": row["lg_nmi_mean"],
            "lg_ari": row["lg_ari_mean"],
            "lg_ari_std": row["lg_ari_std"],
            "sg_acc": row["sg_acc_mean"],
            "sg_nmi": row["sg_nmi_mean"],
            "sg_ari": row["sg_ari_mean"],
            "sg_ari_std": row["sg_ari_std"],
            "delta_ari": row["delta_ari"],
            "p_value": row["p_value"],
            "verdict": row["verdict"],
        })
    app_fields = ["dataset", "data_type", "n_samples", "n_clusters",
                  "lg_acc", "lg_nmi", "lg_ari", "lg_ari_std",
                  "sg_acc", "sg_nmi", "sg_ari", "sg_ari_std",
                  "delta_ari", "p_value", "verdict"]
    write_csv(paper_dir / "appendix_table.csv", app_rows, app_fields)

    # Summary
    print("\nDone.")
    n_lg = sum(1 for r in results if r.get("variant") == "learnable_gate_sched")
    n_sg = sum(1 for r in results if r.get("variant") == "static_gate_full")
    print(f"  Total runs: {len(results)}  (LG={n_lg}, SG={n_sg})")
    print(f"  Datasets in comparison: {len(comp)}")
    if by_type:
        for row in by_type:
            print(f"  [{row['data_type']}] {row['n_datasets']} ds  "
                  f"LG={row['lg_ari_mean']:.3f}  SG={row['sg_ari_mean']:.3f}  "
                  f"Δ={row['delta_mean']:+.3f}  "
                  f"wins={row['wins']}  ties={row['ties']}  losses={row['losses']}")


if __name__ == "__main__":
    main()
