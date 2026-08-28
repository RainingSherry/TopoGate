#!/usr/bin/env python3
"""Smoke test for HVF + Adaptive PCA improvements on TopoGate v2.

Compares 4 configurations across 7 datasets:

  Config A  (v2_baseline):         --n_top_features=0  --knn_pca_mode=fixed   (knn_pca_dim=50)
  Config B  (hvf2000_adaptive):    --n_top_features=2000 --knn_pca_mode=adaptive (knn_pca_dim=500)
  Config C  (full_adaptive):       --n_top_features=0  --knn_pca_mode=adaptive (knn_pca_dim=500)
  Config C_nomix (nomix ablation):  --n_top_features=0  --knn_pca_mode=adaptive (knn_pca_dim=500) + mix_mode=none

Datasets selected to cover:
  - High-dimensional (primary HVF targets): enron, Mouse_retina, hrvatin_filtered, sms_spam
  - nomix-sensitive (full vs nomix delta_ari < -0.01): ISOLET, Quake_Smart-seq2_Lung, iris

Outputs:
  result/hvf_adaptive_pca/<dataset>__<config>.json
  result/hvf_adaptive_pca/comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "baseline" / "CLUBench")):
    if p not in sys.path:
        sys.path.insert(0, p)

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
from methods.TopoGate.learnable_gate.run_npz import run_topogate

LEARNABLE_GATE_ROOT = REPO_ROOT / "methods" / "TopoGate" / "learnable_gate"

# Representative datasets covering a range of dimensionality / variance retention
# + nomix-sensitive datasets (from ablation analysis)
DATASETS = [
    # High-dimensional datasets (primary test targets for HVF + Adaptive PCA)
    "enron",                  # d=4096, PCA(50) retains ~49% variance
    "Mouse_retina",           # d=6198, scRNA, PCA(50) retains ~47%
    "hrvatin_filtered",       # d=25187, scRNA, large dim
    "sms_spam_collection",    # d=500, moderate dim
    # nomix-sensitive datasets (full vs nomix delta_ari < -0.01)
    "ISOLET",                 # d=617, nomix better by +0.078
    "Quake_Smart-seq2_Lung",  # d=23341, nomix better by +0.043
    "iris",                   # d=4, nomix better by +0.132
]

# 5 configurations to compare
VARIANTS = [
    # Config A: v2 baseline (no HVF, fixed PCA dim=50) — original baseline
    ("v2_baseline", {
        "variant": "learnable_gate_sched",
        "epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5,
        "warmup_epochs": 20, "ramp_epochs": 10,
        "n_top_features": 0,
        "knn_pca_mode": "fixed",
        "knn_pca_dim": 50,
        "mix_mode": "reliability",
        "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
    }),
    # Config B: HVF top-2000, adaptive PCA (removes PCA=50 limitation)
    ("hvf2000_adaptive", {
        "variant": "learnable_gate_hvf_adaptive",
        "epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5,
        "warmup_epochs": 20, "ramp_epochs": 10,
        "n_top_features": 2000,
        "knn_pca_mode": "adaptive",
        "knn_pca_dim": 500,
        "mix_mode": "reliability",
        "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
    }),
    # Config C: no HVF (full data), adaptive PCA
    ("full_adaptive", {
        "variant": "learnable_gate_hvf_adaptive",
        "epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5,
        "warmup_epochs": 20, "ramp_epochs": 10,
        "n_top_features": 0,
        "knn_pca_mode": "adaptive",
        "knn_pca_dim": 500,
        "mix_mode": "reliability",
        "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
    }),
    # Config C_nomix: no HVF + adaptive PCA + disable neighbor mixing (nomix ablation)
    ("full_adaptive_nomix", {
        "variant": "learnable_gate_hvf_adaptive",
        "epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5,
        "warmup_epochs": 20, "ramp_epochs": 10,
        "n_top_features": 0,
        "knn_pca_mode": "adaptive",
        "knn_pca_dim": 500,
        "mix_mode": "none",   # disable neighbor mixing (nomix ablation)
        "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
    }),
    # Config D: HVF top-2000 + adaptive PCA + nomix ablation
    ("hvf2000_adaptive_nomix", {
        "variant": "learnable_gate_hvf_adaptive",
        "epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5,
        "warmup_epochs": 20, "ramp_epochs": 10,
        "n_top_features": 2000,
        "knn_pca_mode": "adaptive",
        "knn_pca_dim": 500,
        "mix_mode": "none",
        "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
    }),
]

DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
OUTPUT_DIR = Path("/home/luolie/ToPoGate/result/hvf_adaptive_pca")


def run_one(dataset: str, variant_name: str, overrides: dict, seed: int, gpu: int) -> dict:
    npz = DATA_DIR / f"{dataset}.npz"
    if not npz.exists():
        return {"dataset": dataset, "variant": variant_name, "error": f"{npz} not found"}

    data = np.load(npz)
    X = data["X"] if "X" in data.files else data["x"]
    y = data["y"] if "y" in data.files else None
    n_clusters = int(np.unique(y).size) if y is not None else None

    out_dir = OUTPUT_DIR / f"{dataset}__{variant_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        labels, elapsed, metrics = run_topogate(
            X, n_clusters=n_clusters, y=y, gpu=gpu,
            seed=seed, return_metrics=True,
            save_dir=str(out_dir),
            **overrides,
        )
        summary_path = out_dir / "summary.json"
        beta = None
        if summary_path.exists():
            with open(summary_path) as f:
                beta = json.load(f).get("learned_gate_final_beta", {})
        return {
            "dataset": dataset,
            "variant": variant_name,
            "seed": seed,
            "n_clusters": n_clusters,
            "acc": metrics.get("acc"),
            "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
            "f1": metrics.get("f1"),
            "elapsed": float(elapsed),
            "beta": beta,
            "error": None,
        }
    except Exception as exc:
        return {
            "dataset": dataset, "variant": variant_name, "error": f"{exc}\n{traceback.format_exc()}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Run only these variant names. Skips if all seeds already have results.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_ds = [d for d in DATASETS if (not args.datasets or d in set(args.datasets))]
    selected_vars = {v: o for v, o in VARIANTS} if not args.variants else {v: o for v, o in VARIANTS if v in set(args.variants)}

    rows = []
    total = len(selected_ds) * len(selected_vars) * len(args.seeds)
    print(f"Will run {total} jobs ({len(selected_ds)} ds × {len(selected_vars)} variants × {len(args.seeds)} seeds)")
    print(f"Output: {OUTPUT_DIR}")
    print()

    i = 0
    for ds_name in selected_ds:
        for v_name, v_over in selected_vars.items():
            # Check which seeds are already done
            done_seeds = set()
            for seed in args.seeds:
                if (OUTPUT_DIR / f"{ds_name}__{v_name}__seed{seed}.json").exists():
                    # Load existing result
                    try:
                        with open(OUTPUT_DIR / f"{ds_name}__{v_name}__seed{seed}.json") as ef:
                            rows.append(json.load(ef))
                        done_seeds.add(seed)
                    except:
                        pass
            remaining_seeds = [s for s in args.seeds if s not in done_seeds]
            if not remaining_seeds:
                print(f"  {ds_name}  {v_name}: all {len(args.seeds)} seeds done, skipping")
                continue
            print(f"  {ds_name}  {v_name}: {len(remaining_seeds)}/{len(args.seeds)} seeds to run")
            for seed in remaining_seeds:
                i += 1
                print(f"[{i}/{total}] {ds_name}  {v_name}  seed={seed}", flush=True)
                res = run_one(ds_name, v_name, v_over, seed, args.gpu)
                if res.get("error"):
                    print(f"  !! ERROR: {str(res['error']).splitlines()[-1]}", flush=True)
                else:
                    print(
                        f"  ACC={res['acc']:.4f}  NMI={res['nmi']:.4f}  ARI={res['ari']:.4f}  "
                        f"K={res['n_clusters']}  time={res['elapsed']:.1f}s",
                        flush=True,
                    )
                rows.append(res)
                with open(OUTPUT_DIR / f"{ds_name}__{v_name}__seed{seed}.json", "w") as f:
                    json.dump(res, f, indent=2, default=str)

    csv_path = OUTPUT_DIR / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "seed", "n_clusters", "acc", "nmi", "ari",
                    "f1", "elapsed", "beta_mutual", "beta_snn", "beta_perturb",
                    "beta_uncertainty", "error"])
        for r in rows:
            beta = r.get("beta") or {}
            w.writerow([
                r["dataset"], r["variant"], r.get("seed"),
                r.get("n_clusters"), r.get("acc"), r.get("nmi"), r.get("ari"),
                r.get("f1"), r.get("elapsed"),
                beta.get("beta_mutual"), beta.get("beta_snn"),
                beta.get("beta_perturb"), beta.get("beta_uncertainty"),
                r.get("error"),
            ])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
