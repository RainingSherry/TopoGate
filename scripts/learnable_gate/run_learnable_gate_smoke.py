#!/usr/bin/env python3
"""Smoke test for LearnableGate (the current mainline).

Runs a 3-way comparison on each dataset:
  - static_gate_full           -- StaticGate (4 β fixed as argparse defaults)
  - learnable_gate_schedule0   -- LearnableGate (β frozen at init, gate=0 throughout)
  - learnable_gate_sched       -- LearnableGate (warmup=20, ramp=10, β learns)

Outputs:
  result/learnable_gate_smoke/<dataset>__<variant>.json
  result/learnable_gate_smoke/comparison.csv
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

from methods.TopoGate.learnable_gate.run_npz import run_topogate

STATIC_GATE_ROOT = REPO_ROOT / "methods" / "TopoGate" / "static_gate"
LEARNABLE_GATE_ROOT = REPO_ROOT / "methods" / "TopoGate" / "learnable_gate"

DATASETS = [
    "Mouse_retina",
    "enron",
    "sms_spam_collection",
    "har",
    "breast_cancer_wisconsin_original",
]

VARIANTS = [
    ("static_gate_full", {
        "variant": "static_gate_full",
        "epochs": 150, "mask_ratio": 0.3, "neighbor_k": 5,
        "config_dir": str(STATIC_GATE_ROOT / "configs"),
    }),
    ("learnable_gate_schedule0", {
        "variant": "learnable_gate_sched",
        "epochs": 150, "mask_ratio": 0.3, "neighbor_k": 5,
        "warmup_epochs": 1000, "ramp_epochs": 1,
        "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
    }),
    ("learnable_gate_sched", {
        "variant": "learnable_gate_sched",
        "epochs": 150, "mask_ratio": 0.3, "neighbor_k": 5,
        "warmup_epochs": 20, "ramp_epochs": 10,
        "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
    }),
]

DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
OUTPUT_DIR = Path("/home/luolie/ToPoGate/result/learnable_gate_smoke")


def run_one(dataset: str, variant_name: str, overrides: dict, seed: int, gpu: int) -> dict:
    npz = DATA_DIR / f"{dataset}.npz"
    if not npz.exists():
        return {"dataset": dataset, "variant": variant_name, "error": f"{npz} not found"}
    data = __import__("numpy").load(npz)
    X = data["X"] if "X" in data.files else data["x"]
    y = data["y"] if "y" in data.files else None
    # Always auto-detect K from labels — never hardcode
    import numpy as np
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
                beta = json.load(f).get("learned_gate_final_beta")
        return {
            "dataset": dataset,
            "variant": variant_name,
            "seed": seed,
            "n_clusters": n_clusters,
            "acc": metrics.get("acc"),
            "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
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
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of dataset names; default = all 5")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42],
                        help="Seeds to run; default = [42]")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_ds = [d for d in DATASETS if (not args.datasets or d in set(args.datasets))]

    rows = []
    total = len(selected_ds) * len(VARIANTS) * len(args.seeds)
    print(f"Will run {total} jobs ({len(selected_ds)} ds × {len(VARIANTS)} variants × {len(args.seeds)} seeds)")
    print(f"Output: {OUTPUT_DIR}")
    print()

    i = 0
    for ds_name in selected_ds:
        for v_name, v_over in VARIANTS:
            for seed in args.seeds:
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
                with open(OUTPUT_DIR / f"{ds_name}__{v_name}.json", "w") as f:
                    json.dump(res, f, indent=2, default=str)

    csv_path = OUTPUT_DIR / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "seed", "n_clusters", "acc", "nmi", "ari",
                     "elapsed", "beta_mutual", "beta_snn", "beta_perturb", "beta_uncertainty", "error"])
        for r in rows:
            beta = r.get("beta") or {}
            w.writerow([
                r["dataset"], r["variant"], r.get("seed"),
                r.get("n_clusters"), r.get("acc"), r.get("nmi"), r.get("ari"),
                r.get("elapsed"),
                beta.get("beta_mutual"), beta.get("beta_snn"),
                beta.get("beta_perturb"), beta.get("beta_uncertainty"),
                r.get("error"),
            ])
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
