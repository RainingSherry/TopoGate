#!/usr/bin/env python3
"""Multi-seed verification for v10 = v9_adaptive + nomix_init (all betas = -5.0).

Runs v10_nomix_init on 14 datasets × 3 seeds = 42 runs, with v9_adaptive
as the control arm for comparison.

Key hypothesis: starting from gate ≈ 0 (≈ NoMix) lets the optimizer decide
whether to introduce topology. On datasets where topology hurts (v9 hurts),
betas should go more negative → gate → 0 (recover NoMix).  On datasets where
topology helps (v9 helps), betas should go positive → gate > 0.

Outputs:
  result/v10_learnable_gate/multiseed/<dataset>__v10_nomix_init__seed{N}.json
  result/v10_learnable_gate/multiseed/comparison.csv
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

# 14 datasets: same as v9 but excluding hrvatin_filtered (no npz found in v9 runs)
DATASETS = [
    "enron", "har", "Campbell", "Mouse_retina", "cnae9",
    "reuters", "breast_cancer_wisconsin_original",
    "iris", "mammographic_mass", "ISOLET", "spambase",
    "sms_spam_collection", "first-order-theorem-proving",
]

VARIANTS = {
    # v10: same as v9_adaptive but with learned_gate_init_mode=nomix
    # → all betas initialised to -5.0 → gate ≈ 0 (≈ NoMix at start)
    "v10_nomix_init": {
        "variant": "learnable_gate_v10_nomix_init",
        "epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5,
        "warmup_epochs": 20, "ramp_epochs": 10,
        "n_top_features": 0,
        "knn_pca_mode": "adaptive",
        "knn_pca_dim": 2000,
        "mix_mode": "reliability",
        "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
    },
    # Control: v9_adaptive (zero init) for direct comparison
    "v9_adaptive": {
        "variant": "learnable_gate_v9_adaptive",
        "epochs": 80, "mask_ratio": 0.3, "neighbor_k": 5,
        "warmup_epochs": 20, "ramp_epochs": 10,
        "n_top_features": 0,
        "knn_pca_mode": "adaptive",
        "knn_pca_dim": 2000,
        "mix_mode": "reliability",
        "config_dir": str(LEARNABLE_GATE_ROOT / "configs"),
    },
}

DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
OUTPUT_DIR = Path("/home/luolie/ToPoGate/result/v10_learnable_gate/multiseed")
DEFAULT_SEEDS = [42, 123, 7]


def run_one(dataset: str, variant_name: str, overrides: dict, seed: int, gpu: int) -> dict:
    npz = DATA_DIR / f"{dataset}.npz"
    if not npz.exists():
        return {"dataset": dataset, "variant": variant_name, "seed": seed,
                "error": f"{npz} not found"}
    data = np.load(npz)
    X = data["X"] if "X" in data.files else data["x"]
    y = data["y"] if "y" in data.files else None
    n_clusters = int(np.unique(y).size) if y is not None else None

    out_dir = OUTPUT_DIR / f"{dataset}__{variant_name}__seed{seed}"
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
            "dataset": dataset, "variant": variant_name, "seed": seed,
            "n_clusters": n_clusters, "acc": metrics.get("acc"),
            "nmi": metrics.get("nmi"), "ari": metrics.get("ari"),
            "elapsed": float(elapsed), "beta": beta, "error": None,
        }
    except Exception as exc:
        return {
            "dataset": dataset, "variant": variant_name, "seed": seed,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_ds = [d for d in DATASETS if (not args.datasets or d in set(args.datasets))]
    selected_var = [v for v in VARIANTS if (not args.variants or v in set(args.variants))]
    seeds = args.seeds

    rows = []
    total = len(selected_ds) * len(selected_var) * len(seeds)
    print(f"Will run {total} jobs ({len(selected_ds)} ds × {len(selected_var)} variants × {len(seeds)} seeds)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Seeds: {seeds}")

    i = 0
    for ds_name in selected_ds:
        for v_name in selected_var:
            for seed in seeds:
                i += 1
                t_start = time.time()
                print(f"[{i}/{total}] {ds_name}  {v_name}  seed={seed}", flush=True)
                res = run_one(ds_name, v_name, VARIANTS[v_name], seed, args.gpu)
                if res.get("error"):
                    print(f"  !! ERROR: {str(res['error']).splitlines()[-1]}", flush=True)
                else:
                    beta = res.get("beta") or {}
                    print(
                        f"  ACC={res['acc']:.4f}  NMI={res['nmi']:.4f}  ARI={res['ari']:.4f}  "
                        f"K={res['n_clusters']}  "
                        f"bM={beta.get('beta_mutual', 0):+.3f}  "
                        f"bS={beta.get('beta_snn', 0):+.3f}  "
                        f"bP={beta.get('beta_perturb', 0):+.3f}  "
                        f"time={res['elapsed']:.1f}s  ({time.time()-t_start:.1f}s wall)",
                        flush=True,
                    )
                rows.append(res)
                with open(OUTPUT_DIR / f"{ds_name}__{v_name}__seed{seed}.json", "w") as f:
                    json.dump(res, f, indent=2, default=str)

    csv_path = OUTPUT_DIR / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "seed", "n_clusters", "acc", "nmi", "ari",
                     "elapsed", "beta_mutual", "beta_snn", "beta_perturb",
                     "beta_uncertainty", "effective_gate_max", "error"])
        for r in rows:
            beta = r.get("beta") or {}
            w.writerow([
                r["dataset"], r["variant"], r.get("seed"),
                r.get("n_clusters"), r.get("acc"), r.get("nmi"), r.get("ari"),
                r.get("elapsed"),
                beta.get("beta_mutual"), beta.get("beta_snn"),
                beta.get("beta_perturb"), beta.get("beta_uncertainty"),
                beta.get("effective_gate_max"),
                r.get("error"),
            ])
    print(f"\nWrote {csv_path}")
    print(f"Total runs: {len(rows)}  Errors: {sum(1 for r in rows if r.get('error'))}")


if __name__ == "__main__":
    main()
