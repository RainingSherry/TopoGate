#!/usr/bin/env python3
"""Extended-layer multi-seed comparison: StaticGate vs LearnableGate on ext 10 datasets.

Runs 10 datasets × 2 variants × 3 seeds = 60 runs.
Usage (parallel across 3 GPUs):
  nohup python3 scripts/learnable_gate/run_ext_multiseed.py --gpu_ids 4 5 7 --worker_id 0 > /tmp/ext_ms_w0.log 2>&1 &
  nohup python3 scripts/learnable_gate/run_ext_multiseed.py --gpu_ids 4 5 7 --worker_id 1 > /tmp/ext_ms_w1.log 2>&1 &
  nohup python3 scripts/learnable_gate/run_ext_multiseed.py --gpu_ids 4 5 7 --worker_id 2 > /tmp/ext_ms_w2.log 2>&1 &
"""
from __future__ import annotations

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

STATIC_ROOT = REPO_ROOT / "methods" / "TopoGate" / "static_gate"
LEARNABLE_ROOT = REPO_ROOT / "methods" / "TopoGate" / "learnable_gate"

EXT_DATASETS = [
    "reuters", "ISOLET", "spambase", "cnae9", "Campbell",
    "hrvatin_filtered", "Quake_Smart-seq2_Lung", "mammographic_mass",
    "first-order-theorem-proving", "iris",
]

VARIANTS = {
    "static_gate_full": {
        "variant": "static_gate_full",
        "epochs": 150, "mask_ratio": 0.3, "neighbor_k": 5,
        "config_dir": str(STATIC_ROOT / "configs"),
    },
    "learnable_gate_sched": {
        "variant": "learnable_gate_sched",
        "epochs": 150, "mask_ratio": 0.3, "neighbor_k": 5,
        "warmup_epochs": 20, "ramp_epochs": 10,
        "config_dir": str(LEARNABLE_ROOT / "configs"),
    },
}

PCA_DIM = {"hrvatin_filtered": 500, "Campbell": 500}
SUBSAMPLE_SIZE = {"hrvatin_filtered": 5000}

DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
OUTPUT_DIR = Path("/home/luolie/ToPoGate/result/learnable_gate_smoke/multiseed")


def run_one(dataset: str, variant_name: str, overrides: dict, seed: int, gpu: int) -> dict:
    npz = DATA_DIR / f"{dataset}.npz"
    if not npz.exists():
        return {"dataset": dataset, "variant": variant_name, "seed": seed,
                "error": f"{npz} not found"}

    import numpy as np
    data = np.load(npz)
    X = data["X"] if "X" in data.files else data["x"]
    y = data["y"] if "y" in data.files else None
    n_clusters = int(np.unique(y).size) if y is not None else None

    pca_dim = PCA_DIM.get(dataset)
    if pca_dim and X.shape[1] > pca_dim:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_dim, random_state=seed)
        X = pca.fit_transform(X)

    subsample = SUBSAMPLE_SIZE.get(dataset, 0)
    if subsample and X.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=subsample, replace=False)
        X, y = X[idx], (y[idx] if y is not None else None)

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
            "n_clusters": n_clusters,
            "acc": metrics.get("acc"), "nmi": metrics.get("nmi"),
            "ari": metrics.get("ari"),
            "elapsed": float(elapsed), "beta": beta, "error": None,
        }
    except Exception as exc:
        return {
            "dataset": dataset, "variant": variant_name, "seed": seed,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[4, 5, 7])
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 7])
    args = parser.parse_args()

    gpu = args.gpu_ids[args.worker_id % len(args.gpu_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    selected_var = [v for v in VARIANTS if (not args.variants or v in set(args.variants))]
    seeds = [str(s) for s in args.seeds]

    all_jobs = [(ds, v, s) for ds in EXT_DATASETS for v in selected_var for s in seeds]
    my_jobs = [j for i, j in enumerate(all_jobs) if i % len(args.gpu_ids) == args.worker_id]

    print(f"[worker {args.worker_id} on GPU {gpu}] jobs={len(my_jobs)}  seeds={seeds}  variants={selected_var}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(my_jobs)
    for i, (ds, v_name, seed) in enumerate(my_jobs, 1):
        print(f"[{i:3d}/{total}] {ds}  {v_name}  seed={seed}", flush=True)
        t_start = time.time()
        res = run_one(ds, v_name, VARIANTS[v_name], int(seed), gpu)
        elapsed_wall = time.time() - t_start
        if res.get("error"):
            print(f"  !! ERROR: {str(res['error']).splitlines()[-1]}", flush=True)
        else:
            print(f"  ARI={res['ari']:.4f}  K={res['n_clusters']}  time={res['elapsed']:.1f}s ({elapsed_wall:.1f}s wall)", flush=True)
        rows.append(res)

        out_name = f"{ds}__{v_name}__seed{seed}.json"
        with open(OUTPUT_DIR / out_name, "w") as f:
            json.dump(res, f, indent=2, default=str)

    csv_path = OUTPUT_DIR / "comparison_ext.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "seed", "n_clusters", "acc", "nmi", "ari",
                     "elapsed", "beta_mutual", "beta_snn", "beta_perturb", "beta_uncertainty", "error"])
        for r in rows:
            beta = r.get("beta") or {}
            w.writerow([
                r["dataset"], r["variant"], r.get("seed"), r.get("n_clusters"),
                r.get("acc"), r.get("nmi"), r.get("ari"), r.get("elapsed"),
                beta.get("beta_mutual"), beta.get("beta_snn"),
                beta.get("beta_perturb"), beta.get("beta_uncertainty"),
                r.get("error"),
            ])
    print(f"\n[worker {args.worker_id}] done  rows={len(rows)}  errors={sum(1 for r in rows if r.get('error'))}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
