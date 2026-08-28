#!/usr/bin/env python
"""TopoGate ablation experiment runner (Phase 1 + Phase 2 of ablation plan).

Runs 8 ablation variants × 15 datasets, with optional stratified execution:
  - core layer (5 datasets): all 8 variants
  - extended layer (10 datasets): 4 key variants

All variants share the same hyperparams (epochs / mask_ratio / neighbor_k)
discovered from Phase 0 tuning. Only the variant-specific knobs change.

Outputs:
  result/ablation/<dataset>/<dataset>__<variant>__ep<ep>_mr<mr>_k<k>.json
  result/ablation/core/summary.csv
  result/ablation/ext/summary.csv
  result/ablation/merged_summary.csv

Usage:
  python scripts/run_topogate_ablation.py --gpu_ids 4 5 7 --worker_id 0 &
  python scripts/run_topogate_ablation.py --gpu_ids 4 5 7 --worker_id 1 &
  python scripts/run_topogate_ablation.py --gpu_ids 4 5 7 --worker_id 2 &
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
# scripts/v1/ → repo root requires going up 2 levels (scripts/v1/ → scripts/ → repo root).
REPO_ROOT = SCRIPT_DIR.parent.parent
CLUBENCH_ROOT = REPO_ROOT / "baseline" / "CLUBench"
for p in (str(REPO_ROOT), str(CLUBENCH_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── 15 datasets, stratified ────────────────────────────────────────────
CORE_DATASETS = [
    "Mouse_retina",
    "sms_spam_collection",
    "enron",
    "har",
    "breast_cancer_wisconsin_original",
]

EXT_DATASETS = [
    "reuters",
    "ISOLET",
    "spambase",
    "cnae9",
    "Campbell",
    "hrvatin_filtered",
    "Quake_Smart-seq2_Lung",
    "mammographic_mass",
    "first-order-theorem-proving",
    "iris",
]

ALL_DATASETS = CORE_DATASETS + EXT_DATASETS

# ── 8 ablation variants ─────────────────────────────────────────────────
# static_gate variants.  These map to method/TopoGate/static_gate/configs/static_gate_*.yaml.
ALL_VARIANTS = [
    "static_gate_full",
    "static_gate_nomix",
    "static_gate_random_neighbors",
    "static_gate_far_neighbors",
    "static_gate_constant_gate",
    "static_gate_gate_only",
    "static_gate_edge_only",
    "static_gate_no_topology_features",
]

# 4 key variants for the extended layer (saves 50% of runs)
KEY_VARIANTS = [
    "static_gate_full",
    "static_gate_nomix",
    "static_gate_random_neighbors",
    "static_gate_constant_gate",
]

# Datasets that need PCA preprocessing to avoid GPU OOM
PCA_DIM_OVERRIDE = {
    "hrvatin_filtered": 500,
    "Campbell": 500,
}

# ── Default hyperparameters (overridden by tuning results via CLI args) ─
# These are the 13-dataset dominant hyperparameters from Phase 0 tuning
# (the 2 outlier datasets hrvatin_filtered + Quake were not tuned due to
# OOM / openblas issues, see CHANGELOG_errors.md).
DEFAULT_HPARAMS = dict(
    epochs=150,        # 5/13 votes (vs 40 in 131-dataset Round 2)
    mask_ratio=0.3,    # 6/13 votes (vs 0.3 in 131-dataset, tied with 0.4)
    neighbor_k=5,      # 8/13 votes (vs 10 in 131-dataset)
    hidden_size=128,
    batch_size=256,
    lr=1e-3,
    seed=42,
)


def build_jobs(datasets, variants, **_kwargs):
    """All (dataset, variant) tuples. Hyperparams ignored for job enumeration."""
    return [(ds, var) for ds in datasets for var in variants]


def parse_args():
    p = argparse.ArgumentParser(description="TopoGate ablation experiment")
    p.add_argument("--gpu_ids", type=int, nargs="+", default=[4, 5, 7])
    p.add_argument("--worker_id", type=int, required=True)
    p.add_argument("--layer", choices=["core", "ext", "all"], default="all")
    p.add_argument("--epochs", type=int, default=DEFAULT_HPARAMS["epochs"])
    p.add_argument("--mask_ratio", type=float, default=DEFAULT_HPARAMS["mask_ratio"])
    p.add_argument("--neighbor_k", type=int, default=DEFAULT_HPARAMS["neighbor_k"])
    p.add_argument("--hidden_size", type=int, default=DEFAULT_HPARAMS["hidden_size"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_HPARAMS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULT_HPARAMS["lr"])
    p.add_argument("--seed", type=int, default=DEFAULT_HPARAMS["seed"])
    p.add_argument("--result_dir",
                   default="/home/luolie/ToPoGate/result/ablation")
    p.add_argument("--csv_path",
                   default="/home/luolie/ToPoGate/result/ablation/merged_summary.csv")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def run_one(dataset: str, variant: str, hyperparams: dict, gpu_id: int):
    """One TopoGate ablation run → returns dict row."""
    from sklearn.decomposition import PCA
    from CLUBench import TopoGate
    from CLUBench.tools import load_data as _clubench_load_data, clustering_evaluation
    import numpy as np

    name = dataset
    # Some datasets (hrvatin_filtered, Quake_Smart-seq2_Lung) are added later
    # and are NOT in CLUBench.configs.DATASETS.  We read the .npz directly,
    # matching scripts/run_baseline_comparison.py's approach.
    DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
    npz_path = DATA_DIR / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} does not exist")
    data = np.load(npz_path)
    X = data["x"]
    Y = data["y"]
    X = np.asarray(X)
    Y = np.asarray(Y)
    K = len(set(Y))
    n_samples, n_features = X.shape

    # Optional PCA preprocessing
    pca_dim = PCA_DIM_OVERRIDE.get(name)
    if pca_dim is not None and X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=hyperparams["seed"])
        X = pca.fit_transform(X)

    model = TopoGate(
        n_clusters=K,
        epochs=hyperparams["epochs"],
        batch_size=hyperparams["batch_size"],
        lr=hyperparams["lr"],
        hidden_size=hyperparams["hidden_size"],
        gpu=gpu_id,
        device="cuda",
        variant_name=variant,
        seed=hyperparams["seed"],
        neighbor_k=hyperparams["neighbor_k"],
        mask_ratio=hyperparams["mask_ratio"],
    )

    t0 = time.time()
    labels = model.fit_predict(X)
    runtime = time.time() - t0
    metrics = clustering_evaluation(Y, labels)

    return {
        "dataset": name,
        "variant": variant,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_clusters": int(K),
        "epochs": int(hyperparams["epochs"]),
        "mask_ratio": float(hyperparams["mask_ratio"]),
        "neighbor_k": int(hyperparams["neighbor_k"]),
        "hidden_size": int(hyperparams["hidden_size"]),
        "seed": int(hyperparams["seed"]),
        "gpu": gpu_id,
        "layer": "core" if name in CORE_DATASETS else "ext",
        "runtime_seconds": float(runtime),
        **metrics,
    }


def main():
    args = parse_args()
    if args.worker_id < 0 or args.worker_id >= len(args.gpu_ids):
        raise ValueError(f"worker_id {args.worker_id} out of range")

    hyperparams = dict(
        epochs=args.epochs,
        mask_ratio=args.mask_ratio,
        neighbor_k=args.neighbor_k,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    # Select datasets and variants based on layer
    if args.layer == "core":
        datasets = CORE_DATASETS
        variants = ALL_VARIANTS
    elif args.layer == "ext":
        datasets = EXT_DATASETS
        variants = KEY_VARIANTS
    else:  # "all"
        datasets = ALL_DATASETS
        # Core layer uses all 8, ext layer uses 4 key
        # Build jobs manually so we can do the split
        # Easier: build a list of (dataset, variant) and shard over workers
        all_jobs = []
        for ds in CORE_DATASETS:
            for var in ALL_VARIANTS:
                all_jobs.append((ds, var))
        for ds in EXT_DATASETS:
            for var in KEY_VARIANTS:
                all_jobs.append((ds, var))
        my_jobs = [j for i, j in enumerate(all_jobs) if i % len(args.gpu_ids) == args.worker_id]
        gpu = args.gpu_ids[args.worker_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return _run_jobs(my_jobs, hyperparams, gpu, args)

    gpu = args.gpu_ids[args.worker_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    all_jobs = build_jobs(datasets, variants, **hyperparams)
    my_jobs = [j for i, j in enumerate(all_jobs) if i % len(args.gpu_ids) == args.worker_id]
    return _run_jobs(my_jobs, hyperparams, gpu, args)


def _run_jobs(my_jobs, hyperparams, gpu, args):
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"[worker {args.worker_id} on GPU {gpu}] jobs={len(my_jobs)}  hparams={hyperparams}")

    rows = []
    n_ok, n_skip, n_fail = 0, 0, 0

    for i, (ds, var) in enumerate(my_jobs, 1):
        ep = hyperparams["epochs"]
        mr = hyperparams["mask_ratio"]
        k = hyperparams["neighbor_k"]
        tag = f"{var}__ep{ep}_mr{mr}_k{k}"
        out_path = result_dir / ds / f"{ds}__{tag}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.force:
            n_skip += 1
            try:
                with open(out_path) as f:
                    rows.append(json.load(f))
            except Exception:
                pass
            continue

        print(f"[{i:3d}/{len(my_jobs)}] {ds} {var}  ...", end=" ", flush=True)
        try:
            result = run_one(ds, var, hyperparams, gpu)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            rows.append(result)
            print(f"ACC={result['acc']:.4f} NMI={result['nmi']:.4f} "
                  f"ARI={result['ari']:.4f} ({result['runtime_seconds']:.1f}s)")
            n_ok += 1
        except Exception as e:
            err_path = result_dir / ds / f"{ds}__{tag}.error.json"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, "w") as f:
                json.dump({
                    "dataset": ds, "variant": var, **hyperparams,
                    "error": str(e), "traceback": traceback.format_exc(),
                }, f, indent=2)
            print(f"FAIL: {e}")
            n_fail += 1

    # Append to merged_summary.csv
    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "layer", "variant", "seed", "gpu",
        "n_samples", "n_features", "n_clusters",
        "epochs", "mask_ratio", "neighbor_k", "hidden_size",
        "acc", "nmi", "ari", "f1_macro", "runtime_seconds",
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print()
    print(f"[worker {args.worker_id}] done  ok={n_ok}  skip={n_skip}  fail={n_fail}")
    if rows:
        accs = [r["acc"] for r in rows]
        print(f"[worker {args.worker_id}] mean ACC={sum(accs)/len(accs):.4f}")


if __name__ == "__main__":
    main()
