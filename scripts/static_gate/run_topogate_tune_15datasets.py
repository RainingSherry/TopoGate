#!/usr/bin/env python
"""TopoGate 15-dataset hyperparameter tuning (Phase 0 of ablation plan).

Search grid:
  epochs      ∈ {40, 80, 150}
  mask_ratio  ∈ {0.3, 0.4, 0.5}
  neighbor_k  ∈ {5, 10, 20}
  hidden_size = 128 (fixed)
  seed        = 42  (fixed)

Total: 3 × 3 × 3 = 27 configs × 15 datasets = 405 runs.

Outputs:
  result/tune_15datasets/<dataset>/<dataset>__ep<ep>_mr<mr>_k<k>.json
  result/tune_15datasets/grid.csv             (all 405 rows)
  result/tune_15datasets/best_per_dataset.csv (15 rows, best by ACC)
  result/tune_15datasets/transfer_analysis.md (vs 131-dataset tune)

Usage:
  # Cluster dispatch (each worker gets a tiny subset of (dataset,grid) pairs):
  python scripts/run_topogate_tune_15datasets.py --gpu_ids 4 5 7 --worker_id 0 &
  python scripts/run_topogate_tune_15datasets.py --gpu_ids 4 5 7 --worker_id 1 &
  python scripts/run_topogate_tune_15datasets.py --gpu_ids 4 5 7 --worker_id 2 &
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
REPO_ROOT = SCRIPT_DIR.parent
CLUBENCH_ROOT = REPO_ROOT / "baseline" / "CLUBench"
for p in (str(REPO_ROOT), str(CLUBENCH_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── 15 datasets (from scripts/run_baseline_comparison.py) ──────────────
DATASETS = [
    "Mouse_retina",
    "sms_spam_collection",
    "enron",
    "ISOLET",
    "har",
    "spambase",
    "breast_cancer_wisconsin_original",
    "reuters",
    "cnae9",
    "Campbell",
    "mammographic_mass",
    "first-order-theorem-proving",
    "hrvatin_filtered",
    "Quake_Smart-seq2_Lung",
    "iris",
]

# ── Search grid ────────────────────────────────────────────────────────
GRID_EPOCHS = [40, 80, 150]
GRID_MASK_RATIO = [0.3, 0.4, 0.5]
GRID_NEIGHBOR_K = [5, 10, 20]
FIXED_HIDDEN_SIZE = 128
FIXED_SEED = 42
FIXED_BATCH_SIZE = 256
FIXED_LR = 1e-3

# Datasets that need PCA preprocessing (copy from run_baseline_comparison.py)
PCA_DIM_OVERRIDE = {
    "hrvatin_filtered": 500,
    "Campbell": 500,  # 26774 features — verify if needed
}

# Big-text datasets that CLIP-style embeddings render GPU-cheap
# (just whitelisted for any future use; currently unused)


def build_jobs():
    """All 405 (dataset, epochs, mask_ratio, neighbor_k) tuples."""
    jobs = []
    for ds in DATASETS:
        for ep in GRID_EPOCHS:
            for mr in GRID_MASK_RATIO:
                for k in GRID_NEIGHBOR_K:
                    jobs.append((ds, ep, mr, k))
    return jobs


def parse_args():
    p = argparse.ArgumentParser(description="TopoGate 15-dataset tuning (Phase 0)")
    p.add_argument("--gpu_ids", type=int, nargs="+", default=[4, 5, 7])
    p.add_argument("--worker_id", type=int, required=True)
    p.add_argument("--result_dir",
                   default="/home/luolie/ToPoGate/result/tune_15datasets")
    p.add_argument("--csv_path",
                   default="/home/luolie/ToPoGate/result/tune_15datasets/grid.csv")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if JSON already exists")
    return p.parse_args()


def run_one(dataset: str, epochs: int, mask_ratio: float, neighbor_k: int, gpu_id: int):
    """One TopoGate run → returns dict row."""
    from sklearn.decomposition import PCA
    from CLUBench import TopoGate
    from CLUBench.tools import load_data as _clubench_load_data, clustering_evaluation
    import numpy as np

    name = dataset
    # Some datasets (hrvatin_filtered, Quake_Smart-seq2_Lung) are added later
    # and are NOT in CLUBench.configs.DATASETS.  Read the .npz directly,
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
        pca = PCA(n_components=pca_dim, random_state=FIXED_SEED)
        X = pca.fit_transform(X)

    # Instantiate TopoGate with arbitrary kw — accepts __init__ override
    model = TopoGate(
        n_clusters=K,
        epochs=epochs,
        batch_size=FIXED_BATCH_SIZE,
        lr=FIXED_LR,
        hidden_size=FIXED_HIDDEN_SIZE,
        gpu=gpu_id,
        device="cuda",
        variant_name="topogate_full",
        seed=FIXED_SEED,
        neighbor_k=neighbor_k,
        mask_ratio=mask_ratio,
    )

    t0 = time.time()
    labels = model.fit_predict(X)
    runtime = time.time() - t0
    metrics = clustering_evaluation(Y, labels)

    return {
        "dataset": name,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_clusters": int(K),
        "epochs": int(epochs),
        "mask_ratio": float(mask_ratio),
        "neighbor_k": int(neighbor_k),
        "hidden_size": FIXED_HIDDEN_SIZE,
        "seed": FIXED_SEED,
        "gpu": gpu_id,
        "variant": "topogate_full",
        "runtime_seconds": float(runtime),
        **metrics,
    }


def main():
    args = parse_args()
    if args.worker_id < 0 or args.worker_id >= len(args.gpu_ids):
        raise ValueError(f"worker_id {args.worker_id} out of range for gpu_ids {args.gpu_ids}")
    gpu = args.gpu_ids[args.worker_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    all_jobs = build_jobs()
    # Round-robin shard by worker_id
    my_jobs = [j for i, j in enumerate(all_jobs) if i % len(args.gpu_ids) == args.worker_id]

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"[worker {args.worker_id} on GPU {gpu}] jobs={len(my_jobs)} / {len(all_jobs)}")
    print(f"  grid: epochs={GRID_EPOCHS}  mask_ratio={GRID_MASK_RATIO}  neighbor_k={GRID_NEIGHBOR_K}")

    rows = []
    n_ok, n_skip, n_fail = 0, 0, 0

    for i, (ds, ep, mr, k) in enumerate(my_jobs, 1):
        tag = f"ep{ep}_mr{mr}_k{k}"
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

        print(f"[{i:3d}/{len(my_jobs)}] {ds} {tag}  ...", end=" ", flush=True)
        try:
            result = run_one(ds, ep, mr, k, gpu)
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
                    "dataset": ds, "epochs": ep, "mask_ratio": mr, "neighbor_k": k,
                    "error": str(e), "traceback": traceback.format_exc(),
                }, f, indent=2)
            print(f"FAIL: {e}")
            n_fail += 1

    # Append to grid.csv (each worker appends)
    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "variant", "seed", "gpu",
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
