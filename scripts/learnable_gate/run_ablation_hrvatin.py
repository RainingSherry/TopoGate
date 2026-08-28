#!/usr/bin/env python3
"""Run hrvatin_filtered missing variants with subsample_size=5000."""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, '/home/luolie/ToPoGate/baseline/CLUBench')
sys.path.insert(0, '/home/luolie/ToPoGate')
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import numpy as np
from sklearn.decomposition import PCA
from CLUBench import TopoGate
from CLUBench.tools import clustering_evaluation

DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
ABL_DIR = Path("/home/luolie/ToPoGate/result/ablation")

EPOCHS = 30
MASK_RATIO = 0.3
NEIGHBOR_K = 10
HIDDEN_SIZE = 128
BATCH_SIZE = 256
LR = 1e-3
SEED = 42
SUBSAMPLE_SIZE = 5000

ds = 'hrvatin_filtered'
MISSING_VARIANTS = [
    "static_gate_far_neighbors",
    "static_gate_edge_only",
    "static_gate_gate_only",
    "static_gate_no_topology_features",
]

data = np.load(DATA_DIR / f"{ds}.npz")
X_full, Y = data["x"], data["y"]
n_samples, n_features = X_full.shape
K = len(set(Y))
pca_dim = 500
pca = PCA(n_components=pca_dim, random_state=SEED)
X = pca.fit_transform(X_full)
print(f"{ds}: N={n_samples} d={n_features} K={K} (after PCA: {X.shape})", flush=True)

new_rows = []
for variant in MISSING_VARIANTS:
    tag = f"{variant}__ep{EPOCHS}_mr{MASK_RATIO}_k{NEIGHBOR_K}"
    out_path = ABL_DIR / ds / f"{ds}__{tag}.json"
    if out_path.exists():
        print(f"[SKIP] {variant} exists", flush=True)
        continue
    print(f"[RUN] {variant} ...", end=" ", flush=True)
    t0 = time.time()
    model = TopoGate(
        n_clusters=K,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        hidden_size=HIDDEN_SIZE,
        gpu=0,
        device="cuda",
        variant_name=variant,
        seed=SEED,
        neighbor_k=NEIGHBOR_K,
        mask_ratio=MASK_RATIO,
        mix_neighbors=4,
        subsample_size=SUBSAMPLE_SIZE,
    )
    labels = model.fit_predict(X)
    runtime = time.time() - t0
    metrics = clustering_evaluation(Y, labels)
    row = {
        "dataset": ds,
        "variant": variant,
        "layer": "ext",
        "n_samples": int(n_samples),
        "n_features": int(pca_dim),
        "n_clusters": int(K),
        "seed": SEED,
        "gpu": 4,
        "epochs": EPOCHS,
        "mask_ratio": float(MASK_RATIO),
        "neighbor_k": int(NEIGHBOR_K),
        "hidden_size": HIDDEN_SIZE,
        "runtime_seconds": float(runtime),
        "subsample_size": SUBSAMPLE_SIZE,
        **metrics,
    }
    with open(out_path, "w") as f:
        json.dump(row, f, indent=2)
    new_rows.append(row)
    print(f"ACC={metrics['acc']:.4f} NMI={metrics['nmi']:.4f} ARI={metrics['ari']:.4f} ({runtime:.1f}s)", flush=True)

# Append to CSV
import csv
CSV_PATH = ABL_DIR / "merged_summary.csv"
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerows(new_rows)
print(f"\nAppended {len(new_rows)} rows")