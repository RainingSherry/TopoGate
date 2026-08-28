#!/usr/bin/env bash
# Phase 4-A: Complete ablation table 4 — fill missing variants for ext layer.
# 10 ext datasets × 4 missing variants = 40 runs.
# Variants: static_gate_far_neighbors, static_gate_edge_only,
#           static_gate_gate_only, static_gate_no_topology_features
set -u
cd /home/luolie/ToPoGate

GPU=4
EPOCHS=30
MISSING_VARIANTS=(
  "far_neighbors"
  "edge_only"
  "gate_only"
  "no_topology_features"
)
EXT_DATASETS=(
  "reuters:datasets/reuters.npz"
  "ISOLET:datasets/ISOLET.npz"
  "spambase:datasets/spambase.npz"
  "cnae9:datasets/cnae9.npz"
  "Campbell:datasets/Campbell.npz"
  "hrvatin_filtered:datasets/hrvatin_filtered.npz"
  "Quake_Smart-seq2_Lung:datasets/Quake_Smart-seq2_Lung.npz"
  "mammographic_mass:datasets/mammographic_mass.npz"
  "first-order-theorem-proving:datasets/first-order-theorem-proving.npz"
  "iris:datasets/iris.npz"
)

OUT_DIR=result/learnable_gate_smoke/ablation_ext_complete
mkdir -p "$OUT_DIR"

# Need to reuse the existing ablation structure: result/ablation/<dataset>/<ds>__static_gate_<variant>__ep<ep>_mr<mr>_k<k>.json
ABL_DIR=result/ablation
CSV_PATH="$ABL_DIR/merged_summary.csv"

LOG="$OUT_DIR/run_all.log"
echo "=== Ext layer: complete missing variants ===" > "$LOG"
date >> "$LOG"

# We bypass the run_topogate_ablation.py script (which restricts ext to 4 key variants)
# by directly calling TopoGate wrapper for each (ds, missing_variant) pair.
python3 - <<EOF
import sys, os, time, json, traceback
from pathlib import Path

# Setup
SCRIPT_DIR = Path("/home/luolie/ToPoGate/scripts/static_gate")
REPO_ROOT = SCRIPT_DIR.parent.parent
CLUBENCH_ROOT = REPO_ROOT / "baseline" / "CLUBench"
for p in (str(REPO_ROOT), str(CLUBENCH_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["CUDA_VISIBLE_DEVICES"] = "$GPU"
import numpy as np
from sklearn.decomposition import PCA
from CLUBench import TopoGate
from CLUBench.tools import clustering_evaluation
import csv

DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
ABL_DIR = Path("/home/luolie/ToPoGate/result/ablation")
CSV_PATH = ABL_DIR / "merged_summary.csv"

EPOCHS = $EPOCHS
MASK_RATIO = 0.3
NEIGHBOR_K = 5
HIDDEN_SIZE = 128
BATCH_SIZE = 256
LR = 1e-3
SEED = 42

# Datasets needing PCA preprocessing
PCA_DIM_OVERRIDE = {"hrvatin_filtered": 500, "Campbell": 500}

EXT_DATASETS = [
    "reuters", "ISOLET", "spambase", "cnae9", "Campbell",
    "hrvatin_filtered", "Quake_Smart-seq2_Lung", "mammographic_mass",
    "first-order-theorem-proving", "iris",
]
MISSING_VARIANTS = [
    "static_gate_far_neighbors",
    "static_gate_edge_only",
    "static_gate_gate_only",
    "static_gate_no_topology_features",
]

# Load existing CSV to determine already-known rows
existing_rows = []
if CSV_PATH.exists():
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
existing_keys = {(r["dataset"], r["variant"]) for r in existing_rows}

# Append new rows
new_rows = []
n_ok = n_skip = n_fail = 0
for ds in EXT_DATASETS:
    npz_path = DATA_DIR / f"{ds}.npz"
    if not npz_path.exists():
        print(f"[SKIP] {ds} — missing file", flush=True)
        continue
    for variant in MISSING_VARIANTS:
        tag = f"{variant}__ep{EPOCHS}_mr{MASK_RATIO}_k{NEIGHBOR_K}"
        out_path = ABL_DIR / ds / f"{ds}__{tag}.json"
        if out_path.exists():
            n_skip += 1
            print(f"[SKIP] {ds} {variant} (already on disk)", flush=True)
            continue
        print(f"[RUN] {ds} {variant}", end=" ", flush=True)
        try:
            data = np.load(npz_path)
            X = data["x"]
            Y = data["y"]
            K = len(set(Y))
            n_samples, n_features = X.shape
            pca_dim = PCA_DIM_OVERRIDE.get(ds)
            if pca_dim is not None and X.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim, random_state=SEED)
                X = pca.fit_transform(X)
            t0 = time.time()
            model = TopoGate(
                n_clusters=K,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                hidden_size=HIDDEN_SIZE,
                gpu=int("$GPU"),
                device="cuda",
                variant_name=variant,
                seed=SEED,
                neighbor_k=NEIGHBOR_K,
                mask_ratio=MASK_RATIO,
            )
            labels = model.fit_predict(X)
            runtime = time.time() - t0
            metrics = clustering_evaluation(Y, labels)
            row = {
                "dataset": ds,
                "variant": variant,
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "n_clusters": int(K),
                "epochs": EPOCHS,
                "mask_ratio": float(MASK_RATIO),
                "neighbor_k": int(NEIGHBOR_K),
                "hidden_size": HIDDEN_SIZE,
                "seed": SEED,
                "gpu": int("$GPU"),
                "layer": "ext",
                "runtime_seconds": float(runtime),
                **metrics,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(row, f, indent=2)
            new_rows.append(row)
            print(f"ACC={metrics['acc']:.4f} NMI={metrics['nmi']:.4f} ARI={metrics['ari']:.4f} ({runtime:.1f}s)", flush=True)
            n_ok += 1
        except Exception as e:
            err_path = ABL_DIR / ds / f"{ds}__{tag}.error.json"
            with open(err_path, "w") as f:
                json.dump({
                    "dataset": ds, "variant": variant,
                    "epochs": EPOCHS, "mask_ratio": MASK_RATIO,
                    "neighbor_k": NEIGHBOR_K,
                    "error": str(e), "traceback": traceback.format_exc(),
                }, f, indent=2)
            print(f"FAIL: {e}", flush=True)
            n_fail += 1

# Append new rows to CSV
fieldnames = [
    "dataset", "layer", "variant", "seed", "gpu",
    "n_samples", "n_features", "n_clusters",
    "epochs", "mask_ratio", "neighbor_k", "hidden_size",
    "acc", "nmi", "ari", "f1_macro", "runtime_seconds",
]
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerows(new_rows)

print(f"\n=== Summary ===")
print(f"OK: {n_ok}, Skipped (exists): {n_skip}, Failed: {n_fail}")
print(f"New rows appended to: {CSV_PATH}")
EOF
echo "=== Done ===" >> "$LOG"
date >> "$LOG"
echo "All ext ablation missing variants complete."