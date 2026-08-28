#!/usr/bin/env python3
"""Run remaining ext ablation (5 datasets x 4 missing variants) one at a time.
Direct python (no shell heredoc), proper stdout.
"""
import sys, os, time, json, csv, traceback
from pathlib import Path

SCRIPT_DIR = Path("/home/luolie/ToPoGate/scripts/static_gate")
REPO_ROOT = SCRIPT_DIR.parent.parent
CLUBENCH_ROOT = REPO_ROOT / "baseline" / "CLUBench"
for p in (str(REPO_ROOT), str(CLUBENCH_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np
from sklearn.decomposition import PCA
from CLUBench import TopoGate
from CLUBench.tools import clustering_evaluation

DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
ABL_DIR = Path("/home/luolie/ToPoGate/result/ablation")
CSV_PATH = ABL_DIR / "merged_summary.csv"

EPOCHS = 30
MASK_RATIO = 0.3
NEIGHBOR_K = 5
HIDDEN_SIZE = 128
BATCH_SIZE = 256
LR = 1e-3
SEED = 42

PCA_DIM_OVERRIDE = {"hrvatin_filtered": 500, "Campbell": 500}

REMAINING_DATASETS = [
    "iris",
    "first-order-theorem-proving",
    "mammographic_mass",
    "Quake_Smart-seq2_Lung",
    "hrvatin_filtered",
]
MISSING_VARIANTS = [
    "static_gate_far_neighbors",
    "static_gate_edge_only",
    "static_gate_gate_only",
    "static_gate_no_topology_features",
]

# Load existing CSV
existing_keys = set()
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        existing_keys.add((r["dataset"], r["variant"], r["epochs"]))

new_rows = []
n_ok = n_skip = n_fail = 0
for ds in REMAINING_DATASETS:
    npz_path = DATA_DIR / f"{ds}.npz"
    if not npz_path.exists():
        print(f"[MISS-DATA] {ds}", flush=True)
        continue
    for variant in MISSING_VARIANTS:
        tag = f"{variant}__ep{EPOCHS}_mr{MASK_RATIO}_k{NEIGHBOR_K}"
        out_path = ABL_DIR / ds / f"{ds}__{tag}.json"
        if out_path.exists():
            n_skip += 1
            print(f"[SKIP] {ds} {variant} (exists)", flush=True)
            continue
        if (ds, variant, str(EPOCHS)) in existing_keys:
            n_skip += 1
            print(f"[SKIP-CSV] {ds} {variant}", flush=True)
            continue
        print(f"[RUN] {ds} {variant} ...", end=" ", flush=True)
        try:
            data = np.load(npz_path)
            X, Y = data["x"], data["y"]
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
                gpu=0,
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
                "layer": "ext",
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "n_clusters": int(K),
                "seed": SEED,
                "gpu": 4,
                "epochs": EPOCHS,
                "mask_ratio": float(MASK_RATIO),
                "neighbor_k": int(NEIGHBOR_K),
                "hidden_size": HIDDEN_SIZE,
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
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, "w") as f:
                json.dump({
                    "dataset": ds, "variant": variant,
                    "epochs": EPOCHS, "mask_ratio": MASK_RATIO,
                    "neighbor_k": NEIGHBOR_K,
                    "error": str(e), "traceback": traceback.format_exc(),
                }, f, indent=2)
            print(f"FAIL: {e}", flush=True)
            n_fail += 1

# Append to CSV
if new_rows:
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(new_rows)
    print(f"\n=== Appended {len(new_rows)} rows to {CSV_PATH} ===", flush=True)

print(f"\n=== Summary ===")
print(f"OK: {n_ok}, Skipped: {n_skip}, Failed: {n_fail}")