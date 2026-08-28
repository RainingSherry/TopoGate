#!/usr/bin/env python
"""
TopoGate (learnable_gate) ablation — 3 axes × 5 datasets × 3 seeds.

Ablation axes on top of learnable_gate_sched:
  A. Gate:    learned → none          (Is LearnableGate useful?)
  B. Mix:     reliability → none      (Is neighbor mixing useful?)
  C. Edge:    sim_mutual_snn_distance → none  (Is edge reliability useful?)

4 configs × 5 datasets × 3 seeds = 60 runs.

The "full" config is exactly learnable_gate_sched.yaml with default hparams
from run_npz.py (epochs=80, mask_ratio=0.4, neighbor_k=5, etc.).

All other configs share the SAME hparams as full — only the mechanism knob
is toggled.  This satisfies the ablation design principle: one variable at a time.

Outputs:
  result/ablation_learnable_gate/<dataset>/<dataset>__<config>__seed<seed>.json
  result/ablation_learnable_gate/merged_summary.csv
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
CLUBENCH_ROOT = REPO_ROOT / "baseline" / "CLUBench"
for p in (str(REPO_ROOT), str(CLUBENCH_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 5 core datasets ────────────────────────────────────────────────────────────
DATASETS = [
    "Mouse_retina",
    "sms_spam_collection",
    "enron",
    "har",
    "breast_cancer_wisconsin_original",
]

# ── PCA overrides (same as run_baseline_comparison.py) ────────────────────────
PCA_DIM_OVERRIDE = {}

# ── 4 ablation configs ────────────────────────────────────────────────────────
# All inherit from learnable_gate_sched defaults (epochs=80, mask_ratio=0.4,
# neighbor_k=5, lr=1e-3, batch_size=256, warmup=20, ramp=10, init_mode=zero).
# Only the mechanism-specific knob differs.

CONFIGS = {
    # Full: all three mechanisms enabled (same as learnable_gate_sched)
    "full": dict(
        variant="learnable_gate_sched",
        gate_mode="learned",
        mix_mode="reliability",
        edge_reliability_mode="sim_mutual_snn_distance",
    ),
    # Ablation A: no gate (topology gate disabled → gate_mode=none)
    "no_gate": dict(
        variant="learnable_gate_sched",
        gate_mode="none",          # ← toggled
        mix_mode="reliability",
        edge_reliability_mode="sim_mutual_snn_distance",
    ),
    # Ablation B: no mixing (neighbor mixing disabled → mix_mode=none)
    "no_mix": dict(
        variant="learnable_gate_sched",
        gate_mode="learned",
        mix_mode="none",          # ← toggled
        edge_reliability_mode="sim_mutual_snn_distance",
    ),
    # Ablation C: no edge reliability (edge reliability disabled → mode=none)
    "no_edge": dict(
        variant="learnable_gate_sched",
        gate_mode="learned",
        mix_mode="reliability",
        edge_reliability_mode="none",   # ← toggled
    ),
}

# ── Shared hparams (from learnable_gate_sched defaults + 15-dataset tuning) ──
# NOTE: These match the defaults in run_npz.py AND the values used in
# run_baseline_comparison.py (where TopoGate() is called with no kwargs).
# Using epochs=80 / mask_ratio=0.4 / neighbor_k=5 to stay consistent with
# the main benchmark.  The static_gate ablation used epochs=150 / mask_ratio=0.3
# but those were tuned for static_gate variants and MUST NOT be reused here.
FIXED_HPARAMS = dict(
    epochs=80,
    mask_ratio=0.4,
    neighbor_k=5,
    hidden_size=128,
    batch_size=256,
    lr=1e-3,
    warmup_epochs=20,
    ramp_epochs=10,
    learned_gate_init_mode="zero",
    gate_max=0.15,
    pseudo_weight=0.3,
    knn_pca_dim=50,
    tau=0.2,
    gamma_sim=1.0,
    gamma_mutual=1.0,
    gamma_snn=1.0,
    gamma_distance=1.0,
    beta_mutual=1.0,
    beta_snn=1.0,
    beta_perturb=2.0,
    beta_uncertainty=1.0,
    gate_lr_multiplier=10.0,
    subsample_size=0,
)


def parse_args():
    p = argparse.ArgumentParser(description="TopoGate learnable_gate ablation")
    p.add_argument("--gpu_ids", type=int, nargs="+", default=[4, 5, 7])
    p.add_argument("--worker_id", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--result_dir",
                   default="/home/luolie/ToPoGate/result/ablation_learnable_gate")
    p.add_argument("--csv_path",
                   default="/home/luolie/ToPoGate/result/ablation_learnable_gate/merged_summary.csv")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def run_one(dataset: str, config_name: str, config_overrides: dict,
            hparams: dict, seed: int, gpu_id: int):
    """One TopoGate run → returns dict row."""
    from sklearn.decomposition import PCA
    from methods.TopoGate.learnable_gate.run_npz import run_topogate
    import numpy as np

    DATA_DIR = Path("/data/luolie/ToPoGate/datasets")
    npz_path = DATA_DIR / f"{dataset}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} does not exist")

    data = np.load(npz_path)
    X = data["x"]
    Y = data["y"]
    X = np.asarray(X)
    Y = np.asarray(Y)
    K = int(np.unique(Y).size)
    n_samples, n_features = X.shape

    # PCA preprocessing (same as run_baseline_comparison.py)
    pca_dim = PCA_DIM_OVERRIDE.get(dataset)
    if pca_dim is not None and X.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        X = pca.fit_transform(X)

    # Merge: FIXED_HPARAMS < config_overrides < per-run overrides
    run_cfg = {**hparams, **config_overrides}

    t0 = time.time()
    labels, elapsed, metrics = run_topogate(
        X=X,
        n_clusters=K,
        y=Y,
        gpu=gpu_id,
        variant=run_cfg.pop("variant"),
        seed=seed,
        return_metrics=True,
        config_dir=str(REPO_ROOT / "methods/TopoGate/learnable_gate/configs"),
        **run_cfg,
    )
    runtime = time.time() - t0

    return {
        "dataset": dataset,
        "config": config_name,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_clusters": K,
        "seed": seed,
        "gpu": gpu_id,
        "runtime_seconds": float(runtime),
        **metrics,
    }


def main():
    args = parse_args()
    if args.worker_id < 0 or args.worker_id >= len(args.gpu_ids):
        raise ValueError(f"worker_id {args.worker_id} out of range for gpu_ids {args.gpu_ids}")

    gpu = args.gpu_ids[args.worker_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Build job list: (dataset, config_name, config_overrides, seed)
    all_jobs = []
    for ds in DATASETS:
        for cfg_name, cfg_overrides in CONFIGS.items():
            for seed in args.seeds:
                all_jobs.append((ds, cfg_name, cfg_overrides, seed))

    # Round-robin shard
    my_jobs = [j for i, j in enumerate(all_jobs)
               if i % len(args.gpu_ids) == args.worker_id]

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"[worker {args.worker_id} on GPU {gpu}] jobs={len(my_jobs)} / {len(all_jobs)}")
    print(f"  configs: {list(CONFIGS.keys())}")
    print(f"  seeds: {args.seeds}")

    rows = []
    n_ok, n_skip, n_fail = 0, 0, 0

    for i, (ds, cfg_name, cfg_overrides, seed) in enumerate(my_jobs, 1):
        out_path = result_dir / ds / f"{ds}__{cfg_name}__seed{seed}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.force:
            n_skip += 1
            try:
                with open(out_path) as f:
                    rows.append(json.load(f))
            except Exception:
                pass
            continue

        print(f"[{i:3d}/{len(my_jobs)}] {ds} cfg={cfg_name} seed={seed}  ...", end=" ", flush=True)
        try:
            result = run_one(ds, cfg_name, cfg_overrides, FIXED_HPARAMS, seed, gpu)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            rows.append(result)
            print(f"ACC={result['acc']:.4f} NMI={result['nmi']:.4f} "
                  f"ARI={result['ari']:.4f} ({result['runtime_seconds']:.1f}s)")
            n_ok += 1
        except Exception as e:
            err_path = result_dir / ds / f"{ds}__{cfg_name}__seed{seed}.error.json"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, "w") as f:
                json.dump({
                    "dataset": ds, "config": cfg_name, "seed": seed,
                    "error": str(e), "traceback": traceback.format_exc(),
                }, f, indent=2)
            print(f"FAIL: {e}")
            n_fail += 1

    # Append to merged_summary.csv
    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "config", "seed", "gpu",
        "n_samples", "n_features", "n_clusters",
        "acc", "nmi", "ari", "f1_macro",
        "runtime_seconds",
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
