#!/usr/bin/env python
"""v5 launch — parallel multi-dataset runner.

Variants:
  v5_1g_ste: one_param_scalar + STE mask_learnable
  v5_1g_fixed: one_param_scalar + fixed mask_ratio=0.4
  v5_4f_fixed: all_params_4f + fixed mask_ratio=0.4

Note: v5_4f converges to v5_1g (symmetry, observed in v3) — included for
ablation only.

8 datasets, 3 seeds each, ~30 epochs. ~10min/dataset on small ds.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


DATASETS_8 = [
    "Campbell", "Mouse_retina", "breast_cancer_wisconsin_original",
    "enron", "har", "iris", "mammographic_mass", "spambase",
]


VARIANTS = [
    ("v5_1g_ste",  {"v5_gamma_mode": "one_param_scalar", "mask_ratio_learnable": True}),
    ("v5_1g_fixed", {"v5_gamma_mode": "one_param_scalar", "mask_ratio_learnable": False}),
    ("v5_4f_fixed", {"v5_gamma_mode": "all_params_4f",   "mask_ratio_learnable": False}),
]


def run_one(dataset, variant, seed, epochs, gpu, sub_dir):
    save_dir = ROOT / "result" / "learnable_gate_smoke" / "v5" / sub_dir / f"{dataset}__{variant}__seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "scripts/learnable_gate/run_v5_separate.py",
        "--data_path", str(ROOT / "datasets" / f"{dataset}.npz"),
        "--save_dir", str(save_dir),
        "--dataset_name", dataset,
        "--variant_name", variant,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--gpu", str(gpu),
        "--v5_gamma_mode", VARIANTS[sub_idx_lookup(variant)]["v5_gamma_mode"],
    ]
    if VARIANTS[sub_idx_lookup(variant)]["mask_ratio_learnable"]:
        cmd.append("--mask_ratio_learnable")
    print(f"[{variant}/{dataset}/seed{seed}] launching on gpu={gpu}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT))


def sub_idx_lookup(variant):
    for i, (v, _) in enumerate(VARIANTS):
        if v == variant:
            return i
    raise KeyError(variant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS_8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gpus", nargs="+", type=int, default=[4, 5])
    parser.add_argument("--variant", required=True, choices=[v for v, _ in VARIANTS])
    parser.add_argument("--sub_dir", default="v5_main")
    args = parser.parse_args()

    jobs = []
    next_gpu = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            gpu = args.gpus[next_gpu % len(args.gpus)]
            next_gpu += 1
            jobs.append((dataset, args.variant, seed, args.epochs, gpu, args.sub_dir))

    for j in jobs:
        dataset, variant, seed, epochs, gpu, sub_dir = j
        run_one(dataset, variant, seed, epochs, gpu, sub_dir)


if __name__ == "__main__":
    main()
