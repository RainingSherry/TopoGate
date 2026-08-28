#!/usr/bin/env python
"""v4_baseline_lr10: cleanest v3_lr (no other v3 modifications).

The goal is to verify GateLrDecoupling (lr_multiplier=10) on 15 datasets × 3 seeds
BEFORE adding any other modifications.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from methods.TopoGate.learnable_gate.run_npz import main as run_main


# 15 datasets (same as multiseed test)
DATASETS = [
    "Campbell", "ISOLET", "Mouse_retina", "Quake_Smart-seq2_Lung",
    "breast_cancer_wisconsin_original", "cnae9", "enron",
    "first-order-theorem-proving", "har", "hrvatin_filtered",
    "iris", "mammographic_mass", "reuters", "sms_spam_collection",
    "spambase",
]


def run_one(dataset, seed, lr_mul, epochs=30, gpu=1):
    npz_path = ROOT / "datasets" / f"{dataset}.npz"
    variant = f"v4_lr{int(lr_mul)}"
    save_dir = ROOT / "result" / "learnable_gate_smoke" / "v4_baseline" / f"{dataset}__{variant}__seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--data_path", str(npz_path),
        "--save_dir", str(save_dir),
        "--dataset_name", dataset,
        "--variant_name", variant,
        "--method_name", "TopoGate",
        "--epochs", str(epochs),
        "--batch_size", "256",
        "--neighbor_k", "5",
        "--mask_ratio", "0.3",
        "--seed", str(seed),
        "--gpu", str(gpu),
        "--lightweight_outputs",
        "--warmup_epochs", "10",
        "--ramp_epochs", "10",
        "--freeze_mae_after_epoch", "1000000000",
        # v3_lr ONLY: gate lr 10x, all other v3 changes OFF
        "--gate_mode", "learned",
        "--learnable_gate_max", "false",
        "--gate_lr_multiplier", str(lr_mul),
        "--enhanced_stats", "4",          # NO degree/cluster
        "--learnable_gamma", "false",     # NO learned gamma
        "--learnable_mask_ratio", "false",# NO learned mask
    ]
    saved_argv = sys.argv
    sys.argv = ["run_npz.py"] + argv
    try:
        run_main()
    finally:
        sys.argv = saved_argv

    metrics_path = save_dir / "metrics.json"
    ari = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            ari = json.load(f).get("ari")
    return ari


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 7])
    parser.add_argument("--lr_multiplier", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()

    variant = f"v4_lr{int(args.lr_multiplier)}"
    csv_path = ROOT / "result" / "learnable_gate_smoke" / "v4_baseline" / f"results_{variant}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w") as f:
            f.write("variant,dataset,seed,ari\n")

    for ds in args.datasets:
        for seed in args.seeds:
            tag = f"{variant}/{ds}/seed{seed}"
            print(f"=== {tag} (gpu={args.gpu}) ===", flush=True)
            try:
                ari = run_one(ds, seed, args.lr_multiplier, args.epochs, args.gpu)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                ari = None
            with open(csv_path, "a") as f:
                f.write(f"{variant},{ds},{seed},{ari}\n")
            print(f"  -> ARI={ari}", flush=True)
    print("=== DONE ===")


if __name__ == "__main__":
    main()
