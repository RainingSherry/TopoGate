#!/usr/bin/env python
"""v3_combined: all v3 changes + lr 3x + tighter gate_max bounds.

Conservative combination designed to not destabilise any module.
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


DATASETS = ["Mouse_retina", "enron", "har", "breast_cancer_wisconsin_original", "sms_spam_collection"]


def run_one(dataset, seed, epochs=50, gpu=1, lr_mul=3.0, use_lgm=False):
    npz_path = ROOT / "datasets" / f"{dataset}.npz"
    save_dir = ROOT / "result" / "learnable_gate_smoke" / "v3_combined" / f"{dataset}__v3_combined__seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--data_path", str(npz_path),
        "--save_dir", str(save_dir),
        "--dataset_name", dataset,
        "--variant_name", "v3_combined",
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
        # All v3 flags
        "--gate_mode", "learned",
        "--learnable_gate_max", "true" if use_lgm else "false",
        "--gate_max_min", "0.05",
        "--gate_max_max", "0.6",  # tighter upper bound to avoid runaway
        "--gate_lr_multiplier", str(lr_mul),
        "--enhanced_stats", "6",
        "--learnable_gamma", "true",
        "--gamma_reg_weight", "1e-4",
        "--learnable_mask_ratio", "true",
        "--mask_ratio_min", "0.1",
        "--mask_ratio_max", "0.6",
    ]
    saved_argv = sys.argv
    sys.argv = ["run_npz.py"] + argv
    try:
        run_main()
    finally:
        sys.argv = saved_argv

    metrics_path = save_dir / "metrics.json"
    summary = {}
    ari = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            ari = json.load(f).get("ari")
    if (save_dir / "summary.json").exists():
        with open(save_dir / "summary.json") as f:
            summary = json.load(f)
    return ari, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 7])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--lr_mul", type=float, default=3.0)
    parser.add_argument("--use_lgm", action="store_true",
                        help="If True, also enable learnable_gate_max.  Default False.")
    args = parser.parse_args()

    csv_path = ROOT / "result" / "learnable_gate_smoke" / "v3_combined" / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w") as f:
            f.write("variant,dataset,seed,ari,effective_gate_max,mask_ratio,beta_mutual,beta_snn,beta_perturb,beta_degree,beta_cluster,gamma_sim,gamma_mutual,gamma_snn,gamma_distance\n")

    for ds in args.datasets:
        for seed in args.seeds:
            tag = f"v3_combined/{ds}/seed{seed}"
            print(f"=== {tag} ===", flush=True)
            try:
                ari, summary = run_one(ds, seed, args.epochs, args.gpu, args.lr_mul, args.use_lgm)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                ari, summary = None, None
            fbeta = (summary or {}).get("learned_gate_final_beta") or {}
            fgamma = (summary or {}).get("learned_edge_final_gamma") or {}
            mask = (summary or {}).get("learned_mask_ratio") or ""
            with open(csv_path, "a") as f:
                f.write(f"v3_combined,{ds},{seed},{ari},"
                        f"{fbeta.get('effective_gate_max', '')},"
                        f"{mask},"
                        f"{fbeta.get('beta_mutual', '')},"
                        f"{fbeta.get('beta_snn', '')},"
                        f"{fbeta.get('beta_perturb', '')},"
                        f"{fbeta.get('beta_degree', '')},"
                        f"{fbeta.get('beta_cluster', '')},"
                        f"{fgamma.get('gamma_sim', '')},"
                        f"{fgamma.get('gamma_mutual', '')},"
                        f"{fgamma.get('gamma_snn', '')},"
                        f"{fgamma.get('gamma_distance', '')}\n")
            print(f"  -> ARI={ari} eff_gate_max={fbeta.get('effective_gate_max')} mask_ratio={mask}", flush=True)
    print("=== DONE ===")


if __name__ == "__main__":
    main()