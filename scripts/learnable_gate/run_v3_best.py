#!/usr/bin/env python
"""v3_best: the validated winning recipe.

- gate_lr_multiplier=10.0 (verified +0.0021 from v3_smoke test)
- learnable_gate_max=False (verified lgm destabilises even at lr 3x)
- enhanced_stats=6 (degree + cluster, but cluster only for n<=5000)
- learnable_gamma=True (4 gamma learned)
- learnable_mask_ratio=True (mask_ratio learned)
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


def run_one(dataset, seed, epochs=50, gpu=1):
    npz_path = ROOT / "datasets" / f"{dataset}.npz"
    save_dir = ROOT / "result" / "learnable_gate_smoke" / "v3_best" / f"{dataset}__v3_best__seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--data_path", str(npz_path),
        "--save_dir", str(save_dir),
        "--dataset_name", dataset,
        "--variant_name", "v3_best",
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
        # v3_best: all v3 flags except lgm
        "--gate_mode", "learned",
        "--learnable_gate_max", "false",  # critical: do NOT enable lgm
        "--gate_lr_multiplier", "10.0",   # verified to give +0.0021
        "--enhanced_stats", "6",          # degree + cluster (cluster only if n<=5K)
        "--learnable_gamma", "true",      # 4 gamma learned
        "--gamma_reg_weight", "1e-4",
        "--learnable_mask_ratio", "true", # mask_ratio learned
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
    args = parser.parse_args()

    csv_path = ROOT / "result" / "learnable_gate_smoke" / "v3_best" / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w") as f:
            f.write("variant,dataset,seed,ari,mask_ratio,beta_mutual,beta_snn,beta_perturb,beta_degree,beta_cluster,gamma_sim,gamma_mutual,gamma_snn,gamma_distance\n")

    for ds in args.datasets:
        for seed in args.seeds:
            tag = f"v3_best/{ds}/seed{seed}"
            print(f"=== {tag} ===", flush=True)
            try:
                ari, summary = run_one(ds, seed, args.epochs, args.gpu)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                ari, summary = None, None
            fbeta = (summary or {}).get("learned_gate_final_beta") or {}
            fgamma = (summary or {}).get("learned_edge_final_gamma") or {}
            mask = (summary or {}).get("learned_mask_ratio") or ""
            with open(csv_path, "a") as f:
                f.write(f"v3_best,{ds},{seed},{ari},"
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
            print(f"  -> ARI={ari} mask_ratio={mask}", flush=True)
    print("=== DONE ===")


if __name__ == "__main__":
    main()