#!/usr/bin/env python
"""Quick test: v3_conservative (lr 5x + lgm) vs v3_full (lr 10x + lgm)."""
from __future__ import annotations

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
SEEDS = [42, 123, 7]
VARIANTS = {
    "v3_conservative": {"learnable_gate_max": True, "gate_lr_multiplier": 5.0},
    "v3_lr3": {"learnable_gate_max": True, "gate_lr_multiplier": 3.0},
    "v3_lr10_no_lgm": {"learnable_gate_max": False, "gate_lr_multiplier": 10.0},
}


def run_one(dataset, variant, seed, epochs=50, gpu=1):
    config = VARIANTS[variant]
    npz_path = ROOT / "datasets" / f"{dataset}.npz"
    save_dir = ROOT / "result" / "learnable_gate_smoke" / "v3_tune" / f"{dataset}__{variant}__seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--data_path", str(npz_path),
        "--save_dir", str(save_dir),
        "--dataset_name", dataset,
        "--variant_name", f"v3_{variant}",
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
        "--gate_mode", "learned",
        "--learnable_gate_max", "true" if config["learnable_gate_max"] else "false",
        "--gate_lr_multiplier", str(config["gate_lr_multiplier"]),
    ]
    saved_argv = sys.argv
    sys.argv = ["run_npz.py"] + argv
    try:
        run_main()
    finally:
        sys.argv = saved_argv

    metrics_path = save_dir / "metrics.json"
    if not metrics_path.exists():
        return None, None
    with open(metrics_path) as f:
        metrics = json.load(f)
    summary_path = save_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    return metrics.get("ari"), summary


def main():
    csv_path = ROOT / "result" / "learnable_gate_smoke" / "v3_tune" / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Header
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w") as f:
            f.write("variant,dataset,seed,ari,effective_gate_max,beta_mutual,beta_snn,beta_perturb,beta_uncertainty\n")

    for variant in VARIANTS:
        for ds in DATASETS:
            for seed in SEEDS:
                tag = f"{variant}/{ds}/seed{seed}"
                print(f"=== {tag} ===", flush=True)
                try:
                    ari, summary = run_one(ds, variant, seed)
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)
                    ari, summary = None, None
                fbeta = (summary or {}).get("learned_gate_final_beta") or {}
                with open(csv_path, "a") as f:
                    f.write(f"{variant},{ds},{seed},{ari},"
                            f"{fbeta.get('effective_gate_max', '')},"
                            f"{fbeta.get('beta_mutual', '')},"
                            f"{fbeta.get('beta_snn', '')},"
                            f"{fbeta.get('beta_perturb', '')},"
                            f"{fbeta.get('beta_uncertainty', '')}\n")
                print(f"  -> ARI={ari} eff_gate_max={fbeta.get('effective_gate_max')}", flush=True)

    print("=== DONE ===")


if __name__ == "__main__":
    main()