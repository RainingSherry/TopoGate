#!/usr/bin/env python
"""Single-variant runner for v3_tune.  Used by parallel workers."""
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


def run_one(dataset, variant, config, seed, epochs, gpu):
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
    parser.add_argument("--variant", required=True)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 7])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    VARIANTS = {
        "v3_conservative": {"learnable_gate_max": True, "gate_lr_multiplier": 5.0},
        "v3_lr3": {"learnable_gate_max": True, "gate_lr_multiplier": 3.0},
        "v3_lr10_no_lgm": {"learnable_gate_max": False, "gate_lr_multiplier": 10.0},
    }
    config = VARIANTS[args.variant]

    csv_path = ROOT / "result" / "learnable_gate_smoke" / "v3_tune" / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_path.with_suffix(".lock")
    header = "variant,dataset,seed,ari,effective_gate_max,beta_mutual,beta_snn,beta_perturb,beta_uncertainty\n"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        # Use lock to avoid race on header write
        if not lock_path.exists():
            lock_path.write_text("locked")
            with open(csv_path, "w") as f:
                f.write(header)
            lock_path.unlink()

    for ds in args.datasets:
        for seed in args.seeds:
            tag = f"{args.variant}/{ds}/seed{seed}"
            print(f"=== {tag} ===", flush=True)
            try:
                ari, summary = run_one(ds, args.variant, config, seed, args.epochs, args.gpu)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                ari, summary = None, None
            fbeta = (summary or {}).get("learned_gate_final_beta") or {}
            with open(csv_path, "a") as f:
                f.write(f"{args.variant},{ds},{seed},{ari},"
                        f"{fbeta.get('effective_gate_max', '')},"
                        f"{fbeta.get('beta_mutual', '')},"
                        f"{fbeta.get('beta_snn', '')},"
                        f"{fbeta.get('beta_perturb', '')},"
                        f"{fbeta.get('beta_uncertainty', '')}\n")
            print(f"  -> ARI={ari} eff_gate_max={fbeta.get('effective_gate_max')}", flush=True)
    print("=== DONE ===")


if __name__ == "__main__":
    main()