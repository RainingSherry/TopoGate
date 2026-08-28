#!/usr/bin/env python
"""5 datasets × 4 variants × 3 seeds smoke test runner for v3 changes.

Outputs to result/learnable_gate_smoke/v3_smoke/results.csv (append mode).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Avoid BLAS / OpenMP threads explosion when many workers run
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from methods.TopoGate.learnable_gate.run_npz import main as run_main


DATASETS_5 = [
    "Mouse_retina",
    "enron",
    "har",
    "breast_cancer_wisconsin_original",
    "sms_spam_collection",
]

# Small datasets get epochs=50, big get epochs=50 too — same protocol as before
VARIANTS = {
    "baseline": {
        "gate_mode": "learned",
        "learnable_gate_max": False,
        "gate_lr_multiplier": 1.0,
    },
    "v3_lgm": {
        "gate_mode": "learned",
        "learnable_gate_max": True,
        "gate_lr_multiplier": 1.0,
    },
    "v3_lr": {
        "gate_mode": "learned",
        "learnable_gate_max": False,
        "gate_lr_multiplier": 10.0,
    },
    "v3_full": {
        "gate_mode": "learned",
        "learnable_gate_max": True,
        "gate_lr_multiplier": 10.0,
    },
}


def build_argv(dataset, variant, seed, epochs, gpu):
    config = VARIANTS[variant]
    npz_path = ROOT / "datasets" / f"{dataset}.npz"
    save_dir = ROOT / "result" / "learnable_gate_smoke" / "v3_smoke" / f"{dataset}__{variant}__seed{seed}"
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
        "--gate_mode", config["gate_mode"],
        "--learnable_gate_max", "true" if config["learnable_gate_max"] else "false",
        "--gate_lr_multiplier", str(config["gate_lr_multiplier"]),
    ]
    return argv, save_dir


def run_one(dataset, variant, seed, epochs, gpu):
    argv, save_dir = build_argv(dataset, variant, seed, epochs, gpu)
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


def append_csv(csv_path, variant, ds, seed, ari, summary):
    fbeta = (summary or {}).get("learned_gate_final_beta") or {}
    header = "variant,dataset,seed,ari,effective_gate_max,beta_mutual,beta_snn,beta_perturb,beta_uncertainty\n"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w") as f:
            f.write(header)
    with open(csv_path, "a") as f:
        f.write(f"{variant},{ds},{seed},{ari},"
                f"{fbeta.get('effective_gate_max', '')},"
                f"{fbeta.get('beta_mutual', '')},"
                f"{fbeta.get('beta_snn', '')},"
                f"{fbeta.get('beta_perturb', '')},"
                f"{fbeta.get('beta_uncertainty', '')}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS_5)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 7])
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.datasets = ["iris"]
        args.seeds = [42]
        args.epochs = 5

    csv_path = ROOT / "result" / "learnable_gate_smoke" / "v3_smoke" / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"# Smoke test runner: {len(args.variants)} variants × {len(args.datasets)} datasets × {len(args.seeds)} seeds = {len(args.variants)*len(args.datasets)*len(args.seeds)} runs", flush=True)
    print(f"# CSV: {csv_path}", flush=True)

    for variant in args.variants:
        for ds in args.datasets:
            for seed in args.seeds:
                tag = f"{variant}/{ds}/seed{seed}"
                t0 = time.time()
                print(f"=== {tag} ===", flush=True)
                try:
                    ari, summary = run_one(ds, variant, seed, args.epochs, args.gpu)
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)
                    ari, summary = None, None
                dt = time.time() - t0
                append_csv(csv_path, variant, ds, seed, ari, summary)
                print(f"  -> ARI={ari}  time={dt:.1f}s", flush=True)

    print("=== DONE ===", flush=True)
    # Final summary
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print("\nFinal results:")
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        if r["ari"]:
            by[(r["variant"], r["dataset"])].append(float(r["ari"]))
    print(f"{'Variant':12s} {'Dataset':35s} {'ARI mean':>10s} {'ARI std':>10s}")
    for (v, ds), vals in sorted(by.items()):
        if vals:
            import statistics
            print(f"{v:12s} {ds:35s} {statistics.mean(vals):>10.4f} {statistics.stdev(vals) if len(vals)>1 else 0:>10.4f}")


if __name__ == "__main__":
    main()
