#!/usr/bin/env python
"""Resume Phase 0 tuning: only run hrvatin_filtered + Quake_Smart-seq2_Lung.

These two datasets are not in CLUBench.configs.DATASETS and were skipped by the
original run.  Now uses the same run_one() (npz-direct) as ablation script.
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
REPO_ROOT = SCRIPT_DIR.parent
CLUBENCH_ROOT = REPO_ROOT / "baseline" / "CLUBench"
for p in (str(REPO_ROOT), str(CLUBENCH_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu_ids", type=int, nargs="+", default=[4, 5, 7])
    p.add_argument("--worker_id", type=int, required=True)
    p.add_argument("--result_dir",
                   default="/home/luolie/ToPoGate/result/tune_15datasets")
    p.add_argument("--csv_path",
                   default="/home/luolie/ToPoGate/result/tune_15datasets/grid.csv")
    args = p.parse_args()

    target_datasets = ["hrvatin_filtered", "Quake_Smart-seq2_Lung"]
    gpus = args.gpu_ids
    gpu = gpus[args.worker_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    GRID_EPOCHS = [40, 80, 150]
    GRID_MASK_RATIO = [0.3, 0.4, 0.5]
    GRID_NEIGHBOR_K = [5, 10, 20]

    all_jobs = []
    for ds in target_datasets:
        for ep in GRID_EPOCHS:
            for mr in GRID_MASK_RATIO:
                for k in GRID_NEIGHBOR_K:
                    all_jobs.append((ds, ep, mr, k))

    my_jobs = [j for i, j in enumerate(all_jobs) if i % len(gpus) == args.worker_id]
    print(f"[worker {args.worker_id} on GPU {gpu}] {len(my_jobs)} jobs")

    # Import run_one from the main script
    sys.path.insert(0, str(SCRIPT_DIR))
    from run_topogate_tune_15datasets import run_one

    result_dir = Path(args.result_dir)
    rows = []
    n_ok, n_skip, n_fail = 0, 0, 0

    for i, (ds, ep, mr, k) in enumerate(my_jobs, 1):
        tag = f"ep{ep}_mr{mr}_k{k}"
        out_path = result_dir / ds / f"{ds}__{tag}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip ONLY if a successful .json exists (NOT .error.json)
        if out_path.exists():
            n_skip += 1
            try:
                with open(out_path) as f:
                    rows.append(json.load(f))
            except Exception:
                pass
            continue

        print(f"[{i:3d}/{len(my_jobs)}] {ds} {tag}  ...", end=" ", flush=True)
        try:
            result = run_one(ds, ep, mr, k, gpu)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            rows.append(result)
            print(f"ACC={result['acc']:.4f} ({result['runtime_seconds']:.1f}s)")
            n_ok += 1
        except Exception as e:
            err_path = result_dir / ds / f"{ds}__{tag}.error.json"
            with open(err_path, "w") as f:
                json.dump({
                    "dataset": ds, "epochs": ep, "mask_ratio": mr, "neighbor_k": k,
                    "error": str(e), "traceback": traceback.format_exc(),
                }, f, indent=2)
            print(f"FAIL: {e}")
            n_fail += 1

    # Append to grid.csv
    csv_path = Path(args.csv_path)
    fieldnames = [
        "dataset", "variant", "seed", "gpu",
        "n_samples", "n_features", "n_clusters",
        "epochs", "mask_ratio", "neighbor_k", "hidden_size",
        "acc", "nmi", "ari", "f1_macro", "runtime_seconds",
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[worker {args.worker_id}] done  ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
