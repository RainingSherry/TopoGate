#!/usr/bin/env python3
"""Run Stage 1 sweep only on small datasets (<5000 samples).

3 workers, each on dedicated GPU. Reads existing JSONs and skips them.
Outputs to /home/luolie/ToPoGate/result/learnable_gate_134_sweep/stage1/.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/luolie/ToPoGate")
CSV_PATH = REPO_ROOT / "result" / "dataset_npz_info.csv"
STAGE1_DIR = REPO_ROOT / "result" / "learnable_gate_134_sweep" / "stage1"
RUNNER = REPO_ROOT / "scripts" / "learnable_gate" / "run_learnable_gate_134_sweep.py"

# Stage1 grid: 4 configs
STAGE1_GRID = [
    dict(epochs=80, mask_ratio=0.3, neighbor_k=5,  gate_max=0.15),
    dict(epochs=80, mask_ratio=0.3, neighbor_k=10, gate_max=0.15),
    dict(epochs=80, mask_ratio=0.4, neighbor_k=5,  gate_max=0.15),
    dict(epochs=80, mask_ratio=0.4, neighbor_k=10, gate_max=0.15),
]


def load_small_datasets(max_n=5000):
    sizes = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            sizes[row["dataset_name"]] = int(row["n_samples"])
    return sorted([n for n, s in sizes.items() if s < max_n])


def already_done(datasets):
    """Count how many (ds, cfg) are already complete."""
    done = 0
    for ds in datasets:
        for cfg in STAGE1_GRID:
            tag = f"ep{cfg['epochs']}_mr{cfg['mask_ratio']}_k{cfg['neighbor_k']}_gmax{cfg['gate_max']}"
            p = STAGE1_DIR / f"{ds}__{tag}__seed42.json"
            if p.exists():
                try:
                    json.load(open(p))
                    done += 1
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_n", type=int, default=5000, help="Max n_samples threshold")
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 7, 0])
    ap.add_argument("--logdir", default=str(REPO_ROOT / "result" / "learnable_gate_134_sweep" / "rerun_logs"))
    args = ap.parse_args()

    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    small = load_small_datasets(args.max_n)
    print(f"[small_only] {len(small)} datasets with n<{args.max_n}")
    done = already_done(small)
    total = len(small) * len(STAGE1_GRID)
    print(f"[small_only] {done}/{total} (ds,cfg) already done")

    # We need to filter the dataset list inside the runner. Easy hack: set DATASETS env var?
    # But the runner reads from CSV. Instead, we'll use a wrapper that filters via input.
    # Simpler approach: pass a filtered list via env var (modify the runner later).
    # For now, we'll launch full runner but with --num_workers=1 each, and accept that large
    # datasets get processed too. We just want to validate the pipeline first.

    # Actually, cleanest: write a filtered CSV next to original.
    import shutil
    tmp_csv = REPO_ROOT / "result" / f"dataset_npz_info_small_{args.max_n}.csv"
    with open(CSV_PATH) as fin, open(tmp_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["n_samples"]) < args.max_n:
                writer.writerow(row)
    print(f"[small_only] Wrote filtered CSV: {tmp_csv}")

    # Make the runner use this CSV by setting DATASET_CSV env var
    # (need to modify runner, see below)

    # Launch 3 workers
    pids = []
    for wid, gpu in enumerate(args.gpus):
        log = f"{args.logdir}/small_w{wid}.log"
        env = os.environ.copy()
        env["TMPDIR"] = "/data/luolie/ToPoGate/tmp"
        env["OPENBLAS_NUM_THREADS"] = "8"
        env["DATASET_CSV"] = str(tmp_csv)
        proc = subprocess.Popen(
            ["python3", "-u", str(RUNNER), "--stage", "1",
             "--gpu_ids", str(gpu), "--worker_id", str(wid),
             "--num_workers", str(len(args.gpus))],
            stdout=open(log, "ab"), stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT), env=env,
        )
        pids.append((wid, proc.pid, gpu))
        print(f"[small_only] Worker {wid} PID={proc.pid} on GPU {gpu}, log={log}")

    print(f"[small_only] All workers launched, waiting...")
    for wid, pid, gpu in pids:
        proc.wait()
        rc = proc.returncode
        print(f"[small_only] Worker {wid} (PID={pid}, GPU={gpu}) finished rc={rc}")


if __name__ == "__main__":
    main()