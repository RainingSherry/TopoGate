#!/usr/bin/env python3
"""Run Stage 1 sweep only on medium/large datasets (>=5000 samples).

3 workers, each on dedicated GPU.
"""
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

REPO_ROOT = Path("/home/luolie/ToPoGate")
CSV_PATH = REPO_ROOT / "result" / "dataset_npz_info.csv"
RUNNER = REPO_ROOT / "scripts" / "learnable_gate" / "run_learnable_gate_134_sweep.py"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_n", type=int, default=5000)
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 7, 0])
    ap.add_argument("--logdir", default=str(REPO_ROOT / "result" / "learnable_gate_134_sweep" / "medium_logs"))
    args = ap.parse_args()

    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    # Filter CSV to medium/large
    tmp_csv = REPO_ROOT / "result" / f"dataset_npz_info_medium_{args.min_n}.csv"
    keep = []
    with open(CSV_PATH) as fin, open(tmp_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["n_samples"]) >= args.min_n:
                writer.writerow(row)
                keep.append((row["dataset_name"], int(row["n_samples"])))
    print(f"[medium_only] {len(keep)} datasets with n>={args.min_n}")
    for n, s in keep:
        print(f"  {n}: {s}")
    print(f"[medium_only] CSV: {tmp_csv}")

    pids = []
    for wid, gpu in enumerate(args.gpus):
        log = f"{args.logdir}/medium_w{wid}.log"
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
        print(f"[medium_only] Worker {wid} PID={proc.pid} GPU={gpu}")

    for wid, pid, gpu in pids:
        proc.wait()
        print(f"[medium_only] Worker {wid} (PID={pid}, GPU={gpu}) rc={proc.returncode}")

if __name__ == "__main__":
    main()
