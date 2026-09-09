#!/usr/bin/env python3
"""Resumable, GPU-fenced dispatcher for the frozen CLUBench panel."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ALLOWED_GPUS = (1, 2, 3, 5)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_ids(manifest: Path) -> list[str]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))["datasets"]
    ids = [str(row["dataset_id"]) for row in rows if row.get("panel") == "clubench"]
    if len(ids) != 131 or len(set(ids)) != len(ids):
        raise ValueError(f"expected 131 unique CLUBench ids, got {len(ids)}")
    return ids


def run_one(args: argparse.Namespace, dataset_id: str, gpu: int) -> dict:
    out = args.output / dataset_id
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return {"dataset_id": dataset_id, "status": "locked"}
    os.close(fd)
    try:
        state = out / f"state_{args.stage}.json"
        if state.exists() and json.loads(state.read_text()).get("status") == "completed":
            return {"dataset_id": dataset_id, "status": "reused"}
        previous = sorted(out.glob("attempt_*.log"))
        first_attempt = len(previous) + 1
        for attempt in range(first_attempt, args.max_attempts + 1):
            attempt_dir = out / f"attempt_{attempt}"
            log = attempt_dir.with_suffix(".log")
            attempt_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(state, {"dataset_id": dataset_id, "stage": args.stage, "status": "running", "attempt": attempt, "gpu": gpu, "started": time.time()})
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [args.python, "-m", "methods.TopoGate.V0_RG.tuning", "--manifest", str(args.manifest), "--output-dir", str(args.output), "--dataset-id", dataset_id, "--stage", args.stage, "--device", "cuda", "--gpu", "0", "--epochs", str(args.epochs), "--trials", str(args.trials)]
            with log.open("w", encoding="utf-8") as handle:
                rc = subprocess.run(cmd, cwd=args.repo, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
            status = "completed" if rc == 0 else "failed"
            atomic_json(state, {"dataset_id": dataset_id, "stage": args.stage, "status": status, "attempt": attempt, "gpu": gpu, "returncode": rc, "log": str(log), "finished": time.time()})
            if rc == 0:
                return {"dataset_id": dataset_id, "status": status, "returncode": rc, "attempt": attempt}
        return {"dataset_id": dataset_id, "status": "failed", "returncode": rc, "attempt": args.max_attempts}
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--stage", choices=("search", "final"), required=True)
    p.add_argument("--gpu", type=int, choices=ALLOWED_GPUS, required=True)
    p.add_argument("--dataset-id")
    p.add_argument("--shard-index", type=int, choices=range(len(ALLOWED_GPUS)))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--max-attempts", type=int, default=3)
    args = p.parse_args()
    ids = load_ids(args.manifest)
    if args.shard_index is not None:
        if args.gpu != ALLOWED_GPUS[args.shard_index]:
            raise SystemExit("shard-index must map to its fixed physical GPU")
        ids = ids[args.shard_index::len(ALLOWED_GPUS)]
    if args.dataset_id:
        ids = [x for x in ids if x == args.dataset_id]
        if not ids:
            raise SystemExit("dataset-id is not in the CLUBench manifest")
    results = [run_one(args, dataset_id, args.gpu) for dataset_id in ids]
    atomic_json(args.output / f"dispatcher_{args.stage}_gpu{args.gpu}.json", {"stage": args.stage, "gpu": args.gpu, "results": results})
    if any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
