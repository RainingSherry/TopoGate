#!/usr/bin/env python
"""Start the X-only V19 tuning matrix after the formal matrix is complete."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FORMAL_SEEDS = (42, 123, 7)
EXPECTED_RUNS_PER_SEED = 22
ALLOWED_GPUS = (1, 2, 3, 4, 5, 6)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _formal_status(root: Path) -> dict[str, Any]:
    per_seed = {}
    for seed in FORMAL_SEEDS:
        counts: dict[str, int] = {}
        for path in root.glob(f"**/seed{seed}/run_record.json"):
            try:
                status = str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
            except Exception:
                status = "malformed"
            counts[status] = counts.get(status, 0) + 1
        per_seed[str(seed)] = counts
    complete = all(
        per_seed[str(seed)].get("completed", 0) == EXPECTED_RUNS_PER_SEED
        and per_seed[str(seed)].get("incomplete_compute", 0) == 0
        for seed in FORMAL_SEEDS
    )
    return {"complete": complete, "per_seed": per_seed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch V19 X-only tuning after formal completion")
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(ALLOWED_GPUS))
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if any(gpu not in ALLOWED_GPUS for gpu in args.gpus):
        raise ValueError(f"all GPUs must be in {ALLOWED_GPUS}")
    status_path = args.output_dir / "launcher_status.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        formal = _formal_status(args.formal_root)
        _write(
            status_path,
            {
                "status": "waiting_for_formal_matrix" if not formal["complete"] else "starting_tuning",
                "formal": formal,
                "labels_accessed": False,
                "y_key_read": False,
            },
        )
        if formal["complete"]:
            break
        time.sleep(max(5, int(args.poll_seconds)))

    workers = []
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    for worker_id, gpu in enumerate(args.gpus):
        log_path = args.output_dir / f"launcher_worker{worker_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "V19" / "tune_unsupervised.py"),
            "--manifest",
            str(args.manifest),
            "--config",
            str(args.config),
            "--output-dir",
            str(args.output_dir),
            "--seeds",
            *[str(seed) for seed in FORMAL_SEEDS],
            "--gpu",
            str(gpu),
            "--worker-id",
            str(worker_id),
            "--num-workers",
            str(len(args.gpus)),
            "--max-samples",
            str(int(args.max_samples)),
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        workers.append({"worker_id": worker_id, "gpu": gpu, "pid": process.pid, "log": str(log_path), "handle": log_handle, "process": process})
    _write(
        status_path,
        {
            "status": "tuning_running",
            "formal": _formal_status(args.formal_root),
            "workers": [
                {key: value for key, value in worker.items() if key not in {"handle", "process"}}
                for worker in workers
            ],
            "labels_accessed": False,
            "y_key_read": False,
        },
    )
    return_codes = []
    for worker in workers:
        return_codes.append(worker["process"].wait())
        worker["handle"].close()
    if all(code == 0 for code in return_codes):
        _write(
            status_path,
            {
                "status": "summarizing",
                "formal": _formal_status(args.formal_root),
                "return_codes": return_codes,
                "labels_accessed": False,
                "y_key_read": False,
            },
        )
        summary_log = args.output_dir / "summarizer.log"
        with summary_log.open("w", encoding="utf-8") as handle:
            summary_process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "V19" / "summarize_unsupervised_tuning.py"),
                    "--manifest",
                    str(args.manifest),
                    "--output-dir",
                    str(args.output_dir),
                    "--seeds",
                    *[str(seed) for seed in FORMAL_SEEDS],
                ],
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        summary_code = int(summary_process.returncode)
    else:
        summary_code = None
    _write(
        status_path,
        {
            "status": (
                "selection_completed"
                if all(code == 0 for code in return_codes) and summary_code == 0
                else "tuning_incomplete_compute"
            ),
            "formal": _formal_status(args.formal_root),
            "return_codes": return_codes,
            "summary_return_code": summary_code,
            "labels_accessed": False,
            "y_key_read": False,
        },
    )
    return 0 if all(code == 0 for code in return_codes) and summary_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
