#!/usr/bin/env python
"""Launch the fixed second extension panel only when the first panel is insufficient."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})
PRIMARY_SUMMARY = ROOT / "result" / "V19" / "v19_rg_extended_sparse_ari_v1" / "summary" / "extension_summary.json"
PRIMARY_ROOT = ROOT / "result" / "V19" / "v19_rg_extended_sparse_ari_v1"
BATCH2_MANIFEST = ROOT / "result" / "V19" / "v19_rg_extended_sparse_batch2_manifest_20260811.json"
BATCH2_ROOT = ROOT / "result" / "V19" / "v19_rg_extended_sparse_batch2_ari_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _free_mib() -> dict[int, int]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required for batch-2 launch")
    result = subprocess.run(
        [executable, "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    free: dict[int, int] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            index, used, total = (int(field) for field in fields)
            free[index] = total - used
    return free


def _active_workers() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,cmd="], check=False, capture_output=True, text=True)
    needles = (
        "launch_extended_after_ari.py",
        "run_extended_matrix.py",
        "summarize_extended_matrix.py",
        "run_extended_winner_baselines.py",
    )
    own = str(os.getpid())
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if own not in line and any(needle in line for needle in needles)
    ]


def _wait_for(path: Path, seconds: int) -> dict[str, Any]:
    while not path.is_file():
        time.sleep(max(5, seconds))
    return _read(path)


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"batch-2 launcher lock already exists: {path}") from exc


def _launch(
    *, manifest: Path, output: Path, config: Path, gpus: list[int], log_dir: Path
) -> tuple[dict[str, int], int | None]:
    processes: list[tuple[int, subprocess.Popen[str], Any]] = []
    for worker_id, gpu in enumerate(gpus):
        env = dict(os.environ)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        handle = (log_dir / f"worker{worker_id}_gpu{gpu}.log").open("a", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "V19" / "run_extended_matrix.py"),
            "--manifest", str(manifest), "--output-dir", str(output), "--config", str(config),
            "--worker-id", str(worker_id), "--num-workers", str(len(gpus)), "--gpu", str(gpu),
        ]
        processes.append((worker_id, subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
    exit_codes: dict[str, int] = {}
    for worker_id, process, handle in processes:
        exit_codes[str(worker_id)] = int(process.wait())
        handle.close()
    summary_command = [
        sys.executable, str(ROOT / "scripts" / "V19" / "summarize_extended_matrix.py"),
        "--manifest", str(manifest), "--result-dir", str(output), "--output-dir", str(output / "summary"),
    ]
    summary_result = subprocess.run(summary_command, cwd=ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (log_dir / "summarize.log").write_bytes(summary_result.stdout)
    summary_code = int(summary_result.returncode)
    return exit_codes, summary_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-summary", type=Path, default=PRIMARY_SUMMARY)
    parser.add_argument("--primary-root", type=Path, default=PRIMARY_ROOT)
    parser.add_argument("--manifest", type=Path, default=BATCH2_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=BATCH2_ROOT)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--min-free-mib", type=int, default=30000)
    parser.add_argument("--wait-seconds", type=int, default=30)
    args = parser.parse_args()
    gpus = [int(gpu) for gpu in args.gpus]
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu not in ALLOWED_GPUS for gpu in gpus):
        raise ValueError(f"GPU pool must be distinct physical GPUs in {sorted(ALLOWED_GPUS)}")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    summary = _wait_for(args.primary_summary, int(args.wait_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "batch2_launcher_status.json"
    winners = [row for row in summary.get("datasets", []) if row.get("promotion_rg_win_by_mean_ari") is True]
    if summary.get("audit_ok") is not True:
        _write(status_path, {"status": "blocked_incomplete_primary", "primary_summary": str(args.primary_summary.resolve())})
        return 2
    if len(winners) >= 5:
        _write(status_path, {"status": "not_activated_primary_met", "n_primary_winners": len(winners), "minimum_required": 5})
        return 0
    while _active_workers():
        time.sleep(max(5, int(args.wait_seconds)))
    config = args.primary_root / "resolved_ari_transfer_config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"primary transfer config is missing: {config}")
    free = _free_mib()
    eligible = [gpu for gpu in gpus if int(free.get(gpu, 0)) >= int(args.min_free_mib)]
    if not eligible:
        raise RuntimeError(f"no requested GPU has {args.min_free_mib} MiB free: {free}")
    lock_fd = _acquire_lock(args.output_dir / "batch2_launcher.lock")
    try:
        _write(args.output_dir / "batch2_launch_spec.json", {
            "protocol_id": "v19_rg_extended_sparse_batch2_v1",
            "manifest": str(args.manifest.resolve()),
            "config": str(config.resolve()),
            "primary_summary": str(args.primary_summary.resolve()),
            "primary_winner_count": len(winners),
            "gpus_requested": gpus,
            "gpus_selected": eligible,
            "free_mib_at_launch": {str(gpu): int(free.get(gpu, 0)) for gpu in eligible},
            "activation_rule": "complete second fixed panel only when primary has fewer than five mean-ARI RG wins",
        })
        _write(status_path, {"status": "running", "gpus": eligible, "primary_winner_count": len(winners), "started_at": time.time()})
        log_dir = args.output_dir / "launcher_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        exit_codes, summary_code = _launch(manifest=args.manifest, output=args.output_dir, config=config, gpus=eligible, log_dir=log_dir)
        completed = all(code == 0 for code in exit_codes.values()) and summary_code == 0
        _write(status_path, {"status": "completed" if completed else "incomplete_compute", "gpus": eligible, "exit_codes": exit_codes, "summary_exit_code": summary_code, "finished_at": time.time()})
        return 0 if completed else 1
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
