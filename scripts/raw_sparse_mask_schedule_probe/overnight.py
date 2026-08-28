"""State-machine orchestrator for the bounded overnight probe.

The orchestrator refuses to select a legal GPU that has a foreign process.  It
can therefore finish P0/P1 and leave a truthful ``GPU_WAITING`` state rather
than turning an unavailable resource into a scientific failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import aggregate, benchmark_sparse_compute, protocol, provenance, raw_adapter, run_main


STATE_PATH = protocol.RESULT_ROOT / "OVERNIGHT_STATE.json"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def gpu_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"status": "unavailable", "gpus": [], "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        gpu_cmd = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
        proc = subprocess.run(gpu_cmd, text=True, capture_output=True, check=False, timeout=20)
        if proc.returncode != 0:
            result["error"] = proc.stderr.strip()
            return result
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            idx = int(parts[0])
            app_proc = subprocess.run(["nvidia-smi", "-i", str(idx), "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False, timeout=20)
            app_lines = [app_line.strip() for app_line in app_proc.stdout.splitlines() if app_line.strip()]
            result["gpus"].append({"index": idx, "memory_used_mib": int(float(parts[1])), "memory_total_mib": int(float(parts[2])), "utilization_percent": int(float(parts[3])), "compute_processes_reported": app_lines})
        result["status"] = "completed_valid"
        result["legal_idle_gpus"] = [g["index"] for g in result["gpus"] if g["index"] in protocol.LEGAL_GPU_POOL and not g["compute_processes_reported"] and g["utilization_percent"] <= 5 and g["memory_used_mib"] < 1024]
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def freeze() -> dict[str, Any]:
    protocol.validate_contract()
    protocol.FREEZE_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.MAIN_ROOT.mkdir(parents=True, exist_ok=True)
    for path in (protocol.FIXED_ROOT, protocol.REPR_ROOT, protocol.COMPUTE_ROOT, protocol.FINAL_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    config_path = protocol.FREEZE_ROOT / "resolved_config.json"
    config_path.write_text(json.dumps(protocol.resolved_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance_path = protocol.FREEZE_ROOT / "provenance_manifest.json"
    provenance.write_manifest(provenance_path)
    adapter_path = protocol.FREEZE_ROOT / "adapter_manifest.json"
    raw_adapter.write_adapter_manifest(adapter_path)
    code_paths = sorted((protocol.PROJECT_ROOT / "scripts/raw_sparse_mask_schedule_probe").glob("*.py"))
    test_paths = sorted((protocol.PROJECT_ROOT / "tests/raw_sparse_mask_schedule_probe").glob("*.py"))
    doc_paths = sorted((protocol.REPORT_ROOT).glob("*.md"))
    entries: dict[str, str] = {}
    for path in [config_path, provenance_path, adapter_path, *code_paths, *test_paths, *doc_paths]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries[str(path.relative_to(protocol.PROJECT_ROOT))] = digest
    git = provenance.git_provenance()
    freeze_manifest = {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "plan_version": protocol.PLAN_VERSION, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "git": git, "reproducibility_anchor": "file_hashes_only_unverifiable_git_metadata" if git.get("git_provenance") != "verified" else "git_commit_and_file_hashes", "code_sha256": provenance.code_sha256(), "files_sha256": entries, "gpu_snapshot": gpu_snapshot(), "formal_run_started": False, "labels_loaded": False, "status": "completed_valid"}
    _write(protocol.FREEZE_ROOT / "freeze_manifest.json", freeze_manifest)
    return freeze_manifest


def run_p1() -> dict[str, Any]:
    rows = []
    for dataset in protocol.DATASETS:
        for seed in protocol.SEEDS:
            rows.append(run_main.run_svd(dataset, seed, output_root=protocol.MAIN_ROOT))
    compute = benchmark_sparse_compute.main  # keep import/entrypoint visible for audit
    compute_rows = [benchmark_sparse_compute.benchmark_dataset(dataset) for dataset in protocol.DATASETS]
    _write(protocol.COMPUTE_ROOT / "summary.json", {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "rows": compute_rows, "status": "completed_valid", "labels_used": False})
    return {"svd_rows": len(rows), "compute_rows": len(compute_rows), "status": "completed_valid"}


def _main_queue() -> list[tuple[str, str, int]]:
    # This order is frozen before any result is inspected.
    return [(dataset, arm, seed) for seed in protocol.SEEDS for dataset in protocol.DATASETS for arm in protocol.ARMS]


def queue_assignments(legal_gpus: list[int]) -> dict[int, list[tuple[str, str, int]]]:
    """Deterministic round-robin assignment used by the formal dispatcher."""
    ordered = [int(gpu) for gpu in legal_gpus]
    queue = _main_queue()
    return {gpu: queue[index::len(ordered)] for index, gpu in enumerate(ordered)} if ordered else {}


def dispatch_main(idle_gpus: list[int], *, output_root: Path = protocol.MAIN_ROOT, start_time: float | None = None) -> dict[str, Any]:
    """Run the fixed 90-cell queue with one subprocess per idle legal GPU."""
    import concurrent.futures

    requested = [int(gpu) for gpu in idle_gpus if int(gpu) in protocol.LEGAL_GPU_POOL and int(gpu) not in protocol.FORBIDDEN_GPU_IDS]
    if not requested:
        return {"status": "GPU_WAITING", "launched": 0, "completed": 0, "legal_idle_gpus": [], "occupancy_guard": "no_requested_legal_gpu"}
    # ``idle_gpus`` may have been computed by a caller several seconds earlier.
    # Re-snapshot here so a direct API caller cannot bypass the same occupancy
    # firewall used by ``main --start``.
    dispatch_snapshot = gpu_snapshot()
    observed_idle = sorted(int(gpu) for gpu in dispatch_snapshot.get("legal_idle_gpus", []))
    if any(gpu not in observed_idle for gpu in requested):
        return {
            "status": "GPU_WAITING",
            "launched": 0,
            "completed": 0,
            "requested_gpus": requested,
            "legal_idle_gpus": observed_idle,
            "occupancy_guard": "failed_at_dispatch",
            "gpu_snapshot": dispatch_snapshot,
        }
    legal = requested
    freeze_path = protocol.FREEZE_ROOT / "freeze_manifest.json"
    current_code = provenance.code_sha256()
    if not freeze_path.exists() or json.loads(freeze_path.read_text(encoding="utf-8")).get("code_sha256") != current_code:
        return {"status": "protocol_mismatch", "launched": 0, "completed": 0, "reason": "freeze_code_hash_drift"}
    queue = _main_queue()
    assignments = queue_assignments(legal)
    worker_root = protocol.RESULT_ROOT / "MAIN_WORKERS"
    worker_root.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic() if start_time is None else float(start_time)

    def worker(gpu: int, cells: list[tuple[str, str, int]]) -> dict[str, Any]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        completed = 0
        errors: list[str] = []
        for dataset, arm, seed in cells:
            if time.monotonic() - t0 >= protocol.NEW_LAUNCH_CUTOFF_SECONDS:
                errors.append("new_launch_cutoff")
                break
            # Re-check immediately before each subprocess bind.  This catches
            # a foreign process that appeared after the dispatch snapshot and
            # fails closed without pre-empting it.
            bind_snapshot = gpu_snapshot()
            if gpu not in bind_snapshot.get("legal_idle_gpus", []):
                errors.append(f"{dataset}:{arm}:{seed}:gpu_occupancy_changed_before_launch")
                break
            command = [sys.executable, "-m", "scripts.raw_sparse_mask_schedule_probe.run_main", "--dataset", dataset, "--arm", arm, "--seed", str(seed), "--output-root", str(output_root)]
            log_path = worker_root / f"gpu{gpu}_{dataset.replace(' ', '_')}_{arm}_seed{seed}.log"
            started = time.monotonic()
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.run(command, cwd=protocol.PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=protocol.PER_RUN_TIMEOUT_SECONDS, check=False)
                if proc.returncode != 0:
                    errors.append(f"{dataset}:{arm}:{seed}:returncode={proc.returncode}")
                else:
                    completed += 1
            except subprocess.TimeoutExpired:
                errors.append(f"{dataset}:{arm}:{seed}:timeout")
            if time.monotonic() - t0 >= protocol.HARD_WALL_SECONDS:
                errors.append("hard_wall")
                break
            _ = started
        return {"gpu": gpu, "assigned": len(cells), "completed": completed, "errors": errors}

    with ThreadPoolExecutor(max_workers=len(legal)) as pool:
        futures = [pool.submit(worker, gpu, assignments[gpu]) for gpu in legal]
        reports = [future.result() for future in as_completed(futures)]
    result = {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "status": "completed_valid" if not any(report["errors"] for report in reports) else "incomplete_compute", "legal_idle_gpus": legal, "queue_cells": len(queue), "worker_reports": sorted(reports, key=lambda row: row["gpu"]), "formal_run_started": True}
    _write(protocol.RESULT_ROOT / "MAIN_DISPATCH.json", result)
    return result


def state_snapshot(phase: str, **extra: Any) -> dict[str, Any]:
    value = {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "phase": phase, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), **extra}
    _write(STATE_PATH, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0", action="store_true", help="freeze protocol, adapters, tests, and GPU audit")
    parser.add_argument("--p1", action="store_true", help="run SVD32 and CPU sparse-compute benchmark")
    parser.add_argument("--start", action="store_true", help="run P0/P1 and attempt a guarded MAIN launch")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    if not (args.p0 or args.p1 or args.start):
        parser.error("choose --p0, --p1, or --start")
    if args.p0 or args.start:
        freeze()
        if not args.skip_tests:
            test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/raw_sparse_mask_schedule_probe"], cwd=protocol.PROJECT_ROOT, text=True, capture_output=True, check=False)
            compile_result = subprocess.run([sys.executable, "-m", "compileall", "-q", "scripts/raw_sparse_mask_schedule_probe"], cwd=protocol.PROJECT_ROOT, text=True, capture_output=True, check=False)
            _write(protocol.FREEZE_ROOT / "p0_validation.json", {"pytest_returncode": test.returncode, "pytest_stdout": test.stdout, "pytest_stderr": test.stderr, "compile_returncode": compile_result.returncode, "compile_stdout": compile_result.stdout, "compile_stderr": compile_result.stderr, "status": "completed_valid" if test.returncode == 0 and compile_result.returncode == 0 else "incomplete_compute"})
            if test.returncode != 0 or compile_result.returncode != 0:
                state_snapshot("P0_TEST_FAILURE", status="incomplete_compute")
                return 1
        state_snapshot("P0_COMPLETE", status="completed_valid", gpu=gpu_snapshot())
    if args.p1 or args.start:
        p1 = run_p1()
        state_snapshot("P1_COMPLETE", **p1, gpu=gpu_snapshot())
    if args.start:
        snapshot = gpu_snapshot()
        idle = snapshot.get("legal_idle_gpus", [])
        if not idle:
            state_snapshot("GPU_WAITING", status="blocked_by_external_gpu_occupancy", gpu=snapshot, formal_main_started=False, note="No legal idle GPU; no external process was preempted.")
            print(json.dumps({"status": "GPU_WAITING", "legal_idle_gpus": []}, sort_keys=True))
            return 0
        state_snapshot("MAIN_READY", status="ready", gpu=snapshot, formal_main_started=False)
        dispatch = dispatch_main(idle)
        state_snapshot("MAIN_DISPATCH_COMPLETE", **dispatch, gpu=gpu_snapshot())
        print(json.dumps(dispatch, sort_keys=True))
    else:
        print(json.dumps({"status": "completed_valid"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
