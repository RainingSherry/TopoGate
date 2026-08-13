#!/usr/bin/env python
"""Launch one V19 v2 tuning/reference stage with explicit GPU isolation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_GPUS = (1, 2, 3, 4, 5, 6)
FORMAL_SEEDS = (42, 123, 7)
FORMAL_RG_STAGES = {"mechanism_screen", "mechanism_refine"}


def _acquire_launcher_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "launcher.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another launcher already owns {lock_path}") from exc
    return handle


def _active_pid(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _reject_active_previous_launcher(output_dir: Path) -> None:
    status_path = output_dir / "launcher_status.json"
    if not status_path.is_file():
        return
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if status.get("status") != "running":
        return
    active = [
        int(worker["pid"])
        for worker in status.get("workers", [])
        if _active_pid(int(worker.get("pid", -1)))
    ]
    if active:
        raise RuntimeError(
            f"output root already has active workers {active}; refuse concurrent resume: {output_dir}"
        )
    active_runs = []
    for path in output_dir.rglob("status.json"):
        if path == status_path:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") == "running":
            active_runs.append(str(path))
    if active_runs:
        raise RuntimeError(
            f"output root contains running run records; refuse resume until audited: {active_runs[:3]}"
        )


def _gpu_compute_processes() -> dict[int, list[dict[str, str]]]:
    try:
        gpu_rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        app_rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify GPU occupancy with nvidia-smi") from exc
    uuid_to_index = {}
    for row in gpu_rows:
        fields = [value.strip() for value in row.split(",", 1)]
        if len(fields) == 2:
            uuid_to_index[fields[1]] = int(fields[0])
    occupied: dict[int, list[dict[str, str]]] = {}
    for row in app_rows:
        fields = [value.strip() for value in row.split(",", 3)]
        if len(fields) != 4 or fields[0] not in uuid_to_index:
            continue
        gpu = uuid_to_index[fields[0]]
        occupied.setdefault(gpu, []).append(
            {"pid": fields[1], "process_name": fields[2], "used_memory": fields[3]}
        )
    return occupied


def _assert_gpus_available(
    gpus: list[int], *, allow_shared: bool = False
) -> dict[int, list[dict[str, str]]]:
    if gpus == [-1]:
        return {}
    occupied = _gpu_compute_processes()
    busy = {gpu: occupied[gpu] for gpu in gpus if gpu in occupied}
    if busy and not allow_shared:
        raise RuntimeError(f"requested GPUs are occupied by external compute: {busy}")
    return busy


def _validate_formal_launch(args: argparse.Namespace) -> None:
    if args.kind != "tuning":
        if args.max_samples != 0:
            raise ValueError("formal reference requires --max-samples 0")
        return
    if args.stage not in FORMAL_RG_STAGES:
        raise ValueError(
            "RG-only V19 v2 formal tuning permits mechanism_screen and mechanism_refine only"
        )
    if args.groups:
        raise ValueError("formal V19 v2 launch does not permit --groups")
    if args.max_samples != 0 or args.force:
        raise ValueError("formal V19 v2 launch requires --max-samples 0 and forbids --force")
    if args.stage == "mechanism_screen":
        if tuple(args.seeds) != (42,) or not args.comparable_only:
            raise ValueError("mechanism_screen requires seed 42 and --comparable-only")
        if args.candidate_ids or args.selected_config is not None:
            raise ValueError("mechanism_screen does not accept selected candidates/config")
    else:
        if tuple(args.seeds) != FORMAL_SEEDS or args.comparable_only:
            raise ValueError("mechanism_refine requires seeds 42,123,7 and all 11 layers")
        if args.selected_config is not None or not args.candidate_ids or len(args.candidate_ids) != 12:
            raise ValueError("mechanism_refine requires exactly 12 candidate ids from screen selection")


def _audit_stage_output(
    output_dir: Path,
    expected_runs: int,
    stage_spec: dict[str, Any],
) -> tuple[bool, str]:
    forbidden = [
        path
        for name in ("labels_true.npy", "predictions.npy", "metrics.json")
        for path in output_dir.rglob(name)
    ]
    if forbidden:
        return False, f"label-derived artifacts found: {[str(path) for path in forbidden[:3]]}"
    summaries: dict[str, Path] = {}
    incomplete = []
    expected_keys = {
        f"{stage_spec['stage']}::{dataset_id}::{candidate_id}::seed{int(seed)}"
        for dataset_id in stage_spec["dataset_ids"]
        for candidate_id in stage_spec["candidate_ids"]
        for seed in stage_spec["seeds"]
    }
    if len(expected_keys) != int(expected_runs):
        return False, "stage_spec expected key cardinality mismatch"
    for path in output_dir.rglob("summary.json"):
        if "attempts" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") == "incomplete_compute":
            incomplete.append(str(path))
        if payload.get("status") != "completed":
            continue
        if payload.get("protocol_id") != stage_spec.get("protocol_id"):
            return False, f"protocol audit failed: {path}"
        if payload.get("labels_accessed") is not False or payload.get("y_key_read") is not False:
            return False, f"label audit failed: {path}"
        if payload.get("n_clusters_used") is not None or payload.get("readout_enabled") is not False:
            return False, f"readout audit failed: {path}"
        run_key = str(payload.get("run_key", ""))
        if not run_key or run_key not in expected_keys or run_key in summaries:
            return False, f"duplicate or missing run key: {path}"
        required = (
            "status.json",
            "run_record.json",
            "resolved_config.json",
            "input_profile.json",
            "unsupervised_diagnostics.json",
        )
        if any(not (path.parent / name).is_file() for name in required):
            return False, f"incomplete artifact contract: {path}"
        status_path = path.parent / "status.json"
        record_path = path.parent / "run_record.json"
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        record_payload = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            status_payload.get("status") != "completed"
            or status_payload.get("run_key") != run_key
            or record_payload.get("status") != "completed"
            or record_payload.get("run_key") != run_key
        ):
            return False, f"status/run_record mismatch: {path}"
        summaries[run_key] = path
    if incomplete:
        return False, f"incomplete summaries found: {incomplete[:3]}"
    if set(summaries) != expected_keys:
        missing = sorted(expected_keys - set(summaries))
        extra = sorted(set(summaries) - expected_keys)
        return False, f"expected key audit failed; missing={missing[:3]} extra={extra[:3]}"
    if len(summaries) != int(expected_runs):
        return False, f"completed summary count {len(summaries)} != expected {expected_runs}"
    for path in output_dir.rglob("run_record.json"):
        if "attempts" in path.parts:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            return False, f"non-completed run record remains: {path}"
    return True, "ok"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a V19 v2 label-free stage")
    parser.add_argument("--kind", choices=("reference", "tuning"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("mechanism_screen", "mechanism_refine", "backbone_screen", "joint_refine"))
    parser.add_argument("--selected-config", type=Path, default=None)
    parser.add_argument("--candidate-ids", nargs="*", default=None)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--comparable-only", action="store_true")
    parser.add_argument("--mechanism-count", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument(
        "--schedule",
        choices=("manifest", "small_first"),
        default="small_first",
        help="queue order for tuning workers; small_first defers large source matrices",
    )
    parser.add_argument("--gpus", type=int, nargs="*", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="explicitly allow coexistence with already-running external GPU processes",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.kind == "tuning" and args.stage is None:
        raise ValueError("tuning launch requires --stage")
    if any(int(seed) not in FORMAL_SEEDS for seed in args.seeds):
        raise ValueError(f"seeds must be drawn from {FORMAL_SEEDS}")
    _validate_formal_launch(args)
    if args.cpu:
        gpus = [-1]
    else:
        gpus = [int(gpu) for gpu in (args.gpus or [])]
        if not gpus:
            raise ValueError("provide at least one confirmed free GPU with --gpus, or use --cpu")
        if any(gpu not in ALLOWED_GPUS for gpu in gpus):
            raise ValueError(f"GPU pool must be drawn from {ALLOWED_GPUS}")
        if len(set(gpus)) != len(gpus):
            raise ValueError("GPU pool contains duplicates")
    shared_processes = _assert_gpus_available(gpus, allow_shared=bool(args.allow_shared_gpu))

    _reject_active_previous_launcher(args.output_dir)
    lock_handle = _acquire_launcher_lock(args.output_dir)

    script = (
        ROOT / "scripts" / "V19" / "run_scmae_reference_v2.py"
        if args.kind == "reference"
        else ROOT / "scripts" / "V19" / "tune_unsupervised_v2.py"
    )
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        status_path = args.output_dir / "launcher_status.json"
        workers: list[dict[str, Any]] = []
        for worker_id, gpu in enumerate(gpus):
            log_path = args.output_dir / f"launcher_worker{worker_id}.log"
            command = [
            sys.executable,
            str(script),
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(args.output_dir),
            "--config",
            str(args.config),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--worker-id",
            str(worker_id),
            "--num-workers",
            str(len(gpus)),
            "--max-samples",
            str(int(args.max_samples)),
        ]
            if gpu < 0:
                command.append("--cpu")
            else:
                command.extend(["--gpu", str(gpu)])
            if args.groups:
                command.extend(["--groups", *[str(value) for value in args.groups]])
            if args.comparable_only:
                command.append("--comparable-only")
            if args.kind == "tuning":
                command.extend(["--stage", str(args.stage)])
                command.extend(["--schedule", str(args.schedule)])
                if args.selected_config is not None:
                    command.extend(["--selected-config", str(args.selected_config)])
                if args.candidate_ids:
                    command.extend(["--candidate-ids", *[str(value) for value in args.candidate_ids]])
                command.extend(["--mechanism-count", str(int(args.mechanism_count))])
            if args.force:
                command.append("--force")
            log_handle = log_path.open("w", encoding="utf-8")
            worker_env = dict(os.environ)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                worker_env[name] = "1"
            if gpu >= 0:
                worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=worker_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            workers.append({"worker_id": worker_id, "gpu": gpu, "pid": process.pid, "log": str(log_path), "process": process, "handle": log_handle})
        _write(
            status_path,
            {
                "status": "running",
                "kind": args.kind,
                "stage": args.stage,
                "schedule": args.schedule,
                "workers": [{key: value for key, value in row.items() if key not in {"process", "handle"}} for row in workers],
                "resource_mode": "shared_allowed" if shared_processes else "exclusive_or_cpu",
                "external_compute_at_launch": shared_processes,
                "labels_accessed": False,
                "y_key_read": False,
            },
        )
        return_codes = []
        for worker in workers:
            return_codes.append(int(worker["process"].wait()))
            worker["handle"].close()
        spec_path = args.output_dir / "stage_spec.json"
        expected_runs = None
        if spec_path.is_file():
            expected_runs = int(json.loads(spec_path.read_text(encoding="utf-8")).get("expected_runs", -1))
        audit_ok = expected_runs is not None
        audit_message = "missing stage_spec.json"
        if audit_ok:
            stage_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            audit_ok, audit_message = _audit_stage_output(args.output_dir, expected_runs, stage_spec)
        success = all(code == 0 for code in return_codes) and audit_ok
        _write(
            status_path,
            {
                "status": "completed" if success else "incomplete_compute",
                "kind": args.kind,
                "stage": args.stage,
                "schedule": args.schedule,
                "return_codes": return_codes,
                "workers": [{key: value for key, value in row.items() if key not in {"process", "handle"}} for row in workers],
                "resource_mode": "shared_allowed" if shared_processes else "exclusive_or_cpu",
                "external_compute_at_launch": shared_processes,
                "audit_ok": bool(audit_ok),
                "audit_message": audit_message,
                "labels_accessed": False,
                "y_key_read": False,
            },
        )
        return 0 if success else 1
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
