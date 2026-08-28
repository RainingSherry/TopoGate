"""Shared-resource execution protocol v2 for raw_sparse_mask_schedule_probe.

This is an explicit protocol amendment.  It does not rewrite v1 artifacts or
change the model, data, masks, seeds, or estimands.  The amendment removes only
the user-selected resource gates E1--E4 and E8--E12:

* legal GPUs may be shared with foreign workloads;
* dispatch does not require an idle snapshot and does not re-check occupancy;
* the shared run is eligible for v2 aggregation when its own row audits pass;
* there is no v1 hard-wall/new-launch cutoff or one-shot retry gate;
* v2 can be launched without rerunning the v1 P0/P1 preflight gate.

GPU 0/7 remain forbidden and no foreign process is ever killed or preempted.
The model runner and label firewall are imported unchanged from the v1 code.
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

from scripts.raw_sparse_mask_schedule_probe import aggregate, overnight, protocol, provenance, run_main


V2_PROTOCOL_ID = "raw_sparse_mask_schedule_probe_v2_shared"
V2_PLAN_VERSION = "raw_sparse_mask_schedule_probe_shared_v2"
V2_ROOT = protocol.RESULT_ROOT / "V2_SHARED"
V2_MAIN_ROOT = V2_ROOT / "MAIN"
V2_FREEZE_ROOT = V2_ROOT / "FREEZE"
V2_REPORT_ROOT = protocol.REPORT_ROOT / "v2_shared"
V2_FINAL_ROOT = V2_ROOT / "FINAL"
V2_WORKER_ROOT = V2_ROOT / "WORKERS"
V2_ALLOWED_GPUS = tuple(protocol.LEGAL_GPU_POOL)


def _launcher_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _patch_protocol() -> None:
    """Patch only process-local protocol paths/identity for the v2 child."""
    protocol.PROTOCOL_ID = V2_PROTOCOL_ID
    protocol.PLAN_VERSION = V2_PLAN_VERSION
    protocol.MAIN_ROOT = V2_MAIN_ROOT
    protocol.FREEZE_ROOT = V2_FREEZE_ROOT
    protocol.FIXED_ROOT = V2_ROOT / "FIXED_RATIO_ORACLE"
    protocol.REPR_ROOT = V2_ROOT / "REPR_LOCALIZATION"
    protocol.COMPUTE_ROOT = V2_ROOT / "COMPUTE"
    protocol.FINAL_ROOT = V2_FINAL_ROOT
    # E8 is removed.  Keep the legacy constants numerically valid for the
    # imported contract validator; the v2 dispatcher never consults them.
    protocol.NEW_LAUNCH_CUTOFF_SECONDS = 365 * 24 * 3600
    protocol.HARD_WALL_SECONDS = 366 * 24 * 3600


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _prepare() -> dict[str, Any]:
    _patch_protocol()
    protocol.validate_contract()
    for path in (V2_ROOT, V2_MAIN_ROOT, V2_FREEZE_ROOT, V2_REPORT_ROOT, V2_FINAL_ROOT, V2_WORKER_ROOT):
        path.mkdir(parents=True, exist_ok=True)

    code_hash = provenance.code_sha256()
    manifest = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": V2_PROTOCOL_ID,
        "plan_version": V2_PLAN_VERSION,
        "execution_mode": "shared_resource_allowed",
        "resource_policy": {
            "shared_gpu_allowed": True,
            "idle_gpu_required": False,
            "foreign_process_preemption": False,
            "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
            "legal_gpu_pool": list(V2_ALLOWED_GPUS),
            "occupancy_rechecks": False,
            "hard_wall_enforced": False,
            "p0_p1_gate_required": False,
        },
        "cleared_user_gates": ["E1", "E2", "E3", "E4", "E8", "E9", "E10", "E11", "E12"],
        "code_sha256": code_hash,
        "launcher_sha256": _launcher_hash(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "formal_run_started": False,
        "status": "completed_valid",
        "labels_loaded": False,
        "v1_reference": str(protocol.RESULT_ROOT / "FREEZE" / "freeze_manifest.json"),
    }
    _write_json(V2_FREEZE_ROOT / "freeze_manifest.json", manifest)
    _write_json(V2_FREEZE_ROOT / "resolved_config.json", {
        **protocol.resolved_config(),
        "protocol_id": V2_PROTOCOL_ID,
        "plan_version": V2_PLAN_VERSION,
        "execution_mode": "shared_resource_allowed",
        "shared_gpu_allowed": True,
        "idle_gpu_required": False,
        "occupancy_rechecks": False,
        "hard_wall_enforced": False,
        "p0_p1_gate_required": False,
        "cleared_user_gates": list(manifest["cleared_user_gates"]),
    })
    _write_json(V2_FREEZE_ROOT / "provenance_manifest.json", {
        **provenance.build_manifest(),
        "protocol_id": V2_PROTOCOL_ID,
        "launcher_sha256": _launcher_hash(),
        "execution_mode": "shared_resource_allowed",
    })

    # SVD32 is a separate unchanged baseline.  Reuse the already completed v1
    # rows rather than spending another CPU half-day on an identical transform.
    svd_source = protocol.RESULT_ROOT / "MAIN" / "SVD32"
    svd_link = V2_MAIN_ROOT / "SVD32"
    if svd_source.exists() and not svd_link.exists():
        svd_link.symlink_to(svd_source, target_is_directory=True)
    _write_json(V2_ROOT / "SVD_REUSE_MANIFEST.json", {
        "reused_from": str(svd_source),
        "reason": "same audited X0, SVD32 dimension, seeds, and post-fit readout; v2 changes resource policy only",
        "source_code_sha256": json.loads((protocol.RESULT_ROOT / "FREEZE" / "freeze_manifest.json").read_text(encoding="utf-8")).get("code_sha256"),
        "status": "completed_valid" if svd_source.exists() else "missing_source",
    })
    return manifest


def _run_cell(dataset: str, arm: str, seed: int, gpu: int, epochs: int | None = None) -> dict[str, Any]:
    _patch_protocol()
    gpu = int(gpu)
    if gpu not in V2_ALLOWED_GPUS or gpu in protocol.FORBIDDEN_GPU_IDS:
        raise ValueError(f"illegal v2 GPU: {gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    summary = run_main._run_one(dataset, arm, int(seed), output_root=V2_MAIN_ROOT, use_cpu=False, epochs=epochs)
    run_dir = V2_MAIN_ROOT / dataset / arm / f"seed{int(seed)}"
    # Add explicit resource-mode fields without changing the model result or
    # overriding a failed/incomplete audit from the child runner.
    summary = {**summary, "resource_mode": "shared_resource_allowed", "shared_gpu_allowed": True}
    run_main.write_json_atomic(run_dir / "summary.json", summary)
    audit_path = run_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    audit.update({"resource_mode": "shared_resource_allowed", "shared_gpu_allowed": True})
    if "audit_ok" not in audit:
        audit["audit_ok"] = bool(summary.get("audit_ok", False))
    run_main.write_json_atomic(audit_path, audit)
    return summary


def _cell_command(dataset: str, arm: str, seed: int, gpu: int, epochs: int | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.raw_sparse_mask_schedule_v2_shared",
        "--run-cell",
        "--dataset",
        dataset,
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
    ]
    if epochs is not None:
        command.extend(["--epochs", str(int(epochs))])
    return command


def _run_slot(gpu: int, slot: int, cells: list[tuple[str, str, int]], *, attempts: int, epochs: int | None) -> dict[str, Any]:
    V2_WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures: list[str] = []
    for dataset, arm, seed in cells:
        attempt = 0
        while True:
            attempt += 1
            log_path = V2_WORKER_ROOT / f"gpu{gpu}_slot{slot}_{dataset.replace(' ', '_')}_{arm}_seed{seed}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            command = _cell_command(dataset, arm, seed, gpu, epochs)
            returncode = 1
            failure_text = ""
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n=== attempt {attempt} started {dt.datetime.now(dt.timezone.utc).isoformat()} ===\n")
                try:
                    proc = subprocess.run(command, cwd=str(protocol.PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=protocol.PER_RUN_TIMEOUT_SECONDS, check=False)
                    returncode = int(proc.returncode)
                except subprocess.TimeoutExpired as exc:
                    failure_text = f"TimeoutExpired: {exc}"
                    log.write(f"{failure_text}\n")
                    returncode = 124
                except Exception as exc:  # pragma: no cover - defensive worker boundary
                    failure_text = f"{type(exc).__name__}: {exc}"
                    log.write(f"{failure_text}\n")
                    returncode = 1
                log.write(f"=== attempt {attempt} returncode={returncode} ended {dt.datetime.now(dt.timezone.utc).isoformat()} ===\n")
            if returncode == 0:
                completed += 1
                break
            summary_path = V2_MAIN_ROOT / dataset / arm / f"seed{int(seed)}" / "summary.json"
            if summary_path.exists():
                try:
                    failure_text = f"{failure_text} {summary_path.read_text(encoding='utf-8')}"
                except OSError:
                    pass
            failures.append(f"{dataset}:{arm}:{seed}:attempt={attempt}:returncode={returncode}")
            # Integrity failures are terminal for this cell.  They remain
            # incomplete_compute and are never turned into a success by retry.
            non_retryable_tokens = ("nan", "shape mismatch", "label leakage", "protocol mismatch", "assertion")
            if any(token in failure_text.lower() for token in non_retryable_tokens):
                break
            # attempts=0 means no fixed retry cap (E9 cleared).  The operator
            # can still stop the dispatcher explicitly. E10 is cleared at the
            # scheduler boundary: a later explicit invocation may choose a
            # different retry/resource configuration; this cell is not silently
            # relabeled as equivalent.
            if attempts > 0 and attempt >= attempts:
                break
    return {"gpu": gpu, "slot": slot, "assigned": len(cells), "completed": completed, "failures": failures}


def _dispatch(gpus: tuple[int, ...], workers_per_gpu: int, attempts: int, epochs: int | None) -> dict[str, Any]:
    manifest = _prepare()
    if not gpus:
        raise ValueError("at least one legal GPU is required")
    if any(gpu not in V2_ALLOWED_GPUS or gpu in protocol.FORBIDDEN_GPU_IDS for gpu in gpus):
        raise ValueError(f"GPU list contains forbidden/illegal id: {gpus}")
    queue = overnight._main_queue()
    slots = [(gpu, slot) for gpu in gpus for slot in range(int(workers_per_gpu))]
    assignments = {slot: queue[index::len(slots)] for index, slot in enumerate(slots)}
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(_run_slot, gpu, slot, assignments[(gpu, slot)], attempts=attempts, epochs=epochs) for gpu, slot in slots]
        reports = [future.result() for future in as_completed(futures)]
    failures = [failure for report in reports for failure in report["failures"]]
    result = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": V2_PROTOCOL_ID,
        "execution_mode": "shared_resource_allowed",
        "resource_policy": manifest["resource_policy"],
        "started_at": started,
        "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpus": list(gpus),
        "workers_per_gpu": int(workers_per_gpu),
        "attempts_policy": "unbounded_until_success" if attempts == 0 else f"up_to_{attempts}",
        "queue_cells": len(queue),
        "completed": sum(report["completed"] for report in reports),
        "failures": failures,
        "worker_reports": sorted(reports, key=lambda row: (row["gpu"], row["slot"])),
        "formal_run_started": True,
        "status": "completed_valid" if not failures and sum(report["completed"] for report in reports) == len(queue) else "incomplete_compute",
    }
    _write_json(V2_ROOT / "MAIN_DISPATCH.json", result)
    return result


def _aggregate() -> dict[str, Any]:
    _patch_protocol()
    bundle = aggregate.collect(V2_MAIN_ROOT)
    evaluation = aggregate.evaluate(bundle)
    aggregate.write_outputs(bundle, evaluation, V2_REPORT_ROOT, V2_FINAL_ROOT)
    return {"status": evaluation["status"], "decision": evaluation["decision"], "g0": evaluation["g0"]["passed"], "rows": len(bundle["rows"]), "svd_rows": len(bundle["svd_rows"])}


def main() -> int:
    parser = argparse.ArgumentParser(description="raw sparse mask shared-resource protocol v2")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--run-cell", action="store_true")
    parser.add_argument("--dataset", choices=protocol.DATASETS)
    parser.add_argument("--arm", choices=protocol.ARMS)
    parser.add_argument("--seed", type=int, choices=protocol.SEEDS)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--gpus", default=",".join(str(gpu) for gpu in V2_ALLOWED_GPUS))
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=0, help="0=retry until success; positive value caps retries")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    _patch_protocol()
    if args.prepare:
        print(json.dumps(_prepare(), indent=2, sort_keys=True))
        return 0
    if args.run_cell:
        if args.dataset is None or args.arm is None or args.seed is None or args.gpu is None:
            parser.error("--run-cell requires --dataset --arm --seed --gpu")
        try:
            value = _run_cell(args.dataset, args.arm, args.seed, args.gpu, args.epochs)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"status": value.get("status"), "dataset": args.dataset, "arm": args.arm, "seed": args.seed, "gpu": args.gpu}, sort_keys=True))
        return 0
    if args.dispatch:
        gpus = tuple(int(value.strip()) for value in args.gpus.split(",") if value.strip())
        value = _dispatch(gpus, max(1, int(args.workers_per_gpu)), int(args.attempts), args.epochs)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value["status"] == "completed_valid" else 1
    if args.aggregate:
        print(json.dumps(_aggregate(), indent=2, sort_keys=True))
        return 0
    parser.error("choose --prepare, --dispatch, --aggregate, or --run-cell")


if __name__ == "__main__":
    raise SystemExit(main())
