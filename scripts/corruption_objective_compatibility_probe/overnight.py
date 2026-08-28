"""Unattended E0 -> E4 controller for the independent objective probe.

The controller is intentionally conservative: it only reuses hash-checked
closed controls, launches one process per legal GPU, retries one transient
failure with the exact same command, and converts every timeout or incomplete
job into an explicit ``incomplete_compute`` record.  It never changes the
frozen model, objective, budget, or dataset list in response to results.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import analysis, e0_integrity, e3_raw_audit, protocol


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _job_key(job: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return (job["dataset"], job["arm"], job["objective"], int(job["seed"]), job["stage"])


def _job_dir(root: Path, job: dict[str, Any]) -> Path:
    if job["stage"] == "E1":
        return root / job["dataset"] / job["arm"] / f"seed{job['seed']}"
    return root / job["dataset"] / job["arm"] / job["objective"] / f"seed{job['seed']}"


def _source_run_valid(job: dict[str, Any], root: Path) -> bool:
    return analysis.existing_valid_run(
        _job_dir(root, job),
        dataset=job["dataset"],
        arm=job["arm"],
        objective=job["objective"],
        seed=int(job["seed"]),
        stage=job["stage"],
    )


def _nofit_valid(path: Path, dataset: str, arm: str, seed: int) -> bool:
    try:
        summary = analysis.read_json(path / "summary.json")
        audit = analysis.read_json(path / "audit.json")
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        audit.get("audit_ok") is True
        and summary.get("status") == "completed_valid"
        and summary.get("stage") == "E1b_nofit"
        and summary.get("dataset") == dataset
        and summary.get("arm") == arm
        and int(summary.get("seed", -1)) == int(seed)
        and analysis._source_matches(summary, dataset)
    )


def _cuda_preflight() -> dict[str, Any]:
    """Check that the requested physical GPU pool is visible and legal."""

    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "preflight_unavailable", "error": str(exc), "gpu_runs_started": 0}
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpu_id = int(parts[0])
            memory = int(parts[2])
            memory_used = int(parts[3])
            utilization = int(parts[4])
        except ValueError:
            continue
        rows.append({"id": gpu_id, "name": parts[1], "memory_mib": memory, "memory_used_mib": memory_used, "utilization_pct": utilization})
    visible = {row["id"] for row in rows}
    legal_visible = set(protocol.GPU_POOL).issubset(visible)
    idle_legal_pool = sorted(
        row["id"] for row in rows
        if row["id"] in set(protocol.GPU_POOL)
        and row.get("utilization_pct", 100) == 0
        and row.get("memory_used_mib", 10**9) <= 1024
    )
    # nvidia-smi normally enumerates all physical devices, including the
    # forbidden ones.  That is not a violation; the launch environment is what
    # must exclude them.  If the caller supplied a visibility list, validate
    # that list explicitly.
    requested_text = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        requested = {int(item.strip()) for item in requested_text.split(",") if item.strip()}
    except ValueError:
        requested = set()
    forbidden_requested = sorted(requested & set(protocol.FORBIDDEN_GPU_IDS))
    return {
        "status": "ready" if completed.returncode == 0 and legal_visible and bool(idle_legal_pool) and not forbidden_requested else "failed",
        "returncode": completed.returncode,
        "devices": rows,
        "required_legal_pool": list(protocol.GPU_POOL),
        "forbidden_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "forbidden_visible": sorted(visible & set(protocol.FORBIDDEN_GPU_IDS)),
        "forbidden_requested": forbidden_requested,
        "legal_pool_visible": legal_visible,
        "idle_legal_pool": idle_legal_pool,
        "gpu_runs_started": 0,
        "stderr_tail": completed.stderr[-1000:],
    }


def _error_text(log_path: Path) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")[-8000:].lower()
    except OSError:
        return ""


def _is_retryable(log_path: Path) -> bool:
    text = _error_text(log_path)
    return any(token in text for token in protocol.RETRYABLE_ERROR_TOKENS)


def _mark_incomplete(job: dict[str, Any], root: Path, *, reason: str, attempts: int, gpu_id: int | None = None) -> dict[str, Any]:
    run_dir = _job_dir(root, job)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        **job,
        "status": "incomplete_compute",
        "reason": reason,
        "attempts": int(attempts),
        "gpu_id": gpu_id,
        "started_at": _now(),
        "ended_at": _now(),
        "labels_used_during_fit": False,
    }
    analysis.write_json(run_dir / "incomplete_compute.json", record)
    analysis.write_json(run_dir / "audit.json", {"audit_ok": False, "status": "incomplete_compute", "reason": reason})
    return record


def _launch_gpu_jobs(
    jobs: list[dict[str, Any]],
    *,
    root: Path,
    hard_deadline: float,
    launch_manifest_path: Path,
    gpu_pool: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Run a bounded GPU queue with one worker per physical GPU."""

    pending = list(jobs)
    running: dict[int, dict[str, Any]] = {}
    attempts: dict[tuple[str, str, str, int, str], int] = {}
    records: list[dict[str, Any]] = []
    next_gpu = 0
    fatal = False
    launch_manifest: list[dict[str, Any]] = []
    threads = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

    while pending or running:
        now = time.monotonic()
        if now >= hard_deadline:
            for job in pending:
                records.append(_mark_incomplete(job, root, reason="hard_wall_timeout_before_launch", attempts=attempts.get(_job_key(job), 0)))
            pending.clear()
            for gpu, state in list(running.items()):
                proc = state["process"]
                proc.kill()
                proc.wait(timeout=10)
                records.append(_mark_incomplete(state["job"], root, reason="hard_wall_timeout", attempts=state["attempt"], gpu_id=gpu))
                state["log_handle"].close()
                del running[gpu]
            break

        while pending and len(running) < len(gpu_pool) and not fatal:
            job = pending.pop(0)
            key = _job_key(job)
            attempt = attempts.get(key, 0) + 1
            attempts[key] = attempt
            gpu = gpu_pool[next_gpu % len(gpu_pool)]
            next_gpu += 1
            run_dir = _job_dir(root, job)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / f"attempt{attempt}.log"
            command = [
                sys.executable, "-m", "scripts.corruption_objective_compatibility_probe.runner",
                "--dataset", job["dataset"], "--arm", job["arm"], "--objective", job["objective"],
                "--seed", str(job["seed"]), "--stage", job["stage"], "--output-dir", str(run_dir),
            ]
            env = os.environ.copy()
            env.update(threads)
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=str(protocol.PROJECT_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            state = {"job": job, "attempt": attempt, "gpu": gpu, "process": process, "started": time.monotonic(), "started_at": _now(), "log_handle": handle, "log_path": log_path, "command": command}
            running[gpu] = state
            launch_manifest.append({**job, "attempt": attempt, "gpu_id": gpu, "started_at": state["started_at"], "command": shlex.join(command), "status": "running"})
            analysis.write_json(launch_manifest_path, {"project_id": protocol.PROJECT_ID, "jobs": launch_manifest})

        progressed = False
        for gpu, state in list(running.items()):
            process = state["process"]
            timed_out = time.monotonic() - state["started"] > protocol.PER_RUN_TIMEOUT_SECONDS
            if process.poll() is None and not timed_out:
                continue
            progressed = True
            if timed_out and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
                reason = "per_run_timeout"
                retry = False
                returncode = None
            else:
                returncode = process.returncode
                reason = "runner_nonzero_exit" if returncode else "missing_or_invalid_summary"
                retry = returncode not in (0, None) and attempts[_job_key(state["job"])] <= 1 and _is_retryable(state["log_path"])
            state["log_handle"].close()
            job = state["job"]
            if returncode == 0 and _source_run_valid(job, root):
                record = {**job, "status": "completed_valid", "attempts": state["attempt"], "gpu_id": gpu, "started_at": state["started_at"], "ended_at": _now()}
                records.append(record)
            elif retry:
                pending.insert(0, job)
                launch_manifest.append({**job, "attempt": state["attempt"], "gpu_id": gpu, "ended_at": _now(), "status": "retrying", "reason": "transient_error"})
            else:
                record = _mark_incomplete(job, root, reason=reason, attempts=state["attempt"], gpu_id=gpu)
                records.append(record)
                # A non-transient protocol/runner failure invalidates the stage;
                # queued jobs are not silently run around it.
                if returncode not in (0, None) and not _is_retryable(state["log_path"]):
                    fatal = True
            launch_manifest.append({**job, "attempt": state["attempt"], "gpu_id": gpu, "ended_at": _now(), "status": records[-1]["status"] if records and records[-1].get("dataset") == job["dataset"] and records[-1].get("seed") == job["seed"] else "retrying", "returncode": returncode})
            del running[gpu]

        if fatal:
            for job in pending:
                records.append(_mark_incomplete(job, root, reason="stage_aborted_after_nonretryable_failure", attempts=attempts.get(_job_key(job), 0)))
            pending.clear()
        if not progressed:
            time.sleep(0.2)
    analysis.write_json(launch_manifest_path, {"project_id": protocol.PROJECT_ID, "jobs": launch_manifest, "records": records})
    return records


def _materialize_reuses(root: Path, jobs: list[dict[str, Any]], *, stage: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in jobs:
        out = _job_dir(root, job)
        if _source_run_valid(job, root):
            records.append({**job, "status": "reused", "reused_from": analysis.read_json(out / "summary.json").get("reused_from")})
            continue
        summary = analysis.reuse_c2_run(job["dataset"], job["arm"], int(job["seed"]), out, stage=stage, objective=job["objective"])
        records.append({**job, "status": summary["status"], "reused_from": summary["reused_from"]})
    return records


def _e1_jobs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reuse: list[dict[str, Any]] = []
    new: list[dict[str, Any]] = []
    for dataset in protocol.DEVELOPMENT_PANEL:
        for arm in protocol.E1_ARMS:
            for seed in protocol.PRIMARY_SEEDS:
                job = {"stage": "E1", "dataset": dataset, "arm": arm, "objective": "O0_GlobalMSE", "seed": int(seed)}
                if dataset in protocol.BIOLOGICAL_DATASETS and arm in {"P0_Random", "P2_SupportTarget"}:
                    reuse.append(job)
                else:
                    new.append(job)
    return reuse, new


def _e2_jobs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reuse: list[dict[str, Any]] = []
    new: list[dict[str, Any]] = []
    for dataset in protocol.DEVELOPMENT_PANEL:
        for arm in protocol.E2_CORRUPTIONS:
            for objective in protocol.E2_OBJECTIVES:
                for seed in protocol.PRIMARY_SEEDS:
                    job = {"stage": "E2", "dataset": dataset, "arm": arm, "objective": objective, "seed": int(seed)}
                    (reuse if objective == "O0_GlobalMSE" else new).append(job)
    return reuse, new


def _write_decision(root: Path, e0: dict[str, Any], e1: dict[str, Any], e3: dict[str, Any], e2: dict[str, Any] | None, *, cuda: dict[str, Any], started_at: str, ended_at: str) -> dict[str, Any]:
    if e1["status"] != "completed_valid" or not e1["gate"]["g1_cross_domain_opportunity"]:
        next_step = "STOP_GENERAL_CORRUPTION"
    elif not e1["gate"]["g2_training_amplification"]:
        next_step = "REPRESENTATION_NOT_OBJECTIVE"
    elif e2 is None or e2["status"] != "completed_valid":
        next_step = "STOP_GENERAL_CORRUPTION"
    else:
        next_step = e2["decision"]
    decision = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "completed_valid" if e1["status"] == "completed_valid" and (e2 is None or e2["status"] == "completed_valid") else "incomplete_compute",
        "next": next_step,
        "started_at": started_at,
        "ended_at": ended_at,
        "cuda_preflight": cuda,
        "e0": {"audit_ok": e0["audit"].get("audit_ok"), "corrected_d1_gate_pass": e0["audit"].get("corrected_d1_gate_pass"), "d2_authorized": False, "gpu_runs_started": 0},
        "e1": e1["gate"],
        "e3": e3["audit"],
        "e2": None if e2 is None else {"status": e2["status"], "decision": e2["decision"], "candidates": e2["candidates"]},
        "claim_boundary": "E1/E2 use dense standardized H0; raw-X support is descriptive E3 only; no support-specific causal claim is authorized.",
        "locked_routes": list(protocol.LOCKED_ROUTES),
    }
    analysis.write_json(root / "decision.json", decision)
    lines = [
        "# Overnight Decision — corruption_objective_compatibility_probe",
        "",
        f"- Protocol: `{protocol.PROTOCOL_ID}`",
        f"- Status: **{decision['status']}**",
        f"- Next: **{next_step}**",
        "",
        "| Question | Result | Decision |",
        "|---|---|---|",
        f"| E0 D1 integrity closure | audit_ok={e0['audit'].get('audit_ok')}; corrected gate={e0['audit'].get('corrected_d1_gate_pass')} | support attribution remains frozen; D2 unauthorized |",
        f"| P2 > P0 on non-biological panel | {e1['gate'].get('g1_winner_count')}/3 | {'cross-domain opportunity' if e1['gate'].get('g1_cross_domain_opportunity') else 'stop generic claim'} |",
        f"| training amplification | {e1['gate'].get('g2_winner_count')}/3 | {'learning signal remains plausible' if e1['gate'].get('g2_training_amplification') else 'representation/geometry remains primary'} |",
        f"| E2 objective interaction | {'not run' if e2 is None else e2['decision']} | {'objective candidate' if e2 and e2['decision'] == 'CORRUPTION_AWARE_OBJECTIVE_OPPORTUNITY' else 'freeze objective branch'} |",
        f"| E3 raw-X audit | audit_ok={e3['audit'].get('audit_ok')} | descriptive only; does not alter H0 estimand |",
        "",
        "## Interpretation firewall",
        "",
        "`support` in E1/E2 means the frozen threshold-defined dense H0 proxy. E3 records raw-X zero/nonzero structure descriptively and is not an input to fitting, gates, or the decision.",
        "",
        "No adaptive policy, GAN, learned generator, new Gate, matcher optimization, corruption-rate sweep, or automatic new model was launched.",
    ]
    _write_text(root / "OVERNIGHT_DECISION.md", "\n".join(lines) + "\n")
    return decision


def run_pipeline(*, output_root: Path = protocol.RESULT_ROOT, execute_gpu: bool = True) -> dict[str, Any]:
    protocol.validate_contract()
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    hard_deadline = time.monotonic() + protocol.HARD_WALL_SECONDS
    analysis.write_json(output_root / "resolved_config.json", protocol.resolved_config())
    e0 = e0_integrity.run(output_root / "E0_integrity")
    e3 = e3_raw_audit.run(output_root / "E3_raw_audit")
    if not e0["audit"].get("audit_ok", False):
        # E0 is a technical prerequisite.  Do not create no-fit or GPU
        # records when the closed-line inheritance/immutability audit fails.
        e1 = {
            "status": "incomplete_compute",
            "gate": {
                "complete_matrix": False,
                "g1_cross_domain_opportunity": False,
                "g1_winner_count": 0,
                "g1_winner_datasets": [],
                "g2_training_amplification": False,
                "g2_winner_count": 0,
                "g2_winner_datasets": [],
            },
        }
        decision = _write_decision(output_root, e0, e1, e3, None, cuda=_cuda_preflight(), started_at=started_at, ended_at=_now())
        analysis.write_json(output_root / "pipeline_manifest.json", {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "started_at": started_at, "ended_at": _now(), "e0_failed": True, "decision": decision, "gpu_ids_allowed": list(protocol.GPU_POOL), "gpu_ids_forbidden": list(protocol.FORBIDDEN_GPU_IDS)})
        return {"e0": e0, "e1": e1, "e2": None, "e3": e3, "decision": decision}
    # E1b is CPU-only and is always run before the scientific gate.
    nofit_root = output_root / "E1b_nofit"
    for dataset in protocol.DEVELOPMENT_PANEL:
        for arm in protocol.E1_ARMS:
            for seed in protocol.PRIMARY_SEEDS:
                out = nofit_root / dataset / arm / f"seed{seed}"
                if not _nofit_valid(out, dataset, arm, int(seed)):
                    analysis.nofit_run(dataset, arm, int(seed), out)

    cuda = _cuda_preflight()
    e1_root = output_root / "E1_opportunity"
    reuse_jobs, new_jobs = _e1_jobs(e1_root)
    reuse_records = _materialize_reuses(e1_root, reuse_jobs, stage="E1")
    resumable_records = [
        {**job, "status": "reused", "reused_from": str(_job_dir(e1_root, job) / "summary.json")}
        for job in new_jobs if _source_run_valid(job, e1_root)
    ]
    new_jobs = [job for job in new_jobs if not _source_run_valid(job, e1_root)]
    reuse_records.extend(resumable_records)
    gpu_records: list[dict[str, Any]] = []
    idle_pool = tuple(int(gpu) for gpu in cuda.get("idle_legal_pool", []))
    if execute_gpu and cuda.get("status") == "ready" and idle_pool and time.monotonic() < hard_deadline:
        gpu_records = _launch_gpu_jobs(new_jobs, root=e1_root, hard_deadline=hard_deadline, launch_manifest_path=e1_root / "launch_manifest.json", gpu_pool=idle_pool)
    else:
        for job in new_jobs:
            gpu_records.append(_mark_incomplete(job, e1_root, reason="gpu_launch_not_authorized_or_preflight_failed", attempts=0))
    e1 = analysis.aggregate_e1(e1_root, nofit_root)
    e2: dict[str, Any] | None = None
    if e1["status"] == "completed_valid" and e1["gate"]["g1_cross_domain_opportunity"] and e1["gate"]["g2_training_amplification"]:
        e2_root = output_root / "E2_objective"
        e2_reuse, e2_new = _e2_jobs(e2_root)
        e2_reuse_records = _materialize_reuses(e2_root, e2_reuse, stage="E2")
        e2_resumable = [
            {**job, "status": "reused", "reused_from": str(_job_dir(e2_root, job) / "summary.json")}
            for job in e2_new if _source_run_valid(job, e2_root)
        ]
        e2_new = [job for job in e2_new if not _source_run_valid(job, e2_root)]
        del e2_reuse_records, e2_resumable
        if execute_gpu and cuda.get("status") == "ready" and idle_pool and time.monotonic() < hard_deadline:
            _launch_gpu_jobs(e2_new, root=e2_root, hard_deadline=hard_deadline, launch_manifest_path=e2_root / "launch_manifest.json", gpu_pool=idle_pool)
        else:
            for job in e2_new:
                _mark_incomplete(job, e2_root, reason="gpu_launch_not_authorized_or_preflight_failed", attempts=0)
        e2 = analysis.aggregate_e2(e2_root)
    decision = _write_decision(output_root, e0, e1, e3, e2, cuda=cuda, started_at=started_at, ended_at=_now())
    analysis.write_json(output_root / "pipeline_manifest.json", {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "started_at": started_at,
        "ended_at": _now(),
        "e0": e0["decision"],
        "e1": {"reuse_records": len(reuse_records), "gpu_records": len(gpu_records), "status": e1["status"]},
        "e2_started": e2 is not None,
        "decision": decision,
        "gpu_ids_allowed": list(protocol.GPU_POOL),
        "gpu_ids_forbidden": list(protocol.FORBIDDEN_GPU_IDS),
    })
    return {"e0": e0, "e1": e1, "e2": e2, "e3": e3, "decision": decision}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen overnight corruption-objective probe")
    parser.add_argument("--output-root", type=Path, default=protocol.RESULT_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cpu-dry-run", action="store_true", help="run E0/E1b/E3 but do not launch GPU jobs")
    args = parser.parse_args()
    if args.preflight_only:
        result = _cuda_preflight()
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("status") == "ready" else 2
    result = run_pipeline(output_root=args.output_root, execute_gpu=not args.cpu_dry_run)
    print(json.dumps(result["decision"], sort_keys=True))
    return 0 if result["decision"]["status"] == "completed_valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
