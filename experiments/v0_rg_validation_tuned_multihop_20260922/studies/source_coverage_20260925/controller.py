"""Durable, resumable queue controller for source_coverage_20260925.

``controller.py run``            the scheduler loop (unique via flock on controller.lock).
``controller.py run-task ID``    worker wrapper: validate-or-archive, execute, validate, mark.
``controller.py status``         print a compact state table (read-only).

Guarantees
* Executes only if REVIEW_APPROVED.json (written by the parent, never by this
  file) binds the exact code, queue, prereg and task-file hashes.  Re-checked
  every loop; a mismatch halts new launches.
* Per-task state files under state/tasks/ are written atomically; completed means
  the artifact/hash validation passed and TASK_COMPLETED.json was written.
* Failed attempts are moved to failed_attempts/<id>/ (never deleted); at most
  ``max_attempts`` (default 3 = 1 + 2 retries).  A failed task never blocks
  independent tasks; dependants with dep_mode=all_completed become
  ``blocked_dependency``; all_terminal dependants still run and report partial.
* Live orphan workers (state says running, or an untracked ``run-task ID``
  process) are adopted and waited for, never duplicated and never killed.
* Only this queue's concurrency is adapted; other processes are never touched.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
from common import (AUTHORIZED_GPUS, CODE_FILES, R, ValidationError, atomic_json, atomic_text,  # noqa: E402
                    core_source_hashes, directory_manifest, now, read_json, sha256, validate_task_output)

STATE = EXP / "state"
TASK_STATE = STATE / "tasks"
LOGS = EXP / "logs"
EVENTS = LOGS / "controller_events.jsonl"
RESOURCES = EXP / "resources.json"
DEFAULT_RESOURCES = {
    "gpus": list(AUTHORIZED_GPUS),
    "initial_workers_per_gpu": 2,
    "max_workers_per_gpu": 6,
    "grow_min_free_mb": 12000,
    "start_min_free_mb": 6000,
    "max_existing_util_percent": 90,
    "reserve_mb_per_new_task": 3000,
    "max_new_gpu_launches_per_poll": 1,
    "cpu_threads_per_task": 2,
    "cpu_analysis_concurrency": 8,
    "max_attempts": 3,
    "poll_seconds": 15,
    "min_host_available_mb": 16000,
    "python": sys.executable,
    "pythonpath_extra": ["/home/luolie/ToPoGate"],
}
TERMINAL = {"completed", "failed", "blocked_dependency"}


# ------------------------------------------------------------------ utilities

def event(kind: str, **payload) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"t": now(), "event": kind, **payload}, ensure_ascii=False) + "\n")


def resources() -> dict:
    merged = dict(DEFAULT_RESOURCES)
    if RESOURCES.exists():
        try:
            merged.update(read_json(RESOURCES))
        except Exception as exc:
            event("resources_unreadable", error=repr(exc))
    merged["gpus"] = [g for g in merged["gpus"] if int(g) in AUTHORIZED_GPUS]
    return merged


def load_queue() -> dict:
    return read_json(EXP / "queue.json")


def approval_problems(queue: dict) -> list[str]:
    path = EXP / "REVIEW_APPROVED.json"
    if not path.exists():
        return ["REVIEW_APPROVED.json missing (parent approval required)"]
    approval = read_json(path)
    problems = []
    if approval.get("status") != "approved" or approval.get("reviewer") != "parent":
        problems.append("approval status/reviewer invalid")
    if approval.get("queue_sha256") != sha256(EXP / "queue.json"):
        problems.append("queue hash differs from approval")
    if (EXP / "queue.sha256").read_text().strip() != sha256(EXP / "queue.json"):
        problems.append("queue.sha256 sidecar mismatch")
    if approval.get("prereg_sha256") != sha256(EXP / "prereg" / "prereg.json"):
        problems.append("prereg hash differs from approval")
    approved_code = approval.get("code_sha256", {})
    needed = set(CODE_FILES) | referenced_scripts(queue)
    for name in sorted(needed):
        if not (EXP / name).exists():
            problems.append(f"code file missing: {name}")
        elif approved_code.get(name) != sha256(EXP / name):
            problems.append(f"code hash differs from approval: {name}")
    for rel, expected in queue.get("task_file_sha256", {}).items():
        if sha256(EXP / rel) != expected:
            problems.append(f"task file changed: {rel}")
    for rel, expected in queue.get("auxiliary_locks_sha256", {}).items():
        if sha256(EXP / rel) != expected:
            problems.append(f"auxiliary lock changed: {rel}")
    if approval.get("core_package_sha256") not in (None, queue.get("core_package_sha256")):
        problems.append("approved core hashes differ from queue")
    if core_source_hashes(R) != queue.get("core_package_sha256"):
        problems.append("historical core package changed")
    return problems


def referenced_scripts(queue: dict) -> set[str]:
    return {t["command"][0] for t in queue["tasks"] if t["command"] and t["command"][0].endswith(".py")}


def read_state(task_id: str) -> dict:
    path = TASK_STATE / f"{task_id}.json"
    if path.exists():
        return read_json(path)
    return {"id": task_id, "status": "pending", "attempts": 0, "history": []}


def write_state(rec: dict) -> None:
    rec["updated"] = now()
    atomic_json(TASK_STATE / f"{rec['id']}.json", rec)


def proc_alive(pid) -> bool:
    try:
        pid = int(pid)
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(") ", 1)[1][0] != "Z"
    except (ProcessLookupError, FileNotFoundError, PermissionError, ValueError, TypeError, IndexError):
        return False


def proc_is_worker(pid, task_id: str) -> bool:
    try:
        args = Path(f"/proc/{int(pid)}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except (OSError, ValueError, TypeError):
        return False
    return (any(a.endswith("controller.py") for a in args) and "run-task" in args and task_id in args
            and str(EXP) in Path(f"/proc/{int(pid)}/cwd").resolve().as_posix())


def scan_workers() -> dict[str, int]:
    found = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if "run-task" in args and any(a.endswith("controller.py") for a in args):
            idx = args.index("run-task")
            if idx + 1 < len(args) and proc_is_worker(entry.name, args[idx + 1]):
                found[args[idx + 1]] = int(entry.name)
    return found


def gpu_status() -> dict[int, dict]:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode:
        return {}
    status = {}
    for line in result.stdout.splitlines():
        try:
            idx, free, total, util = (int(x.strip()) for x in line.split(","))
        except ValueError:
            continue
        status[idx] = {"free_mb": free, "total_mb": total, "util": util}
    return status


def host_available_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 10 ** 9


# ------------------------------------------------------------------ worker

def run_task(task_id: str) -> int:
    lock_dir = EXP / "state" / "worker_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{task_id}.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("task already has a live worker", flush=True)
            return 5
        return run_task_locked(task_id)


def run_task_locked(task_id: str) -> int:
    queue = load_queue()
    problems = approval_problems(queue)
    if problems:
        print("refusing to run unapproved task:", problems, flush=True)
        return 3
    task = next(t for t in queue["tasks"] if t["id"] == task_id)
    approved_code = read_json(EXP / "REVIEW_APPROVED.json")["code_sha256"]
    out = EXP / task["out"]
    marker = out / "TASK_COMPLETED.json"
    if marker.exists():
        try:
            saved = read_json(marker)
            if saved.get("task_id") != task_id:
                raise ValidationError("marker task mismatch")
            for rel, meta in saved["files"].items():
                if sha256(out / rel) != meta["sha256"]:
                    raise ValidationError(f"artifact changed after completion: {rel}")
            validate_task_output(task, EXP, approved_code)
            print("already completed and valid", flush=True)
            return 0
        except Exception as exc:
            print("invalid completed artifacts; archiving:", repr(exc), flush=True)
            archive(task_id, out, "invalid_completed")
    elif out.exists() and any(out.iterdir()):
        archive(task_id, out, "stale_partial")
    out.mkdir(parents=True, exist_ok=True)
    res = resources()
    threads = str(int(res["cpu_threads_per_task"]))
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               NUMEXPR_NUM_THREADS=threads, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    if task.get("validator") == "schntl":
        env["PYTHONPATH"] = str(EXP / "baseline_deps") + os.pathsep + env.get("PYTHONPATH", "")
    command = [task.get("python", sys.executable), str(EXP / task["command"][0]), *task["command"][1:]]
    result = subprocess.run(command, cwd=EXP, env=env)
    if result.returncode:
        archive(task_id, out, f"failed_rc{result.returncode}")
        return result.returncode
    try:
        summary = validate_task_output(task, EXP, approved_code)
    except Exception as exc:
        print("post-run validation failed:", repr(exc), flush=True)
        archive(task_id, out, "failed_validation")
        return 4
    status = summary.get("status") if isinstance(summary, dict) else "completed"
    atomic_json(marker, {"task_id": task_id, "status": "completed", "reported_status": status,
                         "files": directory_manifest(out), "finished": now(),
                         "code_sha256": approved_code})
    return 0


def archive(task_id: str, out: Path, reason: str) -> None:
    if not out.exists():
        return
    target = EXP / "failed_attempts" / task_id / f"{int(time.time() * 1000)}_{reason}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out), str(target))
    event("attempt_archived", task=task_id, reason=reason, path=str(target))


# ------------------------------------------------------------------ scheduler

def deps_state(task: dict, states: dict) -> str:
    """ready | wait | blocked."""
    deps = task.get("deps", [])
    mode = task.get("dep_mode", "all_completed")
    statuses = [states[d]["status"] for d in deps]
    if mode == "all_terminal":
        return "ready" if all(s in TERMINAL for s in statuses) else "wait"
    if any(s in ("failed", "blocked_dependency") for s in statuses):
        return "blocked"
    return "ready" if all(s == "completed" for s in statuses) else "wait"


def run_controller() -> int:
    STATE.mkdir(exist_ok=True)
    TASK_STATE.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    lock_handle = (EXP / "controller.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another controller holds controller.lock", flush=True)
        return 0
    queue = load_queue()
    tasks = {t["id"]: t for t in queue["tasks"]}
    problems = approval_problems(queue)
    if problems:
        atomic_json(STATE / "CONTROLLER_HALTED.json", {"t": now(), "problems": problems})
        event("halted_unapproved", problems=problems)
        print("NOT APPROVED:", problems, flush=True)
        return 2
    (STATE / "CONTROLLER_HALTED.json").unlink(missing_ok=True)
    event("controller_start", pid=os.getpid(), tasks=len(tasks))
    states = {tid: read_state(tid) for tid in tasks}
    procs: dict[str, subprocess.Popen | None] = {}
    live = scan_workers()
    for tid, rec in states.items():
        pid = live.get(tid) or (rec.get("pid") if rec["status"] == "running" else None)
        if pid and proc_alive(pid) and proc_is_worker(pid, tid):
            rec.update(status="running", pid=int(pid), adopted=True)
            procs[tid] = None
            write_state(rec)
            event("adopted_orphan", task=tid, pid=pid)
        elif rec["status"] == "running":
            rec["status"] = "pending"
            write_state(rec)
            finalize_attempt(tid, tasks[tid], rec, None)
    gpu_cap_penalty: dict[int, int] = {}
    while True:
        res = resources()
        # Reap finished workers.
        for tid in list(procs):
            rec, proc = states[tid], procs[tid]
            code = proc.poll() if proc is not None else (None if proc_alive(rec["pid"]) else -1)
            if code is None:
                continue
            del procs[tid]
            finalize_attempt(tid, tasks[tid], rec, code)
            if rec["status"] != "completed" and "out of memory" in rec.get("last_error", "").lower():
                gpu = rec.get("gpu")
                if gpu is not None:
                    gpu_cap_penalty[gpu] = gpu_cap_penalty.get(gpu, 0) + 1
                    event("oom_reduce_gpu_cap", gpu=gpu, penalty=gpu_cap_penalty[gpu])
        # Dependency blocking.
        for tid, task in tasks.items():
            rec = states[tid]
            if rec["status"] == "pending" and deps_state(task, states) == "blocked":
                rec["status"] = "blocked_dependency"
                write_state(rec)
                event("blocked_dependency", task=tid)
        pending = [tid for tid, r in states.items() if r["status"] == "pending"]
        if not pending and not procs:
            break
        problems = approval_problems(queue)
        if problems:
            atomic_json(STATE / "CONTROLLER_HALTED.json", {"t": now(), "problems": problems})
            event("halted_approval_drift", problems=problems)
            if not procs:
                return 2
            time.sleep(int(res["poll_seconds"]))
            continue
        gpus = gpu_status()
        ours = {g: sum(1 for t in procs if states[t].get("gpu") == g) for g in res["gpus"]}
        cpu_running = sum(1 for t in procs if tasks[t]["kind"] != "gpu")
        launches_per_gpu = {g: 0 for g in res["gpus"]}
        host_ok = host_available_mb() >= int(res["min_host_available_mb"])
        for tid in sorted(pending, key=lambda t: stage_order(tasks[t])):
            task, rec = tasks[tid], states[tid]
            if deps_state(task, states) != "ready":
                continue
            gpu = None
            if task["kind"] == "gpu":
                if not host_ok:
                    continue
                gpu = pick_gpu(res, gpus, ours, launches_per_gpu, gpu_cap_penalty)
                if gpu is None:
                    continue
                ours[gpu] += 1
                launches_per_gpu[gpu] += 1
                gpus[gpu]["free_mb"] -= int(res["reserve_mb_per_new_task"])
            else:
                if cpu_running >= int(res["cpu_analysis_concurrency"]) or not host_ok:
                    continue
                cpu_running += 1
            procs[tid] = launch(tid, task, rec, gpu, res)
            event("launched", task=tid, gpu=gpu, attempt=rec["attempts"])
        write_summary(states, tasks, procs)
        time.sleep(int(res["poll_seconds"]))
    return finish(states, tasks, queue)


def stage_order(task: dict) -> tuple:
    order = {"refcheck": 0, "cpu_analysis": 1, "train": 2, "posthoc": 3, "extension": 4, "summarize": 9}
    return (order.get(task.get("stage"), 5), task["id"])


def pick_gpu(res, gpus, ours, launches, penalty):
    best = None
    for g in res["gpus"]:
        info = gpus.get(g)
        if info is None or launches[g] >= int(res["max_new_gpu_launches_per_poll"]):
            continue
        cap = max(1, int(res["max_workers_per_gpu"]) - penalty.get(g, 0))
        if ours[g] >= cap or info["util"] >= int(res["max_existing_util_percent"]):
            continue
        if ours[g] < int(res["initial_workers_per_gpu"]):
            if info["free_mb"] < int(res["start_min_free_mb"]):
                continue
        elif info["free_mb"] < int(res["grow_min_free_mb"]):
            continue
        score = info["free_mb"] - 2000 * ours[g]
        if best is None or score > best[0]:
            best = (score, g)
    return None if best is None else best[1]


def launch(tid: str, task: dict, rec: dict, gpu, res) -> subprocess.Popen:
    rec["attempts"] = int(rec.get("attempts", 0)) + 1
    log = LOGS / f"{tid}.attempt{rec['attempts']}.log"
    env = os.environ.copy()
    extra = [str(R), str(EXP), *res.get("pythonpath_extra", [])]
    env["PYTHONPATH"] = os.pathsep.join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    threads = str(int(res["cpu_threads_per_task"]))
    env.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               NUMEXPR_NUM_THREADS=threads, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    with log.open("ab") as stream:
        proc = subprocess.Popen([res.get("python") or sys.executable, str(EXP / "controller.py"), "run-task", tid],
                                cwd=EXP, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
    rec.update(status="running", pid=proc.pid, gpu=gpu, log=str(log), started=now(), adopted=False)
    write_state(rec)
    return proc


def finalize_attempt(tid: str, task: dict, rec: dict, code) -> None:
    out = EXP / task["out"]
    marker = out / "TASK_COMPLETED.json"
    ok = False
    if marker.exists():
        try:
            saved = read_json(marker)
            ok = saved.get("task_id") == tid and all(
                sha256(out / rel) == meta["sha256"] for rel, meta in saved["files"].items())
        except Exception:
            ok = False
    entry = {"attempt": rec.get("attempts"), "returncode": code, "finished": now(), "gpu": rec.get("gpu"),
             "log": rec.get("log")}
    if ok:
        rec["status"] = "completed"
        rec["reported_status"] = read_json(marker).get("reported_status")
    else:
        try:
            tail = Path(rec["log"]).read_text(errors="replace")[-4000:] if rec.get("log") else ""
        except OSError:
            tail = "log unreadable"
        rec["last_error"] = tail
        entry["error_tail"] = tail[-800:]
        rec["status"] = "pending" if int(rec.get("attempts", 0)) < int(resources()["max_attempts"]) else "failed"
    rec.setdefault("history", []).append(entry)
    rec.pop("pid", None)
    write_state(rec)
    event("attempt_finished", task=tid, status=rec["status"], returncode=code)


def write_summary(states, tasks, procs) -> None:
    counts = {}
    for rec in states.values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    atomic_json(STATE / "controller_summary.json", {
        "t": now(), "pid": os.getpid(), "counts": counts,
        "running": {t: {"gpu": states[t].get("gpu"), "attempt": states[t].get("attempts")} for t in procs},
        "failed": [t for t, r in states.items() if r["status"] == "failed"],
        "blocked": [t for t, r in states.items() if r["status"] == "blocked_dependency"]})


def finish(states, tasks, queue) -> int:
    write_summary(states, tasks, {})
    stages: dict[str, dict] = {}
    for tid, task in tasks.items():
        rec = states[tid]
        slot = stages.setdefault(task.get("stage", "other"), {"completed": 0, "total": 0, "not_completed": [],
                                                              "partial_or_blocked": []})
        slot["total"] += 1
        if rec["status"] == "completed":
            slot["completed"] += 1
            if rec.get("reported_status") not in (None, "completed"):
                slot["partial_or_blocked"].append([tid, rec.get("reported_status")])
        else:
            slot["not_completed"].append([tid, rec["status"]])
    core_ok = core_source_hashes(R) == queue.get("core_package_sha256")
    complete = core_ok and all(not s["not_completed"] and not s["partial_or_blocked"] for s in stages.values())
    payload = {"t": now(), "core_source_unchanged": core_ok, "stages": stages,
               "formal_training_completed": stages.get("train", {}).get("completed", 0),
               "note": "refcheck outputs are smoke runs and are not counted as formal training"}
    atomic_json(EXP / ("PIPELINE_COMPLETED.json" if complete else "PIPELINE_PARTIAL.json"), payload)
    event("pipeline_finished", complete=complete)
    return 0


def status() -> None:
    queue = load_queue()
    for task in queue["tasks"]:
        rec = read_state(task["id"])
        print(f"{rec['status']:<20} att={rec.get('attempts', 0)} gpu={rec.get('gpu')} {task['id']}")
    problems = approval_problems(queue)
    print("approval:", "OK" if not problems else problems)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    command = sys.argv[1]
    try:
        if command == "run":
            return run_controller()
        if command == "run-task":
            return run_task(sys.argv[2])
        if command == "status":
            status()
            return 0
    except Exception:
        event("controller_exception", command=command, error=traceback.format_exc()[-4000:])
        raise
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
