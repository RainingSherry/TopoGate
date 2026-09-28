# papers/advantage_story_20260927/controller.py
"""
Remote queue controller for advantage_story_20260927 cohort E3 posthoc analysis.

Execution:
  python controller.py run           - singleton controller with fcntl lock
  python controller.py run-task ID   - worker wrapper
  python controller.py status        - read-only state summary

Safety:
  - Refuses absent or changed REVIEW_APPROVED.json
  - Only GPU 2/4 authorized; exact config match required
  - Adopts live orphan workers by /proc/cmdline match; no duplicate launches
  - Persists state atomically after every change; Popen with shell=False
  - CUDA_VISIBLE_DEVICES isolates one GPU per worker; cuda:0 inside script
  - PYTHONPATH includes base and /home/luolie/ToPoGate
  - Resource-aware admission: max 16 workers per GPU but grow from 4
  - Command runs through "controller.py run-task ID" for lockout
  - Requires record.json with completed status and valid artifact SHA for fit workers
  - CPU worker completion similarly requires verification
  - Retry transient OOM/connection/timeout once; others terminal failed
  - Skip completed only after artifact verification; continue all others
  - Restart preserves old attempt outputs (attempt_001/002/...)
  - No programmatic approval creation; parent updates queue only before approval
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import psutil
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE_BASE = Path("/data/luolie/ToPoGate/experiments/v0_rg_validation_tuned_multihop_20260922")
AUTHORIZED_GPUS = {2, 4}
STATE_DIR = HERE / "controller_state"
LOGS_DIR = HERE / "logs"
TERMINAL = frozenset({"completed", "failed"})

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def now():
    return time.time()

def event_log(kind, **data):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": now(), "event": kind, **data}, ensure_ascii=False) + "\n")

def proc_alive(pid):
    try:
        pid = int(pid)
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(") ", 1)[1][0] != "Z"
    except (ProcessLookupError, FileNotFoundError, PermissionError, ValueError, IndexError):
        return False

def proc_task_id(pid):
    try:
        cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes().decode(errors="replace")
        args = cmdline.split("\0")
        if "run-task" in args:
            idx = args.index("run-task")
            if idx + 1 < len(args):
                return args[idx + 1]
    except (OSError, ValueError):
        pass
    return None

def live_workers():
    """Only recognize this controller's exact run-task command, including attempt."""
    result = {}
    expected = str((HERE / "controller.py").resolve())
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            args = proc.info.get("cmdline") or []
            if len(args) < 5 or str(Path(args[1]).resolve()) != expected or args[2] != "run-task":
                continue
            tid, attempt = args[3], int(args[4])
            result[(tid, attempt)] = proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, OSError):
            continue
    return result

def validate_queue(queue):
    tasks = queue.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("queue must contain tasks")
    ids = [task.get("id") for task in tasks]
    if any(not isinstance(tid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", tid) for tid in ids):
        raise ValueError("task ids must be simple nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    if sum(task.get("kind") == "gpu" for task in tasks) > 21:
        raise ValueError("more than the 21 frozen new training cells")
    known = set(ids)
    for task in tasks:
        if task.get("kind") not in {"gpu", "cpu"}:
            raise ValueError(f"invalid task kind for {task['id']}")
        cmd = task.get("command")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            raise ValueError(f"invalid command for {task['id']}")
        script = (HERE / cmd[0]).resolve()
        if script.parent != HERE.resolve() or script.suffix != ".py":
            raise ValueError(f"task script outside experiment directory: {task['id']}")
        if (script.name == "fit_worker.py") != (task["kind"] == "gpu"):
            raise ValueError(f"training/GPU kind mismatch: {task['id']}")
        if task["kind"] == "gpu" and ("--device" not in cmd or
                                      cmd[cmd.index("--device") + 1:cmd.index("--device") + 2] != ["cuda:0"]):
            raise ValueError(f"GPU task must use visible cuda:0: {task['id']}")
        if "--out" not in cmd or cmd.index("--out") + 1 >= len(cmd):
            raise ValueError(f"task has no --out argument: {task['id']}")
        out = (HERE / task.get("out", "")).resolve()
        if out == HERE.resolve() or HERE.resolve() not in out.parents:
            raise ValueError(f"task output outside experiment directory: {task['id']}")
        deps = task.get("depends_on", [])
        if not isinstance(deps, list) or len(deps) != len(set(deps)) or task["id"] in deps or any(d not in known for d in deps):
            raise ValueError(f"invalid dependencies for {task['id']}")
    visiting, visited = set(), set()
    by_id = {task["id"]: task for task in tasks}
    def visit(tid):
        if tid in visiting:
            raise ValueError("dependency cycle")
        if tid in visited:
            return
        visiting.add(tid)
        for dep in by_id[tid].get("depends_on", []):
            visit(dep)
        visiting.remove(tid)
        visited.add(tid)
    for tid in ids:
        visit(tid)
    return queue

def dependencies_terminal(task, state):
    """Posthoc tasks may run after a failed fit, so success is not required."""
    return all(state["tasks"].get(dep, {}).get("status") in TERMINAL
               for dep in task.get("depends_on", []))

def load_approval():
    path = HERE / "REVIEW_APPROVED.json"
    if not path.exists():
        raise RuntimeError("REVIEW_APPROVED.json missing; parent approval required")
    appr = read_json(path)
    if appr.get("status") != "approved":
        raise RuntimeError("approval status not 'approved'")
    return appr

def verify_approval(queue):
    validate_queue(queue)
    appr = load_approval()
    problems = []
    
    if appr.get("queue_sha256") != sha256(HERE / "queue.json"):
        problems.append("queue hash differs from approval")
    
    approved_code = appr.get("code_sha256", {})
    needed = {"controller.py"}
    for script in queue.get("tasks", []):
        needed.add(script["command"][0])
    needed.update(approved_code)
    
    for name in needed:
        p = (HERE / name).resolve()
        if p.parent != HERE.resolve():
            problems.append(f"code path outside experiment directory: {name}")
            continue
        if not p.exists():
            problems.append(f"code file missing: {name}")
        elif approved_code.get(name) != sha256(p):
            problems.append(f"code hash differs: {name}")

    approved_inputs = appr.get("input_sha256", {})
    if not isinstance(approved_inputs, dict):
        problems.append("input hash lock missing")
        approved_inputs = {}
    needed_inputs = {"REUSE_MANIFEST.json", "COHORT_LOCK_SHA256.txt", "resources.json"}
    for task in queue["tasks"]:
        cmd = task["command"]
        if Path(cmd[0]).name == "fit_worker.py":
            if "--task" not in cmd or cmd.index("--task") + 1 >= len(cmd):
                problems.append(f"fit task JSON missing: {task['id']}")
            else:
                needed_inputs.add(cmd[cmd.index("--task") + 1])
    for name in needed_inputs | set(approved_inputs):
        p = (HERE / name).resolve()
        if p == HERE.resolve() or HERE.resolve() not in p.parents:
            problems.append(f"input path outside experiment directory: {name}")
        elif not p.is_file():
            problems.append(f"input file missing: {name}")
        elif approved_inputs.get(name) != sha256(p):
            problems.append(f"input hash differs: {name}")
    
    core_sha = appr.get("core_sha256", {})
    if not isinstance(core_sha, dict) or not core_sha:
        problems.append("core hash lock missing")
        core_sha = {}
    base_root = REMOTE_BASE
    for rel, expected in core_sha.items():
        full = (base_root / rel) if not Path(rel).is_absolute() else Path(rel)
        if not full.exists():
            problems.append(f"core package missing: {rel}")
        elif sha256(full) != expected:
            problems.append(f"core package changed: {rel}")
    
    if problems:
        raise RuntimeError("Approval verification failed: " + "; ".join(problems))
    
    return appr

def nvidia_smi_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True
        )
        gpus = {}
        for line in result.stdout.strip().split("\n"):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3:
                idx, used, free = int(parts[0]), int(parts[1]), int(parts[2])
                if idx in AUTHORIZED_GPUS:
                    gpus[idx] = {"used_mb": used, "free_mb": free}
        return gpus
    except Exception:
        return {}

def host_available_mb():
    try:
        mem = psutil.virtual_memory()
        return int(mem.available / (1024 * 1024))
    except Exception:
        return 0

def pick_gpu(state, resources, admissions=None, baseline=None):
    if any(rec.get("status") == "running" and rec.get("kind") == "gpu" and
           rec.get("gpu") is None for rec in state["tasks"].values()):
        return None  # unknown orphan assignment: avoid GPU overcommit
    gpus = nvidia_smi_memory()
    if not gpus:
        return None
    admissions = admissions or {}
    
    current_per_gpu = defaultdict(int)
    for tid, rec in state["tasks"].items():
        if rec.get("status") == "running" and rec.get("gpu") is not None:
            current_per_gpu[rec["gpu"]] += 1
    
    max_per_gpu = int(resources.get("max_workers_per_gpu", 16))
    initial = int(resources.get("initial_workers_per_gpu", 4))
    start_free_mb = int(resources.get("start_free_mb", 6000))
    reserve_mb = int(resources.get("reserve_mb_per_task", 4000))
    
    candidates = []
    if baseline is None:
        baseline = dict(current_per_gpu)
    for gpu_id, mem in gpus.items():
        if gpu_id not in resources.get("gpus", []):
            continue
        current = current_per_gpu[gpu_id]
        if current >= max_per_gpu:
            continue
        # Fill to the initial cap, then admit at most one extra per polling pass.
        if baseline.get(gpu_id, 0) < initial:
            if current >= initial:
                continue
        elif admissions.get(gpu_id, 0) >= 1:
            continue
        if mem["free_mb"] - admissions.get(gpu_id, 0) * reserve_mb < start_free_mb + reserve_mb:
            continue
        candidates.append((mem["free_mb"] - admissions.get(gpu_id, 0) * reserve_mb, gpu_id))
    
    return max(candidates, key=lambda x: (x[0], -x[1]))[1] if candidates else None

def load_queue():
    return read_json(HERE / "queue.json")

def load_resources():
    path = HERE / "resources.json"
    if path.exists():
        res = read_json(path)
    else:
        res = {}
    
    defaults = {
        "gpus": [2, 4],
        "max_workers_per_gpu": 16,
        "initial_workers_per_gpu": 4,
        "start_free_mb": 6000,
        "reserve_mb_per_task": 4000,
        "cpu_workers": 4,
        "cpu_threads_per_task": 2,
        "python": sys.executable or "/usr/bin/python3",
        "min_host_available_mb": 16000,
    }
    
    for k, v in defaults.items():
        res.setdefault(k, v)
    
    res["gpus"] = [g for g in res["gpus"] if g in AUTHORIZED_GPUS]
    if not res["gpus"]:
        raise ValueError("no authorized GPUs configured")
    if not 1 <= int(res["max_workers_per_gpu"]) <= 16:
        raise ValueError("max_workers_per_gpu must be within 1..16")
    if not 1 <= int(res["initial_workers_per_gpu"]) <= 4:
        raise ValueError("initial_workers_per_gpu must be within 1..4")
    if int(res["reserve_mb_per_task"]) < 4000 or int(res["start_free_mb"]) < 6000:
        raise ValueError("GPU memory reserve below frozen safe minimum")
    if not 1 <= int(res["cpu_workers"]) <= 4 or int(res["cpu_threads_per_task"]) != 2:
        raise ValueError("CPU worker budget differs from frozen controller policy")
    if int(res["min_host_available_mb"]) < 16000:
        raise ValueError("host memory reserve below frozen safe minimum")
    
    return res

def load_state():
    path = STATE_DIR / "controller_state.json"
    if path.exists():
        st = read_json(path)
    else:
        st = {"tasks": {}, "updated": now()}
    
    return st

def save_state(state):
    state["updated"] = now()
    atomic_json(STATE_DIR / "controller_state.json", state)

def attempt_dir(task, attempt):
    return HERE / task["out"] / f"attempt_{int(attempt):03d}"

def result_path(task_id, attempt):
    return STATE_DIR / "results" / task_id / f"attempt_{int(attempt):03d}.json"

def task_completed(task, out_dir):
    record_path = Path(out_dir) / "record.json"
    if not record_path.exists():
        return False, "record.json missing"
    
    try:
        rec = read_json(record_path)
        if rec.get("status") != "completed":
            return False, f"status={rec.get('status')}"
        
        artifact_sha = rec.get("artifact_sha256")
        if not isinstance(artifact_sha, dict) or not artifact_sha:
            return False, "empty artifact hash map"
        script = Path(task["command"][0]).name
        required = set(task.get("required_artifacts", []))
        if script == "fit_worker.py":
            if (rec.get("labels_used_for_model_fit") is not False or
                    rec.get("test_labels_loaded") is not False or
                    rec.get("core_source_unchanged") is not True or
                    rec.get("K_provenance") != "benchmark_oracle_from_y"):
                return False, "fit label/core provenance mismatch"
            required.update({"model.pt", "embedding.npy", "predictions.npy", "history.json",
                             "plan.npz", "graph.npz", "preprocessor.npz", "source_trace.bin",
                             "source_trace.json", "displacement_by_anchor.npz"})
            if rec.get("task_strategy") == "kwide":
                required.add("reference_plan.npz")
            cmd = task["command"]
            if "--task" in cmd:
                task_file = (HERE / cmd[cmd.index("--task") + 1]).resolve()
                if rec.get("task_sha256") != sha256(task_file):
                    return False, "fit task hash mismatch"
        elif script == "cohort_worker.py":
            if (rec.get("posthoc_full_x_labels_loaded") is not True or
                    rec.get("labels_used_for_model_fit") is not False or
                    rec.get("new_encoder_training") is not False):
                return False, "cohort label/training provenance mismatch"
            required.update({"frozen_readouts_10.npz", "fixed_input_layout.npz",
                             "e3_per_readout.csv", "e3_per_seed.csv", "e3_aggregate.csv"})
            if rec.get("K_provenance") != "benchmark_oracle_from_y":
                return False, "cohort K provenance mismatch"
        if not required.issubset(artifact_sha):
            return False, f"required artifact hashes missing: {sorted(required - set(artifact_sha))}"
        cmd = task["command"]
        if "--dataset" in cmd and rec.get("dataset", rec.get("task_dataset")) != cmd[cmd.index("--dataset") + 1]:
            return False, "dataset mismatch"
        for name, expected in artifact_sha.items():
            p = (Path(out_dir) / name).resolve()
            if Path(out_dir).resolve() not in p.parents:
                return False, f"artifact outside output: {name}"
            if not p.exists():
                return False, f"artifact missing: {name}"
            if sha256(p) != expected:
                return False, f"artifact hash mismatch: {name}"
        
        return True, ""
    except Exception as e:
        return False, str(e)

def run_task_worker(task_id, queue, resources, gpu_id=None, attempt=1):
    task_def = next((t for t in queue["tasks"] if t["id"] == task_id), None)
    if not task_def:
        raise ValueError(f"Task {task_id} not in queue")
    
    cmd = list(task_def["command"])
    out_dir = attempt_dir(task_def, attempt)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd[cmd.index("--out") + 1] = str(out_dir)
    
    log_path = LOGS_DIR / f"{task_id}_attempt_{int(attempt):03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([str(HERE), str(REMOTE_BASE), "/home/luolie/ToPoGate",
                                    env.get("PYTHONPATH", "")])
    
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    
    env["OMP_NUM_THREADS"] = str(resources.get("cpu_threads_per_task", 2))
    env["MKL_NUM_THREADS"] = str(resources.get("cpu_threads_per_task", 2))
    
    python_exe = resources.get("python", sys.executable)
    full_cmd = [python_exe] + cmd
    
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"# Task: {task_id}\n# Command: {full_cmd}\n# GPU: {gpu_id}\n\n")
        log_f.flush()
        
        proc = subprocess.Popen(
            full_cmd,
            cwd=HERE,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            shell=False
        )
        
        proc.wait()
    
    if proc.returncode == 0:
        ok, reason = task_completed(task_def, out_dir)
        if ok:
            return {"status": "completed", "returncode": 0, "out_dir": str(out_dir)}
        else:
            return {"status": "failed", "returncode": 0, "reason": f"validation: {reason}",
                    "out_dir": str(out_dir)}
    else:
        log_content = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        
        transient = proc.returncode in (-9, 137) or any(kw in log_content.lower() for kw in
                        ["oom", "out of memory", "connection", "timeout", "resource temporarily"])
        
        return {
            "status": "failed_transient" if transient else "failed",
            "returncode": proc.returncode,
            "reason": f"exit {proc.returncode}", "out_dir": str(out_dir)
        }

def adopt_orphans(state, queue):
    by_id = {task["id"]: task for task in queue["tasks"]}
    live = live_workers()
    for (tid, attempt), pid in live.items():
        if tid not in by_id:
            event_log("unknown_live_worker", task_id=tid, attempt=attempt, pid=pid)
            continue
        rec = state["tasks"].setdefault(tid, {"id": tid, "status": "pending", "attempts": 0})
        if rec.get("status") in TERMINAL or int(rec.get("attempts", 0)) > attempt:
            event_log("stale_live_worker", task_id=tid, attempt=attempt, pid=pid)
            continue
        if rec.get("status") != "running" or rec.get("pid") != pid:
            event_log("adopt_orphan", task_id=tid, attempt=attempt,
                      old_pid=rec.get("pid"), new_pid=pid)
        rec.update(status="running", attempts=attempt, pid=pid,
                   kind=by_id[tid]["kind"], out_dir=str(attempt_dir(by_id[tid], attempt)))
    return live

def _failure_from_log(task_id, attempt):
    path = LOGS_DIR / f"{task_id}_attempt_{int(attempt):03d}.log"
    content = path.read_text(encoding="utf-8", errors="replace")[-4000:].lower() if path.exists() else ""
    transient = any(word in content for word in
                    ("oom", "out of memory", "connection", "timeout", "resource temporarily"))
    return {"status": "failed_transient" if transient and attempt < 2 else "failed",
            "reason": "worker exited without result", "returncode": None}

def reconcile_state(state, queue, live=None, validated=None):
    """Controller is the sole writer of global state; children write result files."""
    live = live if live is not None else live_workers()
    validated = validated if validated is not None else set()
    by_id = {task["id"]: task for task in queue["tasks"]}
    for tid, task in by_id.items():
        rec = state["tasks"].setdefault(tid, {"id": tid, "status": "pending", "attempts": 0})
        if rec["status"] == "completed":
            marker = (tid, rec.get("out_dir"))
            if marker in validated:
                continue
            ok, reason = task_completed(task, rec.get("out_dir", ""))
            if not ok:
                rec.update(status="failed", reason=f"completed output invalid: {reason}")
                event_log("completed_invalid", task_id=tid, reason=reason)
            else:
                validated.add(marker)
            continue
        if rec["status"] == "failed_transient":
            rec["status"] = "pending" if rec.get("attempts", 0) < 2 else "failed"
        if rec["status"] != "running":
            # Recover a completed attempt if state was lost after a worker wrote its record.
            if rec["status"] == "pending" and rec.get("attempts", 0) == 0:
                for previous in sorted((HERE / task["out"]).glob("attempt_*/record.json"), reverse=True):
                    ok, _ = task_completed(task, previous.parent)
                    if ok:
                        rec.update(status="completed", attempts=int(previous.parent.name.split("_")[-1]),
                                   out_dir=str(previous.parent), end_time=now())
                        validated.add((tid, rec["out_dir"]))
                        break
            continue
        attempt = int(rec.get("attempts", 0))
        if attempt <= 0:
            rec.update(status="failed", reason="running task has no attempt")
            continue
        rp = result_path(tid, attempt)
        if rp.exists():
            payload = read_json(rp)
            if payload.get("task_id") != tid or payload.get("attempt") != attempt:
                rec.update(status="failed", reason="result claim mismatch")
            else:
                rec.update(payload["result"])
                if rec["status"] == "completed":
                    ok, reason = task_completed(task, rec.get("out_dir", ""))
                    if not ok:
                        rec.update(status="failed", reason=f"completion invalid: {reason}")
                    else:
                        validated.add((tid, rec["out_dir"]))
                if rec["status"] == "failed_transient" and attempt >= 2:
                    rec["status"] = "failed"
            rec["end_time"] = now()
            event_log("task_done", task_id=tid, status=rec["status"], attempt=attempt)
        elif (tid, attempt) in live:
            rec["pid"] = live[(tid, attempt)]
        elif not proc_alive(rec.get("pid")):
            out = attempt_dir(task, attempt)
            ok, _ = task_completed(task, out)
            rec.update({"status": "completed", "out_dir": str(out)} if ok else
                       _failure_from_log(tid, attempt))
            if ok:
                validated.add((tid, rec["out_dir"]))
            rec["end_time"] = now()
            event_log("task_died", task_id=tid, status=rec["status"], attempt=attempt)

def _launch_task(state, task, gpu):
    tid = task["id"]
    rec = state["tasks"].setdefault(tid, {"id": tid, "status": "pending", "attempts": 0})
    attempt = int(rec.get("attempts", 0)) + 1
    rec.update(status="running", attempts=attempt, gpu=gpu, kind=task["kind"],
               start_time=now(), pid=None, out_dir=str(attempt_dir(task, attempt)))
    # Persist the GPU assignment before the child can read its claim.
    save_state(state)
    cmd = [sys.executable, str(HERE / "controller.py"), "run-task", tid, str(attempt)]
    try:
        proc = subprocess.Popen(cmd, cwd=HERE, start_new_session=True, shell=False)
    except Exception as exc:
        rec.update(status="failed", reason=f"worker launch failed: {exc}", end_time=now())
        save_state(state)
        event_log("task_launch_failed", task_id=tid, attempt=attempt, reason=str(exc))
        return False
    rec["pid"] = proc.pid
    save_state(state)
    event_log("task_start", task_id=tid, attempt=attempt, task_kind=task["kind"], gpu=gpu, pid=proc.pid)
    return True

def run_controller():
    lock_path = STATE_DIR / "controller.lock"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another controller is running (lock held)")
        sys.exit(1)
    
    event_log("controller_start")
    
    try:
        queue = load_queue()
        verify_approval(queue)
        state = load_state()
        validated = set()
        live = adopt_orphans(state, queue)
        reconcile_state(state, queue, live, validated)
        save_state(state)
        
        while True:
            resources = load_resources()
            queue = load_queue()
            verify_approval(queue)
            live = adopt_orphans(state, queue)
            reconcile_state(state, queue, live, validated)
            save_state(state)
            pending = []
            for t in queue["tasks"]:
                tid = t["id"]
                rec = state["tasks"][tid]
                if rec["status"] == "pending" and dependencies_terminal(t, state):
                    pending.append((tid, t))
            
            running_cpu = sum(1 for r in state["tasks"].values() if r.get("status") == "running" and r.get("kind") == "cpu")
            baseline_gpu = defaultdict(int)
            for rec in state["tasks"].values():
                if rec.get("status") == "running" and rec.get("kind") == "gpu":
                    baseline_gpu[rec.get("gpu")] += 1
            admissions = defaultdict(int)
            
            host_mb = host_available_mb()
            min_host_mb = int(resources.get("min_host_available_mb", 16000))
            
            if host_mb < min_host_mb:
                event_log("host_memory_low", available_mb=host_mb, min_mb=min_host_mb)
            else:
                for tid, t in pending:
                    kind = t.get("kind", "cpu")
                    host_reserve = int(resources.get("reserve_host_mb_per_task", 1024))
                    if host_mb - sum(admissions.values()) * host_reserve - running_cpu * host_reserve < min_host_mb + host_reserve:
                        break
                    
                    if kind == "gpu":
                        gpu = pick_gpu(state, resources, admissions, baseline_gpu)
                        if gpu is None:
                            continue
                        if _launch_task(state, t, gpu):
                            admissions[gpu] += 1
                        
                    elif kind == "cpu":
                        max_cpu = int(resources.get("cpu_workers", 4))
                        if running_cpu >= max_cpu:
                            continue
                        if _launch_task(state, t, None):
                            running_cpu += 1
            
            save_state(state)
            
            all_terminal = all(state["tasks"][t["id"]]["status"] in TERMINAL for t in queue["tasks"])
            
            if all_terminal:
                # Recheck artifacts at closure; the middle polling passes use the
                # successful first hash check to avoid repeatedly reading traces.
                by_id = {t["id"]: t for t in queue["tasks"]}
                for tid, rec in state["tasks"].items():
                    if tid in by_id and rec.get("status") == "completed":
                        ok, reason = task_completed(by_id[tid], rec.get("out_dir", ""))
                        if not ok:
                            rec.update(status="failed", reason=f"final output invalid: {reason}")
                            event_log("completed_invalid", task_id=tid, reason=reason)
                save_state(state)
                completed = sum(state["tasks"][t["id"]]["status"] == "completed" for t in queue["tasks"])
                failed = sum(state["tasks"][t["id"]]["status"] == "failed" for t in queue["tasks"])
                
                final = {
                    "status": "PIPELINE_COMPLETED" if failed == 0 else "PIPELINE_PARTIAL",
                    "completed": completed,
                    "failed": failed,
                    "timestamp": now()
                }
                atomic_json(HERE / "PIPELINE_STATUS.json", final)
                event_log("pipeline_done", **final)
                break
            
            time.sleep(float(resources.get("poll_seconds", 15)))
    
    finally:
        lock_handle.close()

def run_task_entry(task_id, attempt):
    queue = load_queue()
    verify_approval(queue)
    resources = load_resources()
    state = load_state()
    task = next((t for t in queue["tasks"] if t["id"] == task_id), None)
    if task is None:
        raise ValueError("unknown task id")
    rec = state["tasks"].get(task_id, {})
    if rec.get("status") != "running" or int(rec.get("attempts", 0)) != attempt:
        raise RuntimeError("run-task has no matching controller claim")
    gpu = rec.get("gpu")
    if task["kind"] == "gpu" and gpu not in AUTHORIZED_GPUS:
        raise RuntimeError("GPU task has no authorized GPU assignment")
    if task["kind"] == "cpu" and gpu is not None:
        raise RuntimeError("CPU task has a GPU assignment")
    lock_path = STATE_DIR / "locks" / f"{task_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"worker already holds task lock: {task_id}")
        try:
            result = run_task_worker(task_id, queue, resources, gpu, attempt)
        except BaseException as exc:
            result = {"status": "failed", "returncode": None,
                      "reason": f"worker wrapper exception: {exc}",
                      "out_dir": str(attempt_dir(task, attempt))}
            event_log("task_wrapper_exception", task_id=task_id, attempt=attempt,
                      traceback=traceback.format_exc())
        atomic_json(result_path(task_id, attempt),
                    {"task_id": task_id, "attempt": attempt, "result": result})

def print_status():
    state = load_state()
    
    print(f"Tasks: {len(state['tasks'])}")
    
    counts = defaultdict(int)
    for rec in state["tasks"].values():
        counts[rec.get("status", "unknown")] += 1
    
    for status in sorted(counts.keys()):
        print(f"  {status}: {counts[status]}")
    
    running = [r for r in state["tasks"].values() if r.get("status") == "running"]
    if running:
        print(f"\nRunning tasks: {len(running)}")
        for r in running:
            print(f"  {r['id']}: GPU={r.get('gpu')}, PID={r.get('pid')}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "run-task", "status"])
    p.add_argument("task_id", nargs="?")
    p.add_argument("attempt", nargs="?", type=int)
    args = p.parse_args()
    
    if args.command == "run":
        run_controller()
    elif args.command == "run-task":
        if not args.task_id or args.attempt is None or args.attempt < 1:
            print("run-task requires task_id and positive attempt")
            sys.exit(1)
        run_task_entry(args.task_id, args.attempt)
    elif args.command == "status":
        print_status()

if __name__ == "__main__":
    main()
