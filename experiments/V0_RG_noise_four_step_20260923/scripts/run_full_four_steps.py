"""Resume the registered four-case study on physical GPU 4, with bounded concurrency."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
CONFIG = BUNDLE / "config.json"
ROOT = Path(json.loads(CONFIG.read_text(encoding="utf-8"))["output_root"])
PYTHON = Path(json.loads(CONFIG.read_text(encoding="utf-8"))["remote_python"])
LOGS = ROOT / "pipeline_logs"
STATE = ROOT / "PIPELINE_STATUS.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(state):
    state["updated_utc"] = now()
    tmp = STATE.with_name(STATE.name + ".tmp." + str(os.getpid()))
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE)


def phase(state, name, command):
    entry = {"name": name, "command": command, "started_utc": now(), "status": "running"}
    state["phases"].append(entry)
    state["current_phase"] = name
    save(state)
    path = LOGS / (name + ".log")
    entry["log"] = str(path)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1")
    print(now(), name, "started", flush=True)
    try:
        with path.open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=str(BUNDLE), env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        entry["exit_code"] = result.returncode
        entry["status"] = "completed" if result.returncode == 0 else "completed_with_task_failures"
    except Exception:
        entry["status"] = "launcher_failed"
        entry["exception"] = traceback.format_exc()
        with path.open("a", encoding="utf-8") as log:
            log.write(entry["exception"])
    entry["ended_utc"] = now()
    save(state)
    print(now(), name, entry["status"], flush=True)


def stage_command(stage, *args):
    return [str(PYTHON), str(HERE / "run_stages.py"), "--config", str(CONFIG),
            "--stage", stage] + list(args)


def main():
    if sys.platform != "linux":
        raise RuntimeError("This launcher is for the SSH Linux host only.")
    LOGS.mkdir(parents=True, exist_ok=True)
    with (ROOT / "pipeline.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another full four-step launcher already holds pipeline.lock.", flush=True)
            return 2
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        inventory_path = ROOT / "inventories/checkpoints.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        import hashlib
        if inventory.get("config_sha256") != hashlib.sha256(CONFIG.read_bytes()).hexdigest():
            raise RuntimeError("Inventory/config SHA mismatch; rerun inventory before launching.")
        if inventory.get("status_counts", {}).get("ready") != 42:
            raise RuntimeError("Expected 42 effective/reference checkpoints from the audited four-case plan.")
        state = {"schema": "four_step_pipeline_status_v1", "started_utc": now(),
                 "gpu_physical": 4, "workers": 4, "train_new_weights": False,
                 "registered_model_seeds": config["model_seeds"],
                 "registered_noise_seeds": config["noise_seeds"],
                 "registered_conditions": config["conditions"],
                 "expected_observe_results": 42 * len(config["conditions"]) * len(config["noise_seeds"]),
                 "expected_direction_results": 42 * 3 * len(config["direction_multipliers"]),
                 "phases": [], "status": "running"}
        save(state)
        phase(state, "observe_full", stage_command("observe", "--variants", "full",
                                                   "--device", "cuda:4", "--workers", "4"))
        phase(state, "observe_effective_controls", stage_command(
            "observe", "--variants", "base_scmae", "no_edge_modulation",
            "fixed_mix_original_aux", "mixed_observation_reconstruction",
            "--device", "cuda:4", "--workers", "4"))
        phase(state, "summary_after_observe", stage_command("summarize"))
        # The first three steps compare quality and component contrasts. A matched-norm
        # direction probe is needed to distinguish topology-specific response from
        # generic smoothness. The registered four cases are retained without
        # post-hoc cherry-picking from their observe outcomes.
        phase(state, "directions_full", stage_command("directions", "--variants", "full",
                                                    "--device", "cuda:4", "--workers", "4"))
        phase(state, "directions_effective_controls", stage_command(
            "directions", "--variants", "base_scmae", "no_edge_modulation",
            "fixed_mix_original_aux", "mixed_observation_reconstruction",
            "--device", "cuda:4", "--workers", "4"))
        phase(state, "summary_final", stage_command("summarize"))
        state["current_phase"] = None
        state["status"] = "dispatch_finished_file_audit_required"
        state["ended_utc"] = now()
        save(state)
        print(now(), state["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
