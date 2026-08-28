"""Pack-first V26 dispatcher: fill one legal GPU before spilling to another."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V26_support_oracle import protocol


PYTHON = Path("/data/luolie/conda/base/bin/python3")
SAFETY_MIB = 4096
STATE_ROOT = protocol.RESULT_ROOT / "DISPATCH"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def gpu_snapshot() -> list[dict[str, Any]]:
    command = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=20)
    rows = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            continue
        gpu = int(fields[0])
        if gpu in protocol.LEGAL_GPU_POOL and gpu not in protocol.FORBIDDEN_GPU_IDS:
            used, total, utilization = (int(float(fields[index])) for index in (1, 2, 3))
            rows.append({"gpu": gpu, "used_mib": used, "total_mib": total, "free_mib": total - used, "utilization_pct": utilization})
    return rows


def command_for(stage: str, dataset: str, gpu: int | None = None, arm: str | None = None, seed: int | None = None) -> list[str]:
    command = [str(PYTHON), "-m", "methods.TopoGate.V26_support_oracle.run"]
    if stage == "freeze":
        return [*command, "--freeze"]
    if stage == "diagnostics":
        return [*command, "--diagnostics", "--dataset", dataset]
    if stage == "preflight":
        return [*command, "--preflight", "--dataset", dataset, "--gpu", str(gpu)]
    return [*command, "--run-cell", "--dataset", dataset, "--arm", str(arm), "--seed", str(seed), "--gpu", str(gpu)]


def run_blocking(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "MPLCONFIGDIR": "/tmp/v26-mpl"})
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True, check=False).returncode


def choose_anchor() -> int:
    snapshot = gpu_snapshot()
    if not snapshot:
        raise RuntimeError("no legal GPU visible")
    return max(snapshot, key=lambda row: (row["free_mib"], -row["gpu"]))["gpu"]


def run_preparation() -> None:
    if run_blocking(command_for("freeze", "mouse"), STATE_ROOT / "freeze.log") != 0:
        raise RuntimeError("V26 freeze failed")
    for dataset in protocol.DATASET_BY_ID:
        code = run_blocking(command_for("diagnostics", dataset), STATE_ROOT / "diagnostics" / f"{dataset}.log")
        if code != 0:
            raise RuntimeError(f"V26 diagnostics failed for {dataset}")
    anchor = choose_anchor()
    for dataset in protocol.DATASET_BY_ID:
        code = run_blocking(command_for("preflight", dataset, anchor), STATE_ROOT / "preflight" / f"{dataset}.log")
        if code != 0:
            raise RuntimeError(f"V26 preflight failed for {dataset}")


def reservation(dataset: str) -> float:
    path = protocol.RESULT_ROOT / "preflight" / f"{dataset}.json"
    if not path.exists():
        return 8192.0
    return float(json.loads(path.read_text(encoding="utf-8"))["reservation_mib"])


def completed(dataset: str, arm: str, seed: int) -> bool:
    path = protocol.RESULT_ROOT / "runs" / dataset / arm / f"seed{seed}" / "summary.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return (
        summary.get("status") == "completed_valid"
        and summary.get("implementation", {}).get("source_sha256") == protocol.implementation_sha256()
    )


def reserved_active_mib(active: dict[int, dict[str, Any]], gpu: int) -> float:
    return float(sum(float(detail["reservation_mib"]) for detail in active.values() if int(detail["gpu"]) == int(gpu)))


def can_admit(row: dict[str, Any], candidate_reservation_mib: float, active: dict[int, dict[str, Any]], baseline_free_mib: dict[int, float]) -> bool:
    """Admit only if the original non-V26 capacity and live free memory agree.

    The baseline reserves each active V26 process at its measured peak, while
    the live guard notices new external load.  Safety is a per-GPU margin, not
    an arbitrary cap on the number of co-resident programs.
    """
    gpu = int(row["gpu"])
    baseline = float(baseline_free_mib.setdefault(gpu, float(row["free_mib"])))
    planned_free = baseline - reserved_active_mib(active, gpu)
    needed = float(candidate_reservation_mib) + SAFETY_MIB
    return planned_free >= needed and float(row["free_mib"]) >= needed


def launch_matrix() -> dict[str, Any]:
    pending = [(dataset, arm, seed) for dataset in protocol.DATASET_BY_ID for arm in protocol.ARMS for seed in protocol.SEEDS if not completed(dataset, arm, seed)]
    pending.sort(key=lambda job: (-reservation(job[0]), job[0], job[1], job[2]))
    active: dict[int, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    baseline_free_mib: dict[int, float] = {}
    while pending or active:
        for pid, detail in list(active.items()):
            code = detail["process"].poll()
            if code is None:
                continue
            detail["log_handle"].close()
            detail["returncode"] = int(code)
            detail["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            reports.append({key: value for key, value in detail.items() if key != "process"})
            del active[pid]
        launched = False
        if pending:
            snapshot = gpu_snapshot()
            active_by_gpu = {row["gpu"]: sum(1 for value in active.values() if value["gpu"] == row["gpu"]) for row in snapshot}
            # Existing V26 work makes a GPU the preferred anchor.  Otherwise choose max free memory.
            ordering = sorted(snapshot, key=lambda row: (-active_by_gpu[row["gpu"]], -row["free_mib"], row["gpu"]))
            for index, job in enumerate(pending):
                target = next((row for row in ordering if can_admit(row, reservation(job[0]), active, baseline_free_mib)), None)
                if target is None:
                    continue
                dataset, arm, seed = job
                gpu = int(target["gpu"])
                log_path = STATE_ROOT / "workers" / f"gpu{gpu}_{dataset}_{arm}_seed{seed}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "MPLCONFIGDIR": "/tmp/v26-mpl"})
                log_handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(command_for("cell", dataset, gpu, arm, seed), cwd=ROOT, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
                active[process.pid] = {"pid": process.pid, "gpu": gpu, "dataset": dataset, "arm": arm, "seed": seed, "reservation_mib": reservation(dataset), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "log": str(log_path), "log_handle": log_handle, "process": process}
                pending.pop(index)
                launched = True
                time.sleep(3.0)
                break
        state = {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "implementation_revision": protocol.IMPLEMENTATION_REVISION, "pending": len(pending), "active": [{key: value for key, value in item.items() if key not in {"process", "log_handle"}} for item in active.values()], "reserved_active_mib": {str(gpu): reserved_active_mib(active, gpu) for gpu in protocol.LEGAL_GPU_POOL}, "baseline_free_mib": baseline_free_mib, "completed_reports": reports, "gpu_snapshot": gpu_snapshot(), "packing_policy": "pack-first: reserve measured peak memory for every active V26 job on the current legal GPU; spill only when the per-GPU safety margin would be violated", "safety_mib": SAFETY_MIB, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}
        write_json(STATE_ROOT / "state.json", state)
        if not launched:
            time.sleep(10.0)
    failed = [item for item in reports if item.get("returncode") != 0]
    result = {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "reports": reports, "failures": failed, "status": "completed_valid" if not failed else "incomplete_compute", "packing_policy": "pack-first"}
    write_json(STATE_ROOT / "final.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="V26 pack-first dispatcher")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.prepare or args.all:
        run_preparation()
    if args.dispatch or args.all:
        result = launch_matrix()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "completed_valid" else 1
    if not (args.prepare or args.dispatch or args.all):
        parser.error("choose --prepare, --dispatch, or --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
