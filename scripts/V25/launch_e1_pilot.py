#!/usr/bin/env python3
"""Run a frozen V25 E1 pilot or gated confirmation phase on local GPUs.

The manifest stores one row per arm, while the E1 runner executes a complete
N/R/T panel.  This launcher therefore collapses those rows to one resumable
panel key and never launches an individual arm.  Confirmation is admitted only
after the preregistered pilot gate and panel audit are present.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.V25.audit_e1_phase import manifest_expectations


DEFAULT_MANIFEST = ROOT / "result/V25_systematic_mechanism_study/E1/e1_manifest.json"
DEFAULT_OUTPUT = ROOT / "result/V25_systematic_mechanism_study/E1/pilot"
DEFAULT_CONFIRMATION_OUTPUT = ROOT / "result/V25_systematic_mechanism_study/E1/confirmation"
DEFAULT_HOLDOUT_OUTPUT = ROOT / "result/V25_systematic_mechanism_study/PhaseD/E1"
DEFAULT_PILOT_AUDIT = ROOT / "result/V25_systematic_mechanism_study/E1/pilot/Audit/phase_summary.json"
ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})
FORBIDDEN_GPUS = frozenset({0, 7})
SEEDS = (42, 123, 7)
ARMS = ("N", "R", "T")
PROTOCOL_ID = "v25_e1_v21_matched_nrt_v1"
RUNNER = ROOT / "scripts/V25/run_e1_matched_protocol.py"
CORE = ROOT / "methods/TopoGate/V25_systematic_mechanism_study/e1_protocol.py"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path, phase: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    accepted_manifest_ids = {"v25_e1_gated_manifest_v1", "v25_holdout_e1_manifest_v1"}
    if payload.get("manifest_id") not in accepted_manifest_ids:
        raise ValueError(f"unexpected E1 manifest: {payload.get('manifest_id')!r}")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"unexpected E1 protocol: {payload.get('protocol_id')!r}")
    if payload.get("a2_decision") != "retain_e1":
        raise ValueError(f"A2 decision does not admit E1: {payload.get('a2_decision')!r}")
    if payload.get("generated_without_e1_outcomes") is not True:
        raise ValueError("E1 manifest must be outcome-independent")
    if payload.get("manifest_id") == "v25_holdout_e1_manifest_v1":
        if phase != "holdout":
            raise ValueError("holdout manifest can only run phase=holdout")
        phase_payload = payload
    else:
        phase_payload = payload.get("phases", {}).get(phase)
    if not isinstance(phase_payload, dict):
        raise ValueError(f"manifest has no phase {phase!r}")
    jobs = phase_payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"{phase} phase has no arm rows")
    return payload, jobs


def collapse_panels(manifest: dict[str, Any], arm_rows: list[dict[str, Any]], output_root: Path, phase: str = "pilot") -> list[dict[str, Any]]:
    expected = manifest_expectations(manifest, phase)
    if not expected["valid"]:
        raise ValueError(f"invalid frozen {phase} manifest: {expected['errors']}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in arm_rows:
        if row.get("phase") != phase or row.get("arms") != list(ARMS):
            raise ValueError(f"invalid E1 arm row: {row}")
        grouped.setdefault(str(row["panel_run_key"]), []).append(row)
    panels: list[dict[str, Any]] = []
    for panel_key, rows in sorted(grouped.items()):
        by_arm = {str(row.get("arm")): row for row in rows}
        if set(by_arm) != set(ARMS) or len(rows) != len(ARMS):
            raise ValueError(f"panel {panel_key} does not contain exactly N/R/T rows")
        anchor = by_arm["N"]
        for arm in ARMS:
            row = by_arm[arm]
            for key in ("dataset", "input_protocol", "source_path", "source_sha256", "seed"):
                if row.get(key) != anchor.get(key):
                    raise ValueError(f"panel {panel_key} differs across arms for {key}")
            source = Path(str(row["source_path"]))
            if not source.is_file():
                raise FileNotFoundError(source)
            if not row.get("source_sha256") or _sha256(source) != row["source_sha256"]:
                raise ValueError(f"source hash mismatch for {panel_key}: {source}")
        dataset = str(anchor["dataset"])
        seed = int(anchor["seed"])
        output_dir = output_root / dataset.replace(" ", "_") / f"seed{seed}"
        panels.append(
            {
                "panel_run_key": panel_key,
                "phase": phase,
                "dataset": dataset,
                "input_protocol": str(anchor["input_protocol"]),
                "source_path": str(anchor["source_path"]),
                "source_sha256": str(anchor["source_sha256"]),
                "seed": seed,
                "arms": list(ARMS),
                "primary_readout": anchor.get("primary_readout"),
                "K_source": anchor.get("K_source"),
                "output_dir": str(output_dir),
                "status": "queued",
                "attempts": 0,
                "gpu": None,
                "pid": None,
                "started_at": None,
                "finished_at": None,
                "returncode": None,
                "error": None,
            }
        )
    observed_keys = {panel["panel_run_key"] for panel in panels}
    expected_keys = set(expected["panel_keys"])
    if observed_keys != expected_keys or len(panels) != expected["expected_panel_count"]:
        raise ValueError(f"{phase} panel keys/count do not match the frozen manifest")
    if {panel["dataset"] for panel in panels} != set(expected["datasets"]):
        raise ValueError(f"{phase} dataset set does not match the frozen manifest")
    if {panel["seed"] for panel in panels} != set(expected["seeds"]):
        raise ValueError(f"{phase} seeds do not match the frozen seed set")
    for panel in panels:
        spec = expected["panel_map"][panel["panel_run_key"]]
        if panel["dataset"] != spec["dataset"] or panel["seed"] != int(spec["seed"]):
            raise ValueError(f"{phase} panel {panel['panel_run_key']} differs from frozen dataset/seed mapping")
    return panels


def _query_free_memory() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    values: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if line.strip():
            index, memory = (item.strip() for item in line.split(",", 1))
            values[int(index)] = int(float(memory))
    return values


def _required_complete(panel: dict[str, Any]) -> bool:
    output = Path(panel["output_dir"])
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("status") != "completed" or summary.get("protocol_id") != PROTOCOL_ID:
        return False
    if str(summary.get("seed")) != str(panel["seed"]) or int(summary.get("n_clusters", 0)) <= 1:
        return False
    audit_path = output / "audit.json"
    if not audit_path.is_file():
        return False
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("labels_used_during_fit") is not False:
        return False
    shared = audit.get("TR_shared_schedule_hashes", {})
    if any(shared.get(key) is not True for key in ("donor", "eligible", "budget", "selection_noise")):
        return False
    none_contract = audit.get("none_contract", {})
    if none_contract.get("assignment_forward_calls") != 0 or none_contract.get("js_forward_calls") != 0:
        return False
    pairs = summary.get("pairs", {})
    if set(pairs) != {"I_full_ARI", "S_full_ARI", "I_1step_ARI", "S_1step_ARI"}:
        return False
    if any(value is None for value in pairs.values()):
        return False
    for arm in ARMS:
        metrics_path = output / arm / "metrics.json"
        if not metrics_path.is_file():
            return False
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("labels_used_after_fit_only") is not True:
            return False
    return True


def pilot_audit_admits_confirmation(pilot_audit: dict[str, Any], manifest: dict[str, Any]) -> bool:
    """Require a complete, manifest-identical pilot before confirmation."""
    gate = pilot_audit.get("phase_gate", {})
    if gate.get("passes") is not True:
        return False
    expected = manifest_expectations(manifest, "pilot")
    audit_keys = set(pilot_audit.get("expected_panel_keys", []))
    return bool(
        expected["valid"]
        and pilot_audit.get("manifest_id") == manifest.get("manifest_id")
        and pilot_audit.get("phase") == "pilot"
        and pilot_audit.get("coverage_complete") is True
        and int(pilot_audit.get("expected_panel_count", -1)) == expected["expected_panel_count"]
        and int(pilot_audit.get("panel_count", -1)) == expected["expected_panel_count"]
        and int(pilot_audit.get("audit_ok_count", -1)) == expected["expected_panel_count"]
        and set(pilot_audit.get("expected_datasets", [])) == set(expected["datasets"])
        and set(pilot_audit.get("expected_seeds", [])) == set(expected["seeds"])
        and audit_keys == set(expected["panel_keys"])
        and set(pilot_audit.get("observed_panel_keys", [])) == set(expected["panel_keys"])
    )


def _command(panel: dict[str, Any], gpu: int) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--data",
        panel["source_path"],
        "--dataset-name",
        panel["dataset"],
        "--input-protocol",
        panel["input_protocol"],
        "--output-dir",
        panel["output_dir"],
        "--seed",
        str(panel["seed"]),
        "--device",
        "cuda",
        "--gpu",
        str(gpu),
    ]
    if panel.get("K_source") == "explicit_n_clusters" and panel.get("n_clusters") is not None:
        command.extend(["--n-clusters", str(int(panel["n_clusters"]))])
    return command


def _save_state(path: Path, manifest: dict[str, Any], panels: list[dict[str, Any]], started_at: str, status: str, gpu_pool: list[int], phase: str) -> None:
    _write_json(
        path,
        {
            "manifest_id": manifest["manifest_id"],
            "protocol_id": PROTOCOL_ID,
            "phase": phase,
            "ledger_scope": "attempt_local_panel_selection",
            "started_at": started_at,
            "updated_at": _now(),
            "status": status,
            "gpu_pool": gpu_pool,
            "forbidden_gpus": sorted(FORBIDDEN_GPUS),
            "expected_panel_jobs": len(panels),
            "manifest_expected_panel_jobs": (
                manifest.get("expected_panel_jobs")
                if manifest.get("manifest_id") == "v25_holdout_e1_manifest_v1"
                else manifest.get("phases", {}).get(phase, {}).get("expected_panel_jobs")
            ),
            "selected_panel_jobs": len(panels),
            "reused_panels_in_attempt": sum(panel["status"] == "reused" for panel in panels),
            "completed_panels": sum(panel["status"] in {"completed", "reused"} for panel in panels),
            "incomplete_panels": sum(panel["status"] == "incomplete_compute" for panel in panels),
            "queued_panels": sum(panel["status"] == "queued" for panel in panels),
            "panels": panels,
        },
    )


def _load_previous(path: Path, panels: list[dict[str, Any]], phase: str) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("phase") != phase:
        raise ValueError(f"existing queue state belongs to another protocol: {path}")
    old = {str(item.get("panel_run_key")): item for item in payload.get("panels", [])}
    for panel in panels:
        prior = old.get(panel["panel_run_key"], {})
        if prior.get("status") == "running" and prior.get("pid"):
            try:
                os.kill(int(prior["pid"]), 0)
            except ProcessLookupError:
                panel.update(
                    {
                        "status": "queued",
                        "attempts": int(prior.get("attempts", 0)),
                        "error": "retry_after_stale_running_launcher",
                    }
                )
            except PermissionError as exc:
                raise RuntimeError(f"cannot verify active panel process {panel['panel_run_key']} pid={prior['pid']}") from exc
            else:
                raise RuntimeError(f"panel is active in another launcher: {panel['panel_run_key']} pid={prior['pid']}")
        if _required_complete(panel):
            panel.update({"status": "reused", "attempts": int(prior.get("attempts", 0)), "returncode": 0})
        elif prior.get("status") == "incomplete_compute":
            panel.update({"status": "queued", "attempts": int(prior.get("attempts", 0)), "error": "retry_after_incomplete_compute"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase", choices=("pilot", "confirmation", "holdout"), default="pilot")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pilot-audit", type=Path, default=DEFAULT_PILOT_AUDIT)
    parser.add_argument("--gpu-pool", nargs="*", type=int, default=[2])
    parser.add_argument("--min-free-mib", type=int, default=30000)
    parser.add_argument("--max-parallel-per-gpu", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = {
            "pilot": DEFAULT_OUTPUT,
            "confirmation": DEFAULT_CONFIRMATION_OUTPUT,
            "holdout": DEFAULT_HOLDOUT_OUTPUT,
        }[args.phase]
    if not args.gpu_pool or any(gpu in FORBIDDEN_GPUS or gpu not in ALLOWED_GPUS for gpu in args.gpu_pool):
        raise ValueError(f"GPU pool must be a non-empty subset of {sorted(ALLOWED_GPUS)}")
    if args.max_parallel_per_gpu <= 0 or args.min_free_mib <= 0 or args.max_attempts <= 0:
        raise ValueError("parallelism, min-free-mib and max-attempts must be positive")
    if set(args.seeds) != set(SEEDS):
        raise ValueError("E1 launcher is frozen to seeds [42, 123, 7]")
    manifest, arm_rows = _load_manifest(args.manifest, args.phase)
    if args.phase == "confirmation":
        if not args.pilot_audit.is_file():
            raise ValueError(f"confirmation requires pilot audit: {args.pilot_audit}")
        pilot_audit = json.loads(args.pilot_audit.read_text(encoding="utf-8"))
        if not pilot_audit_admits_confirmation(pilot_audit, manifest):
            raise ValueError("pilot panel audit is incomplete or invalid")
    if args.phase == "holdout":
        if manifest.get("manifest_id") != "v25_holdout_e1_manifest_v1":
            raise ValueError("holdout requires the claim-dependent holdout manifest")
        claim_path = ROOT / "result/V25_systematic_mechanism_study/PhaseC/FROZEN_PAPER_CLAIM.json"
        claim_sha = manifest.get("claim_freeze_sha256")
        if not claim_path.is_file() or not claim_sha or _sha256(claim_path) != claim_sha:
            raise ValueError("holdout manifest is not bound to the current frozen claim")
    panels = collapse_panels(manifest, arm_rows, args.output_dir, args.phase)
    if args.datasets is not None:
        selected = set(args.datasets)
        expected_datasets = {panel["dataset"] for panel in panels}
        if selected != expected_datasets:
            raise ValueError(
                f"{args.phase} dataset filtering would violate the frozen full-panel manifest; "
                f"expected {sorted(expected_datasets)}, got {sorted(selected)}"
            )
    if set(args.seeds) != set(SEEDS):
        raise ValueError("seed filter cannot change the frozen E1 seed set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "manifest_snapshot.json", manifest)
    _write_json(args.output_dir / "launcher_contract.json", {
        "protocol_id": PROTOCOL_ID,
        "phase": args.phase,
        "gpu_pool": list(args.gpu_pool),
        "forbidden_gpus": sorted(FORBIDDEN_GPUS),
        "min_free_mib": args.min_free_mib,
        "max_parallel_per_gpu": args.max_parallel_per_gpu,
        "runner_sha256": _sha256(RUNNER),
        "core_sha256": _sha256(CORE),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "created_at": _now(),
    })
    state_path = args.output_dir / "queue_state.json"
    _load_previous(state_path, panels, args.phase)
    started_at = _now()
    if args.dry_run:
        _save_state(state_path, manifest, panels, started_at, "dry_run", list(args.gpu_pool), args.phase)
        print(json.dumps({"status": "dry_run", "panels": panels}, indent=2, ensure_ascii=True, default=str))
        return 0

    logs_root = args.output_dir / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    processes: dict[str, subprocess.Popen[Any]] = {}
    handles: dict[str, Any] = {}
    interrupted = False

    def _stop(_signal: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        for process in processes.values():
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while True:
        for panel in panels:
            key = panel["panel_run_key"]
            process = processes.get(key)
            if process is None:
                continue
            returncode = process.poll()
            if returncode is None:
                continue
            handle = handles.pop(key, None)
            if handle is not None:
                handle.close()
            processes.pop(key, None)
            panel["pid"] = None
            panel["returncode"] = int(returncode)
            panel["finished_at"] = _now()
            if returncode == 0 and _required_complete(panel):
                panel["status"] = "completed"
                panel["error"] = None
            elif int(panel["attempts"]) < args.max_attempts and not interrupted:
                panel["status"] = "queued"
                panel["error"] = f"returncode={returncode}; retrying"
            else:
                panel["status"] = "incomplete_compute"
                panel["error"] = f"returncode={returncode}; log={logs_root / (key.replace(':', '__') + '.log')}"

        if interrupted:
            for panel in panels:
                if panel["panel_run_key"] in processes:
                    panel["status"] = "incomplete_compute"
                    panel["error"] = "launcher_interrupted"
            _save_state(state_path, manifest, panels, started_at, "interrupted", list(args.gpu_pool), args.phase)
            return 130

        active: dict[int, int] = {}
        for panel in panels:
            if panel["status"] == "running" and panel["gpu"] is not None:
                active[int(panel["gpu"])] = active.get(int(panel["gpu"]), 0) + 1
        free_memory = _query_free_memory()
        for panel in panels:
            if panel["status"] != "queued":
                continue
            candidates = [gpu for gpu in args.gpu_pool if active.get(gpu, 0) < args.max_parallel_per_gpu and free_memory.get(gpu, 0) >= args.min_free_mib]
            if not candidates:
                break
            gpu = max(candidates, key=lambda item: free_memory.get(item, 0))
            output = Path(panel["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            _write_json(output / "manifest_record.json", panel)
            log_path = logs_root / (panel["panel_run_key"].replace(":", "__") + ".log")
            handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "MPLCONFIGDIR": str(args.output_dir / "mplconfig")})
            command = _command(panel, gpu)
            _write_json(output / "launch_record.json", {"panel_run_key": panel["panel_run_key"], "gpu": gpu, "command": command, "started_at": _now()})
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes[panel["panel_run_key"]] = process
            handles[panel["panel_run_key"]] = handle
            panel.update({"status": "running", "attempts": int(panel["attempts"]) + 1, "gpu": gpu, "pid": process.pid, "started_at": _now(), "error": None})
            active[gpu] = active.get(gpu, 0) + 1
            free_memory[gpu] = 0

        terminal = all(panel["status"] in {"completed", "reused", "incomplete_compute"} for panel in panels)
        all_ok = terminal and all(panel["status"] in {"completed", "reused"} for panel in panels)
        _save_state(state_path, manifest, panels, started_at, "completed" if all_ok else ("incomplete_compute" if terminal else "running"), list(args.gpu_pool), args.phase)
        if terminal:
            for handle in handles.values():
                handle.close()
            print(json.dumps({"status": "completed" if all_ok else "incomplete_compute", "panels": panels}, indent=2, ensure_ascii=True, default=str))
            return 0 if all_ok else 2
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
