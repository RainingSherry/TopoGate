#!/usr/bin/env python3
"""Run frozen ACCG synthetic jobs; requires --execute and never uses labels in fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MAIN_CONFIG = ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_joint.yaml"
ABLATIONS = {
    "coordinate": ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_coordinate.yaml",
    "shuffled_graph": ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_shuffled_graph.yaml",
    "marginal_only": ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_marginal_only.yaml",
    "abstention_sensitivity": ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_joint_abstain.yaml",
}
CORE_WORLDS = frozenset(
    {"W0_matched_null", "W2_rare_coherent_signal", "W3_coherent_nuisance", "W5_joint_interaction"}
)
ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_jobs(manifest: dict[str, object], output_root: Path, worlds: set[str]) -> list[dict[str, object]]:
    n_clusters = int(manifest["config"]["n_clusters"])
    jobs: list[dict[str, object]] = []
    for record in manifest["records"]:
        if str(record["world"]) not in worlds:
            continue
        source_sha256 = str(record.get("matrix_sha256") or _sha256(Path(str(record["matrix_path"]))))
        base = output_root / str(record["family"]) / str(record["world"]) / f"seed{record['seed']}"
        main = base / "main"
        jobs.append(
            {
                "run_key": f"{record['dataset_id']}::main",
                "record": record,
                "role": "main",
                "config": MAIN_CONFIG,
                "config_sha256": _sha256(MAIN_CONFIG),
                "output": main,
                "n_clusters": n_clusters,
                "source_sha256": source_sha256,
                "reused_from": None,
            }
        )
        if str(record["world"]) == "W5_joint_interaction":
            for name, config in ABLATIONS.items():
                jobs.append(
                    {
                        "run_key": f"{record['dataset_id']}::{name}",
                        "record": record,
                        "role": name,
                        "config": config,
                        "config_sha256": _sha256(config),
                        "output": base / name,
                        "n_clusters": n_clusters,
                        "source_sha256": source_sha256,
                        "reused_from": main,
                    }
                )
    return jobs


def _completed(job: dict[str, object]) -> bool:
    output = Path(job["output"])
    required = [output / "summary.json", output / "runner_profile.json", output / "T_c/metrics.json"]
    if job["role"] == "main":
        required.extend(output / arm / "metrics.json" for arm in ("N", "R", "T_s"))
        required.append(output / "branchpoint.pt")
    if any(not path.is_file() for path in required):
        return False
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        runner = json.loads((output / "runner_profile.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        summary.get("status") == "completed"
        and int(summary.get("seed", -1)) == int(job["record"]["seed"])
        and runner.get("dataset_sha256") == job["source_sha256"]
        and runner.get("config_sha256") == job["config_sha256"]
        and runner.get("labels_used_during_fit") is False
        and bool(runner.get("branchpoint_reused")) == (job["role"] != "main")
    )


def _canonical_control_ready(job: dict[str, object]) -> bool:
    if job.get("reused_from") is None:
        return False
    output = Path(job["reused_from"])
    required = [
        output / "summary.json",
        output / "runner_profile.json",
        output / "resolved_config.json",
        output / "branchpoint.pt",
    ]
    required.extend(output / arm / "metrics.json" for arm in ("N", "R", "T_s", "T_c"))
    if any(not path.is_file() for path in required):
        return False
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        runner = json.loads((output / "runner_profile.json").read_text(encoding="utf-8"))
        resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        summary.get("status") == "completed"
        and summary.get("variant") == "accg_joint"
        and resolved.get("variant") == "accg_joint"
        and int(summary.get("seed", -1)) == int(job["record"]["seed"])
        and runner.get("dataset_sha256") == job["source_sha256"]
        and runner.get("labels_used_during_fit") is False
        and runner.get("branchpoint_reused") is False
    )


def _command(job: dict[str, object], gpu: int | None, epochs: int | None) -> list[str]:
    record = job["record"]
    command = [
        sys.executable,
        "-m",
        "methods.TopoGate.ACCG_action_constrained_gate.run",
        "--data",
        str(record["matrix_path"]),
        "--dataset-name",
        str(record["dataset_id"]),
        "--input-protocol",
        "clubench_bridge",
        "--config",
        str(job["config"]),
        "--output-dir",
        str(job["output"]),
        "--seed",
        str(record["seed"]),
        "--n-clusters",
        str(job["n_clusters"]),
        "--device",
        "cpu" if gpu is None else "cuda",
    ]
    if gpu is not None:
        command.extend(["--gpu", str(gpu)])
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if job["reused_from"] is not None:
        command.extend(["--branchpoint-from", str(job["reused_from"])])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worlds", nargs="+", default=sorted(CORE_WORLDS))
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=("main", "ablation"),
        default=["main"],
        help="run main panels first and ablations only after their branchpoints exist",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None, help="engineering-only override")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--queue-state", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("manifest_id") != "accg_synthetic_w0_w5_v1":
        raise ValueError("incompatible synthetic manifest")
    worlds = set(args.worlds)
    unknown = worlds - set(manifest["worlds"])
    if unknown:
        raise ValueError(f"unknown worlds: {sorted(unknown)}")
    if args.cpu and args.gpu is not None:
        raise ValueError("--cpu and --gpu are mutually exclusive")
    gpu = None if args.cpu else args.gpu
    if args.execute and gpu is None and not args.cpu:
        raise ValueError("execution requires --cpu or an explicit --gpu")
    if gpu is not None and gpu not in ALLOWED_GPUS:
        raise ValueError(f"physical GPU {gpu} is forbidden")
    jobs = build_jobs(manifest, args.output_root, worlds)
    selected_roles = set(args.roles)
    jobs = [job for job in jobs if ("main" if job["role"] == "main" else "ablation") in selected_roles]
    if args.num_workers <= 0 or not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    jobs = [job for index, job in enumerate(jobs) if index % args.num_workers == args.worker_id]
    print(json.dumps({"execute": args.execute, "jobs": len(jobs), "gpu": gpu}, ensure_ascii=True))
    if not args.execute:
        for job in jobs:
            print(json.dumps({"run_key": job["run_key"], "reused_from": None if job["reused_from"] is None else str(job["reused_from"])}, ensure_ascii=True))
        return 0
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    failed = 0
    rows = []
    for index, job in enumerate(jobs, start=1):
        output = Path(job["output"])
        if not args.force and _completed(job):
            row = {"run_key": job["run_key"], "status": "completed", "skipped": True}
            rows.append(row)
            print(f"[{index}/{len(jobs)}] reused {job['run_key']}", flush=True)
            continue
        if job["reused_from"] is not None and not _canonical_control_ready(job):
            row = {"run_key": job["run_key"], "status": "blocked_missing_canonical_control"}
            rows.append(row)
            print(f"[{index}/{len(jobs)}] blocked {job['run_key']}: canonical main panel incomplete", flush=True)
            failed += 1
            continue
        output.mkdir(parents=True, exist_ok=True)
        Path(output / "mpl").mkdir(parents=True, exist_ok=True)
        Path(output / "numba_cache").mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(output / "mpl")
        env["NUMBA_CACHE_DIR"] = str(output / "numba_cache")
        run_record = {
            "run_key": job["run_key"],
            "status": "running",
            "role": job["role"],
            "seed": int(job["record"]["seed"]),
            "physical_gpu": gpu,
            "source_path": job["record"]["matrix_path"],
            "source_sha256": job["source_sha256"],
            "config": str(job["config"]),
            "config_sha256": job["config_sha256"],
            "reused_from": None if job["reused_from"] is None else str(job["reused_from"]),
            "labels_used_during_fit": False,
            "started_at": time.time(),
        }
        _write_json(output / "run_record.json", run_record)
        started = time.time()
        with (output / "launcher.log").open("a", encoding="utf-8") as handle:
            completed = subprocess.run(_command(job, gpu, args.epochs), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        status = "completed" if completed.returncode == 0 and _completed(job) else "incomplete_compute"
        run_record.update(
            {
                "status": status,
                "return_code": int(completed.returncode),
                "wall_seconds": float(time.time() - started),
            }
        )
        _write_json(output / "run_record.json", run_record)
        rows.append({"run_key": job["run_key"], "status": status, "return_code": int(completed.returncode)})
        failed += int(status != "completed")
        if args.queue_state is not None:
            _write_json(
                args.queue_state,
                {
                    "manifest_id": manifest["manifest_id"],
                    "execute": True,
                    "worker_id": int(args.worker_id),
                    "num_workers": int(args.num_workers),
                    "physical_gpu": gpu,
                    "rows": rows,
                    "updated_at": time.time(),
                },
            )
    if args.queue_state is not None:
        _write_json(
            args.queue_state,
            {
                "manifest_id": manifest["manifest_id"],
                "execute": True,
                "worker_id": int(args.worker_id),
                "num_workers": int(args.num_workers),
                "physical_gpu": gpu,
                "rows": rows,
                "updated_at": time.time(),
            },
        )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
