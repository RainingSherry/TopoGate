#!/usr/bin/env python
"""Run one complete V19 seed batch while preserving per-run audit artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PROTOCOL_ID = "v19_rg_selected_advantage_v1"
DEFAULT_CONFIG = ROOT / "methods" / "TopoGate" / "V19_rg_adapter" / "configs" / "v19_rg.yaml"
DEFAULT_VARIANTS = ("scmae_only", "rg_full")
FORMAL_SEEDS = (42, 123, 7)
ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError(
            f"V19 requires protocol_id={EXPECTED_PROTOCOL_ID!r}, got {payload.get('protocol_id')!r}"
        )
    if payload.get("selection_policy", {}).get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("V19 manifest must declare label/outcome-independent selection")
    records = payload.get("datasets", [])
    if len(records) != 11:
        raise ValueError(f"V19 fixed manifest must contain 11 input strata, got {len(records)}")
    return payload


def _run_dir(output_root: Path, record: dict[str, Any], variant: str, seed: int) -> Path:
    return output_root / str(record["dataset_id"]) / variant / f"seed{int(seed)}"


def _is_completed(path: Path) -> bool:
    try:
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return status.get("status") == "completed" and summary.get("status") == "completed"


def _run_one(
    record: dict[str, Any],
    variant: str,
    seed: int,
    output_root: Path,
    *,
    config: Path,
    gpu: int,
    max_samples: int,
    manifest_id: str,
    force: bool,
) -> dict[str, Any]:
    output = _run_dir(output_root, record, variant, seed)
    output.mkdir(parents=True, exist_ok=True)
    run_key = f"{record['dataset_id']}::{variant}::seed{int(seed)}"
    if not force and _is_completed(output):
        return {"status": "completed", "run_key": run_key, "skipped": True}
    run_record: dict[str, Any] = {
        "status": "running",
        "run_key": run_key,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "manifest_id": manifest_id,
        "dataset_id": record["dataset_id"],
        "dataset": record["name"],
        "source_path": record["source_path"],
        "source_hash": record.get("source_hash", "unavailable"),
        "input_protocol": record["input_protocol"],
        "variant": variant,
        "seed": int(seed),
        "gpu": int(gpu),
        "labels_used_during_fit": False,
        "K_source": "benchmark_oracle_from_y",
    }
    _write(output / "run_record.json", run_record)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "V19" / "run.py"),
        "--data-path",
        str(record["source_path"]),
        "--save-dir",
        str(output),
        "--dataset-name",
        str(record["name"]),
        "--dataset-id",
        str(record["dataset_id"]),
        "--input-protocol",
        str(record["input_protocol"]),
        "--variant",
        variant,
        "--config",
        str(config),
        "--seed",
        str(int(seed)),
        "--max-samples",
        str(int(max_samples)),
        "--device",
        "cuda" if gpu >= 0 else "cpu",
        "--gpu",
        str(int(gpu if gpu >= 0 else 1)),
    ]
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    if gpu >= 0:
        if gpu not in ALLOWED_GPUS:
            raise ValueError(f"GPU {gpu} is forbidden; allowed physical GPUs are {sorted(ALLOWED_GPUS)}")
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    log_path = output / "launcher.log"
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=None,
            )
        summary_path = output / "summary.json"
        if completed.returncode == 0 and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["manifest_id"] = manifest_id
            _write(summary_path, summary)
            run_record.update(
                {
                    "status": "completed",
                    "wall_seconds": float(time.time() - started),
                    "metrics": summary.get("metrics", {}),
                    "summary": "summary.json",
                }
            )
        else:
            run_record.update(
                {
                    "status": "incomplete_compute",
                    "wall_seconds": float(time.time() - started),
                    "returncode": int(completed.returncode),
                    "log": str(log_path),
                }
            )
    except Exception as exc:
        run_record.update(
            {
                "status": "incomplete_compute",
                "wall_seconds": float(time.time() - started),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    _write(output / "run_record.json", run_record)
    return run_record


def main() -> int:
    parser = argparse.ArgumentParser(description="V19 one-seed matrix launcher")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "result" / "V19" / "v19_rg_selected_advantage_v1",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest)
    variants = tuple(args.variants or DEFAULT_VARIANTS)
    unknown = set(variants) - set(DEFAULT_VARIANTS)
    if unknown:
        raise ValueError(f"unknown V19 variants: {sorted(unknown)}")
    requested = set(args.datasets or [])
    records = [
        row
        for row in manifest["datasets"]
        if row.get("status") == "eligible"
        and (not requested or str(row.get("dataset_id")) in requested)
    ]
    jobs = [(row, variant) for row in records for variant in variants]
    worker_count = max(1, int(args.num_workers))
    if not 0 <= int(args.worker_id) < worker_count:
        raise ValueError("worker-id must be in [0, num-workers)")
    jobs = [job for index, job in enumerate(jobs) if index % worker_count == int(args.worker_id)]
    if int(args.limit) > 0:
        jobs = jobs[: int(args.limit)]
    header = {
        "manifest_id": manifest.get("manifest_id"),
        "seed_batch": int(args.seed),
        "jobs": len(jobs),
        "variants": list(variants),
        "output": str(args.output_dir),
        "formal_seed_order": list(FORMAL_SEEDS),
    }
    print(json.dumps(header, ensure_ascii=True), flush=True)
    if args.dry_run:
        for row, variant in jobs:
            print(f"{row['dataset_id']}\t{variant}\tseed={int(args.seed)}")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    physical_gpu = -1 if args.cpu else int(args.gpu)
    if physical_gpu >= 0 and physical_gpu not in ALLOWED_GPUS:
        raise ValueError(
            f"GPU {physical_gpu} is forbidden; allowed physical GPUs are {sorted(ALLOWED_GPUS)}"
        )
    rows = []
    for index, (record, variant) in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {record['dataset_id']} {variant} seed={int(args.seed)}",
            flush=True,
        )
        row = _run_one(
            record,
            variant,
            int(args.seed),
            args.output_dir,
            config=args.config,
            gpu=physical_gpu,
            max_samples=int(args.max_samples),
            manifest_id=str(manifest.get("manifest_id", "unknown")),
            force=bool(args.force),
        )
        rows.append(row)
        print(json.dumps({"run_key": row.get("run_key"), "status": row.get("status")}), flush=True)
    worker_summary = {
        **header,
        "completed": sum(row.get("status") == "completed" for row in rows),
        "incomplete_compute": sum(row.get("status") == "incomplete_compute" for row in rows),
        "runs": rows,
    }
    _write(
        args.output_dir / f"matrix_seed{int(args.seed)}_worker{int(args.worker_id)}.json",
        worker_summary,
    )
    print(
        json.dumps(
            {
                "completed": worker_summary["completed"],
                "incomplete_compute": worker_summary["incomplete_compute"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
