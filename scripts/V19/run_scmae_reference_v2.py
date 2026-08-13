#!/usr/bin/env python
"""Produce one fixed, label-free scMAE proxy reference for V19 v2 tuning."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V19_rg_adapter.config import load_config  # noqa: E402
from methods.TopoGate.V19_rg_adapter.run import resolve_runtime_device  # noqa: E402
from methods.TopoGate.V19_rg_adapter.trainer import fit_predict  # noqa: E402
from scripts.V19.tune_unsupervised_v2 import (  # noqa: E402
    ALLOWED_GPUS,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    FORMAL_SEEDS,
    PROTOCOL_ID,
    _load_manifest,
    _prepare_matrix,
    _write_json,
    select_records,
    split_rows,
    underlying_dataset_id,
)


REFERENCE_ID = "scmae_reference"
DEFAULT_REFERENCE_OUTPUT = DEFAULT_OUTPUT.parent / "v19_scmae_xonly_reference_v2_paired_20260809"


def _is_completed(path: Path, run_key: str) -> bool:
    try:
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        status.get("status") == "completed"
        and summary.get("status") == "completed"
        and summary.get("protocol_id") == PROTOCOL_ID
        and summary.get("run_key") == run_key
        and summary.get("variant") == "scmae_only"
        and summary.get("labels_accessed") is False
        and summary.get("y_key_read") is False
        and summary.get("readout_enabled") is False
    )


def _run_one(
    record: dict[str, Any],
    seed: int,
    output_root: Path,
    *,
    config_path: Path,
    gpu: int,
    max_samples: int,
    manifest_id: str,
    force: bool,
) -> dict[str, Any]:
    dataset_id = str(record["dataset_id"])
    output = output_root / dataset_id / REFERENCE_ID / f"seed{int(seed)}"
    output.mkdir(parents=True, exist_ok=True)
    run_key = f"reference::{dataset_id}::seed{int(seed)}"
    if not force and _is_completed(output, run_key):
        return {"status": "completed", "run_key": run_key, "skipped": True}
    run_record: dict[str, Any] = {
        "status": "running",
        "run_key": run_key,
        "protocol_id": PROTOCOL_ID,
        "stage": "reference",
        "manifest_id": manifest_id,
        "dataset_id": dataset_id,
        "underlying_dataset_id": underlying_dataset_id(dataset_id),
        "dataset": str(record["name"]),
        "source_path": str(record["source_path"]),
        "input_protocol": str(record["input_protocol"]),
        "variant": "scmae_only",
        "candidate_id": REFERENCE_ID,
        "seed": int(seed),
        "labels_accessed": False,
        "y_key_read": False,
        "n_clusters_used": None,
        "readout_enabled": False,
    }
    _write_json(output / "run_record.json", run_record)
    _write_json(output / "status.json", {"status": "running", "run_key": run_key, "protocol_id": PROTOCOL_ID})
    started = time.time()
    try:
        config = load_config(
            config_path,
            {"protocol_id": PROTOCOL_ID, "variant": "scmae_only"},
        )
        prepared, input_profile = _prepare_matrix(record, config, int(seed), int(max_samples))
        training_indices, validation_indices, split_seed = split_rows(prepared.shape[0], dataset_id, int(seed))
        np.save(output / "training_row_indices.npy", training_indices)
        np.save(output / "validation_row_indices.npy", validation_indices)
        runtime_device = resolve_runtime_device("cuda" if gpu >= 0 else "cpu", int(gpu))
        _predictions, embedding, diagnostics = fit_predict(
            prepared,
            n_clusters=None,
            config=config,
            seed=int(seed),
            device=runtime_device,
            evaluate_unsupervised=True,
            fit_X=prepared[training_indices],
            evaluation_X=prepared[validation_indices],
        )
        summary = {
            "status": "completed",
            "protocol_id": PROTOCOL_ID,
            "run_key": run_key,
            "stage": "reference",
            "manifest_id": manifest_id,
            "dataset_id": dataset_id,
            "underlying_dataset_id": underlying_dataset_id(dataset_id),
            "dataset": str(record["name"]),
            "input_protocol": str(record["input_protocol"]),
            "variant": "scmae_only",
            "candidate_id": REFERENCE_ID,
            "seed": int(seed),
            "device": str(runtime_device),
            "physical_gpu": int(gpu) if int(gpu) >= 0 else None,
            "n_samples": int(prepared.shape[0]),
            "fit_n_samples": int(training_indices.size),
            "evaluation_n_samples": int(validation_indices.size),
            "split_seed": int(split_seed),
            "validation_fraction": 0.20,
            "labels_accessed": False,
            "y_key_read": False,
            "n_clusters_used": None,
            "readout_enabled": False,
            "resolved_config": config.resolved_dict(),
            "input_profile": input_profile,
            "unsupervised_diagnostics": diagnostics["unsupervised_diagnostics"],
            "training_history": diagnostics["training_history"],
            "graph_profile": diagnostics["graph_profile"],
            "embedding_shape": [int(value) for value in embedding.shape],
            "wall_seconds": float(time.time() - started),
        }
        _write_json(output / "resolved_config.json", config.resolved_dict())
        _write_json(output / "input_profile.json", input_profile)
        _write_json(output / "unsupervised_diagnostics.json", summary["unsupervised_diagnostics"])
        _write_json(output / "summary.json", summary)
        _write_json(output / "status.json", {"status": "completed", "run_key": run_key, "protocol_id": PROTOCOL_ID})
        run_record.update({"status": "completed", "summary": "summary.json", "wall_seconds": summary["wall_seconds"]})
    except Exception as exc:
        run_record.update(
            {
                "status": "incomplete_compute",
                "wall_seconds": float(time.time() - started),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(
            output / "status.json",
            {
                "status": "incomplete_compute",
                "protocol_id": PROTOCOL_ID,
                "run_key": run_key,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
    _write_json(output / "run_record.json", run_record)
    return run_record


def main() -> int:
    parser = argparse.ArgumentParser(description="V19 fixed scMAE X-only reference")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REFERENCE_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--comparable-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or any(seed not in FORMAL_SEEDS for seed in seeds):
        raise ValueError(f"seeds must be drawn from {FORMAL_SEEDS}")
    records = select_records(manifest, args.groups, comparable_only=bool(args.comparable_only))
    worker_count = max(1, int(args.num_workers))
    if not 0 <= int(args.worker_id) < worker_count:
        raise ValueError("worker-id must be in [0, num-workers)")
    jobs = [
        (record, seed)
        for seed in seeds
        for record in records
    ]
    jobs = [job for index, job in enumerate(jobs) if index % worker_count == int(args.worker_id)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "protocol_id": PROTOCOL_ID,
        "stage": "reference",
        "manifest_id": manifest.get("manifest_id"),
        "candidate_id": REFERENCE_ID,
        "dataset_ids": [str(record["dataset_id"]) for record in records],
        "underlying_dataset_ids": sorted({underlying_dataset_id(str(record["dataset_id"])) for record in records}),
        "comparable_only": bool(args.comparable_only),
        "seeds": list(seeds),
        "expected_runs": len(records) * len(seeds),
        "validation_protocol": "fixed_held_out_rows_20pct_per_dataset_seed",
        "selection_uses_labels_or_outcomes": False,
        "labels_accessed": False,
        "y_key_read": False,
        "readout_enabled": False,
    }
    spec_path = args.output_dir / "stage_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise ValueError(f"existing reference stage spec does not match: {spec_path}")
    if not spec_path.exists():
        _write_json(spec_path, spec)
    print(json.dumps({**spec, "worker_id": int(args.worker_id), "jobs_for_worker": len(jobs)}, ensure_ascii=True), flush=True)
    if args.dry_run:
        return 0

    physical_gpu = -1 if args.cpu else int(args.gpu)
    if physical_gpu >= 0 and physical_gpu not in ALLOWED_GPUS:
        raise ValueError(f"GPU {physical_gpu} is forbidden; allowed GPUs are {sorted(ALLOWED_GPUS)}")
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    if physical_gpu >= 0:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    rows = []
    for index, (record, seed) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {record['dataset_id']} reference seed={seed}", flush=True)
        row = _run_one(
            record,
            int(seed),
            args.output_dir,
            config_path=args.config,
            gpu=physical_gpu,
            max_samples=int(args.max_samples),
            manifest_id=str(manifest.get("manifest_id", "unknown")),
            force=bool(args.force),
        )
        rows.append(row)
        print(json.dumps({"run_key": row.get("run_key"), "status": row.get("status")}), flush=True)
    _write_json(
        args.output_dir / f"reference_worker{int(args.worker_id)}_{int(time.time())}.json",
        {
            **spec,
            "worker_id": int(args.worker_id),
            "completed": sum(row.get("status") == "completed" for row in rows),
            "incomplete_compute": sum(row.get("status") == "incomplete_compute" for row in rows),
            "runs": rows,
        },
    )
    return 0 if all(row.get("status") == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
