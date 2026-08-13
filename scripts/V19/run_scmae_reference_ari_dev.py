#!/usr/bin/env python
"""Run the fixed scMAE reference for V19 ARI development evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.V19.tune_ari_dev import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    FORMAL_SEEDS,
    PROTOCOL_ID,
    SELECTION_EVIDENCE,
    _check_gpu,
    _write_json,
    build_stage_spec,
    load_manifest,
    run_ari_job,
    schedule_records,
    target_records,
    write_or_validate_stage_spec,
)


REFERENCE_CANDIDATE = {
    "candidate_id": "scmae_reference",
    "family": "reference",
    "overrides": {},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="V19 fixed scMAE ARI reference runner")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    seeds = tuple(int(value) for value in args.seeds)
    if seeds != FORMAL_SEEDS:
        raise ValueError("reference contract requires seeds 42,123,7")
    records = schedule_records(args.output_dir / "reference", target_records(manifest), "reference")
    candidates = (REFERENCE_CANDIDATE,)
    spec_root = args.output_dir / "reference"
    spec_root.mkdir(parents=True, exist_ok=True)
    spec = build_stage_spec(manifest, records, candidates, "reference", seeds, args.config)
    spec.update(
        {
            "selection_evidence_type": SELECTION_EVIDENCE,
            "reference_role": "fixed_scmae_for_paired_dataset_delta",
            "candidate_count": 1,
        }
    )
    write_or_validate_stage_spec(spec_root, spec)
    worker_count = max(1, int(args.num_workers))
    if not 0 <= int(args.worker_id) < worker_count:
        raise ValueError("worker-id must be in [0, num-workers)")
    jobs = [
        (record, int(seed))
        for record in records
        for seed in seeds
    ]
    jobs = [job for index, job in enumerate(jobs) if index % worker_count == int(args.worker_id)]
    header = {**spec, "worker_id": int(args.worker_id), "num_workers": worker_count, "jobs_for_worker": len(jobs)}
    print(json.dumps(header, ensure_ascii=True), flush=True)
    if args.dry_run:
        for record, seed in jobs:
            print(f"{record['dataset_id']}\tscmae_reference\tseed={seed}")
        return 0
    physical_gpu = -1 if args.cpu else int(args.gpu)
    _check_gpu(physical_gpu)
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    if physical_gpu >= 0:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    rows = []
    for index, (record, seed) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {record['dataset_id']} scmae_reference seed={seed}", flush=True)
        row = run_ari_job(
            record,
            REFERENCE_CANDIDATE,
            seed,
            spec_root,
            stage="reference",
            config_path=args.config,
            gpu=physical_gpu,
            manifest_id=str(manifest.get("manifest_id", "unknown")),
            variant="scmae_only",
        )
        rows.append(row)
        print(json.dumps({"run_key": row.get("run_key"), "status": row.get("status")}), flush=True)
    worker = {
        **header,
        "completed": sum(row.get("status") == "completed" for row in rows),
        "incomplete_compute": sum(row.get("status") == "incomplete_compute" for row in rows),
        "runs": rows,
        "labels_used_for_selection": True,
    }
    _write_json(spec_root / f"worker{int(args.worker_id)}_{int(time.time())}.json", worker)
    return 0 if worker["incomplete_compute"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
