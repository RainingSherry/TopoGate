#!/usr/bin/env python
"""Run every registered V18 run key and preserve completed/incomplete statuses."""

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
DEFAULT_CONFIG = ROOT / "methods" / "TopoGate" / "V18_scmae_latent_gate" / "configs" / "v18_mainline.yaml"
EXPECTED_PROTOCOL_ID = "v18_scmae_mainline_v2_2"
DEFAULT_VARIANTS = [
    "scmae_only", "latent_candidate_spectral", "latent_C_exactzero", "latent_GW_frozen", "v18_full",
    "v18_shuffled_E0", "v18_no_recurrence", "v18_no_stability", "v18_mask04", "v18_leiden",
]
DEFAULT_SEEDS = (42, 123, 7)
ALLOWED_GPUS = {1, 2, 3, 4, 5, 6}


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError(
            f"V18 independent runner requires protocol_id={EXPECTED_PROTOCOL_ID!r}; "
            f"got {payload.get('protocol_id')!r}"
        )
    if payload.get("selection_policy", {}).get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("V18 manifest must declare label/outcome-independent selection")
    return payload


def _run_key_dir(output_root: Path, record: dict[str, Any], variant: str, seed: int) -> Path:
    return output_root / str(record["dataset_id"]) / variant / f"seed{int(seed)}"


def _is_completed(path: Path) -> bool:
    summary = path / "summary.json"
    status = path / "status.json"
    if not summary.exists() or not status.exists():
        return False
    try:
        return json.loads(status.read_text(encoding="utf-8")).get("status") == "completed" and \
            json.loads(summary.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def _run_one(record: dict[str, Any], variant: str, seed: int, output_root: Path, *, config: Path,
             gpu: int, max_samples: int, manifest_id: str, force: bool) -> dict[str, Any]:
    output = _run_key_dir(output_root, record, variant, seed)
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "run_record.json"
    if not force and _is_completed(output):
        return {"status": "completed", "run_key": f"{record['dataset_id']}::{variant}::seed{seed}", "skipped": True}
    run_record = {
        "status": "running", "run_key": f"{record['dataset_id']}::{variant}::seed{seed}",
        "protocol_id": EXPECTED_PROTOCOL_ID, "manifest_id": manifest_id,
        "dataset_id": record["dataset_id"], "dataset": record.get("name"),
        "source_path": record.get("source_path"), "variant": variant, "seed": int(seed),
        "gpu": int(gpu), "labels_used_during_fit": False,
        "k_source": "not_applicable_leiden" if variant == "v18_leiden" else "benchmark_oracle_from_y",
    }
    _write(record_path, run_record)
    log_path = output / "launcher.log"
    command = [sys.executable, str(ROOT / "scripts" / "V18" / "run.py"),
               "--data-path", str(record["source_path"]), "--save-dir", str(output),
               "--dataset-name", str(record.get("name") or record["dataset_id"]),
               "--dataset-id", str(record["dataset_id"]), "--variant", variant,
               "--config", str(config), "--seed", str(seed), "--max-samples", str(max_samples),
               "--device", "cuda" if gpu >= 0 else "cpu"]
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    if gpu >= 0:
        if gpu not in ALLOWED_GPUS:
            raise ValueError(f"GPU {gpu} is forbidden; allowed physical GPUs are {sorted(ALLOWED_GPUS)}")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       check=False, timeout=None)
        summary_path = output / "summary.json"
        if completed.returncode == 0 and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["manifest_id"] = manifest_id
            _write(summary_path, summary)
            run_record.update({"status": "completed", "wall_seconds": time.time() - started,
                               "metrics": summary.get("metrics", {}), "summary": "summary.json"})
        else:
            run_record.update({"status": "incomplete_compute", "wall_seconds": time.time() - started,
                               "returncode": int(completed.returncode), "log": str(log_path)})
    except Exception as exc:
        run_record.update({"status": "incomplete_compute", "wall_seconds": time.time() - started,
                           "error_type": type(exc).__name__, "error": str(exc)})
    _write(record_path, run_record)
    return run_record


def main() -> int:
    parser = argparse.ArgumentParser(description="V18 full matrix launcher")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result" / "V18" / "v18_scmae_mainline_v2_2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest)
    variants = args.variants or DEFAULT_VARIANTS
    seeds = tuple(args.seeds) if args.seeds is not None else DEFAULT_SEEDS
    available = {str(x) for x in variants}
    unknown = available - set(DEFAULT_VARIANTS)
    if unknown:
        raise ValueError(f"unknown V18 variants: {sorted(unknown)}")
    requested = set(args.datasets or [])
    records = [row for row in manifest.get("datasets", []) if row.get("status") == "eligible" and
               (not requested or str(row.get("dataset_id")) in requested)]
    jobs = [(row, variant, seed) for row in records for variant in variants for seed in seeds]
    jobs = [job for index, job in enumerate(jobs) if index % max(1, args.num_workers) == args.worker_id]
    if args.limit > 0:
        jobs = jobs[:args.limit]
    print(json.dumps({"manifest_id": manifest.get("manifest_id"), "jobs": len(jobs), "variants": variants,
                      "seeds": list(seeds), "output": str(args.output_dir)}, ensure_ascii=True))
    if args.dry_run:
        for row, variant, seed in jobs[:20]:
            print(f"{row['dataset_id']}\t{variant}\tseed={seed}")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    physical_gpu = -1 if args.cpu else args.gpu
    if physical_gpu >= 0 and physical_gpu not in ALLOWED_GPUS:
        raise ValueError(f"GPU {physical_gpu} is forbidden; allowed physical GPUs are {sorted(ALLOWED_GPUS)}")
    for index, (record, variant, seed) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {record['dataset_id']} {variant} seed={seed}", flush=True)
        row = _run_one(record, variant, seed, args.output_dir, config=args.config, gpu=physical_gpu,
                       max_samples=args.max_samples, manifest_id=str(manifest.get("manifest_id", "unknown")),
                       force=args.force)
        rows.append(row)
        print(json.dumps({"run_key": row.get("run_key"), "status": row.get("status")}, ensure_ascii=True), flush=True)
    _write(args.output_dir / f"matrix_worker{args.worker_id}.json", {"manifest_id": manifest.get("manifest_id"), "runs": rows})
    print(json.dumps({"completed": sum(x.get("status") == "completed" for x in rows),
                      "incomplete_compute": sum(x.get("status") == "incomplete_compute" for x in rows)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
