#!/usr/bin/env python
"""Run the preregistered sparse/high-dimensional RG/scMAE extension panel."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V19_rg_adapter.config import VARIANTS, load_config
from methods.TopoGate.V19_rg_adapter.run import run_one


ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})
SEEDS = (42, 123, 7)
EXTENSION_PROTOCOLS = frozenset(
    {
        "v19_rg_extended_sparse_v1",
        "v19_rg_extended_sparse_batch2_v1",
    }
)
DEFAULT_CONFIG = ROOT / "methods" / "TopoGate" / "V19_rg_adapter" / "configs" / "v19_rg.yaml"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") not in EXTENSION_PROTOCOLS:
        raise ValueError(f"extended runner requires one of {sorted(EXTENSION_PROTOCOLS)}")
    if payload.get("selection_policy", {}).get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("extension manifest selection must be outcome-independent")
    records = [row for row in payload.get("datasets", []) if row.get("status") == "eligible"]
    if not records:
        raise ValueError("extension manifest has no eligible datasets")
    for row in records:
        for key in ("dataset_id", "name", "source_path", "input_protocol"):
            if not row.get(key):
                raise ValueError(f"manifest record missing {key}: {row}")
        if not Path(str(row["source_path"])).is_file():
            raise FileNotFoundError(f"dataset source is missing: {row['source_path']}")
    return {**payload, "datasets": records}


def _completed(path: Path, run_key: str) -> bool:
    try:
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    expected_suffix = "::".join(run_key.split("::")[-3:])
    actual_key = str(summary.get("run_key", ""))
    return (
        status.get("status") == "completed"
        and summary.get("status") == "completed"
        and (actual_key == run_key or actual_key.endswith(expected_suffix))
    )


def _annotate(output: Path, record: dict[str, Any], manifest_id: str, protocol_id: str) -> None:
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "manifest_id": manifest_id,
            "experiment_protocol_id": protocol_id,
            "run_key": f"{manifest_id}::{record['dataset_id']}::{summary['variant']}::seed{int(summary['seed'])}",
            "source_hash": record.get("processed_sha256", record.get("source_hash", "unavailable")),
            "source_kind": record.get("source_kind"),
            "source_identity": record.get("source_identity"),
            "source_provenance_status": record.get("source_provenance_status", "recorded"),
            "family": record.get("family"),
            "selection_basis": record.get("selection_basis"),
            "comparison_scope": record.get("comparison_scope", "external_highdim_bridge_only"),
            "selection_uses_labels_or_outcomes": False,
            "selection_evidence_type": "fixed pre-registered extension panel",
        }
    )
    _write(summary_path, summary)
    record_path = output / "run_record.json"
    run_record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    run_record.update(
        {
            "manifest_id": manifest_id,
            "experiment_protocol_id": protocol_id,
            "run_key": summary["run_key"],
            "source_hash": summary["source_hash"],
            "source_kind": summary["source_kind"],
            "source_identity": summary["source_identity"],
            "source_provenance_status": summary["source_provenance_status"],
            "family": summary["family"],
            "comparison_scope": summary["comparison_scope"],
            "selection_uses_labels_or_outcomes": False,
        }
    )
    _write(record_path, run_record)


def _jobs(manifest: dict[str, Any], variants: tuple[str, ...], seeds: tuple[int, ...], requested: set[str]) -> list[tuple[dict[str, Any], str, int]]:
    records = [row for row in manifest["datasets"] if not requested or str(row["dataset_id"]) in requested]
    if requested and len(records) != len(requested):
        missing = sorted(requested - {str(row["dataset_id"]) for row in records})
        raise ValueError(f"unknown extension dataset ids: {missing}")
    # Size ordering keeps small tasks useful for early diagnostics while the
    # manifest order and run keys remain unchanged.
    records = sorted(records, key=lambda row: (int(row.get("profile", {}).get("n_samples", 0)) * int(row.get("profile", {}).get("n_features", 0)), str(row["dataset_id"])))
    return [(record, variant, int(seed)) for record in records for variant in variants for seed in seeds]


def _run_job(
    record: dict[str, Any],
    variant: str,
    seed: int,
    *,
    output_root: Path,
    config_path: Path,
    gpu: int,
    max_samples: int,
    manifest_id: str,
    protocol_id: str,
    force: bool,
) -> dict[str, Any]:
    dataset_id = str(record["dataset_id"])
    output = output_root / dataset_id / variant / f"seed{seed}"
    run_key = f"{manifest_id}::{dataset_id}::{variant}::seed{seed}"
    if not force and _completed(output, f"{dataset_id}::{variant}::seed{seed}"):
        return {"status": "completed", "run_key": run_key, "skipped": True}
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = load_config(config_path, {"variant": variant})
    initial = {
        "status": "running",
        "run_key": run_key,
        "protocol_id": config.protocol_id,
        "manifest_id": manifest_id,
        "dataset_id": dataset_id,
        "dataset": record["name"],
        "source_path": record["source_path"],
        "source_hash": record.get("processed_sha256", record.get("source_hash", "unavailable")),
        "input_protocol": record["input_protocol"],
        "variant": variant,
        "seed": seed,
        "gpu": gpu,
        "labels_used_during_fit": False,
        "labels_used_during_preprocessing": False,
        "labels_used_for_graph": False,
        "labels_used_for_gate": False,
        "labels_used_for_loss": False,
        "labels_used_for_selection": False,
        "comparison_scope": record.get("comparison_scope", "external_highdim_bridge_only"),
    }
    _write(output / "run_record.json", initial)
    try:
        summary = run_one(
            record["source_path"],
            output,
            config=config,
            input_protocol=str(record["input_protocol"]),
            seed=seed,
            device="cpu" if gpu < 0 else "cuda:0",
            dataset_name=str(record["name"]),
            dataset_id=dataset_id,
            n_clusters=None,
            max_samples=max_samples,
        )
        _annotate(output, record, manifest_id, protocol_id)
        final = json.loads((output / "run_record.json").read_text(encoding="utf-8"))
        final.update({"status": "completed", "wall_seconds": float(time.time() - started)})
        _write(output / "run_record.json", final)
        return {"status": "completed", "run_key": run_key, "metrics": summary.get("metrics", {})}
    except Exception as exc:
        failure = {
            **initial,
            "status": "incomplete_compute",
            "wall_seconds": float(time.time() - started),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write(output / "run_record.json", failure)
        return {"status": "incomplete_compute", "run_key": run_key, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest)
    variants = tuple(args.variants or sorted(VARIANTS))
    if not variants or set(variants) - set(VARIANTS):
        raise ValueError(f"variants must be a subset of {sorted(VARIANTS)}")
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or any(seed <= 0 for seed in seeds):
        raise ValueError("seeds must be positive")
    if args.cpu:
        gpu = -1
    else:
        gpu = int(args.gpu)
        if gpu not in ALLOWED_GPUS:
            raise ValueError(f"GPU {gpu} is forbidden; allowed physical GPUs are {sorted(ALLOWED_GPUS)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    worker_count = max(1, int(args.num_workers))
    worker_id = int(args.worker_id)
    if not 0 <= worker_id < worker_count:
        raise ValueError("worker-id must be in [0, num-workers)")
    requested = set(args.datasets or [])
    all_jobs = _jobs(manifest, variants, seeds, requested)
    jobs = [job for index, job in enumerate(all_jobs) if index % worker_count == worker_id]
    header = {
        "manifest_id": manifest["manifest_id"],
        "protocol_id": manifest["protocol_id"],
        "jobs_total": len(all_jobs),
        "jobs_for_worker": len(jobs),
        "worker_id": worker_id,
        "num_workers": worker_count,
        "gpu": gpu,
        "variants": list(variants),
        "seeds": list(seeds),
        "environment": {"python": platform.python_version()},
    }
    print(json.dumps(header, ensure_ascii=True), flush=True)
    if args.dry_run:
        for record, variant, seed in jobs:
            print(f"{record['dataset_id']}\t{variant}\tseed={seed}")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, (record, variant, seed) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {record['dataset_id']} {variant} seed={seed}", flush=True)
        row = _run_job(
            record,
            variant,
            seed,
            output_root=args.output_dir,
            config_path=args.config,
            gpu=gpu,
            max_samples=int(args.max_samples),
            manifest_id=str(manifest["manifest_id"]),
            protocol_id=str(manifest["protocol_id"]),
            force=bool(args.force),
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    worker_summary = {
        **header,
        "completed": sum(row.get("status") == "completed" for row in rows),
        "incomplete_compute": sum(row.get("status") == "incomplete_compute" for row in rows),
        "runs": rows,
    }
    _write(args.output_dir / f"matrix_worker{worker_id}.json", worker_summary)
    return 0 if worker_summary["incomplete_compute"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
