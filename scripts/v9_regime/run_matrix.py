#!/usr/bin/env python3
from __future__ import annotations

"""Run the frozen V9 Full/NoMix/control matrix from a locked manifest."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from scripts.v9_regime.protocol import (
    BASE_OVERRIDES,
    CASE_SEEDS,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_TMP_ROOT,
    VARIANT_OVERRIDES,
    get_record,
    load_xy,
    read_manifest,
    standardize_x,
    write_json,
)


def _load_split(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["dataset_id"]: row["split"] for row in payload.get("assignments", [])}


def _load_panel(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("panel must declare selection_uses_labels_or_outcomes=false")
    panel_ids = payload.get("panel_ids")
    if panel_ids is None:
        panel_ids = [row["dataset_id"] for row in payload.get("assignments", [])]
    return {str(dataset_id) for dataset_id in panel_ids}


def _stage_defaults(stage: str) -> tuple[list[str], tuple[int, ...], str | None]:
    if stage == "screen":
        return ["full", "nomix"], (42,), None
    if stage == "main":
        return ["full", "nomix"], DEFAULT_SEEDS, None
    if stage == "ablation":
        return list(VARIANT_OVERRIDES), DEFAULT_SEEDS, "discovery"
    if stage == "confirmation":
        return ["full", "nomix"], DEFAULT_SEEDS, "confirmation"
    if stage == "case":
        return ["full", "nomix"], CASE_SEEDS, "discovery"
    raise ValueError(f"unknown stage: {stage}")


def _sample_data(record: dict[str, Any], seed: int, max_samples: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x, y = load_xy(Path(record["source_path"]))
    if y is None:
        raise ValueError("eligible benchmark dataset has no labels")
    original_k = int(record["n_clusters"])
    # Fit the single X-only scaler on the complete input, matching Stage 0.
    # Sampling is applied only after this step and is deterministic per seed.
    x, preprocessing = standardize_x(x)
    indices = None
    if max_samples > 0 and x.shape[0] > max_samples:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(x.shape[0], size=int(max_samples), replace=False))
        x = x[indices]
        y = y[indices]
    meta = {
        "original_n": int(record["n"]),
        "run_n": int(x.shape[0]),
        "d": int(x.shape[1]),
        "original_k": original_k,
        "run_unique_labels": int(np.unique(y).size),
        "row_sampling": indices is not None,
        "row_sampling_seed": int(seed),
        "labels_used_during_fit": False,
        "preprocessing": preprocessing,
    }
    return x, y, meta


def _completed(path: Path, retry_errors: bool) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("status") == "completed":
        return True
    return payload.get("status") == "error" and not retry_errors


def run_one(
    record: dict[str, Any],
    variant: str,
    seed: int,
    output_root: Path,
    gpu: int,
    no_cuda: bool,
    max_samples: int,
    force: bool,
    retry_errors: bool,
) -> dict[str, Any]:
    dataset_id = str(record["dataset_id"])
    out_dir = output_root / dataset_id / variant / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "run_record.json"
    if not force and _completed(record_path, retry_errors):
        return json.loads(record_path.read_text(encoding="utf-8"))

    started = time.time()
    run_record: dict[str, Any] = {
        "protocol_id": "v9_regime_protocol_v1",
        "dataset_id": dataset_id,
        "dataset": record.get("name"),
        "source_path": record.get("source_path"),
        "source_identity": record.get("source_identity"),
        "source_version": record.get("source_version"),
        "variant": variant,
        "seed": int(seed),
        "status": "running",
        "labels_used_during_fit": False,
        "k_source": "manifest_labels_unique",
        "benchmark_n_clusters": int(record["n_clusters"]),
        "gpu": int(gpu),
        "no_cuda": bool(no_cuda),
        "resolved_overrides": {**BASE_OVERRIDES, **VARIANT_OVERRIDES[variant]},
    }
    write_json(record_path, run_record)
    try:
        x, y, run_meta = _sample_data(record, seed, max_samples)
        run_record.update(run_meta)
        from methods.TopoGate.learnable_gate.run_npz import compute_metrics, run_topogate

        overrides = {**BASE_OVERRIDES, **VARIANT_OVERRIDES[variant]}
        overrides.pop("variant", None)
        overrides.update({"dataset_name": str(record.get("name") or dataset_id)})
        # y is deliberately not passed to the training API.  K is benchmark
        # metadata only; metrics are computed in this outer process.
        predictions, elapsed, _ = run_topogate(
            x,
            n_clusters=int(record["n_clusters"]),
            y=None,
            gpu=int(gpu),
            variant="learnable_gate_v9_adaptive",
            config_dir=overrides.pop("config_dir"),
            seed=int(seed),
            return_metrics=True,
            save_dir=str(out_dir),
            **overrides,
        )
        predictions = np.asarray(predictions, dtype=np.int64)
        metrics = compute_metrics(y, predictions)
        np.save(out_dir / "predictions.npy", predictions)
        np.save(out_dir / "labels_true.npy", np.asarray(y, dtype=np.int64))
        run_record.update(
            {
                "status": "completed",
                "elapsed_seconds": float(elapsed),
                "wall_seconds": float(time.time() - started),
                "metrics": metrics,
                "prediction_count": int(predictions.size),
                "output_semantics": {
                    "predictions": "predictions.npy",
                    "labels_true": "labels_true.npy",
                    "embedding": "embedding_final.npy",
                    "legacy_labels": "disabled",
                },
            }
        )
    except Exception as exc:
        run_record.update(
            {
                "status": "error",
                "wall_seconds": float(time.time() - started),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    write_json(record_path, run_record)
    return run_record


def _select_records(
    manifest: dict[str, Any],
    split: dict[str, str],
    stage: str,
    requested: set[str] | None,
    panel: set[str] | None,
) -> list[dict[str, Any]]:
    _, _, target_split = _stage_defaults(stage)
    if stage == "ablation" and not panel:
        raise ValueError("ablation stage requires a locked X-only --panel")
    records = []
    for record in manifest["datasets"]:
        if record.get("status") != "eligible":
            continue
        dataset_id = str(record["dataset_id"])
        if requested and dataset_id not in requested:
            continue
        if panel and dataset_id not in panel:
            continue
        if target_split and split.get(dataset_id) != target_split:
            continue
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--stage", choices=["screen", "main", "ablation", "confirmation", "case"], default="screen")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    split = _load_split(args.split)
    panel = _load_panel(args.panel)
    default_variants, default_seeds, _ = _stage_defaults(args.stage)
    variants = args.variants or default_variants
    seeds = tuple(args.seeds) if args.seeds is not None else default_seeds
    unknown = sorted(set(variants) - set(VARIANT_OVERRIDES))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    records = _select_records(manifest, split, args.stage, set(args.datasets) if args.datasets else None, panel)
    jobs = [(record, variant, seed) for record in records for variant in variants for seed in seeds]
    jobs = [job for index, job in enumerate(jobs) if index % max(1, args.num_workers) == args.worker_id]
    if args.limit > 0:
        jobs = jobs[: args.limit]
    print(json.dumps({"stage": args.stage, "jobs": len(jobs), "variants": variants, "seeds": list(seeds), "output": str(args.output_dir)}))
    if args.dry_run:
        for record, variant, seed in jobs[:20]:
            print(f"{record['dataset_id']}\t{variant}\tseed={seed}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (record, variant, seed) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {record['dataset_id']} {variant} seed={seed}", flush=True)
        row = run_one(
            record,
            variant,
            seed,
            args.output_dir,
            args.gpu,
            args.no_cuda,
            args.max_samples,
            args.force,
            args.retry_errors,
        )
        rows.append(row)
        if row.get("status") == "completed":
            print(f"  ARI={row['metrics']['ari']:.4f} NMI={row['metrics']['nmi']:.4f}", flush=True)
        else:
            print(f"  ERROR={row.get('error')}", flush=True)
    write_json(args.output_dir / f"{args.stage}_worker{args.worker_id}.json", {"jobs": rows})
    print(json.dumps({"completed": sum(row.get("status") == "completed" for row in rows), "errors": sum(row.get("status") == "error" for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
