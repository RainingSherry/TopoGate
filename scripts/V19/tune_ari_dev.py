#!/usr/bin/env python
"""V19 RG-NeighborMix-scMAE ARI-selected development tuner.

This module is intentionally independent of the historical X-only tuners.
Ground-truth labels are loaded by the outer runner only to derive benchmark K
and to compute post-fit metrics.  They are never passed to preprocessing,
graph construction, reliability, gates, NeighborMix, or the loss.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V19_rg_adapter.config import V19Config, load_config  # noqa: E402
from methods.TopoGate.V19_rg_adapter.run import (  # noqa: E402
    ALLOWED_PHYSICAL_GPUS,
    resolve_runtime_device,
    run_one,
)
from scripts.V19.tune_unsupervised_v2 import MECHANISM_CANDIDATES  # noqa: E402


PROTOCOL_ID = "v19_rg_ari_dev_tuning_v1"
FINAL_PROTOCOL_ID = "v19_rg_ari_dev_final_v1"
MANIFEST_PROTOCOL_ID = "v19_rg_selected_advantage_v1"
DEFAULT_CONFIG = ROOT / "methods" / "TopoGate" / "V19_rg_adapter" / "configs" / "v19_rg_ari_dev.yaml"
DEFAULT_OUTPUT = ROOT / "result" / "V19" / "v19_rg_ari_dev_tuning_v1"
FORMAL_SEEDS = (42, 123, 7)
SCREEN_SEEDS = (42,)
TARGET_DATASET_IDS = (
    "mouse_retina__clubench_bridge",
    "campbell__clubench_bridge",
    "baron_human__clubench_bridge",
    "sms_spam_collection__shared_text",
    "cnae9__shared_text",
    "imdb__shared_text",
    "hate_speech__shared_text",
    "sentiment_labeld_sentences__shared_text",
)
VARIANT_ORDER = (
    "rg_full",
    "scmae_only",
    "rg_default",
    "rg_nomix",
    "rg_reliability_off",
    "rg_constant_gate",
)
SELECTION_EVIDENCE = "ARI-selected development evidence"
LOCKED_BACKBONE = {
    "hidden_size": 128,
    "epochs": 80,
    "batch_size": 256,
    "lr": 1e-3,
    "mask_ratio": 0.4,
    "dropout": 0.0,
    "masked_data_weight": 0.75,
    "mask_loss_weight": 0.7,
    "n_top_features": 1000,
    "target_sum": 10000.0,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("protocol_id") != MANIFEST_PROTOCOL_ID:
        raise ValueError("ARI development requires the frozen V19 selected-data manifest")
    rows = payload.get("datasets", [])
    by_id = {str(row.get("dataset_id")): row for row in rows if row.get("status") == "eligible"}
    if not set(TARGET_DATASET_IDS).issubset(set(by_id)):
        raise ValueError(
            "ARI development manifest must contain all 8 target bridge/shared-text layers; "
            f"missing={sorted(set(TARGET_DATASET_IDS) - set(by_id))}, "
            f"manifest_eligible_count={len(by_id)}"
        )
    for dataset_id in TARGET_DATASET_IDS:
        row = by_id[dataset_id]
        protocol = str(row.get("input_protocol"))
        scope = str(row.get("comparison_scope"))
        if protocol not in {"clubench_bridge", "shared_text"} or scope != "archived_sota_bridge_eligible":
            raise ValueError(f"target layer is not bridge/shared-text comparable: {dataset_id}")
    if payload.get("selection_policy", {}).get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("manifest selection policy must remain label-independent")
    return payload


def target_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = {str(row["dataset_id"]): row for row in manifest["datasets"]}
    return [rows[dataset_id] for dataset_id in TARGET_DATASET_IDS]


def _schedule_cost(record: dict[str, Any]) -> tuple[float, int, str]:
    source = Path(str(record["source_path"]))
    try:
        size = int(source.stat().st_size)
    except OSError:
        size = 1 << 62
    multiplier = {"shared_text": 0.8, "clubench_bridge": 1.6}.get(
        str(record.get("input_protocol")), 1.0
    )
    return float(size) * multiplier, size, str(record["dataset_id"])


def schedule_records(output_root: Path, records: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    ordered = sorted(records, key=_schedule_cost)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "schedule": "small_first",
        "ordered_dataset_ids": [str(row["dataset_id"]) for row in ordered],
        "costs": [
            {
                "dataset_id": str(row["dataset_id"]),
                "input_protocol": str(row["input_protocol"]),
                "source_bytes": int(_schedule_cost(row)[1]),
                "estimated_cost": float(_schedule_cost(row)[0]),
            }
            for row in ordered
        ],
        "label_free": True,
        "purpose": "queue_order_only",
    }
    path = output_root / "schedule_spec.json"
    if path.is_file():
        existing = _read_json(path)
        immutable = ("protocol_id", "stage", "schedule", "ordered_dataset_ids", "label_free", "purpose")
        if any(existing.get(key) != payload.get(key) for key in immutable):
            raise ValueError(f"existing schedule does not match requested stage: {path}")
        payload = existing
    else:
        _write_json(path, payload)
    by_id = {str(row["dataset_id"]): row for row in records}
    return [by_id[str(dataset_id)] for dataset_id in payload["ordered_dataset_ids"]]


def catalog(candidate_ids: list[str] | None = None) -> tuple[dict[str, Any], ...]:
    rows = tuple(MECHANISM_CANDIDATES)
    if candidate_ids is None:
        return rows
    by_id = {str(row["candidate_id"]): row for row in rows}
    unknown = sorted(set(candidate_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown RG mechanism candidate ids: {unknown}")
    return tuple(by_id[value] for value in candidate_ids)


def _config(config_path: Path, candidate: dict[str, Any] | None, variant: str = "rg_full") -> V19Config:
    overrides = {} if candidate is None else dict(candidate.get("overrides", {}))
    forbidden = sorted(set(overrides).intersection(LOCKED_BACKBONE))
    if forbidden:
        raise ValueError(f"ARI development candidate changes frozen scMAE fields: {forbidden}")
    config = load_config(
        config_path,
        {"protocol_id": PROTOCOL_ID, "variant": variant, **overrides},
    )
    resolved = config.resolved_dict()
    for key, expected in LOCKED_BACKBONE.items():
        if float(resolved[key]) != float(expected):
            raise ValueError(f"frozen scMAE field drifted for {key}: {resolved[key]} != {expected}")
    return config


def _run_path(output_root: Path, dataset_id: str, candidate_id: str, seed: int) -> Path:
    return output_root / dataset_id / candidate_id / f"seed{int(seed)}"


def _is_completed(path: Path, run_key: str) -> bool:
    try:
        summary = _read_json(path / "summary.json")
        status = _read_json(path / "status.json")
        record = _read_json(path / "run_record.json")
    except Exception:
        return False
    return bool(
        summary.get("status") == "completed"
        and status.get("status") == "completed"
        and record.get("status") == "completed"
        and summary.get("run_key") == run_key
        and summary.get("ari_dev_protocol_id") == PROTOCOL_ID
        and summary.get("selection_evidence_type") == SELECTION_EVIDENCE
        and summary.get("labels_used_during_fit") is False
        and summary.get("labels_used_for_graph") is False
        and summary.get("labels_used_for_gate") is False
        and summary.get("labels_used_for_loss") is False
        and summary.get("labels_used_for_selection") is True
        and summary.get("metrics", {}).get("labels_available") is True
    )


def _annotate_ari_run(
    output: Path,
    *,
    run_key: str,
    stage: str,
    candidate: dict[str, Any],
    manifest_id: str,
    input_variant: str,
) -> dict[str, Any]:
    summary = _read_json(output / "summary.json")
    summary.update(
        {
            "ari_dev_protocol_id": PROTOCOL_ID,
            "stage": stage,
            "run_key": run_key,
            "candidate_id": str(candidate["candidate_id"]),
            "candidate_family": str(candidate.get("family", "reference")),
            "candidate_overrides": dict(candidate.get("overrides", {})),
            "input_variant": input_variant,
            "manifest_id": manifest_id,
            "selection_evidence_type": SELECTION_EVIDENCE,
            "labels_used_during_fit": False,
            "labels_used_during_preprocessing": False,
            "labels_used_for_graph": False,
            "labels_used_for_gate": False,
            "labels_used_for_loss": False,
            "labels_used_for_selection": True,
            "K_source": "benchmark_oracle_from_y",
            "benchmark_oracle_from_y": True,
        }
    )
    _write_json(output / "summary.json", summary)
    status = _read_json(output / "status.json")
    status.update({"status": "completed", "run_key": run_key, "ari_dev_protocol_id": PROTOCOL_ID})
    _write_json(output / "status.json", status)
    record_path = output / "run_record.json"
    record = _read_json(record_path)
    record.update(
        {
            "status": "completed",
            "run_key": run_key,
            "ari_dev_protocol_id": PROTOCOL_ID,
            "stage": stage,
            "candidate_id": str(candidate["candidate_id"]),
            "selection_evidence_type": SELECTION_EVIDENCE,
            "labels_used_during_fit": False,
            "labels_used_for_graph": False,
            "labels_used_for_gate": False,
            "labels_used_for_loss": False,
            "labels_used_for_selection": True,
            "K_source": "benchmark_oracle_from_y",
        }
    )
    _write_json(record_path, record)
    return summary


def run_ari_job(
    record: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    output_root: Path,
    *,
    stage: str,
    config_path: Path,
    gpu: int,
    manifest_id: str,
    variant: str = "rg_full",
) -> dict[str, Any]:
    dataset_id = str(record["dataset_id"])
    candidate_id = str(candidate["candidate_id"])
    output = _run_path(output_root, dataset_id, candidate_id, int(seed))
    run_key = f"{PROTOCOL_ID}::{stage}::{dataset_id}::{candidate_id}::seed{int(seed)}"
    if _is_completed(output, run_key):
        return {"status": "completed", "run_key": run_key, "skipped": True}
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = _config(config_path, candidate if variant == "rg_full" else None, variant=variant)
    runtime_device = "cpu" if int(gpu) < 0 else resolve_runtime_device("cuda", int(gpu))
    try:
        summary = run_one(
            record["source_path"],
            output,
            config=config,
            input_protocol=str(record["input_protocol"]),
            seed=int(seed),
            device=runtime_device,
            dataset_name=str(record["name"]),
            dataset_id=dataset_id,
            n_clusters=None,
            max_samples=0,
        )
        annotated = _annotate_ari_run(
            output,
            run_key=run_key,
            stage=stage,
            candidate=candidate,
            manifest_id=manifest_id,
            input_variant=variant,
        )
        return {"status": "completed", "run_key": run_key, "summary": annotated}
    except Exception as exc:
        payload = {
            "status": "incomplete_compute",
            "ari_dev_protocol_id": PROTOCOL_ID,
            "stage": stage,
            "run_key": run_key,
            "dataset_id": dataset_id,
            "candidate_id": candidate_id,
            "seed": int(seed),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "wall_seconds": float(time.time() - started),
            "labels_used_during_fit": False,
            "selection_evidence_type": SELECTION_EVIDENCE,
        }
        _write_json(output / "summary.json", payload)
        _write_json(output / "status.json", payload)
        _write_json(output / "run_record.json", payload)
        return {"status": "incomplete_compute", "run_key": run_key, "error": str(exc)}


def build_stage_spec(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    candidates: tuple[dict[str, Any], ...],
    stage: str,
    seeds: tuple[int, ...],
    config_path: Path,
) -> dict[str, Any]:
    expected = [
        f"{PROTOCOL_ID}::{stage}::{record['dataset_id']}::{candidate['candidate_id']}::seed{seed}"
        for record in records
        for candidate in candidates
        for seed in seeds
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "manifest_id": manifest.get("manifest_id"),
        "dataset_ids": [str(row["dataset_id"]) for row in records],
        "candidate_ids": [str(row["candidate_id"]) for row in candidates],
        "candidate_definitions": list(candidates),
        "seed_order": [int(seed) for seed in seeds],
        "expected_runs": len(expected),
        "expected_run_keys": expected,
        "base_config": str(config_path.resolve()),
        "fixed_scmae_config": dict(LOCKED_BACKBONE),
        "input_protocols": {str(row["dataset_id"]): str(row["input_protocol"]) for row in records},
        "selection_evidence_type": SELECTION_EVIDENCE,
        "labels_allowed_only_for": ["benchmark_K", "post_fit_metrics", "candidate_selection"],
        "labels_used_during_fit": False,
        "labels_used_during_preprocessing": False,
        "labels_used_for_graph": False,
        "labels_used_for_gate": False,
        "labels_used_for_loss": False,
        "labels_used_for_selection": True,
        "K_source": "benchmark_oracle_from_y",
        "hash_policy": "do not recompute SHA/hash",
    }


def write_or_validate_stage_spec(root: Path, spec: dict[str, Any]) -> None:
    path = root / "stage_spec.json"
    if path.is_file():
        if _read_json(path) != spec:
            raise ValueError(f"existing stage spec does not match: {path}")
    else:
        _write_json(path, spec)


def _check_gpu(gpu: int) -> None:
    if int(gpu) >= 0 and int(gpu) not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(f"forbidden physical GPU {gpu}; allowed={sorted(ALLOWED_PHYSICAL_GPUS)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="V19 ARI-selected RG development tuner")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("screen", "refine"), required=True)
    parser.add_argument("--candidate-ids", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    records = schedule_records(args.output_dir / args.stage, target_records(manifest), args.stage)
    if args.stage == "screen":
        if args.candidate_ids:
            raise ValueError("screen does not accept candidate ids")
        candidates = catalog()
        seeds = SCREEN_SEEDS if args.seeds is None else tuple(int(v) for v in args.seeds)
        if seeds != SCREEN_SEEDS or len(candidates) != 48:
            raise ValueError("screen contract requires all 48 candidates and seed 42")
    else:
        if not args.candidate_ids or len(args.candidate_ids) != 12:
            raise ValueError("refine contract requires the 12 candidates selected from screen")
        candidates = catalog([str(v) for v in args.candidate_ids])
        seeds = FORMAL_SEEDS if args.seeds is None else tuple(int(v) for v in args.seeds)
        if seeds != FORMAL_SEEDS:
            raise ValueError("refine contract requires seeds 42,123,7")
    if len(records) != 8:
        raise AssertionError("target record selection must contain 8 datasets")
    spec_root = args.output_dir / args.stage
    spec_root.mkdir(parents=True, exist_ok=True)
    spec = build_stage_spec(manifest, records, candidates, args.stage, seeds, args.config)
    write_or_validate_stage_spec(spec_root, spec)
    worker_count = max(1, int(args.num_workers))
    if not 0 <= int(args.worker_id) < worker_count:
        raise ValueError("worker-id must be in [0, num-workers)")
    jobs = [
        (record, candidate, seed)
        for record in records
        for candidate in candidates
        for seed in seeds
    ]
    jobs = [job for index, job in enumerate(jobs) if index % worker_count == int(args.worker_id)]
    header = {**spec, "worker_id": int(args.worker_id), "num_workers": worker_count, "jobs_for_worker": len(jobs)}
    print(json.dumps(header, ensure_ascii=True), flush=True)
    if args.dry_run:
        for record, candidate, seed in jobs:
            print(f"{record['dataset_id']}\t{candidate['candidate_id']}\tseed={seed}")
        return 0
    physical_gpu = -1 if args.cpu else int(args.gpu)
    _check_gpu(physical_gpu)
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    if physical_gpu >= 0:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    rows: list[dict[str, Any]] = []
    for index, (record, candidate, seed) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {record['dataset_id']} {candidate['candidate_id']} seed={seed}", flush=True)
        row = run_ari_job(
            record,
            candidate,
            int(seed),
            spec_root,
            stage=args.stage,
            config_path=args.config,
            gpu=physical_gpu,
            manifest_id=str(manifest.get("manifest_id", "unknown")),
        )
        rows.append(row)
        print(json.dumps({"run_key": row.get("run_key"), "status": row.get("status")}), flush=True)
    worker = {
        **header,
        "completed": sum(row.get("status") == "completed" for row in rows),
        "incomplete_compute": sum(row.get("status") == "incomplete_compute" for row in rows),
        "runs": rows,
    }
    _write_json(spec_root / f"worker{int(args.worker_id)}_{int(time.time())}.json", worker)
    return 0 if worker["incomplete_compute"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
