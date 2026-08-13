#!/usr/bin/env python
"""Run the frozen V19 ARI-selected RG and ablation comparison matrix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V19_rg_adapter.config import V19Config, load_config  # noqa: E402
from methods.TopoGate.V19_rg_adapter.run import (  # noqa: E402
    ALLOWED_PHYSICAL_GPUS,
    resolve_runtime_device,
    run_one,
)
from scripts.V19.tune_ari_dev import (  # noqa: E402
    DEFAULT_CONFIG,
    FINAL_PROTOCOL_ID,
    FORMAL_SEEDS,
    PROTOCOL_ID,
    SELECTION_EVIDENCE,
    TARGET_DATASET_IDS,
    _read_json,
    _write_json,
    load_manifest,
    schedule_records,
    target_records,
    write_or_validate_stage_spec,
)


FINAL_VARIANTS = (
    "rg_full",
    "scmae_only",
    "rg_default",
    "rg_nomix",
    "rg_reliability_off",
    "rg_constant_gate",
)


def _variant_configs(base_config: Path, selected: dict[str, Any]) -> dict[str, V19Config]:
    overrides = selected.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("selected config overrides must be an object")
    locked = load_config(base_config, {"protocol_id": FINAL_PROTOCOL_ID, "variant": "rg_full", **overrides})
    default_rg = load_config(base_config, {"protocol_id": FINAL_PROTOCOL_ID, "variant": "rg_full"})
    scmae = load_config(base_config, {"protocol_id": FINAL_PROTOCOL_ID, "variant": "scmae_only"})
    no_mix = replace(locked, pseudo_weight=0.0)
    reliability_off = replace(
        locked,
        gamma_sim=0.0,
        gamma_mutual=0.0,
        gamma_snn=0.0,
        gamma_distance=0.0,
    )
    constant_gate = replace(locked, gate_min=float(locked.gate_max), gate_max=float(locked.gate_max))
    return {
        "rg_full": locked,
        "scmae_only": scmae,
        "rg_default": default_rg,
        "rg_nomix": no_mix,
        "rg_reliability_off": reliability_off,
        "rg_constant_gate": constant_gate,
    }


def _run_path(root: Path, dataset_id: str, variant: str, seed: int) -> Path:
    return root / dataset_id / variant / f"seed{int(seed)}"


def _is_completed(path: Path, key: str, variant: str) -> bool:
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
        and summary.get("run_key") == key
        and summary.get("final_protocol_id") == FINAL_PROTOCOL_ID
        and summary.get("evaluation_variant") == variant
        and summary.get("metrics", {}).get("labels_available") is True
        and summary.get("labels_used_during_fit") is False
        and summary.get("labels_used_during_preprocessing") is False
    )


def _annotate(
    output: Path,
    *,
    key: str,
    variant: str,
    selected: dict[str, Any],
    manifest_id: str,
    gpu: int,
) -> dict[str, Any]:
    summary = _read_json(output / "summary.json")
    summary.update(
        {
            "run_key": key,
            "final_protocol_id": FINAL_PROTOCOL_ID,
            "tuning_protocol_id": PROTOCOL_ID,
            "evaluation_variant": variant,
            "selected_candidate_id": str(selected["candidate_id"]),
            "selected_overrides": dict(selected.get("overrides", {})),
            "manifest_id": manifest_id,
            "selection_evidence_type": SELECTION_EVIDENCE,
            "selection_status": "selected",
            "labels_used_during_fit": False,
            "labels_used_during_preprocessing": False,
            "labels_used_for_graph": False,
            "labels_used_for_gate": False,
            "labels_used_for_loss": False,
            "labels_used_for_selection": True,
            "K_source": "benchmark_oracle_from_y",
            "benchmark_oracle_from_y": True,
            "resource_gpu": int(gpu) if int(gpu) >= 0 else None,
        }
    )
    _write_json(output / "summary.json", summary)
    status = _read_json(output / "status.json")
    status.update({"status": "completed", "run_key": key, "final_protocol_id": FINAL_PROTOCOL_ID, "evaluation_variant": variant})
    _write_json(output / "status.json", status)
    record = _read_json(output / "run_record.json")
    record.update(
        {
            "status": "completed",
            "run_key": key,
            "final_protocol_id": FINAL_PROTOCOL_ID,
            "evaluation_variant": variant,
            "selected_candidate_id": str(selected["candidate_id"]),
            "manifest_id": manifest_id,
            "labels_used_during_fit": False,
            "labels_used_for_graph": False,
            "labels_used_for_gate": False,
            "labels_used_for_loss": False,
            "labels_used_for_selection": True,
            "K_source": "benchmark_oracle_from_y",
        }
    )
    _write_json(output / "run_record.json", record)
    return summary


def _run_one(
    record: dict[str, Any],
    variant: str,
    config: V19Config,
    seed: int,
    output_root: Path,
    selected: dict[str, Any],
    manifest_id: str,
    gpu: int,
) -> dict[str, Any]:
    dataset_id = str(record["dataset_id"])
    output = _run_path(output_root, dataset_id, variant, seed)
    key = f"{FINAL_PROTOCOL_ID}::{dataset_id}::{variant}::seed{int(seed)}"
    if _is_completed(output, key, variant):
        return {"status": "completed", "run_key": key, "skipped": True}
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        runtime_device = "cpu" if int(gpu) < 0 else resolve_runtime_device("cuda", int(gpu))
        run_one(
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
        summary = _annotate(
            output,
            key=key,
            variant=variant,
            selected=selected,
            manifest_id=manifest_id,
            gpu=gpu,
        )
        return {"status": "completed", "run_key": key, "summary": summary}
    except Exception as exc:
        payload = {
            "status": "incomplete_compute",
            "final_protocol_id": FINAL_PROTOCOL_ID,
            "run_key": key,
            "dataset_id": dataset_id,
            "evaluation_variant": variant,
            "seed": int(seed),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "wall_seconds": float(time.time() - started),
            "labels_used_during_fit": False,
        }
        _write_json(output / "summary.json", payload)
        _write_json(output / "status.json", payload)
        _write_json(output / "run_record.json", payload)
        return {"status": "incomplete_compute", "run_key": key, "error": str(exc)}


def _build_spec(manifest: dict[str, Any], records: list[dict[str, Any]], configs: dict[str, V19Config], selected: dict[str, Any], base_config: Path) -> dict[str, Any]:
    expected = [
        f"{FINAL_PROTOCOL_ID}::{record['dataset_id']}::{variant}::seed{seed}"
        for record in records
        for variant in configs
        for seed in FORMAL_SEEDS
    ]
    return {
        "protocol_id": FINAL_PROTOCOL_ID,
        "tuning_protocol_id": PROTOCOL_ID,
        "manifest_id": manifest.get("manifest_id"),
        "base_config": str(base_config.resolve()),
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_overrides": dict(selected.get("overrides", {})),
        "dataset_ids": [str(row["dataset_id"]) for row in records],
        "variants": list(configs),
        "seeds": list(FORMAL_SEEDS),
        "expected_runs": len(expected),
        "expected_run_keys": expected,
        "configs": {name: config.resolved_dict() for name, config in configs.items()},
        "selection_evidence_type": SELECTION_EVIDENCE,
        "labels_allowed_only_for": ["benchmark_K", "post_fit_metrics"],
        "labels_used_during_fit": False,
        "labels_used_for_graph": False,
        "labels_used_for_gate": False,
        "labels_used_for_loss": False,
        "labels_used_for_selection": True,
        "K_source": "benchmark_oracle_from_y",
        "development_scope": "all 8 target datasets were used in ARI selection",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V19 frozen ARI development final matrix")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    selected = _read_json(args.selected_config)
    if selected.get("protocol_id") != PROTOCOL_ID or selected.get("stage") != "refine":
        raise ValueError("selected config is not a V19 ARI refine selection")
    if selected.get("selection_status") != "selected" or selected.get("no_go") is not False:
        raise ValueError("selected config is not a valid frozen ARI selection")
    if selected.get("labels_used_during_fit") is not False or selected.get("labels_used_for_selection") is not True:
        raise ValueError("selected config label audit failed")
    records = schedule_records(args.output_dir, target_records(manifest), "final")
    configs = _variant_configs(args.config, selected)
    if tuple(configs) != FINAL_VARIANTS:
        raise AssertionError("final variant order drifted")
    spec = _build_spec(manifest, records, configs, selected, args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_stage_spec(args.output_dir, spec)
    worker_count = max(1, int(args.num_workers))
    if not 0 <= int(args.worker_id) < worker_count:
        raise ValueError("worker-id must be in [0, num-workers)")
    jobs = [
        (record, variant, seed)
        for record in records
        for variant in configs
        for seed in FORMAL_SEEDS
    ]
    jobs = [job for index, job in enumerate(jobs) if index % worker_count == int(args.worker_id)]
    header = {**spec, "worker_id": int(args.worker_id), "num_workers": worker_count, "jobs_for_worker": len(jobs)}
    print(json.dumps(header, ensure_ascii=True), flush=True)
    if args.dry_run:
        for record, variant, seed in jobs:
            print(f"{record['dataset_id']}\t{variant}\tseed={seed}")
        return 0
    physical_gpu = -1 if args.cpu else int(args.gpu)
    if physical_gpu >= 0 and physical_gpu not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError(f"forbidden physical GPU {physical_gpu}")
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    if physical_gpu >= 0:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    rows = []
    for index, (record, variant, seed) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {record['dataset_id']} {variant} seed={seed}", flush=True)
        row = _run_one(
            record,
            variant,
            configs[variant],
            int(seed),
            args.output_dir,
            selected,
            str(manifest.get("manifest_id", "unknown")),
            physical_gpu,
        )
        rows.append(row)
        print(json.dumps({"run_key": row.get("run_key"), "status": row.get("status")}), flush=True)
    _write_json(
        args.output_dir / f"worker{int(args.worker_id)}_{int(time.time())}.json",
        {
            **header,
            "completed": sum(row.get("status") == "completed" for row in rows),
            "incomplete_compute": sum(row.get("status") == "incomplete_compute" for row in rows),
            "runs": rows,
        },
    )
    return 0 if all(row.get("status") == "completed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
