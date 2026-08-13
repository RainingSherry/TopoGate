#!/usr/bin/env python
"""Post-freeze V19 RG/scMAE/ablation evaluation matrix.

This entry point is deliberately separate from the label-free tuning stages.
It consumes the single configuration selected by ``mechanism_refine`` and
then permits labels only for the benchmark K and post-fit metrics.  No label
value is passed to ``fit_predict``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V19_rg_adapter.config import V19Config, load_config  # noqa: E402
from methods.TopoGate.V19_rg_adapter.input_adapter import load_npz  # noqa: E402
from methods.TopoGate.V19_rg_adapter.run import (  # noqa: E402
    resolve_runtime_device,
    run_one,
)


PROTOCOL_ID = "v19_rg_final_postfreeze_v1"
TUNING_PROTOCOL_ID = "v19_rg_unsup_tuning_v2"
MANIFEST_PROTOCOL_ID = "v19_rg_selected_advantage_v1"
FORMAL_SEEDS = (42, 123, 7)
ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})
FINAL_VARIANTS = (
    "rg_full",
    "rg_default",
    "scmae_only",
    "rg_nomix",
    "rg_reliability_off",
    "rg_constant_gate",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, default=str),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("protocol_id") != MANIFEST_PROTOCOL_ID:
        raise ValueError("final V19 requires the frozen selected-advantage manifest")
    rows = payload.get("datasets", [])
    if len(rows) != 11 or any(row.get("status") != "eligible" for row in rows):
        raise ValueError("final V19 requires all 11 eligible input strata")
    if payload.get("selection_policy", {}).get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("manifest selection policy is not label-free")
    return payload


def _load_selected(
    path: Path,
    *,
    allow_no_go: bool = False,
    manifest_id: str | None = None,
    base_config: Path | None = None,
) -> dict[str, Any]:
    selected = _read_json(path)
    if selected.get("protocol_id") != TUNING_PROTOCOL_ID:
        raise ValueError("selected config was not produced by V19 v2 tuning")
    if selected.get("stage") != "mechanism_refine":
        raise ValueError("final evaluation requires a mechanism_refine selection")
    if (
        not allow_no_go
        and (
            selected.get("selection_status") != "proxy_supported"
            or selected.get("no_go") is not False
        )
    ):
        raise ValueError("refusing final evaluation: selected config is no_go or unsupported")
    candidate_id = str(selected.get("candidate_id", ""))
    overrides = selected.get("overrides", {})
    if not candidate_id or not isinstance(overrides, dict):
        raise ValueError("selected config lacks one candidate id and overrides")
    if selected.get("candidate_family") != "mechanism":
        raise ValueError("final evaluation requires a mechanism candidate")
    if manifest_id is not None and selected.get("manifest_id") != manifest_id:
        raise ValueError("selected config manifest does not match the frozen manifest")
    if base_config is not None:
        selected_base = Path(str(selected.get("base_config", "")))
        if not selected_base.is_absolute():
            selected_base = ROOT / selected_base
        if selected_base.resolve() != base_config.resolve():
            raise ValueError("selected config base_config does not match the requested base config")
    if selected.get("labels_accessed") is not False or selected.get("y_key_read") is not False:
        raise ValueError("selected config failed its label audit")
    return selected


def _variant_configs(
    base_config: Path,
    selected: dict[str, Any],
) -> dict[str, V19Config]:
    overrides = dict(selected["overrides"])
    locked = load_config(
        base_config,
        {"variant": "rg_full", "protocol_id": PROTOCOL_ID, **overrides},
    )
    default_rg = load_config(
        base_config,
        {"variant": "rg_full", "protocol_id": PROTOCOL_ID},
    )
    scmae = load_config(
        base_config,
        {"variant": "scmae_only", "protocol_id": PROTOCOL_ID},
    )
    # These controls retain the locked backbone and graph construction while
    # removing one RG mechanism at a time.  They are not used for selection.
    no_mix = replace(locked, pseudo_weight=0.0)
    reliability_off = replace(
        locked,
        gamma_sim=0.0,
        gamma_mutual=0.0,
        gamma_snn=0.0,
        gamma_distance=0.0,
    )
    constant_gate = replace(
        locked,
        gate_min=float(locked.gate_max),
        gate_max=float(locked.gate_max),
    )
    return {
        "rg_full": locked,
        "rg_default": default_rg,
        "scmae_only": scmae,
        "rg_nomix": no_mix,
        "rg_reliability_off": reliability_off,
        "rg_constant_gate": constant_gate,
    }


def _select_records(
    manifest: dict[str, Any],
    requested: list[str] | None,
) -> list[dict[str, Any]]:
    requested_set = {str(value) for value in (requested or [])}
    rows = [
        row
        for row in manifest["datasets"]
        if not requested_set or str(row["dataset_id"]) in requested_set
    ]
    if not rows:
        raise ValueError("dataset filter selected no manifest rows")
    unknown = requested_set - {str(row["dataset_id"]) for row in rows}
    if unknown:
        raise ValueError(f"unknown dataset ids: {sorted(unknown)}")
    return rows


def _job_key(dataset_id: str, variant: str, seed: int) -> str:
    return f"{PROTOCOL_ID}::{dataset_id}::{variant}::seed{int(seed)}"


def _run_dir(root: Path, dataset_id: str, variant: str, seed: int) -> Path:
    return root / str(dataset_id) / str(variant) / f"seed{int(seed)}"


def _is_completed(path: Path, key: str) -> bool:
    try:
        summary = _read_json(path / "summary.json")
        status = _read_json(path / "status.json")
    except Exception:
        return False
    return bool(
        summary.get("status") == "completed"
        and status.get("status") == "completed"
        and summary.get("run_key") == key
        and summary.get("evaluation_variant")
        and summary.get("labels_used_during_fit") is False
        and summary.get("metrics", {}).get("labels_available") is True
    )


def _annotate_run(
    output: Path,
    *,
    key: str,
    evaluation_variant: str,
    selected: dict[str, Any],
    manifest_id: str,
    resource_gpu: int,
) -> dict[str, Any]:
    summary_path = output / "summary.json"
    summary = _read_json(summary_path)
    summary.update(
        {
            "run_key": key,
            "evaluation_variant": evaluation_variant,
            "final_protocol_id": PROTOCOL_ID,
            "selected_candidate_id": str(selected["candidate_id"]),
            "selected_overrides": selected["overrides"],
            "manifest_id": manifest_id,
            "labels_used_during_fit": False,
            "labels_used_during_preprocessing": False,
            "K_source": "benchmark_oracle_from_y",
            "benchmark_oracle_from_y": True,
            "selection_status": selected.get("selection_status"),
            "no_go": bool(selected.get("no_go", False)),
            "diagnostic_postfreeze": bool(selected.get("no_go", False)),
            "resource_gpu": int(resource_gpu) if int(resource_gpu) >= 0 else None,
        }
    )
    _write_json(summary_path, summary)
    status_path = output / "status.json"
    status = _read_json(status_path)
    status.update(
        {
            "status": "completed",
            "run_key": key,
            "final_protocol_id": PROTOCOL_ID,
            "evaluation_variant": evaluation_variant,
        }
    )
    _write_json(status_path, status)
    run_record_path = output / "run_record.json"
    run_record = _read_json(run_record_path)
    run_record.update(
        {
            "run_key": key,
            "final_protocol_id": PROTOCOL_ID,
            "evaluation_variant": evaluation_variant,
            "selected_candidate_id": str(selected["candidate_id"]),
            "selected_overrides": selected["overrides"],
            "manifest_id": manifest_id,
            "labels_used_during_fit": False,
            "K_source": "benchmark_oracle_from_y",
        }
    )
    _write_json(run_record_path, run_record)
    return summary


def _run_one(
    record: dict[str, Any],
    evaluation_variant: str,
    config: V19Config,
    seed: int,
    output_root: Path,
    selected: dict[str, Any],
    manifest_id: str,
    gpu: int,
) -> dict[str, Any]:
    dataset_id = str(record["dataset_id"])
    output = _run_dir(output_root, dataset_id, evaluation_variant, seed)
    key = _job_key(dataset_id, evaluation_variant, seed)
    if _is_completed(output, key):
        return {"status": "completed", "run_key": key, "skipped": True}
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        runtime_device = resolve_runtime_device("cuda" if gpu >= 0 else "cpu", gpu)
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
        summary = _annotate_run(
            output,
            key=key,
            evaluation_variant=evaluation_variant,
            selected=selected,
            manifest_id=manifest_id,
            resource_gpu=gpu,
        )
        return {"status": "completed", "run_key": key, "summary": summary}
    except Exception as exc:  # preserve an auditable incomplete-compute record
        payload = {
            "status": "incomplete_compute",
            "run_key": key,
            "final_protocol_id": PROTOCOL_ID,
            "evaluation_variant": evaluation_variant,
            "dataset_id": dataset_id,
            "seed": int(seed),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "wall_seconds": float(time.time() - started),
            "labels_used_during_fit": False,
        }
        _write_json(output / "summary.json", payload)
        _write_json(output / "status.json", payload)
        _write_json(
            output / "run_record.json",
            {
                **payload,
                "manifest_id": manifest_id,
                "source_path": str(record["source_path"]),
            },
        )
        return {"status": "incomplete_compute", "run_key": key, "error": str(exc)}


def _build_spec(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    variants: dict[str, V19Config],
    seeds: tuple[int, ...],
    selected: dict[str, Any],
    base_config: Path,
) -> dict[str, Any]:
    keys = [
        _job_key(str(record["dataset_id"]), variant, seed)
        for record in records
        for variant in variants
        for seed in seeds
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "tuning_protocol_id": TUNING_PROTOCOL_ID,
        "manifest_id": manifest.get("manifest_id"),
        "base_config": str(base_config.resolve()),
        "selected_config_candidate_id": str(selected["candidate_id"]),
        "selected_config_overrides": selected["overrides"],
        "candidate_family": selected.get("candidate_family"),
        "selection_status": selected.get("selection_status"),
        "no_go": bool(selected.get("no_go", False)),
        "diagnostic_postfreeze": bool(selected.get("no_go", False)),
        "selection_uses_labels_or_outcomes": False,
        "variant_selection_uses_labels_or_outcomes": False,
        "dataset_ids": [str(row["dataset_id"]) for row in records],
        "variants": list(variants),
        "seeds": [int(seed) for seed in seeds],
        "expected_runs": len(keys),
        "expected_run_keys": keys,
        "configs": {name: config.resolved_dict() for name, config in variants.items()},
        "labels_allowed_only_for": ["benchmark_K", "post_fit_metrics"],
        "labels_used_during_fit": False,
        "K_source": "benchmark_oracle_from_y",
        "created_by": "scripts/V19/run_final_evaluation.py",
        "python": platform.python_version(),
    }


def _schedule_cost(record: dict[str, Any]) -> tuple[float, int, str]:
    """Return a label-free source-size estimate used only for queue assignment."""

    source = Path(str(record["source_path"]))
    try:
        source_bytes = int(source.stat().st_size)
    except OSError:
        source_bytes = 1 << 62
    multiplier = {
        "shared_text": 0.8,
        "rg_native": 1.0,
        "clubench_bridge": 1.6,
    }.get(str(record.get("input_protocol", "")), 1.2)
    return float(source_bytes) * float(multiplier), source_bytes, str(record["dataset_id"])


def _is_large_record(record: dict[str, Any]) -> bool:
    return _schedule_cost(record)[1] >= 50 * 1024 * 1024


def _assign_jobs(
    jobs: list[tuple[dict[str, Any], str, int]],
    *,
    worker_count: int,
    worker_id: int,
) -> list[tuple[dict[str, Any], str, int]]:
    """Assign a size-ordered final queue across distinct GPU workers."""

    if worker_count <= 1:
        return list(jobs)
    ordered = sorted(
        jobs,
        key=lambda job: (
            _schedule_cost(job[0]),
            str(job[1]),
            int(job[2]),
        ),
    )
    return [
        job
        for index, job in enumerate(ordered)
        if index % worker_count == worker_id
    ]


def _write_worker_spec_if_needed(output_root: Path, spec: dict[str, Any]) -> None:
    path = output_root / "stage_spec.json"
    if path.is_file():
        existing = _read_json(path)
        if existing != spec:
            raise ValueError(f"existing final stage spec does not match: {path}")
    else:
        _write_json(path, spec)


def _run_worker(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    selected = _load_selected(
        args.selected_config,
        allow_no_go=bool(args.allow_no_go),
        manifest_id=str(manifest.get("manifest_id")),
        base_config=args.base_config,
    )
    records = _select_records(manifest, args.datasets)
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or any(seed not in FORMAL_SEEDS for seed in seeds):
        raise ValueError(f"seeds must be drawn from {FORMAL_SEEDS}")
    variants = _variant_configs(args.base_config, selected)
    spec = _build_spec(manifest, records, variants, seeds, selected, args.base_config)
    _write_worker_spec_if_needed(args.output_dir, spec)
    worker_count = max(1, int(args.num_workers))
    worker_id = int(args.worker_id)
    jobs = [
        (record, variant, seed)
        for record in records
        for variant in variants
        for seed in seeds
    ]
    jobs = _assign_jobs(jobs, worker_count=worker_count, worker_id=worker_id)
    rows = []
    for index, (record, variant, seed) in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {record['dataset_id']} {variant} seed={seed}",
            flush=True,
        )
        result = _run_one(
            record,
            variant,
            variants[variant],
            seed,
            args.output_dir,
            selected,
            str(manifest.get("manifest_id", "unknown")),
            int(args.gpu) if not args.cpu else -1,
        )
        rows.append(result)
        print(json.dumps({"run_key": result.get("run_key"), "status": result.get("status")}), flush=True)
    _write_json(
        args.output_dir / f"final_worker{worker_id}_{int(time.time())}.json",
        {
            "protocol_id": PROTOCOL_ID,
            "worker_id": worker_id,
            "num_workers": worker_count,
            "jobs": len(jobs),
            "completed": sum(row.get("status") == "completed" for row in rows),
            "incomplete_compute": sum(row.get("status") == "incomplete_compute" for row in rows),
            "runs": rows,
        },
    )
    return 0 if all(row.get("status") == "completed" for row in rows) else 1


def _audit(output_root: Path, spec: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    expected = {str(value) for value in spec["expected_run_keys"]}
    seen: dict[str, Path] = {}
    incomplete: list[str] = []
    bad_labels: list[str] = []
    bad_shapes: list[str] = []
    for path in sorted(output_root.glob("**/summary.json")):
        if "attempts" in path.parts:
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if payload.get("status") == "incomplete_compute":
            incomplete.append(str(path))
            continue
        if payload.get("status") != "completed":
            continue
        key = str(payload.get("run_key", ""))
        if key in seen or key not in expected:
            return False, f"duplicate or unexpected run key: {key}", {}
        seen[key] = path
        if (
            payload.get("labels_used_during_fit") is not False
            or payload.get("labels_used_during_preprocessing") is not False
            or payload.get("K_source") != "benchmark_oracle_from_y"
        ):
            bad_labels.append(str(path))
        if payload.get("selected_candidate_id") != spec.get("selected_config_candidate_id"):
            return False, f"selected candidate provenance failed: {path}", {}
        if payload.get("manifest_id") != spec.get("manifest_id"):
            return False, f"manifest provenance failed: {path}", {}
        required = (
            "status.json",
            "run_record.json",
            "resolved_config.json",
            "metrics.json",
            "predictions.npy",
            "labels_true.npy",
            "embedding_final.npy",
        )
        if any(not (path.parent / name).is_file() for name in required):
            return False, f"incomplete final artifact contract: {path}", {}
        status_payload = _read_json(path.parent / "status.json")
        record_payload = _read_json(path.parent / "run_record.json")
        if (
            status_payload.get("status") != "completed"
            or status_payload.get("run_key") != key
            or record_payload.get("status") != "completed"
            or record_payload.get("run_key") != key
        ):
            return False, f"final status/run_record mismatch: {path}", {}
        metrics = _read_json(path.parent / "metrics.json")
        for metric in ("ari", "nmi", "acc"):
            value = metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                return False, f"non-finite final metric {metric}: {path}", {}
        try:
            np = __import__("numpy")
            prediction = np.load(path.parent / "predictions.npy", allow_pickle=False)
            truth = np.load(path.parent / "labels_true.npy", allow_pickle=False)
            embedding = np.load(path.parent / "embedding_final.npy", allow_pickle=False)
            if prediction.ndim != 1 or truth.ndim != 1 or prediction.shape != truth.shape:
                bad_shapes.append(str(path))
            if embedding.ndim != 2 or embedding.shape[0] != prediction.shape[0]:
                bad_shapes.append(str(path))
        except Exception:
            bad_shapes.append(str(path))
    missing = sorted(expected - set(seen))
    if incomplete:
        return False, f"incomplete runs: {incomplete[:3]}", {"incomplete": incomplete}
    if missing:
        return False, f"missing {len(missing)} expected runs", {"missing": missing[:10]}
    if bad_labels:
        return False, f"label/K audit failed: {bad_labels[:3]}", {"bad_labels": bad_labels}
    if bad_shapes:
        return False, f"prediction/label shape audit failed: {bad_shapes[:3]}", {"bad_shapes": bad_shapes}
    for path in sorted(output_root.glob("**/run_record.json")):
        if "attempts" in path.parts:
            continue
        payload = _read_json(path)
        if payload.get("status") != "completed":
            return False, f"non-completed final run record: {path}", {}
    return True, "ok", {"completed_runs": len(seen), "expected_runs": len(expected)}


def _launch(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    selected = _load_selected(
        args.selected_config,
        allow_no_go=bool(args.allow_no_go),
        manifest_id=str(manifest.get("manifest_id")),
        base_config=args.base_config,
    )
    records = _select_records(manifest, args.datasets)
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or any(seed not in FORMAL_SEEDS for seed in seeds):
        raise ValueError(f"seeds must be drawn from {FORMAL_SEEDS}")
    variants = _variant_configs(args.base_config, selected)
    spec = _build_spec(manifest, records, variants, seeds, selected, args.base_config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise RuntimeError(
            "final evaluation requires a fresh output root; pass --resume only for an audited retry: "
            f"{args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_worker_spec_if_needed(args.output_dir, spec)
    status_path = args.output_dir / "launcher_status.json"
    if status_path.is_file():
        prior = _read_json(status_path)
        if prior.get("status") == "running":
            active = [
                int(row.get("pid", -1))
                for row in prior.get("workers", [])
                if int(row.get("pid", -1)) > 0 and _pid_alive(int(row["pid"]))
            ]
            if active:
                raise RuntimeError(f"final output root already has active workers: {active}")
    if args.cpu:
        gpus = [-1]
    else:
        gpus = [int(value) for value in args.gpus]
        if not gpus:
            raise ValueError("provide --gpus or --cpu")
        if any(value not in ALLOWED_GPUS for value in gpus):
            raise ValueError(f"GPU pool must be drawn from {sorted(ALLOWED_GPUS)}")
        if len(set(gpus)) != len(gpus):
            raise ValueError("GPU pool contains duplicates")
    workers = []
    for worker_id, gpu in enumerate(gpus):
        log_path = args.output_dir / f"launcher_worker{worker_id}.log"
        environment = dict(os.environ)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        if gpu >= 0:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--manifest", str(args.manifest),
            "--selected-config", str(args.selected_config),
            "--base-config", str(args.base_config),
            "--output-dir", str(args.output_dir),
            "--seeds", *[str(seed) for seed in seeds],
            "--worker-id", str(worker_id),
            "--num-workers", str(len(gpus)),
        ]
        if args.allow_no_go:
            command.append("--allow-no-go")
        if args.resume:
            command.append("--resume")
        if args.datasets:
            command.extend(["--datasets", *[str(value) for value in args.datasets]])
        if gpu < 0:
            command.append("--cpu")
        else:
            command.extend(["--gpu", str(gpu)])
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        workers.append({"worker_id": worker_id, "gpu": gpu, "pid": process.pid, "log": str(log_path), "process": process, "handle": handle})
    _write_json(
        status_path,
        {
            "status": "running",
            "protocol_id": PROTOCOL_ID,
            "workers": [{key: value for key, value in row.items() if key not in {"process", "handle"}} for row in workers],
            "expected_runs": spec["expected_runs"],
            "labels_used_during_fit": False,
            "variant_selection_uses_labels_or_outcomes": False,
            "selection_status": selected.get("selection_status"),
            "no_go": bool(selected.get("no_go", False)),
            "diagnostic_postfreeze": bool(selected.get("no_go", False)),
            "resume": bool(args.resume),
        },
    )
    codes = []
    for worker in workers:
        codes.append(int(worker["process"].wait()))
        worker["handle"].close()
    ok, message, audit = _audit(args.output_dir, spec)
    success = ok and all(code == 0 for code in codes)
    final_status = {
        "status": "completed" if success else "incomplete_compute",
        "protocol_id": PROTOCOL_ID,
        "return_codes": codes,
        "audit_ok": bool(ok),
        "audit_message": message,
        "audit": audit,
        "expected_runs": spec["expected_runs"],
        "labels_used_during_fit": False,
        "variant_selection_uses_labels_or_outcomes": False,
        "selection_status": selected.get("selection_status"),
        "no_go": bool(selected.get("no_go", False)),
        "diagnostic_postfreeze": bool(selected.get("no_go", False)),
        "resume": bool(args.resume),
    }
    _write_json(status_path, final_status)
    _write_json(
        args.output_dir / "matrix_summary.json",
        {
            **final_status,
            "selected_candidate_id": selected["candidate_id"],
            "selected_overrides": selected["overrides"],
            "variants": list(variants),
            "seeds": list(seeds),
            "dataset_ids": [str(row["dataset_id"]) for row in records],
        },
    )
    return 0 if success else 1


def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="V19 post-freeze final evaluation matrix")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--gpus", type=int, nargs="*", default=[])
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--allow-no-go",
        action="store_true",
        help="run a diagnostic post-freeze matrix even when tuning marked no_go",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing audited final root; default requires a fresh root",
    )
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    if args.worker_id is not None:
        if args.cpu:
            args.gpu = -1
        return _run_worker(args)
    if args.cpu:
        args.gpus = []
    elif not args.gpus:
        raise ValueError("top-level launch requires --gpus or --cpu")
    return _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
