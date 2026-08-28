from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V24_conditional_response.calibration import calibrate_estimator
from methods.TopoGate.V24_conditional_response.config import PRIMARY_SEEDS, V24Q1Config, WORLD_NAMES
from methods.TopoGate.V24_conditional_response.contracts import ContractAudit, audit_global_null_panel, audit_world
from methods.TopoGate.V24_conditional_response.decision import decide_q1
from methods.TopoGate.V24_conditional_response.postmortem import run_postmortem
from methods.TopoGate.V24_conditional_response.synthetic import write_panel


ALLOWED_PHYSICAL_GPUS = frozenset({1, 2, 3, 4, 5, 6})
PROBE_CONFIG = ROOT / "methods/TopoGate/V24_conditional_response/configs/q1_probe.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "result/V24_conditional_response/q1_synthetic_v2"
DEFAULT_EXPLORATORY_OUTPUT_ROOT = ROOT / "result/V24_conditional_response/q1_synthetic_v2_exploratory_override"
DEFAULT_V23_ROOT = ROOT / "result/V23_cycle_response/m0_synthetic_protocol_a_v1"
EXPLORATORY_OVERRIDE_REASON = "calibration_gate_failed_by_explicit_user_override"


@dataclass(frozen=True)
class Q1Job:
    world: str
    seed: int
    protocol_id: str
    matrix_path: Path
    labels_path: Path
    run_root: Path

    @property
    def key(self) -> str:
        return f"{self.world}__{self.protocol_id}__frozen_v23_probe__seed{self.seed}"

    @property
    def contract_path(self) -> Path:
        return self.run_root / "contract.json"

    @property
    def fit_dir(self) -> Path:
        return self.run_root / "fit"

    @property
    def profile_dir(self) -> Path:
        return self.run_root / "profile"

    @property
    def analysis_dir(self) -> Path:
        return self.run_root / "analysis"


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _config_payload(config: V24Q1Config) -> dict[str, Any]:
    return _json_ready(config.to_dict())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip()))
    if not values:
        raise ValueError("at least one integer is required")
    return values


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if not values:
        raise ValueError("at least one name is required")
    return values


def resolve_physical_gpus(device: str, raw_gpus: str) -> tuple[int, ...]:
    if device == "cpu":
        return ()
    gpus = _parse_csv_ints(raw_gpus)
    if any(gpu not in ALLOWED_PHYSICAL_GPUS for gpu in gpus):
        raise ValueError("V24 permits only physical GPUs 1..6; 0 and 7 are forbidden")
    return gpus


def build_jobs(
    output_root: Path,
    *,
    seeds: tuple[int, ...],
    worlds: tuple[str, ...],
    protocol_id: str,
    data_root: Path | None = None,
) -> list[Q1Job]:
    source_root = output_root if data_root is None else data_root
    return [
        Q1Job(
            world=world,
            seed=seed,
            protocol_id=protocol_id,
            matrix_path=source_root / "generated_data" / world / f"seed{seed}" / "matrix_only.npz",
            labels_path=source_root / "generated_data" / world / f"seed{seed}" / "labels_true.npy",
            run_root=output_root / "runs" / world / f"seed{seed}",
        )
        for seed in seeds
        for world in worlds
    ]


def _bootstrap_replicates(args: argparse.Namespace, config: V24Q1Config) -> int:
    return int(config.bootstrap_replicates if args.bootstrap_replicates is None else args.bootstrap_replicates)


def build_stage_commands(
    job: Q1Job,
    args: argparse.Namespace,
    *,
    physical_gpu: int | None = None,
    bootstrap_replicates: int | None = None,
) -> dict[str, list[str]]:
    device_args = ["--device", args.device]
    if args.device == "cuda":
        if physical_gpu not in ALLOWED_PHYSICAL_GPUS:
            raise ValueError("CUDA jobs require an assigned physical GPU in 1..6")
        device_args.extend(["--gpu", str(physical_gpu)])
    bootstrap = 200 if bootstrap_replicates is None else int(bootstrap_replicates)
    fit = [
        sys.executable,
        "-m",
        "methods.TopoGate.V23_cycle_response.fit",
        "--matrix",
        str(job.matrix_path),
        "--input-protocol",
        "clubench_bridge",
        "--output-dir",
        str(job.fit_dir),
        "--config",
        str(PROBE_CONFIG),
        "--seed",
        str(job.seed),
        *device_args,
    ]
    if args.epochs is not None:
        fit.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        fit.extend(["--batch-size", str(args.batch_size)])
    profile = [
        sys.executable,
        "-m",
        "methods.TopoGate.V23_cycle_response.profile",
        "--matrix",
        str(job.matrix_path),
        "--fit-dir",
        str(job.fit_dir),
        "--output-dir",
        str(job.profile_dir),
        "--mask-seed",
        str(args.mask_seed),
        "--donor-seed",
        str(args.donor_seed),
        "--corruption-mode",
        "donor_swap",
        *device_args,
    ]
    analyze = [
        sys.executable,
        "-m",
        "methods.TopoGate.V24_conditional_response.analyze",
        "--matrix",
        str(job.matrix_path),
        "--fingerprints",
        str(job.profile_dir / "fingerprints.npz"),
        "--labels",
        str(job.labels_path),
        "--output-dir",
        str(job.analysis_dir),
        "--seed",
        str(job.seed),
        "--bootstrap-replicates",
        str(bootstrap),
    ]
    bootstrap_workers = int(getattr(args, "bootstrap_workers", 1))
    if bootstrap_workers != 1:
        analyze.extend(["--bootstrap-workers", str(bootstrap_workers)])
    return {"fit": fit, "profile": profile, "analyze": analyze}


def _validate_existing_panel(output_root: Path, config: V24Q1Config, seeds: tuple[int, ...]) -> None:
    for seed in seeds:
        manifest_path = output_root / "generated_data" / f"manifest_seed{seed}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("generated V24 panel is absent; run --mode prepare before --mode run")
        manifest = _read_json(manifest_path)
        if manifest.get("generation_config") != _config_payload(config):
            raise ValueError("existing generated panel configuration differs from frozen V24 protocol")
        for record in manifest.get("records", []):
            if not Path(str(record["matrix_path"])).is_file() or not Path(str(record["labels_path"])).is_file():
                raise ValueError("generated V24 panel is incomplete; do not overwrite it")


def _ensure_panel(output_root: Path, config: V24Q1Config, seeds: tuple[int, ...]) -> dict[str, str]:
    generated_root = output_root / "generated_data"
    status: dict[str, str] = {}
    for seed in seeds:
        manifest_path = generated_root / f"manifest_seed{seed}.json"
        if not manifest_path.exists():
            write_panel(generated_root, config, seed=seed)
            status[str(seed)] = "generated"
        else:
            status[str(seed)] = "reused"
    _validate_existing_panel(output_root, config, seeds)
    return status


def _load_contract(job: Q1Job, config: V24Q1Config) -> dict[str, Any]:
    if not job.contract_path.is_file():
        raise FileNotFoundError(f"missing contract audit for {job.key}; run --mode prepare first")
    contract = _read_json(job.contract_path)
    if (
        contract.get("protocol_id") != config.protocol_id
        or contract.get("config") != _config_payload(config)
        or contract.get("world") != job.world
        or int(contract.get("seed", -1)) != job.seed
        or contract.get("valid") is not True
    ):
        raise RuntimeError(f"invalid_design: {job.key}")
    return contract


def _prepare_contracts(jobs: list[Q1Job], config: V24Q1Config) -> dict[str, ContractAudit]:
    results: dict[str, ContractAudit] = {}
    for job in jobs:
        with __import__("numpy").load(job.matrix_path, allow_pickle=False) as loaded:
            matrix = loaded["X"]
        labels = __import__("numpy").load(job.labels_path, allow_pickle=False)
        audit = audit_world(matrix, labels, world=job.world, config=config, seed=job.seed)
        _write_json(
            job.contract_path,
            {
                "protocol_id": config.protocol_id,
                "config": _config_payload(config),
                "world": job.world,
                "seed": job.seed,
                "valid": audit.valid,
                "metrics": audit.metrics,
            },
        )
        results[job.key] = audit
    return results


def _prepare_panel_contracts(
    jobs: list[Q1Job],
    contracts: dict[str, ContractAudit],
    config: V24Q1Config,
) -> dict[str, dict[str, object]]:
    w0_audits = {
        job.seed: contracts[job.key]
        for job in jobs
        if job.world == "W0_global_null" and job.key in contracts
    }
    return {"W0_global_null": audit_global_null_panel(w0_audits, config)}


def _stage_reuse_plan(completed: dict[str, bool]) -> dict[str, bool]:
    """Return reusable stages while respecting fit -> profile -> analysis dependencies."""

    reusable: dict[str, bool] = {}
    upstream_changed = False
    for stage in ("fit", "profile", "analyze"):
        reusable[stage] = bool(completed.get(stage, False) and not upstream_changed)
        if not reusable[stage]:
            upstream_changed = True
    return reusable


def _stage_complete(directory: Path) -> bool:
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        return _read_json(summary_path).get("status") == "completed"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _run_stage(job: Q1Job, stage: str, command: list[str], environment: dict[str, str]) -> None:
    job.run_root.mkdir(parents=True, exist_ok=True)
    log_path = job.run_root / f"{stage}.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )
    except subprocess.CalledProcessError as error:
        _write_json(
            job.run_root / "incomplete_compute.json",
            {
                "status": "incomplete_compute",
                "stage": stage,
                "returncode": int(error.returncode),
                "command": command,
                "log_path": str(log_path),
            },
        )
        raise


def _run_job(
    job: Q1Job,
    args: argparse.Namespace,
    config: V24Q1Config,
    physical_gpu: int | None,
    *,
    bootstrap_replicates: int,
    stages: tuple[str, ...] = ("fit", "profile", "analyze"),
) -> dict[str, Any]:
    requested_stages = tuple(stages)
    if not requested_stages or any(stage not in {"fit", "profile", "analyze"} for stage in requested_stages):
        raise ValueError(f"invalid V24 stage selection: {requested_stages}")
    contract = _load_contract(job, config)
    commands = build_stage_commands(
        job,
        args,
        physical_gpu=physical_gpu,
        bootstrap_replicates=bootstrap_replicates,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = {
        "fit": _stage_complete(job.fit_dir),
        "profile": _stage_complete(job.profile_dir),
        "analyze": _stage_complete(job.analysis_dir),
    }
    reusable = _stage_reuse_plan(completed)
    stage_status: dict[str, str] = {}
    for stage, destination in (("fit", job.fit_dir), ("profile", job.profile_dir), ("analyze", job.analysis_dir)):
        if stage not in requested_stages:
            if completed[stage]:
                stage_status[stage] = "reused"
            continue
        if reusable[stage]:
            stage_status[stage] = "reused"
            continue
        _run_stage(job, stage, commands[stage], environment)
        if not _stage_complete(destination):
            raise RuntimeError(f"{stage} completed without a valid summary for {job.key}")
        stage_status[stage] = "computed"
    if "analyze" not in requested_stages:
        return {
            "world": job.world,
            "seed": job.seed,
            "key": job.key,
            "contract_valid": bool(contract["valid"]),
            "stage_status": stage_status,
            "new_stages": int(sum(value == "computed" for value in stage_status.values())),
            "reused_stages": int(sum(value == "reused" for value in stage_status.values())),
        }
    summary = _read_json(job.analysis_dir / "summary.json")
    bootstrap = dict(summary["bootstrap"])
    return {
        "world": job.world,
        "seed": job.seed,
        "key": job.key,
        "contract_valid": bool(contract["valid"]),
        "delta_auc": float(summary["conditional_pair_utility"]["delta_auc"]),
        "ci95_low": bootstrap["ci95_low"],
        "ci95_high": bootstrap["ci95_high"],
        "bootstrap_replicates_completed": int(bootstrap["replicates_completed"]),
        "stage_status": stage_status,
        "new_stages": int(sum(value == "computed" for value in stage_status.values())),
        "reused_stages": int(sum(value == "reused" for value in stage_status.values())),
    }


def _run_gpu_queue(
    jobs: list[Q1Job],
    args: argparse.Namespace,
    config: V24Q1Config,
    physical_gpu: int,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    return [
        _run_job(job, args, config, physical_gpu, bootstrap_replicates=bootstrap_replicates)
        for job in jobs
    ]


def _run_jobs(
    jobs: list[Q1Job],
    args: argparse.Namespace,
    config: V24Q1Config,
    gpus: tuple[int, ...],
    *,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    if args.device == "cpu":
        return [
            _run_job(job, args, config, None, bootstrap_replicates=bootstrap_replicates)
            for job in jobs
        ]
    queues = [jobs[offset:: len(gpus)] for offset in range(len(gpus))]
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(_run_gpu_queue, queue, args, config, gpu, bootstrap_replicates)
            for gpu, queue in zip(gpus, queues)
            if queue
        ]
        for future in futures:
            records.extend(future.result())
    return sorted(records, key=lambda record: (str(record["world"]), int(record["seed"])))


def _run_exploratory_job(
    job: Q1Job,
    args: argparse.Namespace,
    config: V24Q1Config,
    physical_gpu: int | None,
    *,
    bootstrap_replicates: int,
    stages: tuple[str, ...] = ("fit", "profile", "analyze"),
) -> dict[str, Any]:
    """Run one override job while preserving a per-job failure record."""

    try:
        record = _run_job(
            job,
            args,
            config,
            physical_gpu,
            bootstrap_replicates=bootstrap_replicates,
            stages=stages,
        )
        return dict(record, status="completed", execution_class="exploratory_override", stages=list(stages))
    except Exception as error:  # keep other exploratory jobs running and retain incomplete_compute markers
        return {
            "world": job.world,
            "seed": job.seed,
            "key": job.key,
            "status": "incomplete_compute",
            "execution_class": "exploratory_override",
            "run_root": str(job.run_root),
            "error_type": type(error).__name__,
            "error": str(error),
            "stages": list(stages),
        }


def _run_exploratory_phase(
    jobs: list[Q1Job],
    args: argparse.Namespace,
    config: V24Q1Config,
    gpus: tuple[int, ...],
    *,
    bootstrap_replicates: int,
    stages: tuple[str, ...],
    workers: int | None = None,
) -> list[dict[str, Any]]:
    if stages == ("analyze",):
        worker_count = max(1, min(len(jobs), int(workers or 1)))
        analysis_args = argparse.Namespace(**vars(args))
        analysis_args.device = "cpu"
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _run_exploratory_job,
                    job,
                    analysis_args,
                    config,
                    None,
                    bootstrap_replicates=bootstrap_replicates,
                    stages=stages,
                )
                for job in jobs
            ]
            records = [future.result() for future in futures]
    elif args.device == "cpu":
        records = [
            _run_exploratory_job(
                job,
                args,
                config,
                None,
                bootstrap_replicates=bootstrap_replicates,
                stages=stages,
            )
            for job in jobs
        ]
    else:
        queues = [jobs[offset:: len(gpus)] for offset in range(len(gpus))]
        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [
                executor.submit(
                    lambda queue=queue, gpu=gpu: [
                        _run_exploratory_job(
                            job,
                            args,
                            config,
                            gpu,
                            bootstrap_replicates=bootstrap_replicates,
                            stages=stages,
                        )
                        for job in queue
                    ]
                )
                for gpu, queue in zip(gpus, queues)
                if queue
            ]
            for future in futures:
                records.extend(future.result())
    return sorted(records, key=lambda record: (str(record["world"]), int(record["seed"])))


def _merge_exploratory_phase_records(
    jobs: list[Q1Job],
    fit_profile_records: list[dict[str, Any]],
    analysis_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fit_by_key = {str(record["key"]): record for record in fit_profile_records}
    analysis_by_key = {str(record["key"]): record for record in analysis_records}
    merged: list[dict[str, Any]] = []
    for job in jobs:
        key = job.key
        fit_record = fit_by_key.get(key)
        if fit_record is None:
            merged.append(
                {
                    "world": job.world,
                    "seed": job.seed,
                    "key": key,
                    "status": "incomplete_compute",
                    "execution_class": "exploratory_override",
                    "error_type": "missing_fit_profile_record",
                    "error": "fit/profile phase did not return a record",
                    "run_root": str(job.run_root),
                }
            )
            continue
        if fit_record.get("status") != "completed":
            merged.append(fit_record)
            continue
        analysis_record = analysis_by_key.get(key)
        merged.append(analysis_record if analysis_record is not None else fit_record)
    return sorted(merged, key=lambda record: (str(record["world"]), int(record["seed"])))


def _stage_spec(
    *,
    config: V24Q1Config,
    mode: str,
    seeds: tuple[int, ...],
    worlds: tuple[str, ...],
    gpus: tuple[int, ...],
    jobs: list[Q1Job],
    generated_panels: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "protocol_id": config.protocol_id,
        "variant_id": "frozen_v23_probe_plus_v24_conditional_analysis",
        "mode": mode,
        "seeds": list(seeds),
        "worlds": list(worlds),
        "generated_panels": generated_panels,
        "gpu_pool": list(gpus),
        "forbidden_gpus": [0, 7],
        "label_boundary": (
            "synthetic labels are used only for generator-contract audit and outer pair evaluation; "
            "fit/profile commands receive matrix paths only"
        ),
        "q2_boundary": "DCBoost is frozen and not invoked by V24-Q1",
        "jobs": [{"key": job.key, "world": job.world, "seed": job.seed, "run_root": str(job.run_root)} for job in jobs],
        "config": _config_payload(config),
    }


def _ensure_exploratory_root(output_root: Path) -> None:
    formal_root = DEFAULT_OUTPUT_ROOT.resolve()
    candidate = output_root.resolve()
    if candidate == formal_root:
        raise ValueError("exploratory override must use a separate output root from formal V24-Q1")
    if candidate.is_relative_to(formal_root):
        raise ValueError("exploratory override cannot write inside the formal V24-Q1 output root")
    if candidate.name != DEFAULT_EXPLORATORY_OUTPUT_ROOT.name:
        raise ValueError(
            f"exploratory override output root must end with {DEFAULT_EXPLORATORY_OUTPUT_ROOT.name!r}"
        )
    if (candidate / "q1_decision.json").exists():
        raise ValueError("exploratory override refuses an existing q1_decision.json")


def _ensure_full_request(
    *,
    args: argparse.Namespace,
    config: V24Q1Config,
    seeds: tuple[int, ...],
    worlds: tuple[str, ...],
    bootstrap_replicates: int,
    request_name: str,
) -> None:
    if seeds != config.primary_seeds or worlds != WORLD_NAMES:
        raise ValueError(f"{request_name} requires the frozen six worlds and five primary seeds")
    if args.epochs is not None or args.batch_size is not None:
        raise ValueError(f"{request_name} does not accept shortened training overrides")
    if bootstrap_replicates != config.bootstrap_replicates:
        raise ValueError(f"{request_name} requires the preregistered bootstrap replicate count")


def _ensure_formal_request(
    *,
    args: argparse.Namespace,
    config: V24Q1Config,
    seeds: tuple[int, ...],
    worlds: tuple[str, ...],
    bootstrap_replicates: int,
) -> None:
    _ensure_full_request(
        args=args,
        config=config,
        seeds=seeds,
        worlds=worlds,
        bootstrap_replicates=bootstrap_replicates,
        request_name="formal --mode run",
    )


def _ensure_exploratory_request(
    *,
    args: argparse.Namespace,
    config: V24Q1Config,
    seeds: tuple[int, ...],
    worlds: tuple[str, ...],
    gpus: tuple[int, ...],
    bootstrap_replicates: int,
) -> None:
    _ensure_full_request(
        args=args,
        config=config,
        seeds=seeds,
        worlds=worlds,
        bootstrap_replicates=bootstrap_replicates,
        request_name="exploratory --mode exploratory-override",
    )
    if args.device != "cuda" or gpus != (1, 2, 3):
        raise ValueError("exploratory override requires CUDA on exactly physical GPUs 1,2,3")
    if int(getattr(args, "analysis_workers", 1)) <= 0 or int(getattr(args, "bootstrap_workers", 1)) <= 0:
        raise ValueError("exploratory worker counts must be positive")


def _load_calibration(output_root: Path, config: V24Q1Config) -> dict[str, Any]:
    path = output_root / "calibration.json"
    if not path.is_file():
        raise FileNotFoundError("formal V24-Q1 requires a completed estimator calibration; run --mode calibrate first")
    payload = _read_json(path)
    if payload.get("protocol_id") != config.protocol_id or payload.get("config") != _config_payload(config):
        raise ValueError("calibration artifact does not match the frozen V24 configuration")
    calibration = dict(payload.get("calibration", {}))
    if calibration.get("calibration_passes") is not True:
        raise RuntimeError("estimator calibration did not pass; formal V24-Q1 cannot start")
    return calibration


def _load_prepare_summary(output_root: Path, config: V24Q1Config, jobs: list[Q1Job]) -> dict[str, Any]:
    path = output_root / "prepare_summary.json"
    if not path.is_file():
        raise FileNotFoundError("formal V24-Q1 requires a completed pre-fit contract panel; run --mode prepare first")
    payload = _read_json(path)
    if payload.get("protocol_id") != config.protocol_id or payload.get("config") != _config_payload(config):
        raise ValueError("prepare summary does not match the frozen V24 configuration")
    expected = {job.key for job in jobs}
    contract_validity = dict(payload.get("contracts", {}))
    panel_contracts = dict(payload.get("panel_contracts", {}))
    if payload.get("status") != "completed" or set(contract_validity) != expected:
        raise RuntimeError("pre-fit contract panel is incomplete")
    if not all(value is True for value in contract_validity.values()):
        raise RuntimeError("pre-fit per-seed contract panel did not pass")
    if dict(panel_contracts.get("W0_global_null", {})).get("valid") is not True:
        raise RuntimeError("pre-fit global-null panel centering contract did not pass")
    return payload


def _run_calibration(
    output_root: Path,
    config: V24Q1Config,
    *,
    replicates: int | None,
    workers: int,
) -> dict[str, Any]:
    result = calibrate_estimator(config, replicates=replicates, workers=workers)
    payload = {
        "status": "completed",
        "protocol_id": config.protocol_id,
        "config": _config_payload(config),
        "calibration": result.summary,
        "null_delta_auc": result.null_deltas.tolist(),
        "alternative_delta_auc": result.alternative_deltas.tolist(),
    }
    _write_json(output_root / "calibration.json", payload)
    return payload


def _load_postmortem(output_root: Path) -> dict[str, Any] | None:
    path = output_root / "p0_v23_postmortem" / "summary.json"
    return _read_json(path) if path.is_file() else None


def _write_decision(output_root: Path, records: list[dict[str, Any]], config: V24Q1Config) -> dict[str, Any]:
    calibration = None
    calibration_path = output_root / "calibration.json"
    if calibration_path.is_file():
        calibration = dict(_read_json(calibration_path).get("calibration", {}))
    decision = decide_q1(records, config, calibration=calibration, postmortem=_load_postmortem(output_root))
    _write_json(output_root / "q1_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="V24-Q1 corrected synthetic protocol runner")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "prepare", "calibrate", "p0", "run", "exploratory-override", "decide"),
        default="dry-run",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpus", default="1,2,3,4,5,6")
    parser.add_argument("--seeds", default=",".join(str(value) for value in PRIMARY_SEEDS))
    parser.add_argument("--worlds", default=",".join(WORLD_NAMES))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mask-seed", type=int, default=1701)
    parser.add_argument("--donor-seed", type=int, default=2903)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--bootstrap-workers", type=int, default=1)
    parser.add_argument("--calibration-replicates", type=int, default=None)
    parser.add_argument("--calibration-workers", type=int, default=1)
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=30,
        help="exploratory-only CPU workers for independent outer analyses",
    )
    parser.add_argument("--v23-root", type=Path, default=DEFAULT_V23_ROOT)
    parser.add_argument("--p0-bootstrap-replicates", type=int, default=0)
    args = parser.parse_args()

    config = V24Q1Config()
    config.validate()
    seeds = _parse_csv_ints(args.seeds)
    worlds = _parse_csv_strings(args.worlds)
    if any(world not in WORLD_NAMES for world in worlds):
        raise ValueError("unknown V24 world")
    gpus = resolve_physical_gpus(args.device, args.gpus)
    bootstrap_replicates = _bootstrap_replicates(args, config)
    jobs = build_jobs(args.output_root, seeds=seeds, worlds=worlds, protocol_id=config.protocol_id)

    if args.mode == "dry-run":
        commands = build_stage_commands(
            jobs[0],
            args,
            physical_gpu=(gpus[0] if gpus else None),
            bootstrap_replicates=bootstrap_replicates,
        )
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "writes": False,
                    "job_count": len(jobs),
                    "stage_spec": _stage_spec(
                        config=config,
                        mode=args.mode,
                        seeds=seeds,
                        worlds=worlds,
                        gpus=gpus,
                        jobs=jobs,
                        generated_panels=None,
                    ),
                    "representative_commands": commands,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.mode == "prepare":
        generated = _ensure_panel(args.output_root, config, seeds)
        contracts = _prepare_contracts(jobs, config)
        contract_validity = {key: bool(audit.valid) for key, audit in contracts.items()}
        panel_contracts = _prepare_panel_contracts(jobs, contracts, config)
        panel_validity = {key: bool(value.get("valid")) for key, value in panel_contracts.items()}
        prepared = bool(all(contract_validity.values()) and all(panel_validity.values()))
        _write_json(
            args.output_root / "stage_spec.json",
            _stage_spec(
                config=config,
                mode=args.mode,
                seeds=seeds,
                worlds=worlds,
                gpus=gpus,
                jobs=jobs,
                generated_panels=generated,
            ),
        )
        _write_json(
            args.output_root / "prepare_summary.json",
            {
                "status": "completed" if prepared else "invalid_design",
                "protocol_id": config.protocol_id,
                "config": _config_payload(config),
                "generated_panels": generated,
                "contracts": contract_validity,
                "panel_contracts": panel_contracts,
            },
        )
        if not prepared:
            invalid = [key for key, valid in contract_validity.items() if not valid]
            invalid.extend(key for key, valid in panel_validity.items() if not valid)
            raise RuntimeError(f"invalid_design: {', '.join(invalid)}")
        print(json.dumps({"status": "prepared", "job_count": len(jobs)}, ensure_ascii=False))
        return

    if args.mode == "calibrate":
        payload = _run_calibration(
            args.output_root,
            config,
            replicates=args.calibration_replicates,
            workers=args.calibration_workers,
        )
        print(json.dumps({"status": "completed", "calibration_passes": payload["calibration"]["calibration_passes"]}, ensure_ascii=False))
        return

    if args.mode == "p0":
        result = run_postmortem(
            args.v23_root,
            args.output_root / "p0_v23_postmortem",
            bootstrap_replicates=args.p0_bootstrap_replicates,
        )
        print(json.dumps({"status": result["status"], "records": len(result["records"])}, ensure_ascii=False))
        return

    if args.mode == "exploratory-override":
        _ensure_exploratory_root(args.output_root)
        _ensure_exploratory_request(
            args=args,
            config=config,
            seeds=seeds,
            worlds=worlds,
            gpus=gpus,
            bootstrap_replicates=bootstrap_replicates,
        )
        source_root = DEFAULT_OUTPUT_ROOT
        source_jobs = build_jobs(
            source_root,
            seeds=seeds,
            worlds=worlds,
            protocol_id=config.protocol_id,
        )
        exploratory_jobs = build_jobs(
            args.output_root,
            seeds=seeds,
            worlds=worlds,
            protocol_id=config.protocol_id,
            data_root=source_root,
        )
        _validate_existing_panel(source_root, config, seeds)
        _load_prepare_summary(source_root, config, source_jobs)
        calibration_path = source_root / "calibration.json"
        if not calibration_path.is_file():
            raise FileNotFoundError("exploratory override requires the existing calibration artifact")
        calibration_payload = _read_json(calibration_path)
        if calibration_payload.get("protocol_id") != config.protocol_id or calibration_payload.get("config") != _config_payload(config):
            raise ValueError("calibration artifact does not match the frozen V24 configuration")
        calibration = dict(calibration_payload.get("calibration", {}))
        if calibration.get("calibration_passes") is True:
            raise ValueError("exploratory override is only for a failed calibration gate")
        for source_job, exploratory_job in zip(source_jobs, exploratory_jobs, strict=True):
            source_contract = _load_contract(source_job, config)
            exploratory_job.run_root.mkdir(parents=True, exist_ok=True)
            _write_json(
                exploratory_job.contract_path,
                dict(source_contract, execution_class="exploratory_override", reused_from=str(source_job.contract_path)),
            )
        stage_spec = _stage_spec(
            config=config,
            mode=args.mode,
            seeds=seeds,
            worlds=worlds,
            gpus=gpus,
            jobs=exploratory_jobs,
            generated_panels={str(seed): "reused_from_formal_q1_v2" for seed in seeds},
        )
        stage_spec.update(
            {
                "execution_class": "exploratory_override",
                "calibration_override": True,
                "formal_q1_eligible": False,
                "promotion_to_q2": False,
                "reason": EXPLORATORY_OVERRIDE_REASON,
                "source_formal_root": str(source_root.resolve()),
                "calibration_passes": False,
            }
        )
        _write_json(args.output_root / "stage_spec.json", stage_spec)
        _write_json(
            args.output_root / "exploratory_manifest.json",
            {
                "status": "queued",
                "execution_class": "exploratory_override",
                "calibration_override": True,
                "formal_q1_eligible": False,
                "promotion_to_q2": False,
                "reason": EXPLORATORY_OVERRIDE_REASON,
                "source_formal_root": str(source_root.resolve()),
                "calibration": calibration,
                "jobs": stage_spec["jobs"],
            },
        )
        fit_profile_records = _run_exploratory_phase(
            exploratory_jobs,
            args,
            config,
            gpus,
            bootstrap_replicates=bootstrap_replicates,
            stages=("fit", "profile"),
        )
        successful_fit_profile = {
            str(record["key"])
            for record in fit_profile_records
            if record.get("status") == "completed"
        }
        analysis_jobs = [job for job in exploratory_jobs if job.key in successful_fit_profile]
        analysis_records = _run_exploratory_phase(
            analysis_jobs,
            args,
            config,
            (),
            bootstrap_replicates=bootstrap_replicates,
            stages=("analyze",),
            workers=args.analysis_workers,
        )
        records = _merge_exploratory_phase_records(exploratory_jobs, fit_profile_records, analysis_records)
        completed = sum(record.get("status") == "completed" for record in records)
        failed = len(records) - completed
        summary = {
            "status": "completed" if failed == 0 else "incomplete_compute",
            "execution_class": "exploratory_override",
            "calibration_override": True,
            "formal_q1_eligible": False,
            "promotion_to_q2": False,
            "reason": EXPLORATORY_OVERRIDE_REASON,
            "protocol_id": config.protocol_id,
            "source_formal_root": str(source_root.resolve()),
            "calibration": calibration,
            "records": records,
            "job_count": len(records),
            "completed_job_count": completed,
            "incomplete_job_count": failed,
            "new_stage_count": int(sum(int(record.get("new_stages", 0)) for record in records)),
            "reused_stage_count": int(sum(int(record.get("reused_stages", 0)) for record in records)),
            "q1_decision_written": False,
            "q2_promotion_allowed": False,
        }
        _write_json(args.output_root / "exploratory_summary.json", summary)
        manifest_path = args.output_root / "exploratory_manifest.json"
        manifest = _read_json(manifest_path)
        manifest.update(
            {
                "status": summary["status"],
                "completed_job_count": completed,
                "incomplete_job_count": failed,
                "q1_decision_written": False,
                "q2_promotion_allowed": False,
            }
        )
        _write_json(manifest_path, manifest)
        print(json.dumps({"status": summary["status"], "job_count": len(records), "completed": completed, "incomplete": failed}, ensure_ascii=False))
        return

    if args.mode == "decide":
        run_summary_path = args.output_root / "run_summary.json"
        if not run_summary_path.is_file():
            raise FileNotFoundError("--mode decide requires an existing formal run_summary.json")
        records = list(_read_json(run_summary_path).get("records", []))
        decision = _write_decision(args.output_root, records, config)
        print(json.dumps({"status": "completed", "decision": decision["decision"]}, ensure_ascii=False))
        return

    _ensure_formal_request(
        args=args,
        config=config,
        seeds=seeds,
        worlds=worlds,
        bootstrap_replicates=bootstrap_replicates,
    )
    _validate_existing_panel(args.output_root, config, seeds)
    _load_prepare_summary(args.output_root, config, jobs)
    _load_calibration(args.output_root, config)
    for job in jobs:
        _load_contract(job, config)
    _write_json(
        args.output_root / "stage_spec.json",
        _stage_spec(
            config=config,
            mode=args.mode,
            seeds=seeds,
            worlds=worlds,
            gpus=gpus,
            jobs=jobs,
            generated_panels={str(seed): "preexisting_validated" for seed in seeds},
        ),
    )
    records = _run_jobs(jobs, args, config, gpus, bootstrap_replicates=bootstrap_replicates)
    decision = _write_decision(args.output_root, records, config)
    _write_json(
        args.output_root / "run_summary.json",
        {
            "status": "completed",
            "protocol_id": config.protocol_id,
            "records": records,
            "decision": decision,
            "new_stage_count": int(sum(int(record["new_stages"]) for record in records)),
            "reused_stage_count": int(sum(int(record["reused_stages"]) for record in records)),
        },
    )
    print(json.dumps({"status": "completed", "decision": decision["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
