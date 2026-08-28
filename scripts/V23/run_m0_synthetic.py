from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V23_cycle_response.config import load_config
from methods.TopoGate.V23_cycle_response.data import file_sha256
from methods.TopoGate.V23_cycle_response.synthetic import write_panel


PROTOCOL_ID = "v23_cycle_response_protocol_a_v1"
VARIANT_ID = "canonical_v23_frozen_probe"
DEFAULT_SEEDS = (42, 123, 7)
WORLDS = (
    "cluster_specific_dependency",
    "mean_only_shared_dependency",
    "conditional_dependency_destroyed",
    "global_structure_destroyed_sanity",
)
ALLOWED_PHYSICAL_GPUS = frozenset({1, 2, 3, 4, 5, 6})


@dataclass(frozen=True)
class M0Job:
    world: str
    seed: int
    matrix_path: Path
    labels_path: Path
    run_root: Path

    @property
    def key(self) -> str:
        return f"{self.world}__{PROTOCOL_ID}__{VARIANT_ID}__seed{self.seed}"

    @property
    def fit_dir(self) -> Path:
        return self.run_root / "fit"

    @property
    def profile_dir(self) -> Path:
        return self.run_root / "profile"

    @property
    def evaluate_dir(self) -> Path:
        return self.run_root / "evaluate"


class DigestCache:
    """Compute each artifact digest at most once during one launcher process."""

    def __init__(self) -> None:
        self._values: dict[Path, str] = {}
        self._lock = threading.Lock()

    def __call__(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        with self._lock:
            value = self._values.get(resolved)
            if value is None:
                value = file_sha256(resolved)
                self._values[resolved] = value
            return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_jobs(output_root: Path, seeds: tuple[int, ...], worlds: tuple[str, ...]) -> list[M0Job]:
    jobs: list[M0Job] = []
    for seed in seeds:
        for world in worlds:
            data_root = output_root / "generated_data" / world / f"seed{seed}"
            run_root = output_root / "runs" / world / f"seed{seed}"
            jobs.append(
                M0Job(
                    world=world,
                    seed=int(seed),
                    matrix_path=data_root / "matrix_only.npz",
                    labels_path=data_root / "labels_true.npy",
                    run_root=run_root,
                )
            )
    return jobs


def build_stage_commands(
    job: M0Job,
    args: argparse.Namespace,
    physical_gpu: int | None = None,
) -> dict[str, list[str]]:
    device_args = ["--device", args.device]
    if args.device == "cuda":
        if physical_gpu is None:
            raise ValueError("CUDA stage commands require an assigned physical GPU")
        device_args.extend(["--gpu", str(physical_gpu)])
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
        str(args.config),
        "--seed",
        str(job.seed),
        *device_args,
    ]
    if args.epochs is not None:
        fit.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        fit.extend(["--batch-size", str(args.batch_size)])
    if args.feature_cap is not None:
        fit.extend(["--feature-cap", str(args.feature_cap)])

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
        args.corruption_mode,
        *device_args,
    ]
    if args.fingerprint_masks is not None:
        profile.extend(["--fingerprint-masks", str(args.fingerprint_masks)])
    if args.fingerprint_mask_ratio is not None:
        profile.extend(["--fingerprint-mask-ratio", str(args.fingerprint_mask_ratio)])

    evaluate = [
        sys.executable,
        "-m",
        "methods.TopoGate.V23_cycle_response.evaluate",
        "--fingerprints",
        str(job.profile_dir / "fingerprints.npz"),
        "--labels",
        str(job.labels_path),
        "--output-dir",
        str(job.evaluate_dir),
        "--seed",
        str(job.seed),
    ]
    return {"fit": fit, "profile": profile, "evaluate": evaluate}


def _fit_complete(
    job: M0Job,
    expected_config: dict[str, Any],
    device: str,
    digest: DigestCache,
) -> bool:
    summary = _read_json(job.fit_dir / "summary.json")
    resolved = _read_json(job.fit_dir / "resolved_config.json")
    if summary is None or resolved is None:
        return False
    expected_device = "cpu" if device == "cpu" else "cuda:0"
    return bool(
        summary.get("status") == "completed"
        and summary.get("protocol_id") == PROTOCOL_ID
        and int(summary.get("seed", -1)) == job.seed
        and Path(str(summary.get("matrix_path", ""))).resolve() == job.matrix_path.resolve()
        and summary.get("matrix_sha256") == digest(job.matrix_path)
        and resolved.get("device") == expected_device
        and all(resolved.get(key) == value for key, value in expected_config.items())
    )


def _profile_complete(job: M0Job, args: argparse.Namespace, digest: DigestCache) -> bool:
    summary = _read_json(job.profile_dir / "summary.json")
    mask_config = _read_json(job.profile_dir / "mask_config.json")
    if summary is None or mask_config is None or not (job.profile_dir / "fingerprints.npz").is_file():
        return False
    return bool(
        summary.get("status") == "completed"
        and summary.get("protocol_id") == PROTOCOL_ID
        and int(summary.get("seed", -1)) == job.seed
        and summary.get("matrix_sha256") == digest(job.matrix_path)
        # A checkpoint digest is a conservative provenance guard. PyTorch may
        # serialize identical state differently across builds, so a mismatch
        # safely triggers profile recomputation rather than invalid reuse.
        and summary.get("checkpoint_sha256") == digest(job.fit_dir / "checkpoint.pt")
        and summary.get("preprocessor_sha256") == digest(job.fit_dir / "preprocessor.npz")
        and int(mask_config.get("mask_seed", -1)) == args.mask_seed
        and int(mask_config.get("donor_seed", -1)) == args.donor_seed
        and mask_config.get("corruption_mode") == args.corruption_mode
    )


def _evaluate_complete(job: M0Job, digest: DigestCache) -> bool:
    summary = _read_json(job.evaluate_dir / "summary.json")
    return bool(
        summary is not None
        and summary.get("status") == "completed"
        and int(summary.get("seed", -1)) == job.seed
        and summary.get("primary_scientific_object") == "cycle_repair_standardized"
        and summary.get("fingerprints_sha256") == digest(job.profile_dir / "fingerprints.npz")
        and summary.get("labels_sha256") == digest(job.labels_path)
        and (job.evaluate_dir / "metrics.json").is_file()
        and (job.evaluate_dir / "benchmark_validity_profile.json").is_file()
    )


def _run_stage(job: M0Job, stage: str, command: list[str], environment: dict[str, str]) -> None:
    log_path = job.run_root / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def _record_incomplete_attempt(
    job: M0Job,
    *,
    stage: str,
    returncode: int,
) -> dict[str, Any]:
    attempt_dir = job.run_root / "attempts" / f"{stage}_rc{returncode}_{time.time_ns()}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    current_log = job.run_root / f"{stage}.log"
    archived_log = attempt_dir / f"{stage}.log"
    if current_log.is_file():
        shutil.copy2(current_log, archived_log)
    failure = {
        "status": "incomplete_compute",
        "job_key": job.key,
        "failed_stage": stage,
        "returncode": int(returncode),
        "log_path": str(archived_log.resolve()),
        "attempt_dir": str(attempt_dir.resolve()),
    }
    _write_json(attempt_dir / "incomplete_compute.json", failure)
    _write_json(job.run_root / "incomplete_compute.json", failure)
    return failure


def _retire_incomplete_marker(job: M0Job) -> Path | None:
    marker = job.run_root / "incomplete_compute.json"
    if not marker.is_file():
        return None
    failure = _read_json(marker)
    attempt_dir = failure.get("attempt_dir") if failure else None
    archived = Path(str(attempt_dir)) if attempt_dir else None
    if archived is None or not archived.is_dir():
        archived = job.run_root / "attempts" / f"legacy_{marker.stat().st_mtime_ns}"
        archived.mkdir(parents=True, exist_ok=True)
        shutil.move(str(marker), str(archived / marker.name))
        return archived
    marker.unlink()
    return archived


def _aggregate(output_root: Path, jobs: list[M0Job]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for job in jobs:
        metrics = _read_json(job.evaluate_dir / "metrics.json")
        if metrics is None:
            continue
        for representation, values in dict(metrics.get("representations", {})).items():
            if not isinstance(values, dict):
                continue
            records.append(
                {
                    "world": job.world,
                    "seed": job.seed,
                    "representation": representation,
                    **{key: values.get(key) for key in ("status", "ari", "nmi", "acc", "knn_purity_at_10", "pair_auc")},
                }
            )
    grouped: list[dict[str, Any]] = []
    for world in dict.fromkeys(job.world for job in jobs):
        representations = sorted({row["representation"] for row in records if row["world"] == world})
        for representation in representations:
            rows = [row for row in records if row["world"] == world and row["representation"] == representation]
            aggregate: dict[str, Any] = {
                "world": world,
                "representation": representation,
                "completed_seeds": len(rows),
            }
            for metric in ("ari", "nmi", "acc", "knn_purity_at_10", "pair_auc"):
                values = [float(row[metric]) for row in rows if row.get(metric) is not None]
                if values:
                    mean = sum(values) / len(values)
                    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
                    aggregate[f"{metric}_mean"] = mean
                    aggregate[f"{metric}_std"] = variance**0.5
            grouped.append(aggregate)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "variant_id": VARIANT_ID,
        "status": "descriptive_only_no_automatic_go_decision",
        "records": records,
        "grouped": grouped,
    }
    _write_json(output_root / "aggregate_summary.json", payload)
    return payload


def _ensure_generated_panel(
    output_root: Path,
    *,
    seed: int,
    n_samples: int,
    n_features: int,
    n_clusters: int,
    latent_rank: int,
    zero_fraction: float,
    digest: DigestCache,
) -> str:
    expected = {
        "seed": int(seed),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_clusters": int(n_clusters),
        "latent_rank": int(latent_rank),
        "zero_fraction": float(zero_fraction),
    }
    manifest_path = output_root / f"manifest_seed{seed}.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        write_panel(output_root, **expected)
        return "generated"
    if manifest.get("generation_config") != expected:
        raise ValueError(f"existing synthetic seed{seed} panel uses a different generation protocol")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(WORLDS):
        raise ValueError(f"existing synthetic seed{seed} manifest is incomplete")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"existing synthetic seed{seed} manifest has an invalid record")
        matrix_path = Path(str(record.get("matrix_path", "")))
        labels_path = Path(str(record.get("labels_path", "")))
        if not matrix_path.is_file() or not labels_path.is_file():
            raise ValueError(f"existing synthetic seed{seed} panel is missing data files")
        if record.get("matrix_sha256") != digest(matrix_path):
            raise ValueError(f"existing synthetic seed{seed} matrix changed after generation")
        if record.get("labels_sha256") != digest(labels_path):
            raise ValueError(f"existing synthetic seed{seed} labels changed after generation")
    return "reused"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed V23 M0 synthetic falsification matrix")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "result" / "V23_cycle_response" / "m0_synthetic_protocol_a_v1",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "methods" / "TopoGate" / "V23_cycle_response" / "configs" / "protocol_a.yaml",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--worlds", choices=WORLDS, nargs="+", default=list(WORLDS))
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--n-features", type=int, default=1000)
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--latent-rank", type=int, default=16)
    parser.add_argument("--zero-fraction", type=float, default=0.90)
    parser.add_argument("--mask-seed", type=int, default=1701)
    parser.add_argument("--donor-seed", type=int, default=2903)
    parser.add_argument("--corruption-mode", choices=("donor_swap", "zero"), default="donor_swap")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--feature-cap", type=int, default=None)
    parser.add_argument("--fingerprint-masks", type=int, default=None)
    parser.add_argument("--fingerprint-mask-ratio", type=float, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpus", type=int, nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = _parse_args()
    gpu_pool = tuple(dict.fromkeys(int(gpu) for gpu in args.gpus))
    if args.device == "cuda":
        if not gpu_pool or any(gpu not in ALLOWED_PHYSICAL_GPUS for gpu in gpu_pool):
            raise ValueError("CUDA requires explicit --gpus from 1..6; physical GPUs 0 and 7 are forbidden")
    elif gpu_pool:
        raise ValueError("--gpus is invalid with --device cpu")
    seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    worlds = tuple(dict.fromkeys(str(world) for world in args.worlds))
    jobs = build_jobs(args.output_root, seeds, worlds)
    assignments = {
        job.key: (gpu_pool[index % len(gpu_pool)] if gpu_pool else None)
        for index, job in enumerate(jobs)
    }
    commands = {
        job.key: build_stage_commands(job, args, assignments[job.key])
        for job in jobs
    }
    if args.dry_run:
        payload = {
            "protocol_id": PROTOCOL_ID,
            "variant_id": VARIANT_ID,
            "dry_run": True,
            "job_count": len(jobs),
            "gpu_pool": list(gpu_pool),
            "jobs": [
                {
                    "key": job.key,
                    "physical_gpu": assignments[job.key],
                    "commands": commands[job.key],
                }
                for job in jobs
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    args.output_root.mkdir(parents=True, exist_ok=True)
    digest = DigestCache()
    generated_panels = {
        seed: _ensure_generated_panel(
            args.output_root / "generated_data",
            seed=seed,
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_clusters=args.n_clusters,
            latent_rank=args.latent_rank,
            zero_fraction=args.zero_fraction,
            digest=digest,
        )
        for seed in seeds
    }
    fit_overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "feature_cap": args.feature_cap,
        }.items()
        if value is not None
    }
    expected_fit_config = load_config(args.config, fit_overrides).to_dict()
    expected_config = load_config(
        args.config,
        {
            key: value
            for key, value in {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "feature_cap": args.feature_cap,
                "fingerprint_masks": args.fingerprint_masks,
                "fingerprint_mask_ratio": args.fingerprint_mask_ratio,
            }.items()
            if value is not None
        },
    ).to_dict()
    stage_spec = {
        "protocol_id": PROTOCOL_ID,
        "variant_id": VARIANT_ID,
        "seeds": list(seeds),
        "worlds": list(worlds),
        "label_boundary": "fit/profile commands contain matrix paths only; labels enter evaluate only",
        "gpu_pool": list(gpu_pool),
        "gpu_scheduling": "one_serial_queue_per_physical_gpu" if gpu_pool else "single_cpu_queue",
        "forbidden_gpus": [0, 7],
        "generated_panels": generated_panels,
        "config": expected_config,
        "jobs": [
            {
                "key": job.key,
                "physical_gpu": assignments[job.key],
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(job).items()
                },
            }
            for job in jobs
        ],
    }
    _write_json(args.output_root / "stage_spec.json", stage_spec)
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib-v23",
        }
    )
    status: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "gpu_pool": list(gpu_pool),
        "jobs": {
            job.key: {
                "physical_gpu": assignments[job.key],
                "fit": "queued",
                "profile": "queued",
                "evaluate": "queued",
            }
            for job in jobs
        },
    }
    status_lock = threading.Lock()

    def update_status(job_key: str, stage: str, value: str) -> None:
        with status_lock:
            status["jobs"][job_key][stage] = value
            _write_json(args.output_root / "launcher_status.json", status)

    def run_job(job: M0Job) -> dict[str, Any] | None:
        fit_complete = _fit_complete(job, expected_fit_config, args.device, digest)
        profile_complete = fit_complete and _profile_complete(job, args, digest)
        evaluate_complete = profile_complete and _evaluate_complete(job, digest)
        stages = {"fit": fit_complete, "profile": profile_complete, "evaluate": evaluate_complete}
        for stage, complete in stages.items():
            update_status(job.key, stage, "reused" if complete else "queued")
        for stage in ("fit", "profile", "evaluate"):
            if stages[stage]:
                continue
            try:
                _run_stage(job, stage, commands[job.key][stage], environment)
            except subprocess.CalledProcessError as error:
                failure = _record_incomplete_attempt(
                    job,
                    stage=stage,
                    returncode=int(error.returncode),
                )
                update_status(job.key, stage, "incomplete")
                return failure
            update_status(job.key, stage, "completed")
        _retire_incomplete_marker(job)
        return None

    queues: dict[int | None, list[M0Job]] = {
        worker: [job for job in jobs if assignments[job.key] == worker]
        for worker in (gpu_pool if gpu_pool else (None,))
    }

    def run_queue(queue: list[M0Job]) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for job in queue:
            failure = run_job(job)
            if failure is not None:
                failures.append(failure)
                break
        return failures

    _write_json(args.output_root / "launcher_status.json", status)
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = [executor.submit(run_queue, queue) for queue in queues.values()]
        for future in concurrent.futures.as_completed(futures):
            failures.extend(future.result())
    aggregate = _aggregate(args.output_root, jobs)
    completed = sum(_evaluate_complete(job, digest) for job in jobs)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "variant_id": VARIANT_ID,
        "status": "completed" if completed == len(jobs) else "incomplete_compute",
        "jobs_total": len(jobs),
        "jobs_completed": completed,
        "failed_queues": len(failures),
        "gpu_pool": list(gpu_pool),
        "new_stage_compute": sum(
            state == "completed" for job_state in status["jobs"].values() for state in job_state.values()
        ),
        "reused_stages": sum(state == "reused" for job_state in status["jobs"].values() for state in job_state.values()),
        "automatic_go_decision": False,
        "aggregate_path": str((args.output_root / "aggregate_summary.json").resolve()),
        "aggregate_records": len(aggregate["records"]),
    }
    _write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
