#!/usr/bin/env python
"""Run the label-free V19 RG hyperparameter search.

This entry point reads only feature-matrix members from NPZ files. It never
loads benchmark labels, derives K, runs KMeans, or writes label metrics.
"""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V19_rg_adapter.config import V19Config, load_config
from methods.TopoGate.V19_rg_adapter.input_adapter import (
    load_npz_matrix_only,
    prepare_input,
)
from methods.TopoGate.V19_rg_adapter.run import resolve_runtime_device
from methods.TopoGate.V19_rg_adapter.trainer import fit_predict


PROTOCOL_ID = "v19_rg_unsup_tuning_v1"
FORMAL_SEEDS = (42, 123, 7)
ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})
DEFAULT_CONFIG = ROOT / "methods" / "TopoGate" / "V19_rg_adapter" / "configs" / "v19_rg.yaml"
DEFAULT_OUTPUT = ROOT / "result" / "V19" / "v19_rg_unsup_tuning_v1"


def _candidate(candidate_id: str, **overrides: Any) -> dict[str, Any]:
    return {"candidate_id": candidate_id, "overrides": overrides}


# Fixed, label-free search design. The default and one-factor changes cover
# the main training/graph controls; the final profiles cover reliability and
# gate interactions without opening an unbounded Cartesian product.
CANDIDATES: tuple[dict[str, Any], ...] = (
    _candidate("default"),
    _candidate("epochs40", epochs=40),
    _candidate("epochs120", epochs=120),
    _candidate("lr5e-4", lr=5e-4),
    _candidate("lr2e-3", lr=2e-3),
    _candidate("mask03", mask_ratio=0.3),
    _candidate("mask05", mask_ratio=0.5),
    _candidate("k5", neighbor_k=5),
    _candidate("k20", neighbor_k=20),
    _candidate("mix2", mix_neighbors=2),
    _candidate("mix8", mix_neighbors=8),
    _candidate("tau01", tau=0.1),
    _candidate("tau04", tau=0.4),
    _candidate("gate005", gate_max=0.05),
    _candidate("gate025", gate_max=0.25),
    _candidate("pseudo01", pseudo_weight=0.1),
    _candidate("pseudo05", pseudo_weight=0.5),
    _candidate("rel_uniform", gamma_sim=0.0, gamma_mutual=0.0, gamma_snn=0.0, gamma_distance=0.0),
    _candidate("rel_no_distance", gamma_distance=0.0),
    _candidate(
        "rel_mutual_snn",
        gamma_sim=0.0,
        gamma_mutual=2.0,
        gamma_snn=2.0,
        gamma_distance=0.0,
    ),
    _candidate("gate_strict", beta_mutual=2.0, beta_snn=2.0, beta_perturb=4.0, gate_max=0.1),
    _candidate("gate_open", beta_mutual=2.0, beta_snn=2.0, beta_perturb=1.0),
    _candidate(
        "combo_local",
        neighbor_k=5,
        mix_neighbors=2,
        gate_max=0.1,
        pseudo_weight=0.1,
    ),
    _candidate(
        "combo_sparse",
        neighbor_k=20,
        mix_neighbors=8,
        gate_max=0.1,
        pseudo_weight=0.3,
    ),
)
CANDIDATE_BY_ID = {row["candidate_id"]: row for row in CANDIDATES}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "v19_rg_selected_advantage_v1":
        raise ValueError("unsupervised tuning requires the frozen V19 selected-data manifest")
    if payload.get("selection_policy", {}).get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("manifest selection policy must be label/outcome independent")
    rows = payload.get("datasets", [])
    if len(rows) != 11:
        raise ValueError(f"expected 11 fixed V19 input strata, got {len(rows)}")
    if any(row.get("status") != "eligible" for row in rows):
        raise ValueError("all fixed V19 input strata must be eligible")
    return payload


def _is_completed(path: Path) -> bool:
    try:
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return status.get("status") == "completed" and summary.get("status") == "completed"


def _prepare_matrix(
    record: dict[str, Any],
    config: V19Config,
    seed: int,
    max_samples: int,
) -> tuple[Any, dict[str, Any]]:
    loaded = load_npz_matrix_only(record["source_path"])
    matrix = loaded.X
    row_indices = None
    if int(max_samples) > 0 and matrix.shape[0] > int(max_samples):
        import numpy as np

        sampling_rng = np.random.default_rng(int(seed) + 91_109)
        row_indices = np.sort(
            sampling_rng.choice(matrix.shape[0], size=int(max_samples), replace=False)
        )
        matrix = matrix[row_indices]
    prepared = prepare_input(
        matrix,
        dataset_name=str(record["name"]),
        input_protocol=str(record["input_protocol"]),
        n_top_features=config.n_top_features,
        target_sum=config.target_sum,
    )
    profile = dict(loaded.profile)
    profile.update(
        {
            "dataset_id": str(record["dataset_id"]),
            "dataset_name": str(record["name"]),
            "input_protocol": str(record["input_protocol"]),
            "n_samples_used": int(matrix.shape[0]),
            "row_sampling": row_indices is not None,
            "row_sampling_seed": int(seed) + 91_109 if row_indices is not None else None,
            "labels_accessed": False,
        }
    )
    return prepared, profile


def _run_one(
    record: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    output_root: Path,
    *,
    config_path: Path,
    gpu: int,
    max_samples: int,
    manifest_id: str,
    force: bool,
) -> dict[str, Any]:
    import numpy as np

    candidate_id = str(candidate["candidate_id"])
    output = output_root / str(record["dataset_id"]) / candidate_id / f"seed{int(seed)}"
    output.mkdir(parents=True, exist_ok=True)
    run_key = f"{record['dataset_id']}::{candidate_id}::seed{int(seed)}"
    if not force and _is_completed(output):
        return {"status": "completed", "run_key": run_key, "skipped": True}
    run_record: dict[str, Any] = {
        "status": "running",
        "run_key": run_key,
        "protocol_id": PROTOCOL_ID,
        "manifest_id": manifest_id,
        "dataset_id": str(record["dataset_id"]),
        "dataset": str(record["name"]),
        "source_path": str(record["source_path"]),
        "input_protocol": str(record["input_protocol"]),
        "candidate_id": candidate_id,
        "seed": int(seed),
        "labels_accessed": False,
        "y_key_read": False,
        "n_clusters_used": None,
        "readout_enabled": False,
    }
    _write_json(output / "run_record.json", run_record)
    started = time.time()
    try:
        config = load_config(
            config_path,
            {
                "protocol_id": PROTOCOL_ID,
                "variant": "rg_full",
                **candidate["overrides"],
            },
        )
        prepared, input_profile = _prepare_matrix(record, config, int(seed), int(max_samples))
        runtime_device = resolve_runtime_device("cuda" if gpu >= 0 else "cpu", int(gpu))
        _predictions, embedding, diagnostics = fit_predict(
            prepared.X,
            n_clusters=None,
            config=config,
            seed=int(seed),
            device=runtime_device,
            evaluate_unsupervised=True,
        )
        unsupervised = diagnostics["unsupervised_diagnostics"]
        history = diagnostics["training_history"]
        last = {
            key: (float(value[-1]) if isinstance(value, list) and value else None)
            for key, value in history.items()
            if isinstance(value, list)
        }
        summary = {
            "status": "completed",
            "protocol_id": PROTOCOL_ID,
            "run_key": run_key,
            "manifest_id": manifest_id,
            "dataset_id": str(record["dataset_id"]),
            "dataset": str(record["name"]),
            "input_protocol": str(record["input_protocol"]),
            "candidate_id": candidate_id,
            "seed": int(seed),
            "device": str(runtime_device),
            "n_samples": int(prepared.X.shape[0]),
            "n_features": int(prepared.X.shape[1]),
            "labels_accessed": False,
            "y_key_read": False,
            "n_clusters_used": None,
            "readout_enabled": False,
            "K_used_only_in_readout": False,
            "resolved_config": config.resolved_dict(),
            "input_profile": input_profile,
            "preprocess_profile": prepared.profile,
            "unsupervised_diagnostics": unsupervised,
            "training_last_epoch": last,
            "graph_profile": diagnostics["graph_profile"],
            "edge_weight_summary": diagnostics["edge_summary"],
            "gate_summary": diagnostics["gate_summary"],
            "embedding_shape": [int(value) for value in embedding.shape],
            "wall_seconds": float(time.time() - started),
        }
        _write_json(output / "resolved_config.json", config.resolved_dict())
        _write_json(output / "input_profile.json", input_profile)
        _write_json(output / "preprocess_profile.json", prepared.profile)
        _write_json(output / "training_history.json", history)
        _write_json(output / "unsupervised_diagnostics.json", unsupervised)
        _write_json(output / "summary.json", summary)
        _write_json(output / "status.json", {"status": "completed", "run_key": run_key})
        run_record.update(
            {
                "status": "completed",
                "wall_seconds": summary["wall_seconds"],
                "summary": "summary.json",
            }
        )
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
    parser = argparse.ArgumentParser(description="V19 RG label-free hyperparameter search")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--candidate-ids", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    candidate_ids = args.candidate_ids or list(CANDIDATE_BY_ID)
    unknown = sorted(set(candidate_ids) - set(CANDIDATE_BY_ID))
    if unknown:
        raise ValueError(f"unknown candidate ids: {unknown}")
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or any(seed not in FORMAL_SEEDS for seed in seeds):
        raise ValueError(f"seeds must be drawn from {FORMAL_SEEDS}")
    requested = set(args.datasets or [])
    records = [
        row
        for row in manifest["datasets"]
        if not requested or str(row["dataset_id"]) in requested
    ]
    jobs = [
        (row, CANDIDATE_BY_ID[candidate_id], seed)
        for seed in seeds
        for row in records
        for candidate_id in candidate_ids
    ]
    worker_count = max(1, int(args.num_workers))
    if not 0 <= int(args.worker_id) < worker_count:
        raise ValueError("worker-id must be in [0, num-workers)")
    jobs = [job for index, job in enumerate(jobs) if index % worker_count == int(args.worker_id)]
    if int(args.limit) > 0:
        jobs = jobs[: int(args.limit)]
    header = {
        "protocol_id": PROTOCOL_ID,
        "manifest_id": manifest.get("manifest_id"),
        "candidate_count": len(candidate_ids),
        "candidate_ids": list(candidate_ids),
        "seeds": list(seeds),
        "datasets": len(records),
        "jobs": len(jobs),
        "labels_accessed": False,
        "n_clusters_used": None,
    }
    print(json.dumps(header, ensure_ascii=True), flush=True)
    if args.dry_run:
        for row, candidate, seed in jobs:
            print(f"{row['dataset_id']}\t{candidate['candidate_id']}\tseed={seed}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    physical_gpu = -1 if args.cpu else int(args.gpu)
    if physical_gpu >= 0 and physical_gpu not in ALLOWED_GPUS:
        raise ValueError(f"GPU {physical_gpu} is forbidden; allowed GPUs are {sorted(ALLOWED_GPUS)}")
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    if physical_gpu >= 0:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    rows = []
    for index, (record, candidate, seed) in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {record['dataset_id']} {candidate['candidate_id']} seed={seed}",
            flush=True,
        )
        row = _run_one(
            record,
            candidate,
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
    worker_summary = {
        **header,
        "worker_id": int(args.worker_id),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "incomplete_compute": sum(row.get("status") == "incomplete_compute" for row in rows),
        "runs": rows,
    }
    _write_json(
        args.output_dir / f"tune_worker{int(args.worker_id)}_{int(time.time())}.json",
        worker_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
