#!/usr/bin/env python
"""Label-free, paired RG-full hyperparameter search for V19.

This is deliberately a new protocol and output root.  The v1 tuner remains
usable for reproducing its historical X-only pilot.  V2 fits on a fixed
training-row split and evaluates masked recovery, latent stability, and input
neighborhood preservation on unseen rows.  It never loads labels, derives K,
or runs a clustering readout.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from zlib import crc32

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V19_rg_adapter.config import V19Config, load_config  # noqa: E402
from methods.TopoGate.V19_rg_adapter.input_adapter import (  # noqa: E402
    load_npz_matrix_only,
    prepare_input,
)
from methods.TopoGate.V19_rg_adapter.run import resolve_runtime_device  # noqa: E402
from methods.TopoGate.V19_rg_adapter.trainer import fit_predict  # noqa: E402


PROTOCOL_ID = "v19_rg_unsup_tuning_v2"
MANIFEST_PROTOCOL_ID = "v19_rg_selected_advantage_v1"
FORMAL_SEEDS = (42, 123, 7)
ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})
DEFAULT_CONFIG = ROOT / "methods" / "TopoGate" / "V19_rg_adapter" / "configs" / "v19_rg.yaml"
DEFAULT_OUTPUT = ROOT / "result" / "V19" / "v19_rg_unsup_tuning_v2_paired_20260809"
VAL_FRACTION = 0.20

STAGES = (
    "mechanism_screen",
    "mechanism_refine",
    "backbone_screen",
    "joint_refine",
)
FORMAL_RG_STAGES = ("mechanism_screen", "mechanism_refine")
SCHEDULES = ("manifest", "small_first")


def _candidate(candidate_id: str, family: str, **overrides: Any) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "family": family,
        "overrides": overrides,
    }


def _build_mechanism_candidates() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = [_candidate("default", "mechanism")]
    rows.extend(
        _candidate(f"k{k}", "mechanism", neighbor_k=k)
        for k in (5, 15, 20, 40)
    )
    rows.extend(
        _candidate(f"pca{dim}", "mechanism", knn_pca_dim=dim)
        for dim in (25, 100, 200)
    )
    rows.extend(
        _candidate(f"tau{str(tau).replace('.', '')}", "mechanism", tau=tau)
        for tau in (0.1, 0.3, 0.4)
    )
    rows.extend(
        _candidate(f"gate{str(value).replace('.', '')}", "mechanism", gate_max=value)
        for value in (0.05, 0.10, 0.20, 0.25)
    )
    rows.extend(
        _candidate(f"mix{k}", "mechanism", mix_neighbors=k)
        for k in (1, 2, 6, 8)
    )
    rows.extend(
        _candidate(f"pseudo{str(value).replace('.', '')}", "mechanism", pseudo_weight=value)
        for value in (0.1, 0.2, 0.4, 0.5)
    )
    rows.extend(
        [
            _candidate("rel_uniform", "mechanism", gamma_sim=0.0, gamma_mutual=0.0, gamma_snn=0.0, gamma_distance=0.0),
            _candidate("rel_mutual2", "mechanism", gamma_sim=0.0, gamma_mutual=2.0, gamma_snn=0.0, gamma_distance=0.0),
            _candidate("rel_snn2", "mechanism", gamma_sim=0.0, gamma_mutual=0.0, gamma_snn=2.0, gamma_distance=0.0),
            _candidate("rel_both2", "mechanism", gamma_sim=0.0, gamma_mutual=2.0, gamma_snn=2.0, gamma_distance=0.0),
            _candidate("rel_m2s1", "mechanism", gamma_sim=0.0, gamma_mutual=2.0, gamma_snn=1.0, gamma_distance=0.0),
            _candidate("rel_m1s2", "mechanism", gamma_sim=0.0, gamma_mutual=1.0, gamma_snn=2.0, gamma_distance=0.0),
        ]
    )
    rows.extend(
        [
            _candidate("beta_m0", "mechanism", beta_mutual=0.0),
            _candidate("beta_m2", "mechanism", beta_mutual=2.0),
            _candidate("beta_s0", "mechanism", beta_snn=0.0),
            _candidate("beta_s2", "mechanism", beta_snn=2.0),
            _candidate("beta_p1", "mechanism", beta_perturb=1.0),
            _candidate("beta_p4", "mechanism", beta_perturb=4.0),
            _candidate("beta_m2s2", "mechanism", beta_mutual=2.0, beta_snn=2.0),
            _candidate("beta_m2p4", "mechanism", beta_mutual=2.0, beta_perturb=4.0),
        ]
    )
    rows.extend(
        [
            _candidate("local_conservative", "mechanism", neighbor_k=5, mix_neighbors=1, pseudo_weight=0.1, gate_max=0.1),
            _candidate("wide_conservative", "mechanism", neighbor_k=40, mix_neighbors=2, pseudo_weight=0.1, gate_max=0.1),
            _candidate("sharp_lowmix", "mechanism", tau=0.1, neighbor_k=10, mix_neighbors=2, pseudo_weight=0.2),
            _candidate("soft_highmix", "mechanism", tau=0.4, neighbor_k=20, mix_neighbors=8, pseudo_weight=0.3),
            _candidate("mutual_lowmix", "mechanism", gamma_sim=0.0, gamma_distance=0.0, gamma_mutual=2.0, gamma_snn=1.0, mix_neighbors=2, pseudo_weight=0.1),
            _candidate("snn_sparse", "mechanism", gamma_sim=0.0, gamma_distance=0.0, gamma_mutual=1.0, gamma_snn=2.0, mix_neighbors=2, pseudo_weight=0.1),
            _candidate("strong_reliability", "mechanism", gamma_sim=0.0, gamma_distance=0.0, gamma_mutual=2.0, gamma_snn=2.0, beta_mutual=2.0, beta_snn=2.0, pseudo_weight=0.2),
            _candidate("low_gate_reliability", "mechanism", gamma_sim=0.0, gamma_distance=0.0, gamma_mutual=2.0, gamma_snn=2.0, gate_max=0.05, pseudo_weight=0.1),
            _candidate("high_gate_reliability", "mechanism", gamma_sim=0.0, gamma_distance=0.0, gamma_mutual=2.0, gamma_snn=2.0, gate_max=0.25, pseudo_weight=0.2),
            _candidate("pca_high_local", "mechanism", knn_pca_dim=100, neighbor_k=5, mix_neighbors=2, pseudo_weight=0.1),
            _candidate("pca_low_wide", "mechanism", knn_pca_dim=20, neighbor_k=40, mix_neighbors=4, pseudo_weight=0.2),
        ]
    )
    if len(rows) != 48:
        raise AssertionError(f"expected 48 mechanism candidates, got {len(rows)}")
    return tuple(rows)


MECHANISM_CANDIDATES = _build_mechanism_candidates()
BACKBONE_PROFILES = (
    _candidate("bb_base", "backbone"),
    _candidate("bb_mask03", "backbone", mask_ratio=0.3),
    _candidate("bb_mask05", "backbone", mask_ratio=0.5),
    _candidate("bb_lr5e4", "backbone", lr=5e-4),
    _candidate("bb_lr2e3", "backbone", lr=2e-3),
    _candidate("bb_hidden64", "backbone", hidden_size=64),
    _candidate("bb_hidden256", "backbone", hidden_size=256),
    _candidate("bb_epochs120", "backbone", epochs=120),
)


def _catalog_by_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    materialized = tuple(rows)
    catalog = {str(row["candidate_id"]): row for row in materialized}
    if len(catalog) != len(materialized):
        raise AssertionError("candidate ids must be unique")
    return catalog


def _joint_candidates(mechanism_ids: Iterable[str]) -> tuple[dict[str, Any], ...]:
    mechanism = {row["candidate_id"]: row for row in MECHANISM_CANDIDATES}
    rows: list[dict[str, Any]] = []
    for mechanism_id in mechanism_ids:
        if mechanism_id not in mechanism:
            raise ValueError(f"unknown mechanism candidate for joint stage: {mechanism_id}")
        for profile in BACKBONE_PROFILES:
            candidate_id = f"{mechanism_id}__{profile['candidate_id']}"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "family": "joint",
                    "mechanism_id": mechanism_id,
                    "backbone_id": profile["candidate_id"],
                    "overrides": {
                        **mechanism[mechanism_id]["overrides"],
                        **profile["overrides"],
                    },
                }
            )
    return tuple(rows)


def candidate_catalog_for_stage(
    stage: str,
    *,
    candidate_ids: Iterable[str] | None = None,
    selected_config: Path | None = None,
    mechanism_count: int = 4,
) -> tuple[dict[str, Any], ...]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if stage == "mechanism_screen":
        rows = MECHANISM_CANDIDATES
    elif stage == "backbone_screen":
        if candidate_ids:
            top_mechanisms: list[str] = []
            for candidate_id in candidate_ids:
                mechanism_id = str(candidate_id).split("__", 1)[0]
                if mechanism_id not in top_mechanisms:
                    top_mechanisms.append(mechanism_id)
        else:
            if selected_config is None:
                raise ValueError("backbone_screen requires --selected-config from mechanism_refine")
            selection = json.loads(Path(selected_config).read_text(encoding="utf-8"))
            top_mechanisms = [
                str(value)
                for value in selection.get("top_candidate_ids", [])[: int(mechanism_count)]
            ]
            if not top_mechanisms:
                scores = selection.get("candidate_scores", [])
                top_mechanisms = [str(row["candidate_id"]) for row in scores[: int(mechanism_count)]]
            if not top_mechanisms:
                raise ValueError("mechanism_refine selected config contains no top candidates")
        rows = _joint_candidates(top_mechanisms)
    elif stage == "joint_refine":
        if not candidate_ids:
            raise ValueError("joint_refine requires explicit joint candidate ids")
        mechanism_ids: list[str] = []
        for candidate_id in candidate_ids:
            mechanism_id = str(candidate_id).split("__", 1)[0]
            if mechanism_id not in mechanism_ids:
                mechanism_ids.append(mechanism_id)
        rows = _joint_candidates(mechanism_ids)
    else:
        rows = MECHANISM_CANDIDATES
    if candidate_ids is None:
        return tuple(rows)
    by_id = {str(row["candidate_id"]): row for row in rows}
    unknown = sorted(set(candidate_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown candidate ids for {stage}: {unknown}")
    return tuple(by_id[str(candidate_id)] for candidate_id in candidate_ids)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != MANIFEST_PROTOCOL_ID:
        raise ValueError("v2 requires the frozen V19 selected-data manifest")
    rows = payload.get("datasets", [])
    if len(rows) != 11 or any(row.get("status") != "eligible" for row in rows):
        raise ValueError("v2 requires all 11 eligible V19 input strata")
    if payload.get("selection_policy", {}).get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("manifest selection policy is not label-free")
    return payload


def underlying_dataset_id(dataset_id: str) -> str:
    return str(dataset_id).split("__", 1)[0]


def select_records(
    manifest: dict[str, Any],
    groups: Iterable[str] | None,
    *,
    comparable_only: bool = False,
) -> list[dict[str, Any]]:
    requested = {str(value) for value in (groups or [])}
    rows = [
        row
        for row in manifest["datasets"]
        if (not requested or underlying_dataset_id(str(row["dataset_id"])) in requested)
        and (
            not comparable_only
            or row.get("comparison_scope") == "archived_sota_bridge_eligible"
        )
    ]
    if not rows:
        raise ValueError("dataset/group filter selected no manifest rows")
    return rows


def _prepare_matrix(
    record: dict[str, Any],
    config: V19Config,
    seed: int,
    max_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    loaded = load_npz_matrix_only(record["source_path"])
    matrix = loaded.X
    row_indices = None
    if int(max_samples) > 0 and matrix.shape[0] > int(max_samples):
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
    profile.update(prepared.profile)
    profile["selected_feature_indices"] = [
        int(value) for value in prepared.selected_feature_indices.tolist()
    ]
    profile.update(
        {
            "dataset_id": str(record["dataset_id"]),
            "dataset_name": str(record["name"]),
            "underlying_dataset_id": underlying_dataset_id(str(record["dataset_id"])),
            "input_protocol": str(record["input_protocol"]),
            "n_samples_used": int(matrix.shape[0]),
            "row_sampling": row_indices is not None,
            "row_sampling_seed": int(seed) + 91_109 if row_indices is not None else None,
            "labels_accessed": False,
        }
    )
    return prepared.X, profile


def split_rows(n_samples: int, dataset_id: str, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    if n_samples < 4:
        raise ValueError("held-out row protocol requires at least four samples")
    split_seed = int(seed) + int(crc32(str(dataset_id).encode("utf-8")) & 0x7FFFFFFF) + 71_003
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(int(n_samples))
    n_validation = max(2, int(round(float(n_samples) * VAL_FRACTION)))
    n_validation = min(n_validation, int(n_samples) - 2)
    validation = np.sort(order[:n_validation]).astype(np.int64, copy=False)
    training = np.sort(order[n_validation:]).astype(np.int64, copy=False)
    return training, validation, split_seed


def _cached_pca_embedding(
    output_root: Path,
    dataset_id: str,
    seed: int,
    pca_dim: int,
    fit_data: np.ndarray,
) -> np.ndarray:
    """Share deterministic PCA work across candidates and worker processes."""
    from methods.TopoGate.V19_rg_adapter.graph import build_pca_knn_graph

    cache_dir = output_root / "_graph_cache" / str(dataset_id) / f"seed{int(seed)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fit_pca{int(pca_dim)}"
    cache_path = cache_dir / f"{stem}.npy"
    lock_path = cache_dir / f"{stem}.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_path.is_file():
            cached = np.load(cache_path, allow_pickle=False)
            expected_dim = max(1, min(int(pca_dim), fit_data.shape[1], fit_data.shape[0] - 1))
            if cached.shape == (fit_data.shape[0], expected_dim) and np.all(np.isfinite(cached)):
                return np.ascontiguousarray(cached, dtype=np.float32)
        graph = build_pca_knn_graph(
            fit_data,
            k=1,
            pca_dim=int(pca_dim),
            tau=0.2,
            seed=int(seed),
        )
        temporary = cache_path.with_name(f".{stem}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, graph.embedding, allow_pickle=False)
        os.replace(temporary, cache_path)
        return np.ascontiguousarray(graph.embedding, dtype=np.float32)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def _is_completed(path: Path, run_key: str) -> bool:
    try:
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        run_record = json.loads((path / "run_record.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        status.get("status") == "completed"
        and summary.get("status") == "completed"
        and run_record.get("status") == "completed"
        and summary.get("protocol_id") == PROTOCOL_ID
        and summary.get("run_key") == run_key
        and run_record.get("run_key") == run_key
        and summary.get("labels_accessed") is False
        and summary.get("y_key_read") is False
        and summary.get("readout_enabled") is False
    )


def _preserve_previous_attempt(output: Path) -> None:
    """Keep prior status evidence before a resumable run rewrites its files."""

    existing = [output / name for name in ("run_record.json", "status.json", "summary.json")]
    existing = [path for path in existing if path.is_file()]
    if not existing:
        return
    attempt = output / "attempts" / f"attempt_{int(time.time())}_{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy2(path, attempt / f"{path.stem}.previous.json")


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
    stage: str,
    force: bool,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    dataset_id = str(record["dataset_id"])
    output = output_root / dataset_id / candidate_id / f"seed{int(seed)}"
    output.mkdir(parents=True, exist_ok=True)
    run_key = f"{stage}::{dataset_id}::{candidate_id}::seed{int(seed)}"
    if not force and _is_completed(output, run_key):
        return {"status": "completed", "run_key": run_key, "skipped": True}
    _preserve_previous_attempt(output)
    run_record: dict[str, Any] = {
        "status": "running",
        "run_key": run_key,
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "manifest_id": manifest_id,
        "dataset_id": dataset_id,
        "underlying_dataset_id": underlying_dataset_id(dataset_id),
        "dataset": str(record["name"]),
        "source_path": str(record["source_path"]),
        "input_protocol": str(record["input_protocol"]),
        "candidate_id": candidate_id,
        "candidate_family": str(candidate["family"]),
        "seed": int(seed),
        "labels_accessed": False,
        "y_key_read": False,
        "n_clusters_used": None,
        "readout_enabled": False,
    }
    _write_json(output / "run_record.json", run_record)
    _write_json(output / "status.json", {"status": "running", "run_key": run_key, "protocol_id": PROTOCOL_ID})
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
        reference_eval_config = load_config(
            config_path,
            {
                "protocol_id": PROTOCOL_ID,
                "variant": "scmae_only",
            },
        )
        prepared, input_profile = _prepare_matrix(record, config, int(seed), int(max_samples))
        training_indices, validation_indices, split_seed = split_rows(
            prepared.shape[0], dataset_id, int(seed)
        )
        np.save(output / "training_row_indices.npy", training_indices)
        np.save(output / "validation_row_indices.npy", validation_indices)
        runtime_device = resolve_runtime_device("cuda" if gpu >= 0 else "cpu", int(gpu))
        graph_embedding_cache = {
            int(config.knn_pca_dim): _cached_pca_embedding(
                output_root,
                dataset_id,
                int(seed),
                int(config.knn_pca_dim),
                prepared[training_indices],
            )
        }
        _predictions, embedding, diagnostics = fit_predict(
            prepared,
            n_clusters=None,
            config=config,
            seed=int(seed),
            device=runtime_device,
            evaluate_unsupervised=True,
            fit_X=prepared[training_indices],
            evaluation_X=prepared[validation_indices],
            evaluation_mask_ratio=reference_eval_config.mask_ratio,
            evaluation_graph_config=reference_eval_config,
            precomputed_graph_embeddings=graph_embedding_cache,
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
            "stage": stage,
            "manifest_id": manifest_id,
            "dataset_id": dataset_id,
            "underlying_dataset_id": underlying_dataset_id(dataset_id),
            "dataset": str(record["name"]),
            "input_protocol": str(record["input_protocol"]),
            "candidate_id": candidate_id,
            "candidate_family": str(candidate["family"]),
            "candidate_overrides": candidate["overrides"],
            "seed": int(seed),
            "device": str(runtime_device),
            "physical_gpu": int(gpu) if int(gpu) >= 0 else None,
            "n_samples": int(prepared.shape[0]),
            "fit_n_samples": int(training_indices.size),
            "evaluation_n_samples": int(validation_indices.size),
            "split_seed": int(split_seed),
            "validation_fraction": VAL_FRACTION,
            "labels_accessed": False,
            "y_key_read": False,
            "n_clusters_used": None,
            "readout_enabled": False,
            "K_used_only_in_readout": False,
            "resolved_config": config.resolved_dict(),
            "paired_reference_eval_config": reference_eval_config.resolved_dict(),
            "input_profile": input_profile,
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
        _write_json(output / "training_history.json", history)
        _write_json(output / "unsupervised_diagnostics.json", unsupervised)
        _write_json(output / "summary.json", summary)
        _write_json(output / "status.json", {"status": "completed", "run_key": run_key, "protocol_id": PROTOCOL_ID})
        run_record.update({"status": "completed", "wall_seconds": summary["wall_seconds"], "summary": "summary.json"})
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


def _stage_spec(
    output_root: Path,
    *,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    candidates: tuple[dict[str, Any], ...],
    stage: str,
    seeds: tuple[int, ...],
    comparable_only: bool,
    config_path: Path,
    max_samples: int,
) -> dict[str, Any]:
    spec = {
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "manifest_id": manifest.get("manifest_id"),
        "candidate_ids": [str(row["candidate_id"]) for row in candidates],
        "candidate_definitions": list(candidates),
        "candidate_count": len(candidates),
        "dataset_ids": [str(row["dataset_id"]) for row in records],
        "underlying_dataset_ids": sorted({underlying_dataset_id(str(row["dataset_id"])) for row in records}),
        "comparable_only": bool(comparable_only),
        "seeds": [int(seed) for seed in seeds],
        "expected_runs": len(records) * len(candidates) * len(seeds),
        "config_path": str(config_path.resolve()),
        "max_samples": int(max_samples),
        "validation_protocol": "fixed_held_out_rows_20pct_per_dataset_seed",
        "preprocessing_protocol": "transductive_full_X_label_free_preprocessing",
        "formal_scope": "rg_mechanism_only",
        "selection_uses_labels_or_outcomes": False,
        "labels_accessed": False,
        "y_key_read": False,
        "n_clusters_used": None,
        "readout_enabled": False,
    }
    stage_path = output_root / "stage_spec.json"
    if stage_path.exists():
        existing = json.loads(stage_path.read_text(encoding="utf-8"))
        if existing != spec:
            raise ValueError(f"existing stage_spec.json does not match requested resume protocol: {stage_path}")
    else:
        _write_json(stage_path, spec)
    return spec


def _is_large_record(record: dict[str, Any]) -> bool:
    return int(_schedule_cost(record)[1]) >= 50 * 1024 * 1024


def _assign_jobs(
    jobs: list[tuple[dict[str, Any], dict[str, Any], int]],
    *,
    worker_count: int,
    worker_id: int,
    schedule: str,
) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    """Assign the already size-ordered queue across distinct GPU workers.

    ``small_first`` orders records by a label-free source-size estimate before
    this function is called. Round-robin then lets different physical GPUs
    process large inputs in parallel while keeping one worker process per GPU.
    The run key set and all model inputs remain unchanged.
    """

    if schedule != "small_first" or worker_count <= 1:
        return [
            job
            for index, job in enumerate(jobs)
            if index % worker_count == worker_id
        ]
    return [
        job
        for index, job in enumerate(jobs)
        if index % worker_count == worker_id
    ]


def _schedule_cost(record: dict[str, Any]) -> tuple[float, int, str]:
    """Return a label-free, deterministic estimate used only for queue order.

    The scheduler must not inspect labels or model outcomes.  The source file
    size is a conservative proxy for matrix materialization/graph cost; the
    protocol multiplier keeps bridge inputs behind native/text inputs of the
    same source size because the bridge preprocessing path is more expensive.
    This does not alter a run's configuration, seed, or output key.
    """

    source = Path(str(record["source_path"]))
    try:
        source_bytes = int(source.stat().st_size)
    except OSError:
        source_bytes = 1 << 62
    protocol = str(record.get("input_protocol", ""))
    multiplier = {
        "shared_text": 0.8,
        "rg_native": 1.0,
        "clubench_bridge": 1.6,
    }.get(protocol, 1.2)
    return (float(source_bytes) * float(multiplier), int(source_bytes), str(record["dataset_id"]))


def _schedule_records(
    output_root: Path,
    records: list[dict[str, Any]],
    *,
    stage: str,
    schedule: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
    if schedule == "small_first":
        ordered = sorted(records, key=_schedule_cost)
    else:
        ordered = list(records)
    cost_rows = [
        {
            "dataset_id": str(record["dataset_id"]),
            "input_protocol": str(record.get("input_protocol", "")),
            "estimated_cost": float(_schedule_cost(record)[0]),
            "source_bytes": int(_schedule_cost(record)[1]),
        }
        for record in ordered
    ]
    payload = {
        "protocol_id": PROTOCOL_ID,
        "stage": str(stage),
        "schedule": str(schedule),
        "ordered_dataset_ids": [str(record["dataset_id"]) for record in ordered],
        "costs": cost_rows,
        "label_free": True,
        "purpose": "queue_order_only",
    }
    path = output_root / "schedule_spec.json"
    lock_path = output_root / "schedule_spec.lock"
    output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            immutable = ("protocol_id", "stage", "schedule", "ordered_dataset_ids", "label_free", "purpose")
            if any(existing.get(key) != payload.get(key) for key in immutable):
                raise ValueError(f"existing schedule_spec.json does not match requested schedule: {path}")
            payload = existing
        else:
            _write_json(path, payload)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    by_id = {str(record["dataset_id"]): record for record in records}
    return [by_id[dataset_id] for dataset_id in payload["ordered_dataset_ids"]], payload


def main() -> int:
    parser = argparse.ArgumentParser(description="V19 RG full label-free v2 tuning")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--selected-config", type=Path, default=None)
    parser.add_argument("--candidate-ids", nargs="*", default=None)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument(
        "--comparable-only",
        action="store_true",
        help="retain only bridge-equivalent input layers for SOTA-comparable screens",
    )
    parser.add_argument("--mechanism-count", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default="small_first",
        help="queue order only; small_first defers large source matrices",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stage not in FORMAL_RG_STAGES:
        raise ValueError("formal V19 v2 tuning is RG mechanism-only; backbone/joint stages are disabled")
    if args.groups:
        raise ValueError("formal V19 v2 tuning does not permit --groups")
    if int(args.max_samples) != 0 or int(args.limit) != 0 or args.force:
        raise ValueError("formal V19 v2 tuning requires max_samples=0, limit=0, and no --force")

    manifest = _load_manifest(args.manifest)
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or any(seed not in FORMAL_SEEDS for seed in seeds):
        raise ValueError(f"seeds must be drawn from {FORMAL_SEEDS}")
    records = select_records(manifest, args.groups, comparable_only=bool(args.comparable_only))
    candidates = candidate_catalog_for_stage(
        args.stage,
        candidate_ids=args.candidate_ids,
        selected_config=args.selected_config,
        mechanism_count=int(args.mechanism_count),
    )
    if args.stage == "mechanism_screen":
        if seeds != (42,) or not args.comparable_only or len(records) != 8 or len(candidates) != 48:
            raise ValueError("mechanism_screen contract requires 8 comparable layers, 48 candidates, seed 42")
        if args.candidate_ids or args.selected_config is not None:
            raise ValueError("mechanism_screen does not accept candidate ids or selected config")
    elif (
        seeds != FORMAL_SEEDS
        or args.comparable_only
        or len(records) != 11
        or len(candidates) != 12
        or args.selected_config is not None
        or args.candidate_ids is None
        or len(args.candidate_ids) != 12
    ):
        raise ValueError("mechanism_refine contract requires 11 layers, 12 candidates, and seeds 42,123,7")
    worker_count = max(1, int(args.num_workers))
    if not 0 <= int(args.worker_id) < worker_count:
        raise ValueError("worker-id must be in [0, num-workers)")
    scheduled_records, schedule_spec = _schedule_records(
        args.output_dir,
        records,
        stage=args.stage,
        schedule=str(args.schedule),
    )
    jobs = [
        (record, candidate, seed)
        for record in scheduled_records
        for seed in seeds
        for candidate in candidates
    ]
    jobs = _assign_jobs(
        jobs,
        worker_count=worker_count,
        worker_id=int(args.worker_id),
        schedule=str(args.schedule),
    )
    if int(args.limit) > 0:
        jobs = jobs[: int(args.limit)]
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    spec = _stage_spec(
        output_root,
        manifest=manifest,
        records=records,
        candidates=candidates,
        stage=args.stage,
        seeds=seeds,
        comparable_only=bool(args.comparable_only),
        config_path=args.config,
        max_samples=int(args.max_samples),
    )
    header = {
        **spec,
        "worker_id": int(args.worker_id),
        "num_workers": worker_count,
        "jobs_for_worker": len(jobs),
        "schedule": schedule_spec,
        "assignment_policy": (
            "small_first__distinct_gpu_round_robin"
            if str(args.schedule) == "small_first"
            else "manifest_round_robin"
        ),
    }
    print(json.dumps(header, ensure_ascii=True), flush=True)
    if args.dry_run:
        for record, candidate, seed in jobs:
            print(f"{record['dataset_id']}\t{candidate['candidate_id']}\tseed={seed}")
        return 0

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
        print(f"[{index}/{len(jobs)}] {record['dataset_id']} {candidate['candidate_id']} seed={seed}", flush=True)
        row = _run_one(
            record,
            candidate,
            int(seed),
            output_root,
            config_path=args.config,
            gpu=physical_gpu,
            max_samples=int(args.max_samples),
            manifest_id=str(manifest.get("manifest_id", "unknown")),
            stage=args.stage,
            force=bool(args.force),
        )
        rows.append(row)
        print(json.dumps({"run_key": row.get("run_key"), "status": row.get("status")}), flush=True)
    worker_summary = {
        **header,
        "completed": sum(row.get("status") == "completed" for row in rows),
        "incomplete_compute": sum(row.get("status") == "incomplete_compute" for row in rows),
        "runs": rows,
    }
    _write_json(output_root / f"tune_worker{int(args.worker_id)}_{int(time.time())}.json", worker_summary)
    return 0 if worker_summary["incomplete_compute"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
