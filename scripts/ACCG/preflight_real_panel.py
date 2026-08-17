#!/usr/bin/env python3
"""Run a label-free, resource-bounded preflight for the frozen ACCG real panel.

The preflight deliberately stops after one ACCG forward/backward update.  It
reuses the same input adapter, feature model, epsilon calibration, sample
graph, topology statistics, V21 components, selector, and differentiable
constraint as the formal runner, but it never computes a clustering metric or
loads an NPZ label array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.ACCG_action_constrained_gate.config import ACCGConfig, load_config
from methods.TopoGate.ACCG_action_constrained_gate.protocol import (
    ACCGArm,
    _gate_update_accg,
    _loss_for_arm,
)
from methods.TopoGate.ACCG_action_constrained_gate.calibration import calibrate_epsilon
from methods.TopoGate.ACCG_action_constrained_gate.feature_model import fit_cross_fitted_feature_model
from methods.TopoGate.ACCG_action_constrained_gate.torch_energy import TorchFeatureConstraint
from methods.TopoGate.V21_assignment_adversarial_gate.graph import (
    build_svd_knn_graph,
    compute_topology_statistics,
)
from methods.TopoGate.V21_assignment_adversarial_gate.input_adapter import (
    load_npz_matrix_only,
    prepare_dual_input,
)
from methods.TopoGate.V25_systematic_mechanism_study import e1_protocol as e1


ALLOWED_GPUS = frozenset({1, 2, 3, 4, 5, 6})
SUPPORTED_MANIFEST_IDS = frozenset({"accg_locked_real_panel_v1", "accg_locked_real_panel_v2"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_id") not in SUPPORTED_MANIFEST_IDS:
        raise ValueError(f"unsupported ACCG manifest: {payload.get('manifest_id')!r}")
    if payload.get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("ACCG preflight requires outcome-independent panel selection")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("ACCG manifest has no datasets")
    dataset_ids = [str(row.get("dataset_id")) for row in datasets]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("ACCG manifest contains duplicate dataset ids")
    return payload


def _record_k(record: dict[str, Any]) -> tuple[int, str]:
    explicit = record.get("n_clusters")
    if explicit is not None:
        return int(explicit), "explicit_n_clusters"
    labels_unique = record.get("labels_unique")
    if labels_unique is None:
        raise ValueError(f"manifest record has no K: {record.get('dataset_id')}")
    return int(labels_unique), "benchmark_oracle_from_manifest_metadata"


def _matrix_profile(matrix: Any) -> dict[str, Any]:
    shape = tuple(int(value) for value in matrix.shape)
    if hasattr(matrix, "nnz"):
        zero_fraction = 1.0 - float(matrix.nnz / max(1, shape[0] * shape[1]))
        storage = "csr"
    else:
        values = np.asarray(matrix)
        zero_fraction = float(np.mean(values == 0.0))
        storage = "dense"
    return {
        "shape": [shape[0], shape[1]],
        "storage": storage,
        "dtype": str(matrix.dtype),
        "zero_fraction": zero_fraction,
    }


def _parameter_count(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _device_profile(device: torch.device) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "requested_device": str(device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda":
        profile.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "device_index": int(device.index or 0),
            }
        )
    return profile


def _peak_memory(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "cuda_current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
        )
    return result


def _timed_phase(
    phases: dict[str, Any],
    name: str,
    function: Any,
) -> Any:
    start = time.perf_counter()
    value = function()
    phases[name] = {"status": "completed", "wall_seconds": float(time.perf_counter() - start)}
    return value


def _make_preflight_config(config: ACCGConfig, epsilon_rounds: int | None) -> ACCGConfig:
    if epsilon_rounds is None:
        return config
    values = config.to_dict()
    values["constraint"]["epsilon_rounds"] = int(epsilon_rounds)
    from methods.TopoGate.ACCG_action_constrained_gate.config import FeatureConstraintConfig

    constrained = FeatureConstraintConfig(**values["constraint"])
    updated = ACCGConfig(
        protocol_id=values["protocol_id"],
        variant=values["variant"],
        v21=config.v21,
        constraint=constrained,
    )
    updated.validate()
    return updated


def _run_one(
    record: dict[str, Any],
    *,
    config: ACCGConfig,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    dataset_id = str(record["dataset_id"])
    source = Path(str(record["source_path"])).resolve()
    n_clusters, k_source = _record_k(record)
    phases: dict[str, Any] = {}
    started = time.perf_counter()
    result: dict[str, Any] = {
        "status": "running",
        "evidence_level": "resource_preflight_only",
        "preflight_id": "accg_real_panel_label_free_forward_backward_v1",
        "dataset_id": dataset_id,
        "dataset_name": str(record["name"]),
        "manifest_record": {
            "domain": record.get("domain"),
            "input_protocol": record.get("input_protocol"),
            "source_family": record.get("source_family"),
            "source_sha256_manifest": record.get("source_sha256"),
            "K_source_manifest": record.get("K_source"),
        },
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "seed": int(seed),
        "n_clusters": int(n_clusters),
        "K_source": k_source,
        "labels_loaded": False,
        "labels_used_during_fit": False,
        "labels_used_for_preflight": False,
        "outcomes_used": False,
        "device": _device_profile(device),
        "config": config.to_dict(),
        "phases": phases,
    }
    if result["source_sha256"] != record.get("source_sha256"):
        raise ValueError("source SHA256 does not match the frozen manifest")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    e1._seed_all(seed, device)

    matrix = _timed_phase(phases, "load_matrix_only", lambda: load_npz_matrix_only(source))
    result["input"] = _matrix_profile(matrix)
    expected_shape = [int(record["n_samples"]), int(record["n_features"])]
    if result["input"]["shape"] != expected_shape:
        raise ValueError(f"matrix shape {result['input']['shape']} != manifest {expected_shape}")

    prepared = _timed_phase(
        phases,
        "prepare_dual_input",
        lambda: prepare_dual_input(
            matrix,
            dataset_name=str(record["name"]),
            input_protocol=str(record["input_protocol"]),
        ),
    )
    result["preprocess_profile"] = _jsonable(prepared.profile)
    result["selected_feature_indices_hash"] = e1._hash_array(prepared.selected_feature_indices)
    if prepared.profile.get("labels_used") is not False:
        raise ValueError("input adapter reports label use")

    feature_model = _timed_phase(
        phases,
        "fit_cross_fitted_feature_model",
        lambda: fit_cross_fitted_feature_model(
            prepared.X_model,
            config=config.constraint,
            seed=seed,
        ),
    )
    result["feature_model_profile"] = _jsonable(feature_model.profile)
    if feature_model.profile.get("labels_used") is not False:
        raise ValueError("feature model reports label use")
    z_all = feature_model.transform_matrix(prepared.X_model).astype(np.float64)

    epsilon = _timed_phase(
        phases,
        "calibrate_epsilon",
        lambda: calibrate_epsilon(
            prepared.X_model,
            feature_model,
            mask_ratio=config.v21.assignment_mask_ratio,
            config=config.constraint,
            seed=seed,
        ),
    )
    result["epsilon_profile"] = _jsonable(epsilon.profile)
    if epsilon.profile.get("labels_used") is not False or epsilon.profile.get("outcomes_used") is not False:
        raise ValueError("epsilon calibration reports label or outcome use")

    graph = _timed_phase(
        phases,
        "build_sample_graph",
        lambda: build_svd_knn_graph(
            prepared.X_graph,
            neighbor_k=config.v21.neighbor_k,
            svd_target=config.v21.graph_svd_target,
            svd_min_dim=min(config.v21.graph_svd_min_dim, max(1, prepared.X_model.shape[0] - 1)),
            svd_max_dim=min(config.v21.graph_svd_max_dim, max(1, prepared.X_model.shape[0] - 1)),
            seed=seed,
        ),
    )
    result["sample_graph_profile"] = _jsonable(graph.profile)
    if graph.profile.get("label_leakage_diagnostic") is not False:
        raise ValueError("sample graph leakage diagnostic is not false")

    stats, stats_profile = _timed_phase(
        phases,
        "compute_topology_statistics",
        lambda: compute_topology_statistics(
            prepared.X_model,
            graph,
            block_size=config.v21.stats_block_size,
            cache_dir=output_dir / "cache",
            cache_dtype=config.v21.stats_cache_dtype,
            clip=config.v21.stats_clip,
        ),
    )
    result["stats_profile"] = _jsonable(stats_profile)
    result["topology_statistics_hash"] = e1._hash_array(stats)

    data_device = torch.device("cpu") if device.type == "cuda" else device
    X = torch.as_tensor(prepared.X_model, dtype=torch.float32, device=data_device)
    components = _timed_phase(
        phases,
        "build_v21_accg_components",
        lambda: e1._build_components(X, n_clusters, config.v21, seed, device),
    )
    model = components["model"]
    head = components["head"]
    result["model_profile"] = {
        "model_parameters": _parameter_count(model),
        "head_parameters": _parameter_count(head),
        "gate_parameters": _parameter_count(components["gate"]),
        "optimizer_foreach": components.get("optimizer_foreach"),
        "optimizer_fused": components.get("optimizer_fused", False),
        "model_device": str(next(model.parameters()).device),
        "model_dtype": str(next(model.parameters()).dtype),
        "host_data_device": str(X.device),
    }
    _timed_phase(phases, "initialise_cluster_head", lambda: e1._initialise_head(components, X, config.v21, seed))

    schedule = e1._make_schedule(prepared.X_model.shape[0], config.v21, seed)
    if not schedule.post_branch:
        raise ValueError("schedule has no post-branch entry for the bounded update")
    entry = schedule.post_branch[0]
    row_ids = np.asarray(entry.batch_ids, dtype=np.int64)
    tensors = e1._materialize_schedule(X, entry, config.v21, device)
    stats_batch = e1._stats_batch_on_device(stats, entry.batch_ids, device)
    constraint = TorchFeatureConstraint(feature_model)
    model.train()
    head.train()
    components["gate"].train()
    components["optimizer"].zero_grad(set_to_none=True)
    total, losses, counters = _timed_phase(
        phases,
        "one_accg_forward",
        lambda: _loss_for_arm(
            # The constrained arm is the exact primary action path; this does
            # not run an optimizer schedule or produce a clustering readout.
            ACCGArm.CONSTRAINED,
            components,
            tensors,
            stats_batch,
            row_ids=row_ids,
            z_all=z_all,
            epsilon=epsilon.epsilon,
            feature_model=feature_model,
            config=config,
        ),
    )
    def _backward_step() -> None:
        total.backward()
        components["optimizer"].step()

    _timed_phase(phases, "one_accg_backward_optimizer", _backward_step)
    gate_update = _timed_phase(
        phases,
        "one_accg_gate_backward_optimizer",
        lambda: _gate_update_accg(
            components,
            tensors,
            stats_batch,
            row_ids=row_ids,
            z_all=z_all,
            epsilon=epsilon.epsilon,
            feature_model=feature_model,
            torch_constraint=constraint,
            config=config,
        ),
    )
    result["forward_backward"] = {
        "batch_rows": int(row_ids.size),
        "batch_features": int(tensors["batch"].shape[1]),
        "loss": float(total.detach().cpu()),
        "assignment_forward": bool(counters["assignment_forward"]),
        "joint_delta_mean": float(np.mean(losses["structural"]["joint_delta"])),
        "constraint_infeasible_rate": float(np.mean(losses["selection"].constraint_infeasible)),
        "constraint_violation_rate": float(np.mean(losses["selection"].constraint_violated)),
        "gate_update": _jsonable(gate_update),
    }
    result["resource"] = _peak_memory(device)
    result["status"] = "completed"
    result["wall_seconds"] = float(time.perf_counter() - started)
    result["formal_protocol_equivalent"] = bool(config.constraint.epsilon_rounds == 16)
    result["formal_training_started"] = False
    return _jsonable(result)


def _run_record(
    record: dict[str, Any],
    *,
    config: ACCGConfig,
    seed: int,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    output_dir = output_root / str(record["dataset_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_one(record, config=config, seed=seed, device=device, output_dir=output_dir)
    except Exception as error:  # keep failures auditable and continue the panel
        result = {
            "status": "incomplete_compute",
            "evidence_level": "resource_preflight_only",
            "preflight_id": "accg_real_panel_label_free_forward_backward_v1",
            "dataset_id": str(record.get("dataset_id")),
            "dataset_name": str(record.get("name")),
            "source_path": str(record.get("source_path")),
            "seed": int(seed),
            "labels_loaded": False,
            "labels_used_during_fit": False,
            "formal_training_started": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(limit=20),
            },
        }
    _write_json(output_dir / "preflight.json", result)
    return result


def _select_records(manifest: dict[str, Any], datasets: list[str] | None) -> list[dict[str, Any]]:
    rows = list(manifest["datasets"])
    if not datasets:
        return rows
    wanted = set(datasets)
    found = {str(row["dataset_id"]) for row in rows}
    unknown = wanted - found
    if unknown:
        raise ValueError(f"unknown dataset ids: {sorted(unknown)}")
    return [row for row in rows if str(row["dataset_id"]) in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_joint.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seed", type=int, choices=(42, 123, 7), default=42)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--epsilon-rounds",
        type=int,
        default=None,
        help="engineering-only calibration override; omit to use the frozen 16-round protocol",
    )
    parser.add_argument("--execute", action="store_true", help="required to run preflight computation")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    if args.cpu and args.gpu is not None:
        raise ValueError("--cpu and --gpu are mutually exclusive")
    if args.execute and not args.cpu and args.gpu is None:
        raise ValueError("execution requires --cpu or an explicit --gpu")
    if args.gpu is not None and args.gpu not in ALLOWED_GPUS:
        raise ValueError(f"physical GPU {args.gpu} is forbidden; allowed={sorted(ALLOWED_GPUS)}")
    if args.epsilon_rounds is not None and args.epsilon_rounds < 4:
        raise ValueError("epsilon-rounds must be at least four when overridden")
    if args.num_workers <= 0 or not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    config = _make_preflight_config(load_config(args.config), args.epsilon_rounds)
    records = _select_records(manifest, args.datasets)
    records = [row for index, row in enumerate(records) if index % args.num_workers == args.worker_id]
    gpu = args.gpu if not args.cpu else None
    # Map one logical CUDA device to the requested physical card before the
    # first CUDA query in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    device = torch.device("cpu" if gpu is None else "cuda:0")
    header = {
        "manifest_id": manifest["manifest_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
        "execute": bool(args.execute),
        "worker_id": int(args.worker_id),
        "num_workers": int(args.num_workers),
        "physical_gpu": gpu,
        "datasets_for_worker": [str(row["dataset_id"]) for row in records],
        "seed": int(args.seed),
        "epsilon_rounds": int(config.constraint.epsilon_rounds),
        "formal_epsilon_rounds": 16,
        "formal_training_started": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    print(json.dumps(header, ensure_ascii=True), flush=True)
    if not args.execute:
        for row in records:
            print(json.dumps({"dataset_id": row["dataset_id"], "status": "planned"}, ensure_ascii=True))
        return 0
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record['dataset_id']}", flush=True)
        os.environ.update(
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        rows.append(
            _run_record(
                record,
                config=config,
                seed=args.seed,
                device=device,
                output_root=output_root,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    completed = sum(row.get("status") == "completed" for row in rows)
    aggregate = {
        **header,
        "status": "completed" if completed == len(rows) else "incomplete_compute",
        "completed": int(completed),
        "incomplete": int(len(rows) - completed),
        "rows": rows,
        "updated_at": time.time(),
    }
    _write_json(output_root / f"preflight_worker_{args.worker_id}.json", aggregate)
    return 0 if completed == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
