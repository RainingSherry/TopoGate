"""Run one or all frozen MAIN cells and the SVD32 baseline.

The command is intentionally a single-cell runner so ``overnight.py`` can
assign one legal physical GPU per process without hidden device selection.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import masking, metrics, model, protocol, provenance, raw_adapter


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, encoding="utf-8") as handle:
        tmp = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _device(use_cpu: bool = False) -> tuple[Any, int | None]:
    import torch

    if use_cpu:
        return torch.device("cpu"), None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    values = [v.strip() for v in visible.split(",") if v.strip()]
    if len(values) != 1 or not values[0].isdigit():
        raise RuntimeError("formal jobs require CUDA_VISIBLE_DEVICES to contain exactly one legal physical GPU")
    physical = int(values[0])
    if physical not in protocol.LEGAL_GPU_POOL or physical in protocol.FORBIDDEN_GPU_IDS or not torch.cuda.is_available():
        raise RuntimeError(f"illegal/unavailable formal GPU selection: CUDA_VISIBLE_DEVICES={visible!r}")
    torch.cuda.set_device(0)
    return torch.device("cuda:0"), physical


def _adapter_hash(data: raw_adapter.RawDataset) -> str:
    payload = json.dumps(data.manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return provenance.sha256_bytes(payload)


def _assert_frozen_code(code_hash: str) -> None:
    manifest_path = protocol.FREEZE_ROOT / "freeze_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("formal run requires a completed FREEZE manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("code_sha256") != code_hash:
        raise RuntimeError("protocol mismatch: live code hash differs from FREEZE manifest")


def _batch_preflight(data: raw_adapter.RawDataset, device: Any, seed: int, output_root: Path) -> tuple[int, dict[str, Any]]:
    path = output_root / data.dataset / "batch_preflight.json"
    adapter_hash = _adapter_hash(data)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("adapter_hash") != adapter_hash:
            raise ValueError(f"batch preflight hash drift for {data.dataset}")
        return int(record["selected_batch_size"]), record
    selected, record = model.choose_batch_size(data.x0, data.x0.shape[1], device, seed=seed)
    record = {**record, "dataset": data.dataset, "adapter_hash": adapter_hash, "selected_batch_size": selected, "labels_loaded": False}
    write_json_atomic(path, record)
    return selected, record


def _existing_valid(path: Path, expected: dict[str, Any]) -> bool:
    summary = path / "summary.json"
    if not summary.exists():
        return False
    try:
        value = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return False
    keys = ("project_id", "protocol_id", "dataset", "arm", "seed", "source_sha256", "adapter_hash", "scale_hash", "code_sha256", "status", "labels_loaded_during_fit")
    return value.get("status") == "completed_valid" and all(value.get(k) == expected.get(k) for k in keys)


def _run_one(dataset: str, arm: str, seed: int, *, output_root: Path, use_cpu: bool = False, epochs: int | None = None) -> dict[str, Any]:
    protocol.validate_contract()
    if dataset not in protocol.DATASETS or arm not in protocol.ARMS or int(seed) not in protocol.SEEDS:
        raise ValueError("dataset, arm, or seed is outside the frozen MAIN matrix")
    device, physical_gpu = _device(use_cpu)
    data = raw_adapter.load_dataset(dataset)
    adapter_hash = _adapter_hash(data)
    code_hash = provenance.code_sha256()
    if not use_cpu:
        _assert_frozen_code(code_hash)
    run_dir = output_root / dataset / arm / f"seed{int(seed)}"
    run_status = "engineering_smoke" if use_cpu else "completed_valid"
    base_expected = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "dataset": dataset,
        "arm": arm,
        "seed": int(seed),
        "source_sha256": data.manifest["source_sha256"],
        "adapter_hash": adapter_hash,
        "scale_hash": data.manifest["scale_hash"],
        "code_sha256": code_hash,
        "status": run_status,
        "labels_loaded_during_fit": False,
    }
    if _existing_valid(run_dir, base_expected):
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    batch_size, preflight = _batch_preflight(data, device, int(seed), output_root)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    start_clock = time.monotonic()
    try:
        fit = model.fit_autoencoder(data.x0, data.active, arm=arm, seed=int(seed), device=device, epochs=epochs, batch_size=batch_size)
        # The fit path above only received x0/support.  Labels are loaded at
        # this boundary, after the encoder is complete, for benchmark readout.
        labels, label_meta = raw_adapter.load_labels_after_fit(dataset)
        postfit = metrics.evaluate_after_fit(fit.embedding, labels, int(seed))
        ended = dt.datetime.now(dt.timezone.utc).isoformat()
        summary = {
            **base_expected,
            "domain_role": protocol.ROLE_BY_DATASET[dataset],
            "start_time": started,
            "end_time": ended,
            "wall_seconds": float(time.monotonic() - start_clock),
            "gpu_physical_id": physical_gpu,
            "device": str(device),
            "batch_size": int(fit.batch_size),
            "batch_preflight": preflight,
            "model_init_hash": fit.model_init_hash,
            "model_final_hash": fit.model_final_hash,
            "batch_schedule_hash": fit.batch_schedule_hash,
            "mask_schedule_hash": fit.mask_schedule_hash,
            "peak_gpu_memory_bytes": int(fit.peak_gpu_memory_bytes),
            "labels_loaded_after_fit": True,
            "labels_sha256": label_meta["labels_sha256"],
            "labels_unique": label_meta["labels_unique"],
            "training_history": fit.history,
            "metrics": postfit,
            "adapter_manifest": data.manifest,
            "audit_ok": not use_cpu,
            "labels_used_during_fit": False,
        }
        write_json_atomic(run_dir / "summary.json", summary)
        write_json_atomic(run_dir / "audit.json", {
            "audit_ok": True,
            "project_id": protocol.PROJECT_ID,
            "protocol_id": protocol.PROTOCOL_ID,
            "dataset": dataset,
            "arm": arm,
            "seed": int(seed),
            "labels_loaded_during_fit": False,
            "labels_loaded_after_fit": True,
            "zero_pattern_preserved": data.manifest["zero_pattern_preserved"],
            "gpu_physical_id": physical_gpu,
            "gpu_legal": physical_gpu is None or physical_gpu in protocol.LEGAL_GPU_POOL,
            "paired_hash_fields_present": True,
            "mask_budget_audits": [row for row in fit.history if arm != "CLEAN_AE"],
        })
        write_json_atomic(run_dir / "resolved_config.json", {**protocol.resolved_config(), "dataset": dataset, "arm": arm, "seed": int(seed), "batch_size": int(batch_size)})
        return summary
    except Exception as exc:
        write_json_atomic(run_dir / "summary.json", {**base_expected, "status": "incomplete_compute", "error_type": type(exc).__name__, "error": str(exc), "start_time": started, "end_time": dt.datetime.now(dt.timezone.utc).isoformat(), "gpu_physical_id": physical_gpu})
        raise


def run_svd(dataset: str, seed: int, *, output_root: Path, use_cpu: bool = True) -> dict[str, Any]:
    """Fit the frozen SVD32 baseline without loading labels until transform ends."""
    if dataset not in protocol.DATASETS or int(seed) not in protocol.SEEDS:
        raise ValueError("dataset/seed outside frozen panel")
    data = raw_adapter.load_dataset(dataset)
    _assert_frozen_code(provenance.code_sha256())
    out = output_root / "SVD32" / dataset / f"seed{int(seed)}"
    expected = {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.PROTOCOL_ID, "dataset": dataset, "arm": "SVD32", "seed": int(seed), "source_sha256": data.manifest["source_sha256"], "scale_hash": data.manifest["scale_hash"], "code_sha256": provenance.code_sha256(), "status": "completed_valid", "labels_loaded_during_fit": False}
    if (out / "summary.json").exists():
        existing = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if all(existing.get(k) == v for k, v in expected.items()) and existing.get("svd", {}).get("input_representation") == "csr_zero_preserving_view":
            return existing
    embedding, svd_meta = metrics.svd_embedding(data.x0, int(seed), protocol.SVD_COMPONENTS)
    labels, label_meta = raw_adapter.load_labels_after_fit(dataset)
    result = {
        **expected,
        "domain_role": protocol.ROLE_BY_DATASET[dataset],
        "metrics": metrics.evaluate_after_fit(embedding, labels, int(seed)),
        "svd": svd_meta,
        "labels_loaded_after_fit": True,
        "labels_sha256": label_meta["labels_sha256"],
        "labels_unique": label_meta["labels_unique"],
        "adapter_hash": _adapter_hash(data),
        "audit_ok": True,
        "labels_used_during_fit": False,
    }
    write_json_atomic(out / "summary.json", result)
    write_json_atomic(out / "audit.json", {"audit_ok": True, "labels_loaded_during_fit": False, "labels_loaded_after_fit": True, "source_sha256": data.manifest["source_sha256"]})
    write_json_atomic(out / "resolved_config.json", {**protocol.resolved_config(), "dataset": dataset, "arm": "SVD32", "seed": int(seed)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=protocol.DATASETS)
    parser.add_argument("--arm", choices=protocol.ARMS)
    parser.add_argument("--seed", type=int, choices=protocol.SEEDS)
    parser.add_argument("--all", action="store_true", help="run the sequential MAIN matrix")
    parser.add_argument("--svd", action="store_true", help="run the SVD32 baseline for one dataset/seed")
    parser.add_argument("--output-root", type=Path, default=protocol.MAIN_ROOT)
    parser.add_argument("--cpu", action="store_true", help="engineering smoke only; never a formal GPU claim")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    protocol.validate_contract()
    if args.svd:
        if args.dataset is None or args.seed is None:
            parser.error("--svd requires --dataset and --seed")
        result = run_svd(args.dataset, args.seed, output_root=args.output_root)
        print(json.dumps({"status": result.get("status"), "dataset": args.dataset, "seed": args.seed}, sort_keys=True))
        return 0
    if args.all:
        for seed in protocol.SEEDS:
            for dataset in protocol.DATASETS:
                for arm in protocol.ARMS:
                    _run_one(dataset, arm, seed, output_root=args.output_root, use_cpu=args.cpu, epochs=args.epochs)
        print(json.dumps({"status": "completed_valid", "runs": len(protocol.DATASETS) * len(protocol.ARMS) * len(protocol.SEEDS)}))
        return 0
    if args.dataset is None or args.arm is None or args.seed is None:
        parser.error("one cell requires --dataset --arm --seed; or use --all")
    result = _run_one(args.dataset, args.arm, args.seed, output_root=args.output_root, use_cpu=args.cpu, epochs=args.epochs)
    print(json.dumps({"status": result.get("status"), "dataset": args.dataset, "arm": args.arm, "seed": args.seed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
