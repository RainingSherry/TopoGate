"""CLI for V26 diagnostics, preflight, single cells, and protocol freezing."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import corruption, model, protocol
from .data import SparseDataset, load_dataset, value_only_profiles


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _device(physical_gpu: int | None, cpu: bool) -> tuple[Any, int | None]:
    if cpu:
        import torch
        return torch.device("cpu"), None
    if physical_gpu is None:
        raise ValueError("V26 GPU runs require --gpu")
    if physical_gpu not in protocol.LEGAL_GPU_POOL or physical_gpu in protocol.FORBIDDEN_GPU_IDS:
        raise ValueError(f"forbidden V26 GPU {physical_gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", str(physical_gpu)}:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain only the requested physical GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for V26 GPU run")
    torch.cuda.set_device(0)
    return torch.device("cuda:0"), int(physical_gpu)


def _metrics(embedding: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    prediction = KMeans(n_clusters=int(np.unique(y).size), random_state=seed, n_init=20).fit_predict(embedding)
    contingency = np.zeros((int(prediction.max()) + 1, int(y.max()) + 1), dtype=np.int64)
    np.add.at(contingency, (prediction, y), 1)
    rows, cols = linear_sum_assignment(contingency.max() - contingency)
    acc = float(contingency[rows, cols].sum() / y.size)
    return {
        "ARI": float(adjusted_rand_score(y, prediction)),
        "NMI": float(normalized_mutual_info_score(y, prediction)),
        "ACC": acc,
        "n_clusters": int(np.unique(y).size),
    }


def _implementation_provenance() -> dict[str, Any]:
    import sklearn
    import scipy

    return {
        "revision": protocol.IMPLEMENTATION_REVISION,
        "source_sha256": protocol.implementation_sha256(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def run_diagnostics(identifier: str, output_root: Path) -> dict[str, Any]:
    dataset = load_dataset(identifier, hash_source=True)
    d_eff = min(protocol.SVD_COMPONENTS, dataset.n_samples - 1, dataset.n_features - 1)
    if d_eff < 1:
        raise ValueError("diagnostic SVD has no valid component")
    support = dataset.x.copy().tocsr()
    support.data = np.ones_like(support.data, dtype=np.float32)
    value_embedding = value_only_profiles(dataset.x, protocol.VALUE_PROFILE_QUANTILES)
    support_embedding = TruncatedSVD(n_components=d_eff, random_state=0, n_iter=5).fit_transform(support)
    active = np.diff(dataset.x.indptr)
    pair_budget = np.minimum.reduce([
        np.ceil(protocol.CORRUPTION_RATE * active).astype(np.int64),
        active // 2,
        dataset.n_features - active,
    ])
    result = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "diagnostics",
        "dataset": identifier,
        "source": dataset.source_metadata,
        "implementation": _implementation_provenance(),
        "diagnostic_representation": {
            "support_only": "binary sparse coordinate support -> TruncatedSVD",
            "value_only": "nonzero-value quantile profile with no coordinates or active-count padding",
            "value_profile_quantiles": protocol.VALUE_PROFILE_QUANTILES,
        },
        "support_only": _metrics(support_embedding, dataset.y, 42),
        "value_only": _metrics(value_embedding, dataset.y, 42),
        "support_feasibility": {
            "mean_active": float(np.mean(active)),
            "mean_pair_budget": float(np.mean(pair_budget)),
            "rows_with_pair_budget": int(np.count_nonzero(pair_budget > 0)),
            "fraction_rows_with_pair_budget": float(np.mean(pair_budget > 0)),
        },
        "labels_used_after_transform_for_metrics": True,
        "status": "completed_valid",
    }
    _write_json(output_root / "diagnostics" / identifier / "summary.json", result)
    return result


def preflight(identifier: str, physical_gpu: int, output_root: Path) -> dict[str, Any]:
    dataset = load_dataset(identifier)
    device, gpu = _device(physical_gpu, False)
    batch_size, profile = model.select_batch_size(dataset, device, 42)
    result = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "preflight",
        "dataset": identifier,
        "implementation": _implementation_provenance(),
        "gpu_physical_id": gpu,
        "batch_size": batch_size,
        "profile": profile,
        "reservation_mib": float(max(profile["peak_reserved_mib"] * 1.15, profile["peak_reserved_mib"] + 1024.0)),
        "status": "completed_valid",
    }
    _write_json(output_root / "preflight" / f"{identifier}.json", result)
    return result


def run_cell(identifier: str, arm: str, seed: int, physical_gpu: int, output_root: Path, epochs: int) -> dict[str, Any]:
    dataset = load_dataset(identifier)
    run_dir = output_root / "runs" / identifier / arm / f"seed{seed}"
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed_valid" and existing.get("protocol_id") == protocol.PROTOCOL_ID:
            return {**existing, "reused": True}
    device, gpu = _device(physical_gpu, False)
    preflight_path = output_root / "preflight" / f"{identifier}.json"
    if preflight_path.exists():
        preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
        batch_size = int(preflight_record["batch_size"])
    else:
        batch_size, _ = model.select_batch_size(dataset, device, seed)
    oracle = corruption.build_simple_label_oracle(dataset.x, dataset.y) if arm == "O_LABEL_ORACLE" else None
    fit = model.fit(dataset, arm=arm, seed=seed, device=device, epochs=epochs, batch_size=batch_size, oracle=oracle)
    metrics = _metrics(fit.embedding, dataset.y, seed)
    summary = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": "formal_matrix",
        "dataset": identifier,
        "dataset_display_name": dataset.spec.display_name,
        "domain": dataset.spec.domain,
        "arm": arm,
        "seed": int(seed),
        "epochs": int(epochs),
        "gpu_physical_id": gpu,
        "source": dataset.source_metadata,
        "implementation": _implementation_provenance(),
        "batch_size": fit.batch_size,
        "metrics": metrics,
        "loss_history": fit.history,
        "mask_audit": fit.mask_audit,
        "peak_allocated_mib": fit.peak_allocated_mib,
        "peak_reserved_mib": fit.peak_reserved_mib,
        "label_firewall": {
            "model_fit_receives_y": False,
            "oracle_mask_uses_y": bool(arm == "O_LABEL_ORACLE"),
            "labels_used_after_fit_for_metrics": True,
            "K_source": "benchmark_oracle_from_y_outer_readout",
        },
        "oracle": oracle.metadata if oracle is not None else None,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed_valid",
    }
    _write_json(summary_path, summary)
    return summary


def freeze(output_root: Path) -> dict[str, Any]:
    protocol.validate_protocol()
    rows = []
    for spec in protocol.DATASETS:
        dataset = load_dataset(spec.identifier, hash_source=True)
        rows.append({"dataset": spec.identifier, **dataset.source_metadata, "source_type": spec.source_type, "label_field": spec.label_field})
    result = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "resolved_config": protocol.resolved_config(),
        "implementation": _implementation_provenance(),
        "datasets": rows,
        "status": "completed_valid",
    }
    _write_json(output_root / "FREEZE" / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="V26 Support Oracle Study v1")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-cell", action="store_true")
    parser.add_argument("--dataset", choices=tuple(protocol.DATASET_BY_ID))
    parser.add_argument("--arm", choices=protocol.ARMS)
    parser.add_argument("--seed", type=int, choices=protocol.SEEDS)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--epochs", type=int, default=protocol.EPOCHS)
    parser.add_argument("--output-root", type=Path, default=protocol.RESULT_ROOT)
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze(args.output_root), sort_keys=True))
        return 0
    if args.diagnostics:
        if args.dataset is None:
            parser.error("--diagnostics requires --dataset")
        print(json.dumps(run_diagnostics(args.dataset, args.output_root), sort_keys=True))
        return 0
    if args.preflight:
        if args.dataset is None or args.gpu is None:
            parser.error("--preflight requires --dataset --gpu")
        print(json.dumps(preflight(args.dataset, args.gpu, args.output_root), sort_keys=True))
        return 0
    if args.run_cell:
        if args.dataset is None or args.arm is None or args.seed is None or args.gpu is None:
            parser.error("--run-cell requires --dataset --arm --seed --gpu")
        print(json.dumps(run_cell(args.dataset, args.arm, args.seed, args.gpu, args.output_root, args.epochs), sort_keys=True))
        return 0
    parser.error("choose --freeze, --diagnostics, --preflight, or --run-cell")


if __name__ == "__main__":
    raise SystemExit(main())
