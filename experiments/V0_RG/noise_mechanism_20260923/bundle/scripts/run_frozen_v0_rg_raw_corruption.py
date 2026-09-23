"""Fixed-weight V0_RG input-corruption evaluator.

This program intentionally never calls ``fit_predict``.  It verifies a saved
V0_RG encoder, frozen preprocessor, and clean KMeans centroids before encoding
one corrupted input matrix.  Labels are loaded only after prediction to report
post-hoc clustering metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any


import h5py
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from methods.TopoGate.V0_RG.config import V0_RGConfig
from methods.TopoGate.V0_RG.model import WeightedAutoEncoder
from methods.TopoGate.V0_RG.run import clustering_metrics
from methods.TopoGate.V0_RG.trainer import _extract_embedding
from methods.TopoGate.V0_RG.tuning import SplitPreprocessor


ALLOWED_PHYSICAL_GPUS = {1, 4, 5}
CONFIG_KEYS = {field.name for field in fields(V0_RGConfig)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> tuple[np.ndarray | sp.spmatrix, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        if {"data", "indices", "indptr", "shape"}.issubset(source.files):
            shape = tuple(int(value) for value in np.asarray(source["shape"]).reshape(-1))
            matrix: np.ndarray | sp.spmatrix = sp.csr_matrix(
                (np.asarray(source["data"]), np.asarray(source["indices"], dtype=np.int64), np.asarray(source["indptr"], dtype=np.int64)),
                shape=shape,
            )
        else:
            matrix = next((np.asarray(source[key]) for key in ("X", "x", "features", "data") if key in source.files), None)
        labels = next((np.asarray(source[key]).reshape(-1) for key in ("y", "labels", "label") if key in source.files), None)
    if matrix is None or labels is None:
        raise ValueError(f"{path}: expected a feature matrix and labels")
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    encoded = LabelEncoder().fit_transform(labels.astype(str)).astype(np.int64)
    if matrix.shape[0] != encoded.shape[0]:
        raise ValueError(f"{path}: feature/label row mismatch")
    return matrix, encoded


def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as source:
        matrix = np.asarray(source["X"][:], dtype=np.float32)
        labels = np.asarray(source["Y"][:]).reshape(-1)
    encoded = LabelEncoder().fit_transform(labels.astype(str)).astype(np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != encoded.shape[0]:
        raise ValueError(f"{path}: invalid biological matrix or labels")
    return matrix, encoded


def load_data(path: Path) -> tuple[np.ndarray | sp.spmatrix, np.ndarray]:
    if path.suffix == ".npz":
        return load_npz(path)
    if path.suffix == ".h5":
        return load_h5(path)
    raise ValueError(f"unsupported dataset format: {path}")


def load_frozen_preprocessor(path: Path, input_kind: str, target_sum: float) -> dict[str, np.ndarray | float | str]:
    with np.load(path, allow_pickle=False) as source:
        required = {"features", "mean", "scale"}
        if not required.issubset(source.files):
            raise ValueError(f"{path}: missing {sorted(required - set(source.files))}")
        features = np.asarray(source["features"], dtype=np.int64)
        mean = np.asarray(source["mean"], dtype=np.float64)
        scale = np.asarray(source["scale"], dtype=np.float64)
    if features.ndim != 1 or mean.ndim != 1 or scale.ndim != 1 or len(features) != len(mean) or len(mean) != len(scale):
        raise ValueError(f"{path}: invalid frozen preprocessor arrays")
    if np.any(scale <= 0) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError(f"{path}: non-finite or non-positive scaler values")
    return {"features": features, "mean": mean, "scale": scale, "input_kind": input_kind, "target_sum": float(target_sum)}


def frozen_transform(matrix: np.ndarray | sp.spmatrix, state: dict[str, np.ndarray | float | str]) -> np.ndarray:
    """Apply exactly the saved SplitPreprocessor map without fitting anything."""
    input_kind = str(state["input_kind"])
    features = np.asarray(state["features"], dtype=np.int64)
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    target_sum = float(state["target_sum"])
    width = matrix.shape[1]
    if features.size == 0 or np.any(features < 0) or np.any(features >= width):
        raise ValueError("frozen selected features are incompatible with input width")
    if sp.issparse(matrix):
        normal = matrix.tocsr(copy=True).astype(np.float32)
        normal.sum_duplicates()
        normal.eliminate_zeros()
        if not np.isfinite(normal.data).all():
            raise ValueError("input contains non-finite values")
        if input_kind != "general" and np.any(normal.data < 0):
            raise ValueError("biological input contains negative values")
        if input_kind == "raw_count":
            if not np.allclose(normal.data, np.rint(normal.data), atol=1e-5, rtol=0):
                raise ValueError("raw_count input must remain integer-valued before normalization")
            sums = np.asarray(normal.sum(axis=1)).reshape(-1)
            factors = np.divide(target_sum, sums, out=np.zeros_like(sums), where=sums > 0)
            normal = normal.multiply(factors[:, None]).tocsr()
            normal.data = np.log1p(normal.data)
        selected = normal[:, features].toarray().astype(np.float64, copy=False)
    else:
        raw = np.asarray(matrix, dtype=np.float32)
        if raw.ndim != 2 or not np.isfinite(raw).all():
            raise ValueError("input must be a finite two-dimensional matrix")
        if input_kind != "general" and np.any(raw < 0):
            raise ValueError("biological input contains negative values")
        if input_kind == "raw_count":
            if not np.allclose(raw, np.rint(raw), atol=1e-5, rtol=0):
                raise ValueError("raw_count input must remain integer-valued before normalization")
            sums = raw.sum(axis=1, dtype=np.float64)
            factors = np.divide(target_sum, sums, out=np.zeros_like(sums), where=sums > 0)
            selected = np.log1p(raw[:, features].astype(np.float64, copy=False) * factors[:, None])
        else:
            selected = raw[:, features].astype(np.float64, copy=False)
    transformed = ((selected - mean) / scale).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("frozen preprocessing produced non-finite features")
    return np.ascontiguousarray(transformed)


def validate_preprocessor(clean_raw: np.ndarray | sp.spmatrix, state: dict[str, np.ndarray | float | str], expected: np.ndarray) -> dict[str, float]:
    """Audit the saved arrays against the code used to create them; no model is trained."""
    preprocessor = SplitPreprocessor(str(state["input_kind"]), len(np.asarray(state["features"])), float(state["target_sum"])).fit(clean_raw)
    feature_equal = bool(np.array_equal(preprocessor.features, np.asarray(state["features"])))
    mean_error = float(np.max(np.abs(preprocessor.scaler.mean_ - np.asarray(state["mean"]))))
    scale_error = float(np.max(np.abs(preprocessor.scaler.scale_ - np.asarray(state["scale"]))))
    reconstructed = frozen_transform(clean_raw, state)
    transform_error = float(np.max(np.abs(reconstructed - expected)))
    if not feature_equal or mean_error > 1e-6 or scale_error > 1e-6 or transform_error > 2e-5:
        raise ValueError(f"frozen preprocessor audit failed: features={feature_equal}, mean={mean_error}, scale={scale_error}, transform={transform_error}")
    return {"mean_abs_max": mean_error, "scale_abs_max": scale_error, "transform_abs_max": transform_error}


def load_model(source: Path, config: V0_RGConfig, input_width: int, device: torch.device) -> WeightedAutoEncoder:
    state_path = source / "model.pt"
    payload = torch.load(state_path, map_location=device, weights_only=True)
    model = WeightedAutoEncoder(
        num_genes=input_width,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        masked_data_weight=config.masked_data_weight,
        mask_loss_weight=config.mask_loss_weight,
    ).to(device)
    model.load_state_dict(payload, strict=True)
    model.eval()
    return model


def nearest_centroid(embedding: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # Equivalent to KMeans.predict for Euclidean distance, without refitting centroids.
    distances = ((embedding[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(np.int64)


def corrupt_model_input(clean: np.ndarray, kind: str, severity: float, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    if kind == "clean":
        return clean.copy(), {"mean_abs_displacement": 0.0, "fraction_changed": 0.0}
    if kind == "gaussian":
        if severity <= 0:
            raise ValueError("Gaussian severity must be positive")
        reference_scale = clean.std(axis=0, dtype=np.float64)
        noisy = clean + (float(severity) * reference_scale[None, :] * rng.standard_normal(clean.shape)).astype(np.float32)
    elif kind == "replacement":
        if not 0.0 < severity < 1.0:
            raise ValueError("replacement severity must be in (0,1)")
        mask = rng.random(clean.shape) < float(severity)
        donors = rng.integers(0, clean.shape[0], size=clean.shape)
        columns = np.broadcast_to(np.arange(clean.shape[1]), clean.shape)
        noisy = clean.copy()
        noisy[mask] = clean[donors[mask], columns[mask]]
    else:
        raise ValueError(f"unsupported model-space corruption: {kind}")
    changed = noisy != clean
    return np.ascontiguousarray(noisy, dtype=np.float32), {"mean_abs_displacement": float(np.mean(np.abs(noisy - clean))), "fraction_changed": float(np.mean(changed))}


def corrupt_raw_input(clean: np.ndarray | sp.spmatrix, kind: str, severity: float, seed: int, input_kind: str):
    if input_kind != "general":
        raise ValueError("raw-input Gaussian/Replacement panel currently supports general NPZ inputs only")
    dense = clean.toarray() if sp.issparse(clean) else np.asarray(clean)
    return corrupt_model_input(np.asarray(dense, dtype=np.float32), kind, severity, seed)


def binomial_thinning(matrix: np.ndarray | sp.spmatrix, retain_probability: float, seed: int) -> np.ndarray | sp.spmatrix:
    if not 0.0 < retain_probability < 1.0:
        raise ValueError("thinning retain probability must be in (0,1)")
    rng = np.random.default_rng(seed)
    if sp.issparse(matrix):
        output = matrix.tocsr(copy=True)
        if not np.allclose(output.data, np.rint(output.data), atol=1e-5, rtol=0):
            raise ValueError("sparse raw count matrix is not integer-valued")
        output.data = rng.binomial(np.rint(output.data).astype(np.int64), retain_probability).astype(np.float32)
        output.eliminate_zeros()
        return output
    raw = np.asarray(matrix)
    if not np.allclose(raw, np.rint(raw), atol=1e-5, rtol=0):
        raise ValueError("dense raw count matrix is not integer-valued")
    output = np.empty(raw.shape, dtype=np.float32)
    for start in range(0, raw.shape[0], 256):
        stop = min(start + 256, raw.shape[0])
        output[start:stop] = rng.binomial(np.rint(raw[start:stop]).astype(np.int64), retain_probability)
    return output


def resolve_device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if not visible:
        raise ValueError("set CUDA_VISIBLE_DEVICES to one permitted physical GPU before GPU inference")
    physical = {int(item) for item in visible}
    if not physical.issubset(ALLOWED_PHYSICAL_GPUS):
        raise ValueError(f"allowed physical GPUs are {sorted(ALLOWED_PHYSICAL_GPUS)}, got {sorted(physical)}")
    if raw != "cuda:0":
        raise ValueError("GPU inference must use cuda:0 after one-GPU visibility isolation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after visibility isolation")
    return torch.device(raw)


def cohort_entry(path: Path, dataset: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["entries"]:
        if row["dataset"] == dataset:
            return row, sha256(path)
    raise ValueError(f"{dataset} is not present in {path}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    entry, cohort_hash = cohort_entry(Path(args.cohort), args.dataset)
    checkpoint_root = entry.get("checkpoint_root")
    if not checkpoint_root:
        raise ValueError(f"{args.dataset}: no verified frozen V0_RG checkpoint is registered")
    full_root = Path(checkpoint_root)
    checkpoint = full_root.parents[1] / args.variant / "current" / f"seed_{args.model_seed}"
    required = ["model.pt", "cluster_centers.npy", "preprocessor.npz", "embedding_all.npy", "predictions_all.npy", "test_indices.npy", "labels_true_test.npy", "final_summary.json"]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{checkpoint}: missing {missing}")
    output = Path(args.save_root) / "V0_RG_raw_input" / args.dataset / args.variant / f"model_seed_{args.model_seed}" / args.corruption / f"severity_{args.severity:g}" / f"noise_seed_{args.noise_seed}" / "result.json"
    if output.is_file():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior.get("status") == "completed":
            return {"status": "skipped_existing", "result": str(output)}
    started = time.time()
    clean_raw, labels = load_data(Path(entry["data_path"]))
    summary = json.loads((checkpoint / "final_summary.json").read_text(encoding="utf-8"))
    config_payload = dict(summary.get("config", {}))
    config = V0_RGConfig(**{key: config_payload[key] for key in CONFIG_KEYS if key in config_payload})
    state = load_frozen_preprocessor(checkpoint / "preprocessor.npz", str(entry["input_kind"]), config.target_sum)
    clean_input = frozen_transform(clean_raw, state)
    if args.skip_preprocessor_audit:
        preprocessor_audit: dict[str, Any] = {
            "performed_this_run": False,
            "reference": args.audit_reference,
        }
    else:
        preprocessor_audit = validate_preprocessor(clean_raw, state, clean_input)
        preprocessor_audit["performed_this_run"] = True
    device = resolve_device(args.device)
    model = load_model(checkpoint, config, clean_input.shape[1], device)
    saved_embedding = np.load(checkpoint / "embedding_all.npy")
    saved_pred = np.asarray(np.load(checkpoint / "predictions_all.npy"), dtype=np.int64)
    if args.skip_clean_replay:
        clean_embedding = saved_embedding
        clean_pred = saved_pred
        embedding_error: float | None = None
        prediction_ari: float | None = None
    else:
        clean_embedding = _extract_embedding(model, clean_input, config.batch_size, device)
        embedding_error = float(np.max(np.abs(clean_embedding - saved_embedding)))
        if clean_embedding.shape != saved_embedding.shape or not np.allclose(clean_embedding, saved_embedding, atol=2e-5, rtol=2e-5):
            raise ValueError(f"{args.dataset} seed {args.model_seed}: frozen clean embedding mismatch ({embedding_error})")
    centers = np.asarray(np.load(checkpoint / "cluster_centers.npy"), dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != clean_embedding.shape[1]:
        raise ValueError("saved cluster centers are incompatible with frozen embedding")
    if not args.skip_clean_replay:
        clean_pred = nearest_centroid(clean_embedding, centers)
        prediction_ari = float(adjusted_rand_score(saved_pred, clean_pred))
        if prediction_ari != 1.0:
            raise ValueError(f"frozen centroid prediction differs from saved prediction (ARI={prediction_ari})")
    test_indices = np.asarray(np.load(checkpoint / "test_indices.npy"), dtype=np.int64)
    saved_test_labels = np.asarray(np.load(checkpoint / "labels_true_test.npy"), dtype=np.int64)
    if not np.array_equal(labels[test_indices], saved_test_labels):
        raise ValueError("current source labels do not match frozen checkpoint test labels")
    clean_metrics = clustering_metrics(saved_test_labels, clean_pred[test_indices])
    if args.corruption == "thinning":
        raise ValueError("thinning belongs to a separate raw-count protocol")
    noisy_raw, noise_summary = corrupt_raw_input(clean_raw, args.corruption, float(args.severity), int(args.noise_seed), str(entry["input_kind"]))
    noisy_input = frozen_transform(noisy_raw, state)
    noisy_embedding = _extract_embedding(model, noisy_input, config.batch_size, device)
    noisy_pred = nearest_centroid(noisy_embedding, centers)
    noisy_metrics = clustering_metrics(saved_test_labels, noisy_pred[test_indices])
    result = {
        "status": "completed",
        "protocol": "v0_rg_frozen_encoder_clean_centroids_raw_input_corruption_v1",
        "dataset": args.dataset,
        "variant": args.variant,
        "input_kind": entry["input_kind"],
        "data_path": entry["data_path"],
        "model_seed": int(args.model_seed),
        "noise_seed": int(args.noise_seed),
        "corruption": args.corruption,
        "severity": float(args.severity),
        "fitting_performed": False,
        "readout": "saved clean KMeans centroids, nearest-centroid prediction",
        "labels_used_only_after_prediction": True,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256(checkpoint / "model.pt"),
        "cohort_sha256": cohort_hash,
        "preprocessor_audit": preprocessor_audit,
        "clean_embedding_abs_max_error": embedding_error,
        "saved_prediction_ari": prediction_ari,
        "clean_replay_verified_this_run": not args.skip_clean_replay,
        "audit_reference": args.audit_reference,
        "clean_metrics": clean_metrics,
        "noisy_metrics": noisy_metrics,
        "delta_clean_minus_noisy": {key: float(clean_metrics[key] - noisy_metrics[key]) for key in ("ari", "nmi", "acc")},
        "noise_summary": noise_summary,
        "source_data_sha256": sha256(Path(entry["data_path"])),
        "prediction_scope": "all_rows_saved; metrics_on_saved_test_indices",
        "test_indices_sha256": sha256(checkpoint / "test_indices.npy"),
        "inference_note": "V0_RG graph and gate are training-time mechanisms; frozen inference encodes each corrupted sample without dynamic graph mixing.",
        "wall_seconds": time.time() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output.parent / "predictions.npy", noisy_pred)
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--model-seed", type=int, required=True, choices=(42, 123, 7))
    parser.add_argument("--corruption", required=True, choices=("clean", "gaussian", "replacement", "thinning"))
    parser.add_argument("--severity", type=float, required=True)
    parser.add_argument("--noise-seed", type=int, required=True)
    parser.add_argument("--save-root", default="result/v0rg_frozen_corruption_top5")
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
    parser.add_argument("--skip-preprocessor-audit", action="store_true")
    parser.add_argument("--skip-clean-replay", action="store_true")
    parser.add_argument("--audit-reference")
    args = parser.parse_args()
    try:
        result = evaluate(args)
    except Exception as error:
        payload = {"status": "failed", "dataset": args.dataset, "model_seed": args.model_seed, "corruption": args.corruption, "severity": args.severity, "noise_seed": args.noise_seed, "error_type": type(error).__name__, "error": str(error)}
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
