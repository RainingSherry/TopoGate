"""C2 formal GPU matrix for the frozen static corruption library.

The runner deliberately keeps the experiment small and auditable.  Every arm
uses the same reconstruction probe, optimizer and budget; only the frozen
static corruption principle changes.  Labels are loaded after fitting for the
benchmark-known-K outer readout and metrics only.

The raw score arrays used by P4/P5 are local score artifacts.  They are needed
to reproduce the matrix but are excluded from the compact publication bundle.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import protocol
from .corruption_library import compact_audit, corrupt_matrix, geometry_importance


PROJECT_ROOT = protocol.PROJECT_ROOT
H0_ROOT = protocol.H0_ROOT
LABEL_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S1_oracle_v2"
DEFAULT_OUTPUT = protocol.RESULT_ROOT / "C2_static_matrix"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in fields} for row in materialized)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        pass


def _cuda_visible_is_legal() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        ids = {int(item.strip()) for item in visible.split(",") if item.strip()}
    except ValueError:
        return False
    return len(ids) == 1 and ids.issubset(set(protocol.LEGAL_GPU_POOL)) and ids.isdisjoint(set(protocol.FORBIDDEN_GPU_IDS))


def _device_or_fail() -> tuple[Any, int]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime environment
        raise RuntimeError("C2 requires PyTorch with CUDA") from exc
    if not torch.cuda.is_available() or not _cuda_visible_is_legal():
        raise RuntimeError("formal C2 jobs require CUDA_VISIBLE_DEVICES to contain exactly one legal physical GPU")
    physical_gpu = int(os.environ["CUDA_VISIBLE_DEVICES"].strip())
    torch.cuda.set_device(0)
    return torch.device("cuda:0"), physical_gpu


def _load_h0(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    h0_path = H0_ROOT / dataset / "H0.npy"
    budget_path = H0_ROOT / dataset / "budget_manifest.json"
    if not h0_path.exists() or not budget_path.exists():
        raise FileNotFoundError(f"missing audited S0 H0/budget artifact for {dataset}")
    h0 = np.asarray(np.load(h0_path), dtype=np.float32)
    if h0.ndim != 2 or h0.shape[1] < 1 or h0.shape[1] > 128 or not np.isfinite(h0).all():
        raise ValueError(f"unexpected H0 for {dataset}: shape={h0.shape}")
    return h0, {
        "H0_path": str(h0_path.resolve()),
        "H0_sha256": sha256_file(h0_path),
        "budget_manifest_path": str(budget_path.resolve()),
        "budget_manifest_sha256": sha256_file(budget_path),
        "shape": list(h0.shape),
        "source": "representation_consumer_probe/S0_freeze",
        "labels_used": False,
    }


def _load_labels(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = LABEL_ROOT / dataset / "seed42" / "R" / "labels_true.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing post-fit benchmark labels for {dataset}")
    labels = np.asarray(np.load(path), dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"labels must be one-dimensional for {dataset}")
    return labels, {
        "labels_path": str(path.resolve()),
        "labels_sha256": sha256_file(path),
        "labels_loaded_after_fit": True,
    }


def _standardize(h0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(h0, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(h0, axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((h0 - mean) / std).astype(np.float32), mean, std


class _SmallMAE:
    def __init__(self, device: Any, input_dim: int) -> None:
        import torch
        from torch import nn

        self.device = device
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        ).to(device)
        self.encoder = nn.Sequential(*list(self.model.children())[:4]).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(protocol.BACKBONE_CONTRACT["learning_rate"]),
        )

    def fit_epoch(self, x: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> float:
        import torch

        self.model.train()
        order = rng.permutation(x.shape[0])
        loss_sum = 0.0
        count = 0
        batch_size = int(protocol.BACKBONE_CONTRACT["batch_size"])
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            xb = torch.from_numpy(np.asarray(x[idx], dtype=np.float32)).to(self.device)
            yb = torch.from_numpy(np.asarray(target[idx], dtype=np.float32)).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            self.optimizer.step()
            loss_sum += float(loss.detach().cpu()) * idx.size
            count += int(idx.size)
        return loss_sum / max(count, 1)

    def predict(self, x: np.ndarray, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        import torch

        self.model.eval()
        embeddings: list[np.ndarray] = []
        recon: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = torch.from_numpy(np.asarray(x[start : start + batch_size], dtype=np.float32)).to(self.device)
                embeddings.append(self.encoder(xb).detach().cpu().numpy())
                recon.append(self.model(xb).detach().cpu().numpy())
        return np.concatenate(embeddings, axis=0), np.concatenate(recon, axis=0)


def _warmup_residual(h0_scaled: np.ndarray, seed: int, device: Any) -> np.ndarray:
    """Build the explicit frozen P4 residual artifact from a common warm-up."""

    _seed_everything(seed + 991)
    model = _SmallMAE(device, h0_scaled.shape[1])
    rng = np.random.default_rng(seed + 991)
    for _ in range(int(protocol.BACKBONE_CONTRACT["warmup_epochs"])):
        model.fit_epoch(h0_scaled, h0_scaled, rng)
    _, reconstruction = model.predict(h0_scaled)
    residual = np.abs(reconstruction - h0_scaled).astype(np.float32)
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass
    return residual


def _prepare_geometry_artifact(dataset: str, root: Path) -> dict[str, Any]:
    h0, source = _load_h0(dataset)
    path = root / dataset / "geometry_scores.npy"
    meta_path = root / dataset / "geometry_metadata.json"
    if path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        scores = np.asarray(np.load(path), dtype=np.float32)
        if metadata.get("H0_sha256") == source["H0_sha256"] and scores.shape == h0.shape and np.isfinite(scores).all():
            return metadata
    scores = geometry_importance(h0, k=protocol.GEOMETRY_K)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, scores.astype(np.float32))
    metadata = {
        "stage": "C2_score_preparation",
        "score_kind": "geometry_importance",
        "dataset": dataset,
        "H0_sha256": source["H0_sha256"],
        "shape": list(scores.shape),
        "k": int(protocol.GEOMETRY_K),
        "labels_used": False,
        "raw_artifact_local_only": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }
    write_json(meta_path, metadata)
    return metadata


def _prepare_residual_artifact(dataset: str, seed: int, root: Path, physical_gpu: int) -> dict[str, Any]:
    h0, source = _load_h0(dataset)
    path = root / dataset / f"seed{seed}" / "residual_scores.npy"
    meta_path = root / dataset / f"seed{seed}" / "residual_metadata.json"
    if path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        residual = np.asarray(np.load(path), dtype=np.float32)
        if (
            metadata.get("H0_sha256") == source["H0_sha256"]
            and metadata.get("warmup_seed") == int(seed + 991)
            and residual.shape == h0.shape
            and np.isfinite(residual).all()
        ):
            return metadata

    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    try:
        h0_scaled, _, _ = _standardize(h0)
        device, actual_gpu = _device_or_fail()
        residual = _warmup_residual(h0_scaled, seed, device)
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, residual.astype(np.float32))
    metadata = {
        "stage": "C2_score_preparation",
        "score_kind": "frozen_warmup_residual",
        "dataset": dataset,
        "seed": int(seed),
        "warmup_seed": int(seed + 991),
        "warmup_epochs": int(protocol.BACKBONE_CONTRACT["warmup_epochs"]),
        "H0_sha256": source["H0_sha256"],
        "shape": list(residual.shape),
        "physical_gpu": int(actual_gpu),
        "labels_used": False,
        "raw_artifact_local_only": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }
    write_json(meta_path, metadata)
    return metadata


def prepare_score_artifacts(root: Path, physical_gpu: int) -> dict[str, Any]:
    """Prepare deterministic P4/P5 scores before any performance result is seen."""

    root.mkdir(parents=True, exist_ok=True)
    geometry = [_prepare_geometry_artifact(dataset, root / "geometry") for dataset in protocol.DEVELOPMENT_PANEL]
    residuals = [
        _prepare_residual_artifact(dataset, seed, root / "residual", physical_gpu)
        for dataset in protocol.DEVELOPMENT_PANEL
        for seed in protocol.PRIMARY_SEEDS
    ]
    manifest = {
        "stage": "C2_score_preparation",
        "project_id": protocol.PROJECT_ID,
        "c2_protocol_id": protocol.C2_PROTOCOL_ID,
        "geometry_artifacts": geometry,
        "residual_artifacts": residuals,
        "labels_used": False,
        "raw_artifacts_local_only": True,
        "support_interpretation_firewall": protocol.resolved_config()["support_interpretation_firewall"],
    }
    write_json(root / "score_manifest.json", manifest)
    return manifest


def _score_path(score_root: Path, dataset: str, principle: str, seed: int) -> tuple[Path | None, Path | None]:
    geometry_path = score_root / "geometry" / dataset / "geometry_scores.npy"
    residual_path = score_root / "residual" / dataset / f"seed{seed}" / "residual_scores.npy"
    return (
        residual_path if principle == "P4_ResidualHard" else None,
        geometry_path if principle == "P5_GeometryHard" else None,
    )


def _clustering_acc(labels: np.ndarray, predictions: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    y_true = np.unique(labels, return_inverse=True)[1]
    y_pred = np.unique(predictions, return_inverse=True)[1]
    matrix = confusion_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(-matrix)
    return float(matrix[rows, cols].sum() / max(labels.size, 1))


def _embedding_diagnostics(embedding: np.ndarray) -> dict[str, float]:
    z = np.asarray(embedding, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("embedding is non-finite")
    centered = z - np.mean(z, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(z.shape[0], 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(np.sum(eigenvalues))
    if total <= 1e-12:
        effective_rank = 0.0
    else:
        p = eigenvalues / total
        effective_rank = float(np.exp(-np.sum(np.where(p > 0.0, p * np.log(p), 0.0))))
    std = np.std(z, axis=0)
    return {
        "effective_rank": effective_rank,
        "variance_floor": float(np.min(std)) if std.size else 0.0,
        "variance_median": float(np.median(std)) if std.size else 0.0,
        "low_variance_dimension_ratio": float(np.mean(std < 1e-5)) if std.size else 1.0,
    }


def positive_control() -> dict[str, Any]:
    rng = np.random.default_rng(20260818)
    matrix = np.zeros((24, 16), dtype=np.float32)
    matrix[:, :4] = rng.uniform(1.0, 2.0, size=(24, 4))
    matrix[:, 4:8] = rng.uniform(-1.0, 1.0, size=(24, 4))
    geometry = geometry_importance(matrix, k=5)
    residual = np.abs(rng.normal(size=matrix.shape)).astype(np.float32)
    checks: dict[str, bool] = {}
    arm_audits: dict[str, Any] = {}
    for principle in protocol.PRINCIPLES:
        corrupted, audit = corrupt_matrix(
            matrix,
            principle,
            np.random.default_rng(7),
            residual_scores=residual,
            geometry_scores=geometry,
        )
        compact = compact_audit(audit)
        checks[f"{principle}_finite"] = bool(np.isfinite(corrupted).all())
        checks[f"{principle}_exact_budget"] = bool(compact["exact_budget"])
        arm_audits[principle] = compact
    checks["P1_support_preserved"] = arm_audits["P1_SupportPreserve"]["support_change_rate"] == 0.0
    checks["P2_support_target_changes_support"] = arm_audits["P2_SupportTarget"]["support_change_rate"] > 0.0
    return {
        "stage": "C2_positive_control",
        "status": "completed_valid" if all(checks.values()) else "protocol_insensitive",
        "labels_used": False,
        "checks": checks,
        "arms": arm_audits,
    }


def run_job(
    dataset: str,
    principle: str,
    seed: int,
    output_dir: Path,
    *,
    score_root: Path,
) -> dict[str, Any]:
    protocol.validate_c2_authorization()
    if dataset not in protocol.DEVELOPMENT_PANEL or principle not in protocol.PRINCIPLES or seed not in protocol.PRIMARY_SEEDS:
        raise ValueError("dataset, principle or seed is outside the frozen C2 matrix")
    _seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device, physical_gpu = _device_or_fail()
    h0_raw, source = _load_h0(dataset)
    h0_scaled, mean, std = _standardize(h0_raw)
    residual_path, geometry_path = _score_path(score_root, dataset, principle, seed)
    residual_scores = None
    geometry_scores = None
    score_refs: dict[str, Any] = {}
    if residual_path is not None:
        if not residual_path.exists():
            raise FileNotFoundError(f"missing frozen residual artifact: {residual_path}")
        residual_scores = np.asarray(np.load(residual_path), dtype=np.float32)
        score_refs["residual_path"] = str(residual_path.resolve())
        score_refs["residual_sha256"] = sha256_file(residual_path)
    if geometry_path is not None:
        if not geometry_path.exists():
            raise FileNotFoundError(f"missing frozen geometry artifact: {geometry_path}")
        geometry_scores = np.asarray(np.load(geometry_path), dtype=np.float32)
        score_refs["geometry_path"] = str(geometry_path.resolve())
        score_refs["geometry_sha256"] = sha256_file(geometry_path)
    for score in (residual_scores, geometry_scores):
        if score is not None and (score.shape != h0_raw.shape or not np.isfinite(score).all()):
            raise ValueError("frozen score artifact shape/finite contract failed")

    model = _SmallMAE(device, h0_raw.shape[1])
    rng = np.random.default_rng(seed)
    epoch_rows: list[dict[str, Any]] = []
    for epoch in range(int(protocol.BACKBONE_CONTRACT["epochs"])):
        corrupted_raw, audit = corrupt_matrix(
            h0_raw,
            principle,
            rng,
            residual_scores=residual_scores,
            geometry_scores=geometry_scores,
        )
        corrupted_scaled = ((corrupted_raw - mean) / std).astype(np.float32)
        train_loss_before_step = model.fit_epoch(corrupted_scaled, h0_scaled, rng)
        compact = compact_audit(audit)
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss_before_step": train_loss_before_step,
                "exact_budget": compact["exact_budget"],
                "effective_changed_coordinate_rate": compact["effective_changed_coordinate_rate"],
                "support_change_rate": compact["support_change_rate"],
                "value_change_rate": compact["value_change_rate"],
                "total_absolute_change": compact["total_absolute_change"],
            }
        )

    embedding, reconstruction = model.predict(h0_scaled)
    # Labels are intentionally introduced only after all fit steps are over.
    labels, label_source = _load_labels(dataset)
    if labels.size != h0_raw.shape[0]:
        raise ValueError(f"label/H0 mismatch for {dataset}: {labels.size} != {h0_raw.shape[0]}")
    k = int(np.unique(labels).size)
    predictions = KMeans(n_clusters=k, n_init=20, random_state=int(seed)).fit_predict(embedding)
    metrics = {
        "ARI": float(adjusted_rand_score(labels, predictions)),
        "NMI": float(normalized_mutual_info_score(labels, predictions)),
        "ACC": _clustering_acc(labels, predictions),
        "L_rec": float(np.mean((reconstruction - h0_scaled) ** 2)),
        "last_epoch_train_loss_before_step": float(epoch_rows[-1]["train_loss_before_step"]),
        **_embedding_diagnostics(embedding),
    }
    exact_budget = bool(all(bool(row["exact_budget"]) for row in epoch_rows))
    corruption_audit = {
        "effective_changed_coordinate_rate_mean": float(np.mean([row["effective_changed_coordinate_rate"] for row in epoch_rows])),
        "support_change_rate_mean": float(np.mean([row["support_change_rate"] for row in epoch_rows])),
        "value_change_rate_mean": float(np.mean([row["value_change_rate"] for row in epoch_rows])),
        "total_absolute_change_mean": float(np.mean([row["total_absolute_change"] for row in epoch_rows])),
        "exact_budget_all_epochs": exact_budget,
        "epochs_audited": len(epoch_rows),
    }
    summary = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.C2_PROTOCOL_ID,
        "base_protocol_id": protocol.PROTOCOL_ID,
        "stage": "C2_static_matrix",
        "dataset": dataset,
        "role": protocol.ROLE_BY_DATASET[dataset],
        "principle": principle,
        "seed": int(seed),
        "status": "completed_valid" if exact_budget and np.isfinite(list(metrics.values())).all() else "protocol_mismatch",
        "device": str(device),
        "physical_gpu": physical_gpu,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "K": k,
        "K_source": "benchmark_oracle_from_y_outer_readout_only",
        "metrics": metrics,
        "corruption_audit": corruption_audit,
        "source": {**source, **label_source, "mean_std_fit_on_clean_H0_only": True},
        "score_artifacts": score_refs,
        "backbone": dict(protocol.BACKBONE_CONTRACT),
        "input_dim": int(h0_raw.shape[1]),
        "support_definition": protocol.resolved_config()["support_definition"],
        "support_interpretation_firewall": protocol.resolved_config()["support_interpretation_firewall"],
        "raw_arrays_persisted": False,
    }
    audit = {
        "audit_ok": summary["status"] == "completed_valid",
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.C2_PROTOCOL_ID,
        "stage": "C2_static_matrix",
        "dataset": dataset,
        "principle": principle,
        "seed": int(seed),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "cuda_visible_is_legal": _cuda_visible_is_legal(),
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "embedding_finite": bool(np.isfinite(embedding).all()),
        "prediction_count": int(np.unique(predictions).size),
        "exact_budget_all_epochs": exact_budget,
        "raw_artifacts_persisted": False,
        "score_artifacts_label_free": True,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "audit.json", audit)
    write_json(
        output_dir / "resolved_config.json",
        {
            **protocol.resolved_config(),
            "stage": "C2_static_matrix",
            "c2_protocol_id": protocol.C2_PROTOCOL_ID,
            "dataset": dataset,
            "principle": principle,
            "seed": int(seed),
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "score_artifacts": score_refs,
        },
    )
    _write_csv(output_dir / "training_metrics.csv", epoch_rows)
    return summary


def _run_one_subprocess(
    dataset: str,
    principle: str,
    seed: int,
    root: Path,
    score_root: Path,
    gpu_id: int,
) -> dict[str, Any]:
    run_dir = root / dataset / principle / f"seed{seed}"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "4"
    command = [
        sys.executable,
        "-m",
        "scripts.sparse_corruption_principle_probe.c2_matrix",
        "--dataset",
        dataset,
        "--principle",
        principle,
        "--seed",
        str(seed),
        "--output-dir",
        str(run_dir),
        "--score-root",
        str(score_root),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        write_json(
            run_dir / "failure.json",
            {
                "dataset": dataset,
                "principle": principle,
                "seed": int(seed),
                "status": "incomplete_compute",
                "returncode": completed.returncode,
                "gpu": int(gpu_id),
                "stderr_tail": completed.stderr[-6000:],
                "stdout_tail": completed.stdout[-2000:],
            },
        )
        return {"dataset": dataset, "principle": principle, "seed": int(seed), "status": "incomplete_compute"}
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"dataset": dataset, "principle": principle, "seed": int(seed), "status": "incomplete_compute", "reason": "missing_summary"}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _existing_run_valid(run_dir: Path, dataset: str, principle: str, seed: int) -> bool:
    summary_path = run_dir / "summary.json"
    audit_path = run_dir / "audit.json"
    config_path = run_dir / "resolved_config.json"
    if not summary_path.exists() or not audit_path.exists() or not config_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        current_h0, current_source = _load_h0(dataset)
        current_labels, current_label_source = _load_labels(dataset)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    del current_h0, current_labels
    recorded_source = summary.get("source", {})
    return bool(
        summary.get("status") == "completed_valid"
        and audit.get("audit_ok") is True
        and summary.get("stage") == "C2_static_matrix"
        and summary.get("protocol_id") == protocol.C2_PROTOCOL_ID
        and summary.get("dataset") == dataset
        and summary.get("principle") == principle
        and int(summary.get("seed", -1)) == int(seed)
        and config.get("dataset") == dataset
        and config.get("principle") == principle
        and int(config.get("seed", -1)) == int(seed)
        and recorded_source.get("H0_sha256") == current_source.get("H0_sha256")
        and recorded_source.get("budget_manifest_sha256") == current_source.get("budget_manifest_sha256")
        and recorded_source.get("labels_sha256") == current_label_source.get("labels_sha256")
    )


def _quarantine_invalid_run(run_dir: Path, root: Path) -> None:
    if not run_dir.exists():
        return
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = root / "_attempts" / f"{run_dir.parent.parent.name}_{run_dir.parent.name}_{run_dir.name}_{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(destination))


def _coarse_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    audit = summary.get("corruption_audit", {})
    return {
        "dataset": summary.get("dataset"),
        "role": summary.get("role"),
        "principle": summary.get("principle"),
        "seed": summary.get("seed"),
        "status": summary.get("status"),
        "ARI": metrics.get("ARI"),
        "NMI": metrics.get("NMI"),
        "ACC": metrics.get("ACC"),
        "L_rec": metrics.get("L_rec"),
        "effective_changed_coordinate_rate": audit.get("effective_changed_coordinate_rate_mean"),
        "support_change_rate": audit.get("support_change_rate_mean"),
        "value_change_rate": audit.get("value_change_rate_mean"),
        "total_absolute_change": audit.get("total_absolute_change_mean"),
        "exact_budget_all_epochs": audit.get("exact_budget_all_epochs"),
        "physical_gpu": summary.get("physical_gpu"),
    }


def aggregate(root: Path, positive: dict[str, Any], *, score_manifest: dict[str, Any], gpu_pool: tuple[int, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in protocol.DEVELOPMENT_PANEL:
        for principle in protocol.PRINCIPLES:
            for seed in protocol.PRIMARY_SEEDS:
                path = root / dataset / principle / f"seed{seed}" / "summary.json"
                if path.exists():
                    try:
                        summary = json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        summary = {}
                    if summary.get("status") == "completed_valid":
                        rows.append(_coarse_row(summary))
                        continue
                rows.append({"dataset": dataset, "role": protocol.ROLE_BY_DATASET[dataset], "principle": principle, "seed": int(seed), "status": "incomplete_compute"})

    expected = len(protocol.DEVELOPMENT_PANEL) * len(protocol.PRINCIPLES) * len(protocol.PRIMARY_SEEDS)
    valid = [row for row in rows if row.get("status") == "completed_valid"]
    complete = len(valid) == expected
    dataset_rows: list[dict[str, Any]] = []
    by_dataset_principle: dict[tuple[str, str], dict[str, float]] = {}
    for dataset in protocol.DEVELOPMENT_PANEL:
        p0_by_seed = {
            int(row["seed"]): row
            for row in valid
            if row["dataset"] == dataset and row["principle"] == "P0_Random"
        }
        p0_complete = set(p0_by_seed) == set(protocol.PRIMARY_SEEDS)
        p0_ari = float(np.mean([p0_by_seed[seed]["ARI"] for seed in protocol.PRIMARY_SEEDS])) if p0_complete else float("nan")
        for principle in protocol.PRINCIPLES:
            selected = [row for row in valid if row["dataset"] == dataset and row["principle"] == principle]
            selected_by_seed = {int(row["seed"]): row for row in selected}
            paired_seeds = sorted(set(selected_by_seed) & set(p0_by_seed))
            if not p0_complete or len(selected_by_seed) != len(protocol.PRIMARY_SEEDS) or set(paired_seeds) != set(protocol.PRIMARY_SEEDS):
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        "role": protocol.ROLE_BY_DATASET[dataset],
                        "principle": principle,
                        "seed_count": len(selected_by_seed),
                        "paired_seed_count": len(paired_seeds),
                        "status": "incomplete_compute",
                    }
                )
                continue
            paired_rows = [selected_by_seed[seed] for seed in protocol.PRIMARY_SEEDS]
            means = {key: float(np.mean([row[key] for row in paired_rows])) for key in ("ARI", "NMI", "ACC", "L_rec", "effective_changed_coordinate_rate", "support_change_rate", "value_change_rate", "total_absolute_change")}
            row = {
                "dataset": dataset,
                "role": protocol.ROLE_BY_DATASET[dataset],
                "principle": principle,
                "seed_count": len(selected),
                "paired_seed_count": len(paired_seeds),
                "status": "completed_valid",
                "ARI_mean": means["ARI"],
                "NMI_mean": means["NMI"],
                "ACC_mean": means["ACC"],
                "L_rec_mean": means["L_rec"],
                "effective_changed_coordinate_rate_mean": means["effective_changed_coordinate_rate"],
                "support_change_rate_mean": means["support_change_rate"],
                "value_change_rate_mean": means["value_change_rate"],
                "total_absolute_change_mean": means["total_absolute_change"],
                "delta_ARI_vs_P0": means["ARI"] - p0_ari if np.isfinite(p0_ari) else float("nan"),
            }
            dataset_rows.append(row)
            by_dataset_principle[(dataset, principle)] = row

    structured = [principle for principle in protocol.PRINCIPLES if principle != "P0_Random"]
    material_count_by_principle = {
        principle: sum(
            float(by_dataset_principle.get((dataset, principle), {}).get("delta_ARI_vs_P0", float("nan"))) >= protocol.MATERIAL_DELTA_ARI
            for dataset in protocol.DEVELOPMENT_PANEL
        )
        for principle in structured
    }
    winners: dict[str, dict[str, Any]] = {}
    for dataset in protocol.DEVELOPMENT_PANEL:
        candidates = [
            (float(by_dataset_principle.get((dataset, principle), {}).get("delta_ARI_vs_P0", float("nan"))), principle)
            for principle in structured
            if (dataset, principle) in by_dataset_principle
        ]
        candidates = [item for item in candidates if np.isfinite(item[0])]
        if candidates:
            value, principle = max(candidates, key=lambda item: (item[0], item[1]))
            winners[dataset] = {"principle": principle, "delta_ARI_vs_P0": value, "material": value >= protocol.MATERIAL_DELTA_ARI}
    material_winners = {dataset: row for dataset, row in winners.items() if row["material"]}
    distinct_material_winners = sorted({row["principle"] for row in material_winners.values()})
    simple_principles = [principle for principle, count in material_count_by_principle.items() if count >= 2]
    if not complete:
        status = "incomplete_compute"
    elif simple_principles:
        status = "simple_static_principle_sufficient"
    elif len(material_winners) >= 2 and len(distinct_material_winners) >= 2:
        status = "heterogeneous_static_principles"
    elif material_winners:
        status = "development_only_material_effect"
    else:
        status = "no_reproducible_static_principle"

    principle_rows: list[dict[str, Any]] = []
    for principle in protocol.PRINCIPLES:
        values = [float(row.get("delta_ARI_vs_P0")) for row in dataset_rows if row.get("principle") == principle and row.get("status") == "completed_valid" and np.isfinite(float(row.get("delta_ARI_vs_P0")))]
        principle_rows.append(
            {
                "principle": principle,
                "dataset_count": len(values),
                "delta_ARI_mean": float(np.mean(values)) if values else float("nan"),
                "delta_ARI_median": float(np.median(values)) if values else float("nan"),
                "material_dataset_count": material_count_by_principle.get(principle, 0),
            }
        )

    _write_csv(root / "c2_run_summary.csv", rows)
    _write_csv(root / "c2_dataset_summary.csv", dataset_rows)
    _write_csv(root / "c2_principle_summary.csv", principle_rows)
    decision = {
        "stage": "C2_static_matrix",
        "status": status,
        "expected_runs": expected,
        "completed_valid_runs": len(valid),
        "simple_principles": simple_principles,
        "material_count_by_principle": material_count_by_principle,
        "dataset_best_principle": winners,
        "distinct_material_winner_principles": distinct_material_winners,
        "tested_static_library_opportunity_is_not_oracle": True,
        "adaptive_policy_unlocked": False,
        "c3_holdout_runs_unlocked": False,
        "support_interpretation_firewall": protocol.resolved_config()["support_interpretation_firewall"],
    }
    write_json(root / "decision.json", decision)
    audit = {
        "audit_ok": bool(complete and positive.get("status") == "completed_valid" and all(row.get("exact_budget_all_epochs") is True for row in valid)),
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.C2_PROTOCOL_ID,
        "stage": "C2_static_matrix",
        "expected_run_count": expected,
        "completed_valid_run_count": len(valid),
        "all_jobs_completed_valid": complete,
        "positive_control_passed": positive.get("status") == "completed_valid",
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "gpu_pool": list(gpu_pool),
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "raw_arrays_persisted": False,
        "score_artifacts_local_only": True,
        "support_interpretation_firewall": protocol.resolved_config()["support_interpretation_firewall"],
    }
    write_json(root / "audit.json", audit)
    report_lines = [
        "# C2 Static Corruption Principle Matrix",
        "",
        f"Status: `{status}`; completed-valid runs: `{len(valid)}/{expected}`.",
        "",
        "> Support in C2 denotes the frozen threshold-defined support of dense H0, not raw-X zero/nonzero support; raw sparse-support claims require a separate validation.",
        "",
        "Primary endpoint: `Delta_P = ARI(P) - ARI(P0_Random)`; seeds are paired repeats and the dataset is the analysis unit.",
        "",
        "## Dataset-level summary",
        "",
        "| Dataset | Principle | ARI mean | Delta vs P0 | Support change | Value change | Sum abs delta | L_rec |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_rows:
        if row.get("status") != "completed_valid":
            continue
        report_lines.append(
            f"| {row['dataset']} | {row['principle']} | {row['ARI_mean']:.6f} | {row['delta_ARI_vs_P0']:.6f} | {row['support_change_rate_mean']:.6f} | {row['value_change_rate_mean']:.6f} | {row['total_absolute_change_mean']:.6f} | {row['L_rec_mean']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            f"- Material descriptive margin: `{protocol.MATERIAL_DELTA_ARI}` ARI.",
            f"- Simple-principle candidates (material on at least two development datasets): `{', '.join(simple_principles) if simple_principles else 'none'}`.",
            f"- Dataset best arms: `{json.dumps(winners, sort_keys=True)}`.",
            "- Score representation caveat: P4 uses standardized clean H0 residuals frozen per dataset×seed; P5 uses raw clean H0 geometry scores frozen per dataset.",
            "- Adaptive policy, GAN, learned generator and C3 holdout runs remain locked; any future unlock requires a new explicit protocol.",
            "",
            "Raw score arrays, H0, labels, embeddings, predictions, checkpoints and logs remain local and are not publication artifacts.",
            "",
        ]
    )
    (root / "C2_RESULTS.md").write_text("\n".join(report_lines), encoding="utf-8")
    write_json(
        root / "resolved_config.json",
        {
            **protocol.resolved_config(),
            "stage": "C2_static_matrix",
            "c2_protocol_id": protocol.C2_PROTOCOL_ID,
            "score_manifest": score_manifest,
            "gpu_pool_used": list(gpu_pool),
            "raw_artifacts_published": False,
        },
    )
    return {"decision": decision, "audit": audit, "rows": rows, "dataset_rows": dataset_rows, "principle_rows": principle_rows}


def run_matrix(root: Path = DEFAULT_OUTPUT, *, gpu_pool: tuple[int, ...] = (2, 3, 4, 5, 6)) -> dict[str, Any]:
    protocol.validate_c2_authorization()
    if not gpu_pool or any(gpu not in protocol.LEGAL_GPU_POOL for gpu in gpu_pool) or set(gpu_pool) & set(protocol.FORBIDDEN_GPU_IDS):
        raise ValueError(f"illegal C2 GPU pool: {gpu_pool}")
    root.mkdir(parents=True, exist_ok=True)
    positive = positive_control()
    write_json(root / "positive_control.json", positive)
    if positive.get("status") != "completed_valid":
        return aggregate(root, positive, score_manifest={}, gpu_pool=gpu_pool)

    score_root = root / "score_artifacts"
    score_manifest = prepare_score_artifacts(score_root, gpu_pool[0])
    jobs = [(dataset, principle, seed) for dataset in protocol.DEVELOPMENT_PANEL for principle in protocol.PRINCIPLES for seed in protocol.PRIMARY_SEEDS]
    ledger: list[dict[str, Any]] = []
    to_run: list[tuple[str, str, int]] = []
    for dataset, principle, seed in jobs:
        run_dir = root / dataset / principle / f"seed{seed}"
        if _existing_run_valid(run_dir, dataset, principle, seed):
            ledger.append({"dataset": dataset, "principle": principle, "seed": seed, "status": "reused"})
        else:
            if run_dir.exists():
                _quarantine_invalid_run(run_dir, root)
            ledger.append({"dataset": dataset, "principle": principle, "seed": seed, "status": "queued"})
            to_run.append((dataset, principle, seed))
    write_json(root / "launch_manifest.json", {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.C2_PROTOCOL_ID,
        "stage": "C2_static_matrix",
        "expected_jobs": len(jobs),
        "new_jobs": len(to_run),
        "reused_jobs": len(jobs) - len(to_run),
        "gpu_pool": list(gpu_pool),
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "score_preparation_is_not_formal_matrix_run": True,
        "jobs": ledger,
    })
    with ThreadPoolExecutor(max_workers=len(gpu_pool)) as executor:
        futures = {
            executor.submit(_run_one_subprocess, dataset, principle, seed, root, score_root, gpu_pool[index % len(gpu_pool)]): (dataset, principle, seed)
            for index, (dataset, principle, seed) in enumerate(to_run)
        }
        for future in as_completed(futures):
            dataset, principle, seed = futures[future]
            try:
                result = future.result()
                status = "completed" if result.get("status") == "completed_valid" else "incomplete"
            except Exception as exc:  # pragma: no cover - scheduler boundary
                result = {"dataset": dataset, "principle": principle, "seed": seed, "status": "incomplete_compute", "error": str(exc)}
                status = "incomplete"
            for row in ledger:
                if row["dataset"] == dataset and row["principle"] == principle and int(row["seed"]) == int(seed):
                    row.update({"status": status, "result_status": result.get("status")})
                    break
            write_json(root / "launch_manifest.json", {
                "project_id": protocol.PROJECT_ID,
                "protocol_id": protocol.C2_PROTOCOL_ID,
                "stage": "C2_static_matrix",
                "expected_jobs": len(jobs),
                "new_jobs": len(to_run),
                "reused_jobs": len(jobs) - len(to_run),
                "gpu_pool": list(gpu_pool),
                "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
                "score_preparation_is_not_formal_matrix_run": True,
                "jobs": ledger,
            })
    result = aggregate(root, positive, score_manifest=score_manifest, gpu_pool=gpu_pool)
    result["launch_manifest"] = ledger
    write_json(root / "run_manifest.json", {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.C2_PROTOCOL_ID,
        "stage": "C2_static_matrix",
        "expected_jobs": len(jobs),
        "completed_valid": result["audit"]["completed_valid_run_count"],
        "status": result["decision"]["status"],
        "publication_scope": "compact summaries and audits only",
    })
    # Compact exact-tree hashes exclude raw local score arrays and attempt logs.
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_hashes.json" or "_attempts" in path.parts:
            continue
        if path.suffix in {".npy", ".npz", ".log"} or "score_artifacts" in path.parts:
            continue
        files[str(path.relative_to(root))] = sha256_file(path)
    write_json(root / "artifact_hashes.json", {"stage": "C2_static_matrix", "files": files, "raw_local_exclusions": ["score_artifacts/**/*.npy", "_attempts/**"], "exact_tree_policy": "compact_non-array_files_only"})
    write_json(root / "run_manifest.json", {"project_id": protocol.PROJECT_ID, "protocol_id": protocol.C2_PROTOCOL_ID, "stage": "C2_static_matrix", "expected_jobs": len(jobs), "completed_valid": result["audit"]["completed_valid_run_count"], "status": result["decision"]["status"], "publication_scope": "compact summaries and audits only"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=protocol.DEVELOPMENT_PANEL)
    parser.add_argument("--principle", choices=protocol.PRINCIPLES)
    parser.add_argument("--seed", type=int, choices=protocol.PRIMARY_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score-root", type=Path)
    parser.add_argument("--gpu-pool", default="2,3,4,5,6")
    parser.add_argument("--positive-control", action="store_true")
    parser.add_argument("--prepare-scores", action="store_true")
    parser.add_argument("--run-matrix", action="store_true")
    args = parser.parse_args()
    if args.positive_control:
        print(json.dumps(positive_control(), indent=2, sort_keys=True, default=_json_default))
    elif args.prepare_scores:
        pool = tuple(int(item) for item in args.gpu_pool.split(",") if item.strip())
        print(json.dumps(prepare_score_artifacts(args.output_dir, pool[0]), indent=2, sort_keys=True, default=_json_default))
    elif args.run_matrix:
        pool = tuple(int(item) for item in args.gpu_pool.split(",") if item.strip())
        print(json.dumps(run_matrix(args.output_dir, gpu_pool=pool), indent=2, sort_keys=True, default=_json_default))
    elif args.dataset and args.principle and args.seed is not None:
        if args.score_root is None:
            parser.error("direct jobs require --score-root")
        print(json.dumps(run_job(args.dataset, args.principle, args.seed, args.output_dir, score_root=args.score_root), indent=2, sort_keys=True, default=_json_default))
    else:
        parser.error("choose --positive-control, --prepare-scores, --run-matrix, or --dataset/--principle/--seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
