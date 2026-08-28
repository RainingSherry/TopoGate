"""Matched small-MAE runner for E1/E2.

The runner never loads labels until every fit epoch and checkpoint embedding
has been produced.  It is intentionally a single-job entry point; the
overnight orchestrator owns GPU assignment, timeout, retry and reuse.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import corruption, protocol


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for row in rows)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _cuda_visible_is_legal() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        ids = {int(item.strip()) for item in visible.split(",") if item.strip()}
    except ValueError:
        return False
    return len(ids) == 1 and ids.issubset(set(protocol.GPU_POOL)) and ids.isdisjoint(set(protocol.FORBIDDEN_GPU_IDS))


def _device_or_fail() -> tuple[Any, int]:
    import torch

    if not torch.cuda.is_available() or not _cuda_visible_is_legal():
        raise RuntimeError("formal jobs require one legal physical GPU in CUDA_VISIBLE_DEVICES")
    physical = int(os.environ["CUDA_VISIBLE_DEVICES"].strip())
    torch.cuda.set_device(0)
    return torch.device("cuda:0"), physical


def _load_h0(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    h0_path = protocol.INPUT_ROOT / dataset / "H0.npy"
    budget_path = protocol.INPUT_ROOT / dataset / "budget_manifest.json"
    if not h0_path.exists() or not budget_path.exists():
        raise FileNotFoundError(f"missing frozen H0/budget for {dataset}")
    h0 = np.asarray(np.load(h0_path), dtype=np.float32)
    if h0.ndim != 2 or not np.isfinite(h0).all():
        raise ValueError(f"invalid H0 for {dataset}: {h0.shape}")
    return h0, {
        "H0_sha256": _sha256(h0_path),
        "budget_manifest_sha256": _sha256(budget_path),
        "shape": list(h0.shape),
        "source": "representation_consumer_probe/S0_freeze",
        "labels_used": False,
    }


def _load_labels(dataset: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = protocol.LABEL_ROOT / dataset / "seed42" / "R" / "labels_true.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing post-fit labels for {dataset}")
    labels = np.asarray(np.load(path), dtype=np.int64)
    return labels, {"labels_sha256": _sha256(path), "labels_loaded_after_fit": True}


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
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim),
        ).to(device)
        self.encoder = nn.Sequential(*list(self.model.children())[:4]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(protocol.BACKBONE["learning_rate"]))

    @staticmethod
    def _loss_components(pred: Any, target: Any, changed: Any) -> tuple[Any, Any, Any]:
        import torch

        error = (pred - target) ** 2
        all_loss = torch.mean(error)
        changed_count = torch.sum(changed)
        unchanged_count = torch.sum(~changed)
        changed_loss = torch.sum(error * changed) / torch.clamp(changed_count, min=1)
        unchanged_loss = torch.sum(error * (~changed)) / torch.clamp(unchanged_count, min=1)
        return all_loss, changed_loss, unchanged_loss

    def fit_epoch(self, x: np.ndarray, target: np.ndarray, changed: np.ndarray, objective: str, rng: np.random.Generator) -> float:
        import torch

        self.model.train()
        order = rng.permutation(x.shape[0])
        loss_sum = 0.0
        count = 0
        batch_size = int(protocol.BACKBONE["batch_size"])
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            xb = torch.as_tensor(x[idx], dtype=torch.float32, device=self.device)
            yb = torch.as_tensor(target[idx], dtype=torch.float32, device=self.device)
            mb = torch.as_tensor(changed[idx], dtype=torch.bool, device=self.device)
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(xb)
            all_loss, changed_loss, unchanged_loss = self._loss_components(pred, yb, mb)
            if objective == "O0_GlobalMSE":
                loss = all_loss
            elif objective == "O1_ChangedOnlyMSE":
                loss = changed_loss
            elif objective == "O2_BalancedMSE":
                loss = 0.5 * changed_loss + 0.5 * unchanged_loss
            else:
                raise ValueError(f"unknown objective {objective}")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss for objective {objective}")
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
                xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=self.device)
                embeddings.append(self.encoder(xb).cpu().numpy())
                recon.append(self.model(xb).cpu().numpy())
        return np.concatenate(embeddings, axis=0), np.concatenate(recon, axis=0)


def _embedding_diagnostics(embedding: np.ndarray) -> dict[str, float]:
    z = np.asarray(embedding, dtype=np.float64)
    centered = z - np.mean(z, axis=0, keepdims=True)
    eig = np.clip(np.linalg.eigvalsh(centered.T @ centered / max(z.shape[0], 1)), 0.0, None)
    total = float(np.sum(eig))
    p = eig / total if total > 1e-12 else np.zeros_like(eig)
    return {
        "effective_rank": float(np.exp(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))) if total > 1e-12 else 0.0,
        "variance_median": float(np.median(np.std(z, axis=0))),
        "variance_floor": float(np.min(np.std(z, axis=0))),
        "low_variance_dimension_ratio": float(np.mean(np.std(z, axis=0) < 1e-5)),
    }


def _acc(labels: np.ndarray, predictions: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    yt = np.unique(labels, return_inverse=True)[1]
    yp = np.unique(predictions, return_inverse=True)[1]
    matrix = confusion_matrix(yt, yp)
    rows, cols = linear_sum_assignment(-matrix)
    return float(matrix[rows, cols].sum() / max(labels.size, 1))


def _checkpoint_row(epoch: int, embedding: np.ndarray, reconstruction: np.ndarray, target: np.ndarray, changed: np.ndarray) -> dict[str, Any]:
    err = (reconstruction - target) ** 2
    changed_count = int(np.sum(changed))
    unchanged_count = int(np.sum(~changed))
    row = {
        "epoch": int(epoch),
        "all_mse": float(np.mean(err)),
        "changed_mse": float(np.sum(err * changed) / max(changed_count, 1)),
        "unchanged_mse": float(np.sum(err * (~changed)) / max(unchanged_count, 1)),
    }
    row.update(_embedding_diagnostics(embedding))
    return row


def run_job(dataset: str, arm: str, objective: str, seed: int, output_dir: Path, *, stage: str) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    protocol.validate_contract()
    if dataset not in protocol.DEVELOPMENT_PANEL or seed not in protocol.PRIMARY_SEEDS:
        raise ValueError("dataset/seed outside frozen contract")
    if arm not in protocol.E1_ARMS:
        raise ValueError(f"unsupported arm {arm}")
    if stage == "E1" and objective != "O0_GlobalMSE":
        raise ValueError("E1 is frozen to O0_GlobalMSE")
    if stage == "E2" and (arm not in protocol.E2_CORRUPTIONS or objective not in protocol.E2_OBJECTIVES[1:]):
        raise ValueError("E2 is frozen to P0/P2 x O1/O2")
    _seed_everything(seed)
    device, physical_gpu = _device_or_fail()
    clean_raw, source = _load_h0(dataset)
    clean_scaled, mean, std = _standardize(clean_raw)
    model = _SmallMAE(device, clean_raw.shape[1])
    rng = np.random.default_rng(seed)
    checkpoint_embeddings: dict[int, np.ndarray] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    exact_budget = True
    last_corrupted = clean_scaled
    last_changed = np.zeros_like(clean_raw, dtype=bool)
    for epoch in range(1, int(protocol.BACKBONE["epochs"]) + 1):
        corrupted_raw, audit = corruption.make_corruption(clean_raw, arm, rng)
        changed = np.asarray(audit.get("changed_mask", np.abs(corrupted_raw - clean_raw) > 1e-7), dtype=bool)
        if arm != "Clean" and not bool(audit.get("exact_budget", True)):
            exact_budget = False
        corrupted_scaled = ((corrupted_raw - mean) / std).astype(np.float32)
        train_loss = model.fit_epoch(corrupted_scaled, clean_scaled, changed, objective, rng)
        last_corrupted, last_changed = corrupted_scaled, changed
        epoch_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "effective_changed_coordinate_rate": float(np.mean(changed)),
            "support_change_rate": float(audit.get("support_change_rate", 0.0)),
            "value_change_rate": float(audit.get("value_change_rate", 0.0)),
            "total_absolute_change": float(audit.get("total_absolute_change", 0.0)),
            "exact_budget": bool(audit.get("exact_budget", True)),
        })
        if epoch in protocol.CHECKPOINT_EPOCHS:
            embedding, _ = model.predict(clean_scaled)
            _, reconstruction = model.predict(last_corrupted)
            checkpoint_embeddings[epoch] = embedding
            checkpoint_rows.append(_checkpoint_row(epoch, embedding, reconstruction, clean_scaled, last_changed))

    # Label boundary: this is the first label read in the entire job.
    labels, label_source = _load_labels(dataset)
    if labels.size != clean_raw.shape[0]:
        raise ValueError("label/H0 row count mismatch")
    k = int(np.unique(labels).size)
    checkpoint_metrics: list[dict[str, Any]] = []
    for row in checkpoint_rows:
        epoch = int(row["epoch"])
        predictions = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(checkpoint_embeddings[epoch])
        metrics = {
            "epoch": epoch,
            "ARI": float(adjusted_rand_score(labels, predictions)),
            "NMI": float(normalized_mutual_info_score(labels, predictions)),
            "ACC": _acc(labels, predictions),
            **{key: row[key] for key in ("all_mse", "changed_mse", "unchanged_mse", "effective_rank", "variance_median", "variance_floor", "low_variance_dimension_ratio")},
        }
        checkpoint_metrics.append(metrics)
    final = checkpoint_metrics[-1]
    ended_at = datetime.now(timezone.utc).isoformat()
    try:
        peak_gpu_memory = int(__import__("torch").cuda.max_memory_allocated(device))
    except Exception:
        peak_gpu_memory = None
    summary = {
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": stage,
        "dataset": dataset,
        "role": protocol.ROLE_BY_DATASET[dataset],
        "arm": arm,
        "objective": objective,
        "seed": int(seed),
        "status": "completed_valid" if exact_budget and all(np.isfinite(float(v)) for v in final.values() if isinstance(v, (float, int))) else "protocol_mismatch",
        "physical_gpu": physical_gpu,
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": started_at,
        "ended_at": ended_at,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "runner_source_sha256": _sha256(Path(__file__)),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "K": k,
        "K_source": "benchmark_oracle_from_y_outer_readout_only",
        "metrics": final,
        "checkpoint_metrics": checkpoint_metrics,
        "training_metrics": epoch_rows,
        "source": {**source, **label_source, "mean_std_fit_on_clean_H0_only": True},
        "backbone": dict(protocol.BACKBONE),
        "checkpoint_epochs": list(protocol.CHECKPOINT_EPOCHS),
        "raw_arrays_persisted": False,
        "support_semantics": "threshold_defined_dense_H0_only; raw_X_support_not_used",
    }
    audit = {
        "audit_ok": summary["status"] == "completed_valid",
        "project_id": protocol.PROJECT_ID,
        "protocol_id": protocol.PROTOCOL_ID,
        "stage": stage,
        "dataset": dataset,
        "arm": arm,
        "objective": objective,
        "seed": int(seed),
        "labels_used_during_fit": False,
        "labels_used_for_outer_metrics": True,
        "cuda_visible_is_legal": _cuda_visible_is_legal(),
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "exact_budget_all_epochs": exact_budget,
        "checkpoint_epochs_complete": [int(row["epoch"]) for row in checkpoint_metrics] == list(protocol.CHECKPOINT_EPOCHS),
        "raw_arrays_persisted": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "audit.json", audit)
    _write_json(output_dir / "resolved_config.json", {
        **protocol.resolved_config(),
        "stage": stage,
        "dataset": dataset,
        "arm": arm,
        "objective": objective,
        "seed": int(seed),
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runner_source_sha256": _sha256(Path(__file__)),
    })
    _write_csv(output_dir / "training_metrics.csv", epoch_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=protocol.DEVELOPMENT_PANEL)
    parser.add_argument("--arm", required=True, choices=protocol.E1_ARMS)
    parser.add_argument("--objective", required=True, choices=protocol.E2_OBJECTIVES)
    parser.add_argument("--seed", required=True, type=int, choices=protocol.PRIMARY_SEEDS)
    parser.add_argument("--stage", required=True, choices=("E1", "E2"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_job(args.dataset, args.arm, args.objective, args.seed, args.output_dir, stage=args.stage)
    print(json.dumps({"status": result["status"], "dataset": args.dataset, "arm": args.arm, "objective": args.objective, "seed": args.seed}, sort_keys=True))
    return 0 if result["status"] == "completed_valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
