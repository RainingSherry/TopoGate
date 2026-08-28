"""The frozen small MLP autoencoder and training-time audit path."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import masking, protocol


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def state_hash(state: dict[str, Any]) -> str:
    """Hash a PyTorch state dict without relying on pickle ordering."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if hasattr(value, "detach"):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes(order="C"))
        else:
            digest.update(name.encode("utf-8"))
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def hash_orders(orders: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for order in orders:
        value = np.asarray(order, dtype=np.int64)
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def hash_mask_audits(audits: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for audit in audits:
        digest.update(json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def make_model(input_dim: int, seed: int, device: Any = "cpu") -> Any:
    import torch
    from torch import nn

    seed_everything(seed)
    model = nn.Sequential(
        nn.Linear(int(input_dim), 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, int(input_dim)),
    ).to(device)
    return model


def encoder_from_model(model: Any) -> Any:
    import torch.nn as nn

    # The first four modules end at the 32-dimensional ReLU embedding.
    return nn.Sequential(*list(model.children())[:4])


def batch_orders(n_rows: int, epochs: int, batch_size: int, seed: int) -> list[np.ndarray]:
    orders: list[np.ndarray] = []
    for epoch in range(int(epochs)):
        rng = np.random.default_rng(int(seed) + 1_000_003 * (epoch + 1))
        orders.append(rng.permutation(int(n_rows)).astype(np.int64))
    return orders


def choose_batch_size(x: np.ndarray, input_dim: int, device: Any, candidates: tuple[int, ...] | None = None, seed: int = 42) -> tuple[int, dict[str, Any]]:
    """Outcome-independent forward/backward memory preflight.

    The first candidate that completes is frozen for all formal arms of the
    dataset.  The smoke never reads labels or metrics.
    """
    import torch

    candidates = tuple(candidates or protocol.BACKBONE["batch_size_candidates"])
    rows = int(np.asarray(x).shape[0])
    attempts: list[dict[str, Any]] = []
    for candidate in candidates:
        bs = min(int(candidate), rows)
        if bs <= 0:
            continue
        model = None
        try:
            model = make_model(input_dim, seed, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(protocol.BACKBONE["learning_rate"]))
            xb = torch.zeros((bs, int(input_dim)), dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(xb).square().mean()
            loss.backward()
            optimizer.step()
            attempts.append({"batch_size": int(bs), "status": "passed"})
            del model, optimizer, xb, loss
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            return int(bs), {"candidates": attempts, "selected": int(bs), "status": "completed_valid"}
        except (RuntimeError, MemoryError) as exc:
            attempts.append({"batch_size": int(bs), "status": "failed", "error": str(exc)[:300]})
            del model
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    raise RuntimeError(f"no batch size passed memory preflight: {attempts}")


@dataclass
class FitResult:
    embedding: np.ndarray
    history: list[dict[str, Any]]
    model_init_hash: str
    model_final_hash: str
    batch_schedule_hash: str
    mask_schedule_hash: str
    batch_size: int
    peak_gpu_memory_bytes: int


def _to_tensor(value: np.ndarray, device: Any) -> Any:
    import torch

    return torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)


def fit_autoencoder(
    x0: np.ndarray,
    active: np.ndarray,
    *,
    arm: str,
    seed: int,
    device: Any = "cpu",
    epochs: int | None = None,
    batch_size: int = 512,
    init_state: dict[str, Any] | None = None,
    fixed_ratio: float | None = None,
    loss_mode: str = "selected",
) -> FitResult:
    """Fit one frozen arm and return only in-memory embedding + compact audits."""
    import torch

    value = np.asarray(x0, dtype=np.float32)
    support = np.asarray(active, dtype=bool)
    if value.ndim != 2 or support.shape != value.shape:
        raise ValueError("x0/active shape mismatch")
    if arm not in protocol.ARMS and arm not in ("Z_FIXED", "Z_VARIABLE"):
        raise ValueError(f"arm is not in frozen MAIN matrix: {arm}")
    epochs = int(epochs if epochs is not None else protocol.BACKBONE["epochs"])
    batch_size = max(1, min(int(batch_size), value.shape[0]))
    seed_everything(seed)
    model = make_model(value.shape[1], seed, device)
    if init_state is not None:
        model.load_state_dict(init_state)
    initial_hash = state_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(protocol.BACKBONE["learning_rate"]), weight_decay=float(protocol.BACKBONE["weight_decay"]))
    orders = batch_orders(value.shape[0], epochs, batch_size, seed)
    histories: list[dict[str, Any]] = []
    mask_audits: list[dict[str, Any]] = []
    model.train()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for epoch, order in enumerate(orders):
        loss_sum = 0.0
        all_mse_sum = 0.0
        unchanged_mse_sum = 0.0
        seen = 0
        epoch_audits: list[dict[str, Any]] = []
        for start in range(0, value.shape[0], batch_size):
            idx = order[start : start + batch_size]
            clean = value[idx]
            support_batch = support[idx]
            if fixed_ratio is None:
                mask_batch = masking.mask_for_arm(clean, support_batch, arm, seed=seed, epoch=epoch, stream=start)
            else:
                target, schedule = masking.arm_to_spec(arm)
                mask_batch = None if target is None else masking.make_mask(clean, support_batch, target_space=target, schedule=schedule, seed=seed, epoch=epoch, stream=start, fixed_ratio=fixed_ratio)
            if mask_batch is None:
                corrupted = clean
                mask = np.ones(clean.shape, dtype=bool)
                audit = {
                    "target_space": "ALL_COORDINATES",
                    "schedule": "CLEAN",
                    "selected_mask_count_total": int(mask.size),
                    "selected_nonzero_count_total": int(np.count_nonzero(clean)),
                    "selected_nonzero_fraction": float(np.count_nonzero(clean) / max(mask.size, 1)),
                    "actual_value_change_count_total": 0,
                    "actual_value_change_fraction": 0.0,
                    "masked_target_zero_fraction": float(np.count_nonzero(clean == 0.0) / max(mask.size, 1)),
                    "zero_budget_rows": int(np.count_nonzero(support_batch.sum(axis=1) == 0)),
                    "mean_sampled_mask_ratio": 0.0,
                    "std_sampled_mask_ratio": 0.0,
                    "sampled_ratio_min": 0.0,
                    "sampled_ratio_max": 0.0,
                }
            else:
                corrupted = mask_batch.corrupted
                mask = mask_batch.mask
                audit = dict(mask_batch.audit)
                epoch_audits.append(audit)
                mask_audits.append(audit)
            xb = _to_tensor(corrupted, device)
            yb = _to_tensor(clean, device)
            mb = torch.from_numpy(np.asarray(mask, dtype=bool)).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            if arm == "CLEAN_AE" or loss_mode == "all":
                loss = torch.mean((prediction - yb) ** 2)
            else:
                loss = masking.masked_mse(prediction, yb, mb)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                all_mse = torch.mean((prediction - yb) ** 2)
                unchanged = ~mb
                unchanged_mse = torch.mean((prediction[unchanged] - yb[unchanged]) ** 2) if bool(unchanged.any()) else torch.tensor(0.0, device=device)
            count = int(idx.size)
            loss_sum += float(loss.detach().cpu()) * count
            all_mse_sum += float(all_mse.detach().cpu()) * count
            unchanged_mse_sum += float(unchanged_mse.detach().cpu()) * count
            seen += count
        row = {
            "epoch": int(epoch + 1),
            "loss": float(loss_sum / max(seen, 1)),
            "masked_mse": float(loss_sum / max(seen, 1)),
            "all_coordinate_mse": float(all_mse_sum / max(seen, 1)),
            "unchanged_coordinate_mse": float(unchanged_mse_sum / max(seen, 1)),
        }
        if arm == "CLEAN_AE":
            row.update({
                "requested_mask_count_total": 0,
                "selected_mask_count_total": 0,
                "selected_nonzero_count_total": 0,
                "selected_nonzero_fraction": 0.0,
                "actual_value_change_count_total": 0,
                "actual_value_change_fraction": 0.0,
                "masked_target_zero_fraction": 0.0,
                "zero_budget_rows": int(np.count_nonzero(support.sum(axis=1) == 0)),
                "mean_sampled_mask_ratio": 0.0,
                "std_sampled_mask_ratio": 0.0,
                "mask_count_exact": True,
                "sampled_ratio_min": 0.0,
                "sampled_ratio_max": 0.0,
            })
        else:
            totals = {key: 0.0 for key in ("requested_mask_count_total", "selected_mask_count_total", "selected_nonzero_count_total", "actual_value_change_count_total", "zero_budget_rows")}
            for audit in epoch_audits:
                for key in totals:
                    totals[key] += float(audit.get(key, 0.0))
            total_selected = max(totals["selected_mask_count_total"], 1.0)
            row.update(totals)
            row["selected_nonzero_fraction"] = totals["selected_nonzero_count_total"] / total_selected
            row["actual_value_change_fraction"] = totals["actual_value_change_count_total"] / max(float(value.size), 1.0)
            row["masked_target_zero_fraction"] = max(0.0, (totals["selected_mask_count_total"] - totals["selected_nonzero_count_total"]) / total_selected)
            row["mean_sampled_mask_ratio"] = float(np.mean([a["mean_sampled_mask_ratio"] for a in epoch_audits])) if epoch_audits else 0.0
            row["std_sampled_mask_ratio"] = float(np.mean([a["std_sampled_mask_ratio"] for a in epoch_audits])) if epoch_audits else 0.0
            row["mask_count_exact"] = bool(all(bool(a.get("mask_count_exact", False)) for a in epoch_audits))
            row["sampled_ratio_min"] = float(min((a.get("sampled_ratio_min", 0.0) for a in epoch_audits), default=0.0))
            row["sampled_ratio_max"] = float(max((a.get("sampled_ratio_max", 0.0) for a in epoch_audits), default=0.0))
        histories.append(row)
    encoder = encoder_from_model(model).to(device)
    model.eval()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, value.shape[0], max(batch_size, 1024)):
            embeddings.append(encoder(_to_tensor(value[start : start + max(batch_size, 1024)], device)).detach().cpu().numpy())
    embedding = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 32), dtype=np.float32)
    peak = 0
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        peak = int(torch.cuda.max_memory_allocated(device))
    return FitResult(
        embedding=embedding.astype(np.float32, copy=False),
        history=histories,
        model_init_hash=initial_hash,
        model_final_hash=state_hash(model.state_dict()),
        batch_schedule_hash=hash_orders(orders),
        mask_schedule_hash=hash_mask_audits(mask_audits),
        batch_size=int(batch_size),
        peak_gpu_memory_bytes=peak,
    )
