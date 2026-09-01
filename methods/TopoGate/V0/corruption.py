"""Unified vicinal corruption operator used by V0's F and T settings."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config import normalize_parameterization
from .graph import NeighborGraph


def compute_node_gate(
    graph: NeighborGraph,
    *,
    parameterization: str,
    alpha: float = 0.90,
    gate_min: float = 0.0,
    gate_max: float = 0.15,
    beta_mutual: float = 1.0,
    beta_snn: float = 1.0,
    beta_perturb: float = 2.0,
    beta_uncertainty: float = 1.0,
    uncertainty: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return node mixing gates and pseudo-loss weights.

    F has a fixed convex-combination coefficient, ``gate = 1 - alpha``, and
    intentionally returns unit sample weights.  T uses the analytic topology
    score from the retired RG implementation and normalizes its gate by the
    empirical maximum gate for the weighted pseudo-loss, matching the training
    path in :func:`make_pseudo_batch`.
    """

    canonical = normalize_parameterization(parameterization)
    n_samples, k = graph.indices.shape
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if float(gate_min) < 0.0 or float(gate_max) < float(gate_min):
        raise ValueError("gate_min/gate_max must satisfy 0 <= gate_min <= gate_max")

    if canonical == "fixed":
        gate = np.full(n_samples, 1.0 - float(alpha), dtype=np.float32)
        sample_weight = np.ones(n_samples, dtype=np.float32)
        perturb = np.zeros(n_samples, dtype=np.float32)
        summary = {
            "parameterization": "fixed",
            "gate_mode": "fixed",
            "alpha": float(alpha),
            "gate_min": float(1.0 - float(alpha)),
            "gate_max": float(1.0 - float(alpha)),
            "mean_node_gate": float(np.mean(gate)) if gate.size else 0.0,
            "min_node_gate": float(np.min(gate)) if gate.size else 0.0,
            "max_node_gate": float(np.max(gate)) if gate.size else 0.0,
            "fraction_gate_lt_0p01": float(np.mean(gate < 0.01)) if gate.size else 1.0,
            "fraction_gate_gt_90pct_max": float(np.mean(gate > 0.9 * max(gate.max(), 1e-8)))
            if gate.size
            else 0.0,
            "sample_weight_mode": "unit",
            "uncertainty_enabled": False,
            "mean_perturb_proxy": 0.0,
        }
        return gate, sample_weight, summary

    # T's gate is defined even when the graph has no edges; in that case the
    # only sensible corruption is zero, which also keeps pseudo loss disabled.
    if k == 0:
        gate = np.zeros(n_samples, dtype=np.float32)
        perturb = np.zeros(n_samples, dtype=np.float32)
    else:
        mutual_ratio = graph.mutual.mean(axis=1).astype(np.float32)
        snn_avg = graph.snn.mean(axis=1).astype(np.float32)
        perturb = (1.0 - np.sum(graph.probs * graph.similarity, axis=1)).astype(np.float32)
        if uncertainty is None:
            unc = np.zeros(n_samples, dtype=np.float32)
        else:
            unc = np.asarray(uncertainty, dtype=np.float32).reshape(-1)
            if unc.shape != (n_samples,):
                raise ValueError(f"uncertainty must have shape ({n_samples},), got {unc.shape}")
            if not np.all(np.isfinite(unc)):
                raise ValueError("uncertainty must be finite")
        logits = (
            float(beta_mutual) * mutual_ratio
            + float(beta_snn) * snn_avg
            - float(beta_perturb) * perturb
            - float(beta_uncertainty) * unc
        )
        # Clipping avoids overflow for explicitly large diagnostic coefficients.
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        gate = (
            float(gate_min) + (float(gate_max) - float(gate_min)) * sigmoid
        ).astype(np.float32)
    empirical_max_gate = float(np.max(gate)) if gate.size else 0.0
    sample_weight = np.clip(
        gate / max(empirical_max_gate, 1e-8), 0.0, 1.0
    ).astype(np.float32)
    summary = {
        "parameterization": "topology",
        "gate_mode": "topology",
        "alpha": float(alpha),
        "gate_min": float(gate_min),
        "gate_max": float(gate_max),
        "mean_node_gate": float(np.mean(gate)) if gate.size else 0.0,
        "min_node_gate": float(np.min(gate)) if gate.size else 0.0,
        "max_node_gate": float(np.max(gate)) if gate.size else 0.0,
        "fraction_gate_lt_0p01": float(np.mean(gate < 0.01)) if gate.size else 1.0,
        "fraction_gate_gt_90pct_max": float(np.mean(gate > 0.9 * float(gate_max)))
        if gate.size
        else 0.0,
        "sample_weight_mode": "gate_over_empirical_max",
        "sample_weight_normalizer": "max_observed_gate",
        "sample_weight_max_gate": empirical_max_gate,
        "uncertainty_enabled": bool(uncertainty is not None),
        "uncertainty_source": "disabled" if uncertainty is None else "unsupervised",
        "mean_perturb_proxy": float(np.mean(perturb)) if perturb.size else 0.0,
    }
    return gate, sample_weight, summary


def _row_and_probabilities(
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    cell: int,
    parameterization: str,
) -> tuple[np.ndarray, np.ndarray]:
    row = graph.indices[cell]
    if row.size == 0:
        return row, np.zeros(0, dtype=np.float32)
    probs = graph.probs[cell] if normalize_parameterization(parameterization) == "fixed" else edge_weights[cell]
    # NumPy's Generator.choice checks the probability sum tightly.  Keep this
    # local sampling representation in float64 so a float32 graph row whose sum
    # is off by a few ulps cannot make an otherwise valid batch fail.
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / np.clip(float(probs.sum()), 1e-12, None)
    return row, probs


def make_pseudo_batch(
    data_np: np.ndarray,
    batch_indices: np.ndarray,
    batch_x: torch.Tensor,
    *,
    parameterization: str,
    graph: NeighborGraph,
    edge_weights: np.ndarray,
    node_gate: np.ndarray,
    mix_neighbors: int,
    alpha: float = 0.90,
    rng: np.random.Generator,
    neighbor_estimator: str = "current",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Construct a detached pseudo-cell view for one batch.

    ``current`` reproduces both historical runners: sample neighbour positions
    from the row distribution and re-normalize the picked probabilities as the
    interpolation weights.  ``uniform_sample`` and ``full`` are explicit
    diagnostics retained for parity with the retired T runner.
    """

    canonical = normalize_parameterization(parameterization)
    if neighbor_estimator not in {"current", "uniform_sample", "full"}:
        raise ValueError(f"unknown neighbor_estimator: {neighbor_estimator!r}")
    data = np.ascontiguousarray(np.asarray(data_np, dtype=np.float32))
    indices = np.asarray(batch_indices, dtype=np.int64).reshape(-1)
    if batch_x.ndim != 2 or batch_x.shape[0] != indices.shape[0] or batch_x.shape[1] != data.shape[1]:
        raise ValueError("batch_x, batch_indices, and data_np have incompatible shapes")
    if np.any(indices < 0) or np.any(indices >= data.shape[0]):
        raise IndexError("batch_indices contains an out-of-range cell")
    if graph.indices.shape[0] != data.shape[0]:
        raise ValueError("graph and data_np must contain the same number of cells")
    if graph.indices.shape[1] == 0 or int(mix_neighbors) <= 0:
        unit = torch.ones(batch_x.shape[0], dtype=batch_x.dtype, device=batch_x.device)
        return batch_x.detach(), unit, {
            "parameterization": canonical,
            "mean_node_gate": 0.0,
            "mean_perturb_norm": 0.0,
            "fraction_zero_gate": 1.0,
            "neighbor_estimator": neighbor_estimator,
        }

    k = int(graph.indices.shape[1])
    m = max(1, min(int(mix_neighbors), k))
    if neighbor_estimator == "full":
        neighbor_mean = np.empty((indices.shape[0], data.shape[1]), dtype=np.float32)
        for position, cell_value in enumerate(indices):
            row, probs = _row_and_probabilities(graph, edge_weights, int(cell_value), canonical)
            neighbor_mean[position] = np.sum(data[row] * probs[:, None], axis=0).astype(np.float32)
    else:
        sampled = np.empty((indices.shape[0], m), dtype=np.int64)
        interpolation_weights = np.empty((indices.shape[0], m), dtype=np.float32)
        for position, cell_value in enumerate(indices):
            row, probs = _row_and_probabilities(graph, edge_weights, int(cell_value), canonical)
            choices = rng.choice(row.shape[0], size=m, replace=True, p=probs)
            sampled[position] = row[choices]
            picked = probs[choices]
            if neighbor_estimator == "current":
                interpolation_weights[position] = picked / max(float(picked.sum()), 1e-12)
            else:
                interpolation_weights[position] = 1.0 / float(m)
        neighbor_mean = np.sum(
            data[sampled] * interpolation_weights[:, :, None], axis=1
        ).astype(np.float32)

    gates = np.asarray(node_gate, dtype=np.float32).reshape(-1)
    if gates.shape != (data.shape[0],):
        raise ValueError(f"node_gate must have shape ({data.shape[0]},), got {gates.shape}")
    if canonical == "fixed":
        gate_batch = np.full(indices.shape[0], 1.0 - float(alpha), dtype=np.float32)
        sample_weight_np = np.ones(indices.shape[0], dtype=np.float32)
    else:
        gate_batch = gates[indices]
        sample_weight_np = np.clip(
            gate_batch / max(float(np.max(gates)) if gates.size else 1.0, 1e-8),
            0.0,
            1.0,
        ).astype(np.float32)
    anchor = data[indices]
    mixed = (1.0 - gate_batch[:, None]) * anchor + gate_batch[:, None] * neighbor_mean
    perturb = np.linalg.norm(neighbor_mean - anchor, axis=1) / (
        np.linalg.norm(anchor, axis=1) + 1e-6
    )
    pseudo = torch.as_tensor(mixed, dtype=batch_x.dtype, device=batch_x.device)
    sample_weight = torch.as_tensor(
        sample_weight_np, dtype=batch_x.dtype, device=batch_x.device
    )
    info = {
        "parameterization": canonical,
        "mean_node_gate": float(np.mean(gate_batch)) if gate_batch.size else 0.0,
        "mean_perturb_norm": float(np.mean(perturb)) if perturb.size else 0.0,
        "fraction_zero_gate": float(np.mean(gate_batch <= 0.0)) if gate_batch.size else 1.0,
        "neighbor_estimator": neighbor_estimator,
    }
    return pseudo.detach(), sample_weight, info


def apply_scmae_noise(
    x: torch.Tensor,
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the historical elementwise row-swap corruption and effective mask."""

    if x.ndim != 2:
        raise ValueError(f"x must be two-dimensional, got {tuple(x.shape)}")
    if not 0.0 <= float(mask_ratio) <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1]")
    selected = torch.rand(
        x.shape, dtype=torch.float32, device=x.device, generator=generator
    ) < float(mask_ratio)
    replacement = (
        x
        if x.shape[0] <= 1
        else x[torch.randperm(x.shape[0], device=x.device, generator=generator)]
    )
    corrupted = torch.where(selected, replacement, x)
    return corrupted, (corrupted != x).to(dtype=x.dtype)


__all__ = [
    "apply_scmae_noise",
    "compute_node_gate",
    "make_pseudo_batch",
]
