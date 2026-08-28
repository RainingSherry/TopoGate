"""Deterministic full-neighborhood aggregation for reliable graph mixing."""

from __future__ import annotations

import torch


def _resolve_anchor_indices(
    values: torch.Tensor,
    neighbor_indices: torch.Tensor,
    anchor_indices: torch.Tensor | None,
) -> torch.Tensor:
    num_anchors = int(neighbor_indices.shape[0])
    if anchor_indices is None:
        if num_anchors != values.shape[0]:
            raise ValueError(
                "anchor_indices is required when neighbor rows are a subset of values."
            )
        return torch.arange(num_anchors, device=values.device)
    resolved = anchor_indices.to(device=values.device, dtype=torch.long).reshape(-1)
    if resolved.shape[0] != num_anchors:
        raise ValueError("anchor_indices must have one entry per neighbor row.")
    return resolved


def aggregate_neighbors(
    values: torch.Tensor,
    neighbor_indices: torch.Tensor,
    edge_gates: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    anchor_indices: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate every valid neighbor exactly once using normalized gates.

    This is a deterministic full estimator.  No edge sampling or probability
    reweighting is performed.  When all gates of an anchor are closed, its
    aggregate falls back to the anchor value itself.

    Returns
    -------
    neighbor_mean:
        Tensor with shape ``[num_anchors, value_dim]``.
    normalized_weights:
        Per-edge weights with the same shape as ``edge_gates``.
    """

    if values.ndim != 2:
        raise ValueError("values must have shape [n_nodes, value_dim].")
    if neighbor_indices.ndim != 2 or edge_gates.shape != neighbor_indices.shape:
        raise ValueError("neighbor_indices and edge_gates must have the same 2D shape.")
    indices = neighbor_indices.to(device=values.device, dtype=torch.long)
    gates = edge_gates.to(device=values.device, dtype=values.dtype)
    if valid_mask is None:
        valid = indices >= 0
    else:
        if valid_mask.shape != indices.shape:
            raise ValueError("valid_mask must have the same shape as neighbor_indices.")
        valid = valid_mask.to(device=values.device, dtype=torch.bool) & (indices >= 0)
    if torch.any(indices[valid] >= values.shape[0]):
        raise ValueError("neighbor_indices contains a node id outside values.")
    anchors = _resolve_anchor_indices(values, indices, anchor_indices)
    safe_indices = indices.clamp_min(0)
    effective_gates = gates.clamp_min(0.0) * valid.to(dtype=values.dtype)
    denominator = effective_gates.sum(dim=1, keepdim=True)
    normalized = torch.where(
        denominator > float(epsilon),
        effective_gates / denominator.clamp_min(float(epsilon)),
        torch.zeros_like(effective_gates),
    )
    neighbor_values = values[safe_indices]
    neighbor_mean = (neighbor_values * normalized.unsqueeze(-1)).sum(dim=1)
    closed = denominator.squeeze(1) <= float(epsilon)
    neighbor_mean = torch.where(closed.unsqueeze(1), values[anchors], neighbor_mean)
    return neighbor_mean, normalized


# Explicit name retained for readers who want the estimator property in the API.
full_neighbor_aggregate = aggregate_neighbors


def mix_with_reliable_neighbors(
    values: torch.Tensor,
    neighbor_indices: torch.Tensor,
    edge_gates: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    anchor_indices: torch.Tensor | None = None,
    mix_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Create a conservative graph view using edge reliability as mix amount.

    The normalized gates select the neighbor message.  Their per-node mean
    controls how far the anchor moves toward that message, so closing every
    edge yields exact NoMix behavior without detached sample weights.
    """

    if not 0.0 <= float(mix_scale) <= 1.0:
        raise ValueError("mix_scale must be in [0, 1].")
    anchors = _resolve_anchor_indices(values, neighbor_indices, anchor_indices)
    neighbor_mean, normalized = aggregate_neighbors(
        values,
        neighbor_indices,
        edge_gates,
        valid_mask,
        anchor_indices=anchors,
    )
    if valid_mask is None:
        valid = neighbor_indices.to(device=values.device) >= 0
    else:
        valid = valid_mask.to(device=values.device, dtype=torch.bool) & (
            neighbor_indices.to(device=values.device) >= 0
        )
    valid_count = valid.sum(dim=1).clamp_min(1).to(dtype=values.dtype)
    gates = edge_gates.to(device=values.device, dtype=values.dtype).clamp(0.0, 1.0)
    node_strength = (gates * valid).sum(dim=1) / valid_count
    node_strength = (node_strength * float(mix_scale)).clamp(0.0, 1.0)
    self_weight = 1.0 - node_strength
    anchor_values = values[anchors]
    mixed = anchor_values + node_strength.unsqueeze(1) * (neighbor_mean - anchor_values)
    perturbation = torch.linalg.vector_norm(mixed - anchor_values, dim=1)
    anchor_norm = torch.linalg.vector_norm(anchor_values, dim=1).clamp_min(1e-8)
    info = {
        "normalized_weights": normalized,
        "node_strength": node_strength,
        "self_weight": self_weight,
        "effective_neighbor_count": (normalized > 1e-6).sum(dim=1),
        "gate_closed_fraction": ((gates <= 1e-4) & valid).sum().to(values.dtype)
        / valid.sum().clamp_min(1).to(values.dtype),
        "gate_open_fraction": ((gates >= 1.0 - 1e-4) & valid).sum().to(values.dtype)
        / valid.sum().clamp_min(1).to(values.dtype),
        "mean_node_strength": node_strength.mean() if node_strength.numel() else values.sum() * 0.0,
        "mean_relative_perturbation": (
            (perturbation / anchor_norm).mean() if perturbation.numel() else values.sum() * 0.0
        ),
    }
    return mixed, info


def gather_edge_assignments(
    assignments: torch.Tensor,
    neighbor_indices: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    anchor_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather source/destination assignment tensors for valid directed edges."""

    if assignments.ndim != 2:
        raise ValueError("assignments must have shape [n_nodes, n_clusters].")
    indices = neighbor_indices.to(device=assignments.device, dtype=torch.long)
    anchors = _resolve_anchor_indices(assignments, indices, anchor_indices)
    valid = indices >= 0 if valid_mask is None else valid_mask.to(assignments.device).bool() & (indices >= 0)
    source = assignments[anchors].unsqueeze(1).expand(-1, indices.shape[1], -1)
    destination = assignments[indices.clamp_min(0)]
    return source[valid], destination[valid], valid
