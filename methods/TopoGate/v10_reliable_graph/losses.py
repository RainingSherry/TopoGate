"""Objective terms that align V10 representation learning with clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


Reduction = Literal["none", "mean", "sum"]


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _probabilities(values: torch.Tensor, from_logits: bool, epsilon: float) -> torch.Tensor:
    if from_logits:
        return torch.softmax(values, dim=-1)
    if torch.any(values < 0):
        raise ValueError("Probabilities must be non-negative; pass from_logits=True for logits.")
    return values / values.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    masked_weight: float = 1.0,
    visible_weight: float = 0.1,
    sample_weight: torch.Tensor | None = None,
    kind: Literal["mse", "huber"] = "mse",
) -> torch.Tensor:
    """Compute a position- and optionally sample-weighted reconstruction loss."""

    if reconstruction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("reconstruction, target, and mask must have identical shapes.")
    if reconstruction.ndim != 2:
        raise ValueError("Reconstruction tensors must have shape [batch, features].")
    if float(masked_weight) < 0 or float(visible_weight) < 0:
        raise ValueError("masked_weight and visible_weight must be non-negative.")
    if kind == "mse":
        element_loss = (reconstruction - target).square()
    elif kind == "huber":
        element_loss = F.smooth_l1_loss(reconstruction, target, reduction="none")
    else:
        raise ValueError(f"Unknown reconstruction loss kind: {kind!r}.")
    binary_mask = mask.to(dtype=reconstruction.dtype).clamp(0.0, 1.0)
    weights = binary_mask * float(masked_weight) + (1.0 - binary_mask) * float(visible_weight)
    if sample_weight is not None:
        samples = sample_weight.to(device=reconstruction.device, dtype=reconstruction.dtype)
        if samples.ndim == 1:
            samples = samples.unsqueeze(1)
        if samples.shape not in {(reconstruction.shape[0], 1), reconstruction.shape}:
            raise ValueError("sample_weight must have shape [batch], [batch, 1], or [batch, features].")
        weights = weights * samples
    denominator = weights.sum()
    return (element_loss * weights).sum() / denominator.clamp_min(1e-8)


def view_consistency_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    kind: Literal["cosine", "mse"] = "cosine",
    stop_gradient: bool = False,
) -> torch.Tensor:
    """Align two independently corrupted views of the same samples."""

    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("z1 and z2 must have the same [batch, latent_dim] shape.")
    right = z2.detach() if stop_gradient else z2
    if kind == "cosine":
        left_norm = F.normalize(z1, dim=1)
        right_norm = F.normalize(right, dim=1)
        return (1.0 - (left_norm * right_norm).sum(dim=1)).mean()
    if kind == "mse":
        return F.mse_loss(z1, right)
    raise ValueError(f"Unknown view consistency kind: {kind!r}.")


def edge_assignment_js_loss(
    q_src: torch.Tensor,
    q_dst_or_indices: torch.Tensor,
    edge_gates: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    *,
    reduction: Reduction = "mean",
    from_logits: bool = False,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Gate-weighted Jensen-Shannon divergence of neighboring assignments.

    Two calling conventions are supported:

    ``edge_assignment_js_loss(q_src, q_dst, gates)`` for flattened/batched
    edge pairs, and ``edge_assignment_js_loss(q_all, indices, gates, mask)``
    for a complete fixed-width graph.
    """

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unknown reduction: {reduction!r}.")
    graph_form = q_dst_or_indices.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if graph_form:
        if q_src.ndim != 2 or q_dst_or_indices.ndim != 2:
            raise ValueError("Graph form requires q_all [n, c] and indices [n, k].")
        indices = q_dst_or_indices.to(device=q_src.device, dtype=torch.long)
        if indices.shape[0] != q_src.shape[0]:
            raise ValueError("indices and q_all must have the same node dimension.")
        valid = indices >= 0
        if valid_mask is not None:
            if valid_mask.shape != indices.shape:
                raise ValueError("valid_mask must have the same shape as indices.")
            valid = valid & valid_mask.to(device=q_src.device, dtype=torch.bool)
        safe_indices = indices.clamp_min(0)
        source = q_src.unsqueeze(1).expand(-1, indices.shape[1], -1)
        destination = q_src[safe_indices]
        if edge_gates is None:
            gates = valid.to(dtype=q_src.dtype)
        else:
            if edge_gates.shape != indices.shape:
                raise ValueError("edge_gates must have the same shape as indices.")
            gates = edge_gates.to(device=q_src.device, dtype=q_src.dtype) * valid
    else:
        destination = q_dst_or_indices
        source = q_src
        if source.shape != destination.shape or source.ndim < 2:
            raise ValueError("Pair form requires q_src and q_dst with identical [..., c] shape.")
        edge_shape = source.shape[:-1]
        valid = torch.ones(edge_shape, device=source.device, dtype=torch.bool)
        if valid_mask is not None:
            if valid_mask.shape != edge_shape:
                raise ValueError("valid_mask must match q_src leading dimensions.")
            valid = valid & valid_mask.to(device=source.device, dtype=torch.bool)
        if edge_gates is None:
            gates = valid.to(dtype=source.dtype)
        else:
            if edge_gates.shape != edge_shape:
                raise ValueError("edge_gates must match q_src leading dimensions.")
            gates = edge_gates.to(device=source.device, dtype=source.dtype) * valid

    p = _probabilities(source, from_logits, epsilon).clamp_min(epsilon)
    q = _probabilities(destination, from_logits, epsilon).clamp_min(epsilon)
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * (p * (p.log() - midpoint.log())).sum(dim=-1)
    divergence = divergence + 0.5 * (q * (q.log() - midpoint.log())).sum(dim=-1)
    weighted = divergence * gates
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    return weighted.sum() / gates.sum().clamp_min(epsilon)


def entropy_balance_loss(
    assignments: torch.Tensor,
    *,
    from_logits: bool = False,
    prior: torch.Tensor | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Promote confident samples while matching a declared cluster prior.

    With ``prior=None`` this retains the historical uniform-balance form.  A
    non-uniform prior lets the runner preserve cluster-size evidence obtained
    without labels from the warmup KMeans partition.
    """

    if assignments.ndim != 2:
        raise ValueError("assignments must have shape [n_samples, n_clusters].")
    probabilities = _probabilities(assignments, from_logits, epsilon).clamp_min(epsilon)
    sample_entropy = -(probabilities * probabilities.log()).sum(dim=1).mean()
    marginal = probabilities.mean(dim=0)
    if prior is not None:
        target = prior.to(device=assignments.device, dtype=assignments.dtype).reshape(-1)
        if target.shape[0] != assignments.shape[1] or torch.any(target < 0):
            raise ValueError("prior must be a non-negative vector with one value per cluster.")
        target = target / target.sum().clamp_min(epsilon)
        marginal_kl = (
            marginal.clamp_min(epsilon)
            * (marginal.clamp_min(epsilon).log() - target.clamp_min(epsilon).log())
        ).sum()
        return sample_entropy + marginal_kl
    marginal_entropy = -(marginal * marginal.clamp_min(epsilon).log()).sum()
    return sample_entropy - marginal_entropy


def gate_budget_loss(
    gates: torch.Tensor,
    target: float,
    valid_mask: torch.Tensor | None = None,
    *,
    mode: Literal["upper_bound", "target"] = "upper_bound",
) -> torch.Tensor:
    """Constrain the mean edge openness without forcing graph use.

    ``upper_bound`` is the V10 default: it penalizes only gate means above the
    declared budget, so a dataset can legitimately reject all graph edges.
    ``target`` is retained for controlled ablations that deliberately require
    a fixed mean openness.
    """

    if not 0.0 <= float(target) <= 1.0:
        raise ValueError("target must be in [0, 1].")
    if valid_mask is None:
        values = gates.reshape(-1)
    else:
        if valid_mask.shape != gates.shape:
            raise ValueError("valid_mask and gates must have identical shapes.")
        values = gates[valid_mask]
    if values.numel() == 0:
        return _zero(gates)
    mean = values.mean()
    if mode == "upper_bound":
        return F.relu(mean - float(target)).square()
    if mode == "target":
        return (mean - float(target)).square()
    raise ValueError(f"Unknown gate budget mode: {mode!r}.")


def gate_stability_loss(
    gates: torch.Tensor,
    stability: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Use an independently computed edge recurrence target for the gate."""

    if gates.shape != stability.shape:
        raise ValueError("gates and stability must have identical shapes.")
    valid = (
        torch.ones_like(gates, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(device=gates.device, dtype=torch.bool)
    )
    if valid.shape != gates.shape:
        raise ValueError("valid_mask and gates must have identical shapes.")
    if not torch.any(valid):
        return _zero(gates)
    prediction = gates[valid].clamp(epsilon, 1.0 - epsilon)
    target = stability.to(device=gates.device, dtype=gates.dtype)[valid].clamp(0.0, 1.0)
    return F.binary_cross_entropy(prediction, target)


def gate_regularization(
    gates: torch.Tensor,
    stability: torch.Tensor,
    *,
    budget_target: float = 0.25,
    valid_mask: torch.Tensor | None = None,
    budget_weight: float = 1.0,
    stability_weight: float = 1.0,
    budget_mode: Literal["upper_bound", "target"] = "upper_bound",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine interpretable edge-budget and stability regularizers."""

    budget = gate_budget_loss(gates, budget_target, valid_mask, mode=budget_mode)
    stable = gate_stability_loss(gates, stability, valid_mask)
    total = float(budget_weight) * budget + float(stability_weight) * stable
    return total, {"gate_budget": budget, "gate_stability": stable, "gate_regularization": total}


@dataclass(frozen=True, slots=True)
class V10LossWeights:
    """Weights for the complete V10 training objective."""

    reconstruction: float = 1.0
    view_consistency: float = 0.1
    edge_assignment: float = 0.2
    entropy_balance: float = 0.05
    gate_budget: float = 0.01
    gate_temporal: float = 0.05


def combine_v10_losses(
    parts: dict[str, torch.Tensor],
    weights: V10LossWeights,
    *,
    graph_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Combine the canonical V10 terms with one graph schedule application."""

    graph = (
        weights.edge_assignment * parts["edge_assignment"]
        + weights.entropy_balance * parts["entropy_balance"]
        + weights.gate_budget * parts["gate_budget"]
        + weights.gate_temporal * parts["gate_temporal"]
    )
    return (
        weights.reconstruction * parts["reconstruction"]
        + weights.view_consistency * parts["view_consistency"]
        + graph_scale * graph
    )


class V10Objective(nn.Module):
    """Compose reconstruction, view, clustering, and reliable-edge losses."""

    def __init__(
        self,
        weights: V10LossWeights | None = None,
        *,
        budget_target: float = 0.25,
        masked_weight: float = 1.0,
        visible_weight: float = 0.1,
        reconstruction_kind: Literal["mse", "huber"] = "mse",
    ) -> None:
        super().__init__()
        self.weights = weights or V10LossWeights()
        self.budget_target = float(budget_target)
        self.masked_weight = float(masked_weight)
        self.visible_weight = float(visible_weight)
        self.reconstruction_kind = reconstruction_kind

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        z_view1: torch.Tensor,
        z_view2: torch.Tensor,
        assignments: torch.Tensor,
        neighbor_indices: torch.Tensor,
        edge_gates: torch.Tensor,
        temporal_target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        graph_scale: float | torch.Tensor = 1.0,
        cluster_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate the complete objective and return named scalar components."""

        parts = {
            "reconstruction": masked_reconstruction_loss(
                reconstruction,
                target,
                mask,
                masked_weight=self.masked_weight,
                visible_weight=self.visible_weight,
                kind=self.reconstruction_kind,
            ),
            "view_consistency": view_consistency_loss(z_view1, z_view2),
            "edge_assignment": edge_assignment_js_loss(
                assignments,
                neighbor_indices,
                edge_gates,
                valid_mask,
            ),
            "entropy_balance": entropy_balance_loss(assignments, prior=cluster_prior),
            "gate_budget": gate_budget_loss(edge_gates, self.budget_target, valid_mask),
            "gate_temporal": gate_stability_loss(edge_gates, temporal_target, valid_mask),
        }
        total = combine_v10_losses(parts, self.weights, graph_scale=graph_scale)
        parts["total"] = total
        return total, parts
