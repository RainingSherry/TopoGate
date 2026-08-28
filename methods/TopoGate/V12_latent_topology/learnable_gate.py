"""Differentiable edge-level topology gate for V12."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


EDGE_FEATURE_NAMES = ("similarity", "mutual", "snn", "distance")


def _as_edge_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; NumPy conversion is not allowed")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [N, K], got {tuple(value.shape)}")
    return value.to(dtype=torch.float32)


def build_gate_stats_tensor(
    graph_similarity: torch.Tensor,
    graph_mutual: torch.Tensor,
    graph_snn: torch.Tensor,
    graph_distance: torch.Tensor,
) -> torch.Tensor:
    """Build edge features without reducing the neighbour dimension."""

    tensors = [
        _as_edge_tensor(graph_similarity, "graph_similarity"),
        _as_edge_tensor(graph_mutual, "graph_mutual"),
        _as_edge_tensor(graph_snn, "graph_snn"),
        _as_edge_tensor(graph_distance, "graph_distance"),
    ]
    shape = tensors[0].shape
    if any(value.shape != shape for value in tensors[1:]):
        raise ValueError("all graph edge features must have the same [N, K] shape")
    return torch.stack(tensors, dim=-1)


class LearnableGate(nn.Module):
    """Map candidate edges to differentiable self/null and edge weights.

    ``self_null`` is the V12 default.  Its softmax contains one learned
    self/null logit and the K edge logits, so unreliable rows can retain their
    anchor representation instead of being forced toward a neighbour mean.
    ``edge_only`` is retained as an explicit ablation and returns zero self
    mass with the edge logits normalized over K candidates.
    """

    def __init__(
        self,
        feature_dim: int = len(EDGE_FEATURE_NAMES),
        hidden_dim: int = 32,
        temperature: float = 1.0,
        dropout: float = 0.0,
        self_init_weight: float = 0.8,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < float(self_init_weight) < 1.0:
            raise ValueError("self_init_weight must be strictly between 0 and 1")
        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)
        self.self_init_weight = float(self_init_weight)
        self.network = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.self_network = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        final = self.network[-1]
        if isinstance(final, nn.Linear):
            nn.init.normal_(final.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(final.bias)
        self_final = self.self_network[-1]
        if isinstance(self_final, nn.Linear):
            # Start with roughly 80% self/null mass while keeping the branch
            # learnable from the first topology-loss update.
            nn.init.zeros_(self_final.weight)
            logit = torch.logit(torch.tensor(self.self_init_weight))
            nn.init.constant_(self_final.bias, float(logit))

    def logits(
        self, edge_stats: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if edge_stats.ndim != 3 or edge_stats.shape[-1] != self.feature_dim:
            raise ValueError(
                f"edge_stats must have shape [B, K, {self.feature_dim}], "
                f"got {tuple(edge_stats.shape)}"
            )
        values = self.network(edge_stats).squeeze(-1) / self.temperature
        if valid_mask is not None:
            if valid_mask.shape != values.shape:
                raise ValueError("valid_mask must have shape [B, K]")
            values = values.masked_fill(~valid_mask.to(dtype=torch.bool), -torch.inf)
        return values

    def self_logits(self, edge_stats: torch.Tensor) -> torch.Tensor:
        """Return one self/null logit per row without NumPy conversion."""

        if edge_stats.ndim != 3 or edge_stats.shape[-1] != self.feature_dim:
            raise ValueError(
                f"edge_stats must have shape [B, K, {self.feature_dim}], "
                f"got {tuple(edge_stats.shape)}"
            )
        if edge_stats.shape[1] == 0:
            # An empty graph has no edge features to pool.  A zero feature
            # vector keeps the self branch well-defined and differentiable.
            pooled = edge_stats.new_zeros((edge_stats.shape[0], self.feature_dim))
        else:
            pooled = edge_stats.mean(dim=1)
        return self.self_network(pooled).squeeze(-1) / self.temperature

    @staticmethod
    def _masked_edge_softmax(
        values: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if values.shape[1] == 0:
            return values
        if valid_mask is None:
            return F.softmax(values, dim=-1)
        valid = valid_mask.to(dtype=torch.bool, device=values.device)
        masked = values.masked_fill(~valid, -torch.inf)
        probabilities = torch.softmax(masked, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        denominator = probabilities.sum(dim=-1, keepdim=True)
        valid_count = valid.sum(dim=-1, keepdim=True)
        valid_fallback = valid.to(dtype=values.dtype) / valid_count.clamp_min(1.0)
        all_edge_fallback = torch.full_like(values, 1.0 / float(values.shape[1]))
        fallback = torch.where(valid_count > 0, valid_fallback, all_edge_fallback)
        return torch.where(denominator > 0.0, probabilities / denominator.clamp_min(1e-8), fallback)

    def forward(
        self,
        edge_stats: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        topology_mode: str = "self_null",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(self_weight, edge_weights)``.

        Both outputs retain gradients to the gate parameters.  In
        ``self_null`` mode the concatenated self/edge softmax sums to one. In
        ``edge_only`` mode self mass is exactly zero and the edge branch is
        normalized independently, preserving the historical ablation.
        """

        if topology_mode not in {"self_null", "edge_only"}:
            raise ValueError("topology_mode must be 'self_null' or 'edge_only'")

        values = self.logits(edge_stats, valid_mask=valid_mask)
        edge_weights = self._masked_edge_softmax(values, valid_mask)
        if topology_mode == "edge_only":
            return torch.zeros(values.shape[0], dtype=values.dtype, device=values.device), edge_weights

        self_values = self.self_logits(edge_stats)
        if values.shape[1] > 0:
            # ``self_init_weight`` is the desired mass against K initially
            # near-zero edge logits.  Add log(K) so the categorical softmax
            # does not divide that prior by the number of candidates.
            self_values = self_values + values.new_tensor(float(values.shape[1])).log()
        combined = torch.cat([self_values.unsqueeze(1), values], dim=1)
        if valid_mask is not None:
            combined_valid = torch.cat(
                [
                    torch.ones(
                        (combined.shape[0], 1), dtype=torch.bool, device=combined.device
                    ),
                    valid_mask.to(dtype=torch.bool, device=combined.device),
                ],
                dim=1,
            )
            combined = combined.masked_fill(
                ~combined_valid,
                -torch.inf,
            )
        probabilities = torch.softmax(combined, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        denominator = probabilities.sum(dim=-1, keepdim=True)
        # The self logit is always valid, so denominator is normally one. The
        # fallback is defensive for unusual empty/invalid tensors.
        probabilities = torch.where(
            denominator > 0.0,
            probabilities / denominator.clamp_min(1e-8),
            torch.cat(
                [
                    torch.ones((combined.shape[0], 1), dtype=combined.dtype, device=combined.device),
                    torch.zeros_like(values),
                ],
                dim=1,
            ),
        )
        return probabilities[:, 0], probabilities[:, 1:]

    @staticmethod
    def entropy(weights: torch.Tensor) -> torch.Tensor:
        if weights.ndim != 2:
            raise ValueError("weights must have shape [B, K]")
        return -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)


def topology_alignment_loss(
    z_anchor: torch.Tensor,
    z_neighbors: torch.Tensor,
    edge_weights: torch.Tensor,
    *,
    self_weight: torch.Tensor | None = None,
    z_self: torch.Tensor | None = None,
    detach_neighbors: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align anchors to a self/null + weighted-neighbour target.

    Clean target representations are detached, but self/edge weights are not,
    so the topology loss continues to train the gate.  Omitting ``self_weight``
    and ``z_self`` selects the historical edge-only target for ablations.
    """

    if z_anchor.ndim != 2 or z_neighbors.ndim != 3 or edge_weights.ndim != 2:
        raise ValueError("expected z_anchor=[B,H], z_neighbors=[B,K,H], weights=[B,K]")
    if z_neighbors.shape[:2] != edge_weights.shape or z_neighbors.shape[0] != z_anchor.shape[0]:
        raise ValueError("anchor, neighbour, and edge-weight shapes are inconsistent")
    values = z_neighbors.detach() if detach_neighbors else z_neighbors
    edge_target = (values * edge_weights.unsqueeze(-1)).sum(dim=1)
    if self_weight is None and z_self is None:
        target = edge_target
    elif self_weight is not None and z_self is not None:
        if self_weight.ndim != 1 or self_weight.shape[0] != z_anchor.shape[0]:
            raise ValueError("self_weight must have shape [B]")
        if z_self.shape != z_anchor.shape:
            raise ValueError("z_self must have the same shape as z_anchor")
        self_target = z_self.detach() if detach_neighbors else z_self
        target = self_weight.unsqueeze(-1) * self_target + edge_target
    else:
        raise ValueError("self_weight and z_self must be provided together")
    return F.mse_loss(z_anchor, target), target


def rank_alignment_loss(
    edge_weights: torch.Tensor,
    edge_reliability: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Per-row pairwise margin: edges with higher reliability should receive
    higher gate weight than edges with lower reliability, within each row.

    ``edge_reliability`` is detached; gradients flow back to gate parameters
    only. ``edge_weights`` is allowed to be a softmax distribution: the loss
    takes ``log(edge_weights + eps)`` so the gradient is well-defined even when
    the softmax outputs are already close to uniform.

    Args:
        edge_weights: (B, K) softmax-style edge weights with gradients.
        edge_reliability: (B, K) detached reliability target (e.g. sum of
            similarity + mutual + SNN per edge).
        margin: minimum log-weight gap required between higher-reliability
            edges and lower-reliability edges before the loss stops
            penalising them. A positive margin makes the loss only fire when
            the ranking is reversed by more than ``exp(margin)``.

    Returns:
        Scalar tensor; mean over rows of the pairwise hinge penalty.
    """

    if edge_weights.ndim != 2 or edge_reliability.ndim != 2:
        raise ValueError(
            "edge_weights and edge_reliability must be 2-D tensors with shape [B, K]"
        )
    if edge_weights.shape != edge_reliability.shape:
        raise ValueError(
            f"edge_weights shape {tuple(edge_weights.shape)} must match "
            f"edge_reliability shape {tuple(edge_reliability.shape)}"
        )
    if float(margin) < 0.0:
        raise ValueError("margin must be non-negative")

    reliability = edge_reliability.detach()
    # Operating in log-space gives a non-vanishing gradient even when
    # softmax outputs are nearly uniform: d log(w_i) / d logit_i = 1.
    log_weights = (edge_weights + 1e-12).log()
    diff_reliability = reliability.unsqueeze(2) - reliability.unsqueeze(1)
    diff_log_weight = log_weights.unsqueeze(2) - log_weights.unsqueeze(1)
    # Penalise pairs where a higher-reliability edge received a lower log
    # weight than a lower-reliability edge by more than ``margin``.
    violation = torch.clamp(float(margin) - diff_log_weight, min=0.0)
    # Suppress pairs whose reliability is equal (the gate is allowed to break
    # ties) and pairs whose ranking is already at least ``margin`` wider.
    active = (diff_reliability > 0.0).to(dtype=violation.dtype)
    penalty = (violation * active).sum(dim=(1, 2))
    pair_count = active.sum(dim=(1, 2)).clamp_min(1.0)
    return (penalty / pair_count).mean()

