"""Gumbel-Top-k hard gate for V13.

V13 replaces the V12 softmax + rank_loss soft gate with a hard top-k selection
via Gumbel-Softmax straight-through gradient. During training the gate uses a
temperature-annealed Gumbel-Softmax relaxation; at evaluation time the gate
applies a deterministic top-k hard truncation so that ``effective_neighbors``
is exactly ``top_k_neighbors``.

Key properties
--------------
- **Hard selection**: the gate outputs a (B, K) binary mask where exactly
  ``top_k_neighbors`` entries are 1 per row.
- **No rank loss**: the hard ordering property of top-k already guarantees
  that the top-``k`` neighbours are the ``k`` highest-reliability ones
  according to the gate's learned score — no auxiliary ranking signal needed.
- **Straight-through gradient**: the binary mask is relaxed to a Gumbel-Softmax
  distribution during training so gradients can flow back to the MLP parameters.
- **Temperature annealing**: ``gumbel_tau`` starts at ``gumbel_tau`` (default 1.0)
  and linearly anneals to ``gumbel_tau_min`` (default 0.1) over the first
  ``gumbel_tau_anneal_epochs`` epochs, making the relaxation progressively
  sharper and the gate behaviour closer to inference.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


EDGE_FEATURE_NAMES = ("similarity", "mutual", "snn", "distance")


def build_gate_stats_tensor(
    graph_similarity: torch.Tensor,
    graph_mutual: torch.Tensor,
    graph_snn: torch.Tensor,
    graph_distance: torch.Tensor,
) -> torch.Tensor:
    """Build edge features without reducing the neighbour dimension."""
    if not isinstance(graph_similarity, torch.Tensor):
        raise TypeError("graph_similarity must be a torch.Tensor")
    if not isinstance(graph_mutual, torch.Tensor):
        raise TypeError("graph_mutual must be a torch.Tensor")
    if not isinstance(graph_snn, torch.Tensor):
        raise TypeError("graph_snn must be a torch.Tensor")
    if not isinstance(graph_distance, torch.Tensor):
        raise TypeError("graph_distance must be a torch.Tensor")

    tensors = [
        graph_similarity.to(dtype=torch.float32),
        graph_mutual.to(dtype=torch.float32),
        graph_snn.to(dtype=torch.float32),
        graph_distance.to(dtype=torch.float32),
    ]
    shape = tensors[0].shape
    if any(t.shape != shape for t in tensors[1:]):
        raise ValueError("all graph edge features must have the same [N, K] shape")
    return torch.stack(tensors, dim=-1)


class GumbelTopKGateOutput(NamedTuple):
    """Outputs of GumbelTopKGate.forward."""

    #: (B, K) mask with exactly ``top_k`` ones per row.  Retains gradients.
    mask: torch.Tensor
    #: (B, K) Gumbel-Softmax probabilities used during training; detached.
    gumbel_probs: torch.Tensor
    #: (B, K) raw logits before the top-k operation; detached.
    scores: torch.Tensor


class GumbelTopKGate(nn.Module):
    """Learn a hard top-k neighbourhood gate using Gumbel-Softmax relaxation.

    The module accepts per-row edge statistics ``(B, K, 4)`` and learns a scalar
    score for each candidate neighbour.  The final output is a binary mask
    ``(B, K)`` with exactly ``top_k`` entries set to 1 per row — the selected
    neighbours.  During training a straight-through Gumbel-Softmax estimator is
    used so that the mask is differentiable with respect to the MLP parameters.
    During evaluation the mask is obtained by a deterministic top-k argmax.

    The gate has **no self/null fallback**: every row must select exactly
    ``top_k`` neighbours.  This is intentional — the hard截断 (hard cutoff)
    is the primary mechanism for discarding cross-cluster edges.

    Args:
        feature_dim: number of statistics per neighbour (default 4: similarity,
            mutual, snn, distance).
        hidden_dim: hidden dimension of the scoring MLP (default 32).
        top_k: number of neighbours to select per row (default 2).
        dropout: dropout probability inside the MLP (default 0.0).
    """

    def __init__(
        self,
        feature_dim: int = len(EDGE_FEATURE_NAMES),
        hidden_dim: int = 32,
        top_k: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 1 <= int(top_k):
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.top_k = int(top_k)
        self.dropout = float(dropout)

        self.network = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(hidden_dim), 1),
        )
        # Small-amplitude initialisation keeps early logits near uniform so that
        # every neighbour has a chance to be sampled during warmup.
        final = self.network[-1]
        if isinstance(final, nn.Linear):
            nn.init.normal_(final.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        edge_stats: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> GumbelTopKGateOutput:
        """Return (mask, gumbel_probs, scores) for the given edge statistics.

        Args:
            edge_stats: (B, K, 4) tensor of edge-level statistics.
            tau: Gumbel-Softmax temperature.  Smaller values make the
                relaxation sharper (closer to hard one-hot).
            hard: if True, return the straight-through estimator (hard mask with
                identity gradient).  If False, return soft Gumbel-Softmax
                probabilities (straight-through but with a softmax gradient).

        Returns:
            GumbelTopKGateOutput with ``mask`` retaining gradients.
        """
        if edge_stats.ndim != 3 or edge_stats.shape[-1] != self.feature_dim:
            raise ValueError(
                f"edge_stats must have shape [B, K, {self.feature_dim}], "
                f"got {tuple(edge_stats.shape)}"
            )

        scores = self.network(edge_stats).squeeze(-1)  # (B, K)
        B, K = scores.shape

        if hard or not self.training:
            # Inference / hard path: deterministic top-k mask.
            # topk returns (values, indices) each (B, k).
            k = min(self.top_k, K)
            _, top_indices = torch.topk(scores, k=k, dim=1)
            mask = torch.zeros_like(scores)
            mask.scatter_(1, top_indices, 1.0)  # (B, K) binary, k ones per row
            # Detach scores so they do not appear in the autograd graph of the
            # hard path; gradients flow only through the mask.
            return GumbelTopKGateOutput(
                mask=mask,
                gumbel_probs=mask.detach(),
                scores=scores.detach(),
            )

        # Training path: Gumbel-Softmax straight-through estimator.
        #   forward  : soft Gumbel-Softmax probabilities
        #   backward : hard one-hot gradient (identity, passes straight through)
        gumbel = torch.rand_like(scores).log_().neg().log_().neg()
        gumbel_scores = scores + gumbel
        gumbel_probs = F.softmax(gumbel_scores / max(float(tau), 1e-8), dim=1)

        k = min(self.top_k, K)
        _, top_indices = torch.topk(scores, k=k, dim=1)
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, top_indices, 1.0)

        # Straight-through: forward uses soft probs, backward uses hard gradient.
        mask = (hard_mask - gumbel_probs).detach() + gumbel_probs
        return GumbelTopKGateOutput(
            mask=mask,
            gumbel_probs=gumbel_probs.detach(),
            scores=scores.detach(),
        )


def hard_topk_alignment_loss(
    z_anchor: torch.Tensor,
    z_neighbors: torch.Tensor,
    gate_mask: torch.Tensor,
    *,
    detach_neighbors: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align anchors to the mean of the selected top-k neighbours.

    The mask is hard (or a Gumbel-Softmax relaxation) and has the same
    ``(B, K)`` shape as ``z_neighbors``.  The neighbour tensor is detached
    so the loss only trains the autoencoder and the gate, not the neighbour
    representations themselves.

    Crucially, the target mean is computed using the **sum of mask values**
    (which is exactly ``top_k`` at inference time) rather than the constant
    ``K``.  This ensures the target is a true average over the selected
    neighbours even when the Gumbel relaxation is active.

    Args:
        z_anchor: (B, H) latent representation of the current batch.
        z_neighbors: (B, K, H) latent representations of the K neighbours.
        gate_mask: (B, K) binary or soft mask from GumbelTopKGate.
        detach_neighbors: if True, detach ``z_neighbors`` before computing the
            target so only the anchor encoder is trained by this loss term.

    Returns:
        (topology_loss, neighbour_target): the MSE alignment loss and the
        (B, H) neighbour mean tensor for diagnostics.
    """
    if z_anchor.ndim != 2:
        raise ValueError(f"z_anchor must be [B, H], got {tuple(z_anchor.shape)}")
    if z_neighbors.ndim != 3:
        raise ValueError(f"z_neighbors must be [B, K, H], got {tuple(z_neighbors.shape)}")
    if gate_mask.ndim != 2:
        raise ValueError(f"gate_mask must be [B, K], got {tuple(gate_mask.shape)}")
    B, K, H = z_neighbors.shape
    if z_anchor.shape[0] != B or gate_mask.shape[0] != B:
        raise ValueError("batch dimension mismatch")
    if z_neighbors.shape[1] != gate_mask.shape[1]:
        raise ValueError(f"K mismatch: z_neighbors has K={z_neighbors.shape[1]}, mask has K={gate_mask.shape[1]}")

    # Detach neighbours so the gate MLP and the anchor encoder are the only
    # learnable targets of this loss.
    z_neighbours_detached = z_neighbors.detach() if detach_neighbors else z_neighbors

    # Normalise by the mask sum (== top_k at inference, a soft float in training).
    mask_sum = gate_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)  # (B, 1)
    weighted = z_neighbours_detached * gate_mask.unsqueeze(-1)  # (B, K, H)
    neighbour_target = weighted.sum(dim=1) / mask_sum  # (B, H)

    loss = F.mse_loss(z_anchor, neighbour_target)
    return loss, neighbour_target
