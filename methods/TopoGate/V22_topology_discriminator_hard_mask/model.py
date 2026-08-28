from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional

from methods.NeighborMix_scMAE.model import AutoEncoder


class V22AutoEncoder(AutoEncoder):
    """The project scMAE backbone with an explicit encoder/reconstruction API."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check_expression_shape(x, "x")
        return self.encoder(x)

    def reconstruction_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.to(dtype=reconstruction.dtype, device=reconstruction.device)
        target = target.to(dtype=reconstruction.dtype, device=reconstruction.device)
        raw_mse = functional.mse_loss(reconstruction, target, reduction="none")
        weights = mask * self.masked_data_weight + (1.0 - mask) * (1.0 - self.masked_data_weight)
        return (1.0 - self.mask_loss_weight) * (weights * raw_mse).mean()

    def loss_encoder(
        self,
        corrupted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        latent, mask_logits, reconstruction = self.forward_mask(corrupted)
        rec_loss = self.reconstruction_loss(reconstruction, target, mask)
        mask_target = mask.to(dtype=mask_logits.dtype, device=mask_logits.device)
        mask_loss = self.mask_loss_weight * functional.binary_cross_entropy_with_logits(mask_logits, mask_target)
        total = rec_loss + mask_loss
        return latent, {
            "reconstruction_loss": rec_loss.detach(),
            "mask_loss": mask_loss.detach(),
            "total_loss": total.detach(),
            "mask_positive_rate": mask_target.mean().detach(),
            "loss": total,
        }


class CoordinateDiscriminator(nn.Module):
    """A shared coordinate discriminator for matched real/fake candidates.

    The API deliberately accepts only context, feature coordinates, topology
    context, and the candidate value.  It never receives the source mask or a
    Hint-presence bit, so real and fake pairs cannot be separated by a mask
    shortcut.
    """

    def __init__(
        self,
        n_features: int,
        context_dim: int,
        hidden_size: int = 96,
        coordinate_embedding_dim: int = 16,
        topology_dim: int = 4,
    ) -> None:
        super().__init__()
        if min(n_features, context_dim, hidden_size, coordinate_embedding_dim, topology_dim) <= 0:
            raise ValueError("discriminator dimensions must be positive")
        self.n_features = int(n_features)
        self.context_dim = int(context_dim)
        self.topology_dim = int(topology_dim)
        self.coordinate_embedding = nn.Embedding(self.n_features, int(coordinate_embedding_dim))
        input_dim = self.context_dim + int(coordinate_embedding_dim) + self.topology_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(
        self,
        context: torch.Tensor,
        feature_indices: torch.Tensor,
        topology_context: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape [n_pairs, context_dim]")
        if feature_indices.ndim != 1 or feature_indices.shape[0] != context.shape[0]:
            raise ValueError("feature_indices must have shape [n_pairs]")
        if topology_context.ndim != 2 or topology_context.shape != (context.shape[0], self.topology_dim):
            raise ValueError("topology_context must have shape [n_pairs, topology_dim]")
        values = values.reshape(-1)
        if values.shape[0] != context.shape[0]:
            raise ValueError("values must have one scalar per pair")
        indices = feature_indices.to(dtype=torch.long, device=context.device)
        if bool((indices < 0).any()) or bool((indices >= self.n_features).any()):
            raise ValueError("feature index is outside discriminator coordinate range")
        coordinate = self.coordinate_embedding(indices)
        inputs = torch.cat(
            [context, coordinate, topology_context.to(dtype=context.dtype), values[:, None].to(dtype=context.dtype)],
            dim=1,
        )
        return self.net(inputs).squeeze(1)


class CoordinateGate(nn.Module):
    """Shared feature gate with optional topology statistics.

    When topology is disabled, callers pass zeros for the four statistics; the
    learned coordinate embedding remains available as the non-topology control.
    """

    def __init__(self, n_features: int, hidden_size: int = 64, coordinate_embedding_dim: int = 16) -> None:
        super().__init__()
        if min(n_features, hidden_size, coordinate_embedding_dim) <= 0:
            raise ValueError("gate dimensions must be positive")
        self.n_features = int(n_features)
        self.coordinate_embedding = nn.Embedding(self.n_features, int(coordinate_embedding_dim))
        self.net = nn.Sequential(
            nn.Linear(int(coordinate_embedding_dim) + 4, int(hidden_size)),
            nn.LayerNorm(int(hidden_size)),
            nn.GELU(),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(self, topology_stats: torch.Tensor) -> torch.Tensor:
        if topology_stats.ndim != 3 or topology_stats.shape[2] != 4:
            raise ValueError("topology_stats must have shape [batch, features, 4]")
        if topology_stats.shape[1] != self.n_features:
            raise ValueError("topology_stats feature count does not match gate")
        batch = topology_stats.shape[0]
        feature_ids = torch.arange(self.n_features, device=topology_stats.device)
        coordinates = self.coordinate_embedding(feature_ids)[None, :, :].expand(batch, -1, -1)
        inputs = torch.cat([coordinates, topology_stats.to(dtype=coordinates.dtype)], dim=2)
        return self.net(inputs).squeeze(2)


def random_topk_mask(
    shape: tuple[int, int],
    mask_ratio: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if not 0.0 <= float(mask_ratio) <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1]")
    if float(mask_ratio) == 0.0:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    budget = max(1, min(shape[1], int(round(float(mask_ratio) * shape[1]))))
    scores = torch.rand(shape, device=device, generator=generator)
    indices = torch.topk(scores, k=budget, dim=1, largest=True, sorted=False).indices
    return torch.zeros_like(scores).scatter(1, indices, 1.0)


def cyclic_donor(x: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    if x.shape[0] <= 1:
        return x.clone()
    offset = int(torch.randint(1, x.shape[0], (1,), device=x.device, generator=generator).item())
    return torch.roll(x, shifts=offset, dims=0)


def straight_through_topk(
    logits: torch.Tensor,
    eligible: torch.Tensor,
    mask_ratio: float,
    *,
    generator: torch.Generator,
    gumbel_scale: float = 1.0,
    tau: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select an exact per-row budget, with a differentiable soft backward path."""
    if logits.ndim != 2 or eligible.shape != logits.shape:
        raise ValueError("logits and eligible must be matching 2D tensors")
    if not 0.0 < float(mask_ratio) <= 1.0 or tau <= 0.0:
        raise ValueError("mask_ratio must be in (0, 1] and tau must be positive")
    eligible_bool = eligible.to(dtype=torch.bool, device=logits.device)
    counts = eligible_bool.sum(dim=1)
    budgets = torch.ceil(counts.to(dtype=logits.dtype) * float(mask_ratio)).to(torch.long)
    budgets = torch.minimum(budgets, counts)
    max_budget = int(budgets.max().item()) if budgets.numel() else 0
    if max_budget == 0:
        zero = logits * 0.0
        return zero, torch.zeros_like(logits), budgets
    uniform = torch.rand(logits.shape, device=logits.device, generator=generator, dtype=logits.dtype).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    noisy = logits + float(gumbel_scale) * gumbel
    masked_noisy = noisy.masked_fill(~eligible_bool, -torch.inf)
    top_values, top_indices = torch.topk(masked_noisy, k=max_budget, dim=1, largest=True, sorted=True)
    ranks = torch.arange(max_budget, device=logits.device)[None, :] < budgets[:, None]
    hard = torch.zeros_like(logits).scatter(1, top_indices, ranks.to(dtype=logits.dtype))
    hard = hard * eligible_bool.to(dtype=logits.dtype)
    threshold = top_values.gather(1, (budgets - 1).clamp_min(0)[:, None])
    valid_rows = budgets.gt(0)[:, None]
    soft = torch.sigmoid((noisy - threshold.detach()) / float(tau))
    soft = soft * eligible_bool.to(dtype=logits.dtype) * valid_rows.to(dtype=logits.dtype)
    return hard + soft - soft.detach(), hard, budgets


def coverage_concentration(mask: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    rates = mask.mean(dim=0)
    active = eligible.to(torch.bool).any(dim=0)
    if not bool(active.any()):
        return rates.sum() * 0.0
    selected = rates[active]
    return (selected - selected.mean()).square().mean()
