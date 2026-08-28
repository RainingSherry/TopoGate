from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.cluster import KMeans

from methods.NeighborMix_scMAE.model import AutoEncoder


class V21AutoEncoder(AutoEncoder):
    """The unchanged scMAE backbone plus a gradient-enabled encoder API."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check_expression_shape(x, "x")
        return self.encoder(x)

    def reconstruction_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        effective_mask: torch.Tensor,
    ) -> torch.Tensor:
        effective_mask = effective_mask.to(dtype=reconstruction.dtype, device=reconstruction.device)
        target = target.to(dtype=reconstruction.dtype, device=reconstruction.device)
        raw_mse = functional.mse_loss(reconstruction, target, reduction="none")
        weights = effective_mask * self.masked_data_weight + (1.0 - effective_mask) * (1.0 - self.masked_data_weight)
        return (1.0 - self.mask_loss_weight) * (weights * raw_mse).mean()

    def loss_encoder(
        self,
        corrupted: torch.Tensor,
        target: torch.Tensor,
        effective_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        latent, mask_logits, reconstruction = self.forward_mask(corrupted)
        rec_loss = self.reconstruction_loss(reconstruction, target, effective_mask)
        mask_target = effective_mask.to(dtype=mask_logits.dtype, device=mask_logits.device)
        mask_loss = self.mask_loss_weight * functional.binary_cross_entropy_with_logits(mask_logits, mask_target)
        total = rec_loss + mask_loss
        return latent, {
            "reconstruction_loss": rec_loss.detach(),
            "mask_loss": mask_loss.detach(),
            "total_loss": total.detach(),
            "mask_positive_rate": mask_target.mean().detach(),
            "loss": total,
        }


class StudentTClusterHead(nn.Module):
    """DEC-style Student-t soft assignments with K-means initialisation."""

    def __init__(
        self,
        n_clusters: int,
        latent_dim: int,
        alpha: float = 1.0,
        distance_reduction: str = "mean",
    ) -> None:
        super().__init__()
        if n_clusters <= 1 or latent_dim <= 0 or alpha <= 0.0:
            raise ValueError("n_clusters > 1, latent_dim > 0, and alpha > 0 are required")
        self.n_clusters = int(n_clusters)
        self.latent_dim = int(latent_dim)
        self.alpha = float(alpha)
        if distance_reduction not in {"mean", "sum"}:
            raise ValueError("distance_reduction must be mean or sum")
        self.distance_reduction = distance_reduction
        self.centres = nn.Parameter(torch.randn(self.n_clusters, self.latent_dim) * 0.02)
        self.register_buffer("initialised", torch.tensor(False))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"z must have shape [batch, {self.latent_dim}]")
        squared = (z[:, None, :] - self.centres[None, :, :]).square()
        distance = squared.sum(dim=2) if self.distance_reduction == "sum" else squared.mean(dim=2)
        numerator = torch.pow(1.0 + distance / self.alpha, -0.5 * (self.alpha + 1.0))
        return numerator / numerator.sum(dim=1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def initialise(self, embeddings: np.ndarray, *, seed: int, n_init: int) -> None:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.latent_dim:
            raise ValueError("embedding shape does not match cluster head")
        if values.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be at least n_clusters")
        km = KMeans(n_clusters=self.n_clusters, n_init=int(n_init), random_state=int(seed))
        km.fit(values)
        centres = torch.as_tensor(km.cluster_centers_, dtype=self.centres.dtype, device=self.centres.device)
        self.centres.copy_(centres)
        self.initialised.fill_(True)


class FeatureGate(nn.Module):
    """One shared 2 -> hidden -> 1 MLP applied independently per feature."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, int(hidden_size)), nn.ReLU(), nn.Linear(int(hidden_size), 1))

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        if stats.ndim < 2 or stats.shape[-1] != 2:
            raise ValueError(f"stats must have shape [..., 2], got {tuple(stats.shape)}")
        return self.net(stats).squeeze(-1)


def random_bernoulli_mask(
    shape: tuple[int, int],
    mask_ratio: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if not 0.0 <= float(mask_ratio) <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1]")
    return (torch.rand(shape, device=device, generator=generator) < float(mask_ratio)).to(torch.float32)


def random_topk_mask(
    shape: tuple[int, int],
    k: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    k = max(1, min(int(k), int(shape[1])))
    scores = torch.rand(shape, device=device, generator=generator)
    indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices
    return torch.zeros(shape, dtype=scores.dtype, device=device).scatter(1, indices, 1.0)


def cyclic_donor(x: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    if x.shape[0] <= 1:
        return x.clone()
    offset = int(torch.randint(1, x.shape[0], (1,), device=x.device, generator=generator).item())
    return torch.roll(x, shifts=offset, dims=0)


def straight_through_changeable_topk(
    logits: torch.Tensor,
    eligible: torch.Tensor,
    mask_ratio: float,
    *,
    generator: torch.Generator,
    gumbel_scale: float,
    tau_ste: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select an exact ratio of donor-different positions in every row.

    The hard forward pass never selects an ineligible position. The soft
    backward path uses a row-specific threshold so Gate gradients remain
    available when the number of changeable features differs by sample.
    """

    if logits.ndim != 2 or eligible.shape != logits.shape:
        raise ValueError("logits and eligible must be matching 2D tensors")
    if not 0.0 < float(mask_ratio) <= 1.0 or tau_ste <= 0.0:
        raise ValueError("mask_ratio must be in (0, 1] and tau_ste must be positive")
    eligible_bool = eligible.to(dtype=torch.bool, device=logits.device)
    counts = eligible_bool.sum(dim=1)
    budgets = torch.ceil(counts.to(dtype=logits.dtype) * float(mask_ratio)).to(torch.long)
    budgets = torch.minimum(budgets, counts)
    max_budget = int(budgets.max().item()) if budgets.numel() else 0
    if max_budget == 0:
        zero = logits * 0.0
        return zero, torch.zeros_like(logits), budgets

    uniform = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device, generator=generator).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    noisy = logits + float(gumbel_scale) * gumbel
    masked_noisy = noisy.masked_fill(~eligible_bool, -torch.inf)
    top_values, top_indices = torch.topk(masked_noisy, k=max_budget, dim=1, largest=True, sorted=True)
    ranks = torch.arange(max_budget, device=logits.device)[None, :] < budgets[:, None]
    hard = torch.zeros_like(logits).scatter(1, top_indices, ranks.to(dtype=logits.dtype))
    hard = hard * eligible_bool.to(dtype=logits.dtype)

    threshold_indices = (budgets - 1).clamp_min(0)[:, None]
    threshold = top_values.gather(1, threshold_indices)
    valid_rows = budgets.gt(0)[:, None]
    soft = torch.sigmoid((noisy - threshold.detach()) / float(tau_ste))
    soft = soft * eligible_bool.to(dtype=logits.dtype) * valid_rows.to(dtype=logits.dtype)
    return hard + soft - soft.detach(), hard, budgets


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError("p and q must be matching [batch, clusters] tensors")
    p_safe = p.clamp_min(1e-8)
    q_safe = q.clamp_min(1e-8)
    midpoint = 0.5 * (p_safe + q_safe)
    kl_p = torch.sum(p_safe * (torch.log(p_safe) - torch.log(midpoint)), dim=1)
    kl_q = torch.sum(q_safe * (torch.log(q_safe) - torch.log(midpoint)), dim=1)
    return 0.5 * (kl_p + kl_q).mean()


def information_maximization_loss(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 2:
        raise ValueError("q must have shape [batch, clusters]")
    q_safe = q.clamp_min(1e-8)
    conditional_entropy = -(q_safe * torch.log(q_safe)).sum(dim=1).mean()
    marginal = q_safe.mean(dim=0)
    marginal_entropy = -(marginal * torch.log(marginal)).sum()
    return conditional_entropy - marginal_entropy


def coverage_concentration(mask_st: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    rates = mask_st.mean(dim=0)
    active = eligible.to(torch.bool).any(dim=0)
    if not bool(active.any()):
        return rates.sum() * 0.0
    active_rates = rates[active]
    return (active_rates - active_rates.mean()).square().mean()


def theoretical_js_upper_bound() -> float:
    return math.log(2.0)
