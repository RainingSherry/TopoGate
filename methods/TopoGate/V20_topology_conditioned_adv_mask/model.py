from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional

from methods.NeighborMix_scMAE.model import AutoEncoder


class V20AutoEncoder(AutoEncoder):
    """scMAE contract with an exposed differentiable reconstruction component."""

    def reconstruction_loss(self, reconstruction: torch.Tensor, target: torch.Tensor, effective_mask: torch.Tensor) -> torch.Tensor:
        effective_mask = effective_mask.to(dtype=reconstruction.dtype, device=reconstruction.device)
        target = target.to(dtype=reconstruction.dtype, device=reconstruction.device)
        raw_mse = functional.mse_loss(reconstruction, target, reduction="none")
        weights = effective_mask * self.masked_data_weight + (1.0 - effective_mask) * (1.0 - self.masked_data_weight)
        return (1.0 - self.mask_loss_weight) * (weights * raw_mse).mean()

    def loss_encoder(self, corrupted: torch.Tensor, target: torch.Tensor, effective_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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


class FeatureGate(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, int(hidden_size)), nn.ReLU(), nn.Linear(int(hidden_size), 1))

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        if stats.ndim < 2 or stats.shape[-1] != 2:
            raise ValueError(f"stats must have shape [..., 2], got {tuple(stats.shape)}")
        return self.net(stats).squeeze(-1)


def _topk_hard(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = max(1, min(int(k), int(scores.shape[1])))
    indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(scores)
    return mask.scatter(1, indices, 1.0)


def random_topk_mask(scores_shape: tuple[int, int], k: int, *, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    scores = torch.rand(scores_shape, device=device, generator=generator)
    return _topk_hard(scores, k)


def random_bernoulli_mask(
    shape: tuple[int, int],
    mask_ratio: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample the original scMAE per-position corruption mask."""
    if not 0.0 <= float(mask_ratio) <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1]")
    return (torch.rand(shape, device=device, generator=generator) < float(mask_ratio)).to(torch.float32)


def straight_through_topk(logits: torch.Tensor, k: int, *, generator: torch.Generator, gumbel_scale: float, tau_ste: float) -> tuple[torch.Tensor, torch.Tensor]:
    if tau_ste <= 0.0:
        raise ValueError("tau_ste must be positive")
    uniform = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device, generator=generator).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    noisy = logits + float(gumbel_scale) * gumbel
    hard = _topk_hard(noisy, k)
    kth = torch.topk(noisy, k=max(1, min(k, noisy.shape[1])), dim=1, largest=True, sorted=True).values[:, -1:].detach()
    soft = torch.sigmoid((noisy - kth) / float(tau_ste))
    return hard + soft - soft.detach(), hard


def cyclic_donor(x: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    if x.shape[0] <= 1:
        return x.clone()
    offset = int(torch.randint(1, x.shape[0], (1,), device=x.device, generator=generator).item())
    return torch.roll(x, shifts=offset, dims=0)
