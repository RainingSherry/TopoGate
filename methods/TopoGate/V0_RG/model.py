from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as functional

from methods.NeighborMix_scMAE.model import AutoEncoder as BaseAutoEncoder


class WeightedAutoEncoder(BaseAutoEncoder):
    """The original scMAE backbone with RG's per-sample pseudo loss weighting."""

    def loss_mask_weighted(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        self._check_expression_shape(x, "x")
        self._check_expression_shape(target, "target")
        self._check_expression_shape(mask, "mask")
        if x.shape != target.shape or x.shape != mask.shape:
            raise ValueError("x, target, and mask must have identical shapes")
        mask = mask.to(dtype=x.dtype, device=x.device)
        target = target.to(dtype=x.dtype, device=x.device)
        latent, mask_logits, reconstruction = self.forward_mask(x)
        raw_mse = functional.mse_loss(reconstruction, target, reduction="none")
        weights = mask * self.masked_data_weight + (1.0 - mask) * (
            1.0 - self.masked_data_weight
        )
        weighted_mse = weights * raw_mse
        if self.normalize_reconstruction_by_weight:
            reconstruction_per_sample = weighted_mse.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
        else:
            reconstruction_per_sample = weighted_mse.mean(dim=1)
        reconstruction_per_sample = (1.0 - self.mask_loss_weight) * reconstruction_per_sample
        mask_per_sample = functional.binary_cross_entropy_with_logits(
            mask_logits, mask, reduction="none"
        ).mean(dim=1)
        mask_per_sample = self.mask_loss_weight * mask_per_sample
        total_per_sample = reconstruction_per_sample + mask_per_sample
        if sample_weight is None:
            loss = total_per_sample.mean()
        else:
            weight = sample_weight.to(dtype=x.dtype, device=x.device).view(-1)
            loss = (total_per_sample * weight).sum() / weight.sum().clamp_min(1e-8)
        parts = {
            "reconstruction_loss": reconstruction_per_sample.mean().detach(),
            "mask_loss": mask_per_sample.mean().detach(),
            "total_loss": loss.detach(),
            "mask_positive_rate": mask.mean().detach(),
            "per_sample_loss": total_per_sample.detach(),
        }
        return latent, loss, parts


def apply_scmae_noise(
    x: torch.Tensor,
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply scMAE corruption with an optional independent random stream."""
    should_swap = torch.rand(
        x.shape,
        dtype=torch.float32,
        device=x.device,
        generator=generator,
    ) < float(mask_ratio)
    replacement = x if x.shape[0] <= 1 else x[
        torch.randperm(x.shape[0], device=x.device, generator=generator)
    ]
    corrupted = torch.where(should_swap, replacement, x)
    return corrupted, (corrupted != x).to(dtype=x.dtype)
