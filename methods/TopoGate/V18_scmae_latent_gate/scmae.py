from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss


LossReturn = Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
]


class MaskedAutoencoder(nn.Module):
    """Small scMAE-compatible front end without pseudo-cell mixing."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        dropout: float = 0.0,
        masked_data_weight: float = 0.75,
        mask_loss_weight: float = 0.70,
    ) -> None:
        super().__init__()
        if num_features <= 0 or hidden_size <= 0:
            raise ValueError("num_features and hidden_size must be positive")
        self.num_features = int(num_features)
        self.hidden_size = int(hidden_size)
        self.masked_data_weight = float(masked_data_weight)
        self.mask_loss_weight = float(mask_loss_weight)
        width = max(256, 2 * self.hidden_size)
        self.encoder = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.num_features, width),
            nn.LayerNorm(width),
            nn.Mish(inplace=True),
            nn.Linear(width, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.mask_predictor = nn.Linear(self.hidden_size, self.num_features)
        self.decoder = nn.Linear(self.hidden_size + self.num_features, self.num_features)

    def _check(self, x: torch.Tensor, name: str) -> None:
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(f"{name} must have shape [batch, {self.num_features}], got {tuple(x.shape)}")

    def forward_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check(x, "x")
        z = self.encoder(x)
        mask_logits = self.mask_predictor(z)
        reconstruction = self.decoder(torch.cat([z, mask_logits], dim=1))
        return z, mask_logits, reconstruction

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_mask(x)

    def loss_mask(
        self,
        corrupted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_parts: bool = False,
    ) -> LossReturn:
        self._check(corrupted, "corrupted")
        self._check(target, "target")
        self._check(mask, "mask")
        z, mask_logits, reconstruction = self.forward_mask(corrupted)
        mask = mask.to(dtype=corrupted.dtype)
        weights = mask * self.masked_data_weight + (1.0 - mask) * (1.0 - self.masked_data_weight)
        reconstruction_loss = (weights * mse_loss(reconstruction, target, reduction="none")).mean()
        reconstruction_loss = (1.0 - self.mask_loss_weight) * reconstruction_loss
        mask_loss = self.mask_loss_weight * binary_cross_entropy_with_logits(mask_logits, mask)
        total = reconstruction_loss + mask_loss
        if not return_parts:
            return z, total
        parts = {
            "reconstruction_loss": reconstruction_loss.detach(),
            "mask_loss": mask_loss.detach(),
            "mae_loss": total.detach(),
        }
        return z, total, parts

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x, "x")
        return self.encoder(x)


def masked_view(
    x: torch.Tensor,
    ratio: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a reproducible cell-wise mask view; labels never enter this path."""
    if not 0.0 <= float(ratio) < 1.0:
        raise ValueError("mask ratio must be in [0, 1)")
    mask = torch.rand(x.shape, generator=generator, device=x.device) < float(ratio)
    if x.shape[0] > 1:
        permutation = torch.randperm(x.shape[0], generator=generator, device=x.device)
        donor = x[permutation]
    else:
        donor = x
    corrupted = torch.where(mask, donor, x)
    # Match the locked scMAE contract: a position is masked only when the
    # replacement changed its value. This matters for sparse inputs where
    # donor and target are often both zero.
    effective_mask = (corrupted != x).to(dtype=x.dtype)
    return corrupted, effective_mask
