"""Masked autoencoder used by the V12 latent-topology variant."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class AutoEncoder(nn.Module):
    """Deterministic masked autoencoder with an explicitly exposed latent."""

    def __init__(
        self,
        num_genes: int,
        hidden_size: int = 128,
        dropout: float = 0.0,
        masked_data_weight: float = 0.75,
        mask_loss_weight: float = 0.1,
        mask_loss_mode: str = "additive",
        decoder_mode: str = "legacy_mask_conditioned",
        decoder_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if int(num_genes) <= 0 or int(hidden_size) <= 0:
            raise ValueError("num_genes and hidden_size must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= float(masked_data_weight) <= 1.0:
            raise ValueError("masked_data_weight must be in [0, 1]")
        if not 0.0 <= float(mask_loss_weight) <= 1.0:
            raise ValueError("mask_loss_weight must be in [0, 1]")
        if mask_loss_mode not in {"additive", "legacy_weighted"}:
            raise ValueError("mask_loss_mode must be 'additive' or 'legacy_weighted'")
        if decoder_mode not in {"legacy_mask_conditioned", "latent_only"}:
            raise ValueError(
                "decoder_mode must be 'legacy_mask_conditioned' or 'latent_only'"
            )

        self.num_genes = int(num_genes)
        self.hidden_size = int(hidden_size)
        self.masked_data_weight = float(masked_data_weight)
        self.mask_loss_weight = float(mask_loss_weight)
        self.mask_loss_mode = str(mask_loss_mode)
        self.decoder_mode = str(decoder_mode)
        self.encoder_width = max(256, self.hidden_size * 2)
        self.decoder_hidden_size = int(decoder_hidden_size or self.encoder_width)

        self.encoder = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.num_genes, self.encoder_width),
            nn.LayerNorm(self.encoder_width),
            nn.Mish(inplace=True),
            nn.Linear(self.encoder_width, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.mask_predictor = nn.Linear(self.hidden_size, self.num_genes)
        if self.decoder_mode == "legacy_mask_conditioned":
            # Preserve the scMAE decoder contract. The V12 change is the
            # topology objective, not an unrelated decoder redesign.
            self.decoder = nn.Linear(
                self.hidden_size + self.num_genes,
                self.num_genes,
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.decoder_hidden_size),
                nn.LayerNorm(self.decoder_hidden_size),
                nn.Mish(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(self.decoder_hidden_size, self.num_genes),
            )

    def _check_expression_shape(self, x: torch.Tensor, name: str) -> None:
        if x.ndim != 2 or x.shape[1] != self.num_genes:
            raise ValueError(
                f"{name} must have shape [batch, {self.num_genes}], got {tuple(x.shape)}"
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation with gradients enabled."""

        self._check_expression_shape(x, "x")
        return self.encoder(x)

    def forward_mask(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(latent, mask_logits, reconstruction)``."""

        latent = self.encode(x)
        mask_logits = self.mask_predictor(latent)
        if self.decoder_mode == "legacy_mask_conditioned":
            reconstruction = self.decoder(torch.cat([latent, mask_logits], dim=1))
        else:
            reconstruction = self.decoder(latent)
        return latent, mask_logits, reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_mask(x)

    def _reconstruction_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample masked reconstruction MSE."""

        self._check_expression_shape(reconstruction, "reconstruction")
        self._check_expression_shape(target, "target")
        self._check_expression_shape(mask, "mask")
        mask = mask.to(dtype=reconstruction.dtype, device=reconstruction.device)
        target = target.to(dtype=reconstruction.dtype, device=reconstruction.device)
        weights = mask * self.masked_data_weight + (1.0 - mask) * (
            1.0 - self.masked_data_weight
        )
        return (weights * F.mse_loss(reconstruction, target, reduction="none")).mean(dim=1)

    def loss_mask_weighted(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        mask_loss_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute reconstruction plus a low-weight mask-prediction loss."""

        self._check_expression_shape(x, "x")
        self._check_expression_shape(y, "y")
        self._check_expression_shape(mask, "mask")
        if x.shape != y.shape or x.shape != mask.shape:
            raise ValueError("x, y, and mask must have identical shapes")

        latent, mask_logits, reconstruction = self.forward_mask(x)
        mask_t = mask.to(dtype=x.dtype, device=x.device)
        rec_per = self._reconstruction_loss(reconstruction, y, mask_t)
        mask_per = F.binary_cross_entropy_with_logits(
            mask_logits, mask_t, reduction="none"
        ).mean(dim=1)
        weighted_mask_per = self.mask_loss_weight * float(mask_loss_scale) * mask_per
        if self.mask_loss_mode == "additive":
            rec_term = rec_per
            total_per = rec_per + weighted_mask_per
        else:
            rec_term = (1.0 - self.mask_loss_weight) * rec_per
            total_per = rec_term + weighted_mask_per

        if sample_weight is None:
            loss = total_per.mean()
        else:
            weights = sample_weight.to(dtype=x.dtype, device=x.device).reshape(-1)
            if weights.shape[0] != total_per.shape[0]:
                raise ValueError("sample_weight must have one value per sample")
            loss = (total_per * weights).sum() / weights.sum().clamp_min(1e-8)

        parts: Dict[str, torch.Tensor] = {
            # Keep both raw and weighted terms explicit.  The raw values make
            # additive-vs-legacy comparisons auditable without reverse
            # engineering the configured weight from a summary file.
            "raw_reconstruction_loss": rec_per.mean().detach(),
            "raw_mask_loss": mask_per.mean().detach(),
            "reconstruction_loss": rec_term.mean().detach(),
            "mask_loss": weighted_mask_per.mean().detach(),
            "mask_loss_weighted": weighted_mask_per.mean().detach(),
            "total_loss": loss.detach(),
            "mask_positive_rate": mask_t.mean().detach(),
            "per_sample_loss": total_per.detach(),
        }
        return latent, loss, parts

    @torch.no_grad()
    def feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract clean embeddings for evaluation or detached graph targets."""

        return self.encode(x)
