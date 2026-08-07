"""Scalable masked autoencoder used by TopoGate V10."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class V10AutoEncoder(nn.Module):
    """Deterministic masked autoencoder with a low-rank decoder.

    Unlike the historical scMAE-compatible implementation, the decoder never
    concatenates predicted mask logits.  When ``condition_on_mask=True``, the
    *observed corruption mask* is projected through its own low-rank path and
    added to the decoder state.  Parameter growth is
    ``O(num_features * (hidden_dim + decoder_rank))`` rather than ``O(d^2)``.
    """

    def __init__(
        self,
        num_features: int | None = None,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        decoder_rank: int = 64,
        dropout: float = 0.1,
        condition_on_mask: bool = False,
        *,
        input_dim: int | None = None,
        reconstruction_kind: Literal["mse", "huber"] = "mse",
        masked_weight: float = 1.0,
        visible_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if num_features is None:
            num_features = input_dim
        elif input_dim is not None and int(num_features) != int(input_dim):
            raise ValueError("num_features and input_dim disagree.")
        if num_features is None or int(num_features) <= 0:
            raise ValueError("num_features must be a positive integer.")
        for name, value in (
            ("latent_dim", latent_dim),
            ("hidden_dim", hidden_dim),
            ("decoder_rank", decoder_rank),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if reconstruction_kind not in {"mse", "huber"}:
            raise ValueError("reconstruction_kind must be 'mse' or 'huber'.")
        if float(masked_weight) < 0 or float(visible_weight) < 0:
            raise ValueError("Reconstruction weights must be non-negative.")

        self.num_features = int(num_features)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.decoder_rank = int(decoder_rank)
        self.condition_on_mask = bool(condition_on_mask)
        self.reconstruction_kind = reconstruction_kind
        self.masked_weight = float(masked_weight)
        self.visible_weight = float(visible_weight)

        self.encoder = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )
        self.latent_to_rank = nn.Sequential(
            nn.Linear(self.latent_dim, self.decoder_rank),
            nn.GELU(),
        )
        self.mask_to_rank = (
            nn.Linear(self.num_features, self.decoder_rank, bias=False)
            if self.condition_on_mask
            else None
        )
        self.output_projection = nn.Linear(self.decoder_rank, self.num_features)

    def _validate_features(self, value: torch.Tensor, name: str) -> None:
        if value.ndim != 2 or value.shape[1] != self.num_features:
            raise ValueError(
                f"{name} must have shape [batch, {self.num_features}], got {tuple(value.shape)}."
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a clean or corrupted batch into deterministic latent vectors."""

        self._validate_features(x, "x")
        return self.encoder(x)

    def decode(
        self,
        latent: torch.Tensor,
        corruption_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode latent vectors, optionally conditioned on the true mask."""

        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent must have shape [batch, {self.latent_dim}], got {tuple(latent.shape)}."
            )
        decoder_state = self.latent_to_rank(latent)
        if self.condition_on_mask:
            if corruption_mask is None:
                raise ValueError("corruption_mask is required when condition_on_mask=True.")
            self._validate_features(corruption_mask, "corruption_mask")
            if corruption_mask.shape[0] != latent.shape[0]:
                raise ValueError("corruption_mask and latent must have the same batch size.")
            assert self.mask_to_rank is not None
            decoder_state = decoder_state + self.mask_to_rank(
                corruption_mask.to(dtype=latent.dtype)
            )
        return self.output_projection(decoder_state)

    def forward(
        self,
        x_corrupted: torch.Tensor,
        corruption_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(latent, reconstruction)`` for a corrupted input batch."""

        latent = self.encode(x_corrupted)
        reconstruction = self.decode(latent, corruption_mask)
        return latent, reconstruction

    def reconstruction_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute a normalized masked reconstruction loss.

        ``sample_weight`` is optional and never derived from a detached gate by
        this module.  If supplied by an experiment, it remains differentiable.
        """

        from .losses import masked_reconstruction_loss

        return masked_reconstruction_loss(
            reconstruction,
            target,
            mask,
            masked_weight=self.masked_weight,
            visible_weight=self.visible_weight,
            sample_weight=sample_weight,
            kind=self.reconstruction_kind,
        )

    def parameter_profile(self) -> dict[str, int]:
        """Return parameter counts for scalability diagnostics."""

        total = sum(parameter.numel() for parameter in self.parameters())
        decoder = sum(parameter.numel() for parameter in self.latent_to_rank.parameters())
        decoder += sum(parameter.numel() for parameter in self.output_projection.parameters())
        if self.mask_to_rank is not None:
            decoder += sum(parameter.numel() for parameter in self.mask_to_rank.parameters())
        return {"total_parameters": int(total), "decoder_parameters": int(decoder)}
