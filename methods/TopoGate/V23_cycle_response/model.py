from __future__ import annotations

import torch
import torch.nn as nn

from methods.NeighborMix_scMAE.model import AutoEncoder


class CycleAutoEncoder(AutoEncoder):
    """Canonical scMAE backbone with explicit frozen-probe methods."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check_expression_shape(x, "x")
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_mask(x)[2]


class LatentLinearDecoder(nn.Module):
    """Simple decoder control trained on a frozen canonical encoder."""

    def __init__(self, hidden_size: int, n_features: int) -> None:
        super().__init__()
        if min(hidden_size, n_features) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.linear = nn.Linear(int(hidden_size), int(n_features))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.linear(latent)
