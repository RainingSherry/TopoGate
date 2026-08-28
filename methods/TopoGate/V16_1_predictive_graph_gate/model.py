from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from methods.NeighborMix_scMAE.model import AutoEncoder as ScMAEAutoEncoder


class SparseCountMAE(nn.Module):
    """scMAE-contract model with a count-likelihood reconstruction head.

    The encoder block is the frozen project's scMAE encoder contract.  The
    decoder receives ``[latent, mask_logits]`` just like the historical model,
    while the trainer supplies a count-aware likelihood instead of dense MSE.
    No topology tensor is accepted by this class.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        encoder = ScMAEAutoEncoder(
            num_genes=int(input_dim),
            hidden_size=int(hidden_dim),
            dropout=float(dropout),
            masked_data_weight=1.0,
            mask_loss_weight=0.0,
            decoder_use_sigmoid_mask=False,
            detach_decoder_mask=False,
            normalize_reconstruction_by_weight=False,
        )
        self.encoder = encoder.encoder
        self.projection = nn.Linear(int(hidden_dim), int(latent_dim))
        self.mask_predictor = nn.Linear(int(latent_dim), int(input_dim))
        self.decoder = nn.Linear(int(latent_dim) + int(input_dim), int(input_dim))
        self._latent_dim = int(latent_dim)

    def forward_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        latent = self.projection(hidden)
        mask_logits = self.mask_predictor(latent)
        reconstruction = self.decoder(torch.cat([latent, mask_logits], dim=1))
        return latent, mask_logits, reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(x))

    @property
    def latent_dim(self) -> int:
        return self._latent_dim


class SphericalPrototypeHead(nn.Module):
    def __init__(self, n_clusters: int, latent_dim: int, temperature: float = 0.1):
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.temperature = float(temperature)
        self.centres = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        values = F.normalize(z, dim=1)
        centres = F.normalize(self.centres, dim=1)
        return F.softmax(values @ centres.t() / self.temperature, dim=1)

    @torch.no_grad()
    def initialise(self, embeddings: np.ndarray, seed: int, n_init: int = 10) -> None:
        values = np.asarray(embeddings, dtype=np.float32)
        km = KMeans(n_clusters=self.n_clusters, n_init=int(n_init), random_state=int(seed))
        km.fit(values)
        centres = torch.as_tensor(km.cluster_centers_, dtype=self.centres.dtype, device=self.centres.device)
        self.centres.copy_(centres)


def masked_poisson_loss(
    log_rate: torch.Tensor,
    target_counts: torch.Tensor,
    observed_mask: torch.Tensor,
    zero_mask: torch.Tensor,
) -> torch.Tensor:
    """Count likelihood restricted to hidden nonzeros and sampled zeros."""
    rate_log = log_rate.clamp(-12.0, 12.0)
    nll = torch.exp(rate_log) - target_counts * rate_log
    weights = observed_mask.to(dtype=log_rate.dtype) + 0.1 * zero_mask.to(dtype=log_rate.dtype)
    numerator = (nll * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (numerator / denominator).mean()
