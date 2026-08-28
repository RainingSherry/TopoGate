from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class SparseCountMAE(nn.Module):
    """Dense mini-batch MLP fed by sparse count rows."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


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
    """Count likelihood on hidden nonzeros plus a small sampled-zero term."""
    rate_log = log_rate.clamp(-12.0, 12.0)
    nll = torch.exp(rate_log) - target_counts * rate_log
    weights = observed_mask.to(dtype=log_rate.dtype) + 0.1 * zero_mask.to(dtype=log_rate.dtype)
    numerator = (nll * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (numerator / denominator).mean()
