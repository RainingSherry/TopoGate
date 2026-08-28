"""MC Dropout uncertainty estimation for TopoGate LearnableGate.

Uncertainty is one of the four topology stats (mutual, snn, perturb, uncertainty)
passed to the LearnableGate.  Previously this was always None, making the
4th stat a zero vector and beta_uncertainty a dead parameter.

This module computes a per-node "structural instability" score via Monte Carlo
Dropout: we run the encoder forward n_passes times with dropout enabled, then
measure the standard deviation of the latent representations across passes.

Nodes whose latent embedding is unstable under dropout are structurally ambiguous —
they sit near decision boundaries — so a higher uncertainty score justifies
mixing more aggressively with their neighbors.
"""
from __future__ import annotations

import torch
import numpy as np


def compute_mc_dropout_uncertainty(
    model: torch.nn.Module,
    X_tensor: torch.Tensor,
    n_passes: int = 5,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Monte Carlo Dropout uncertainty via encoder latent variance.

    Args:
        model: AutoEncoder model. Must have an `encoder` attribute.
        X_tensor: (n, d) input tensor.
        n_passes: Number of MC forward passes.
        device: Device to run on.

    Returns:
        uncertainty: (n,) per-sample uncertainty scores in [0, 1].
    """
    model.train()  # keep dropout active
    X = X_tensor.to(device)

    with torch.no_grad():
        preds = []
        for _ in range(n_passes):
            latent = model.encoder(X)  # (n, hidden_size)
            preds.append(latent.float())

    preds = torch.stack(preds, dim=0)  # (n_passes, n, hidden)
    # Variance across MC passes, then average over latent dims
    uncertainty = preds.std(dim=0).mean(dim=1)  # (n,)

    # Normalise to [0, 1] per batch (min-max)
    u_min = uncertainty.min()
    u_max = uncertainty.max()
    if u_max > u_min:
        uncertainty = (uncertainty - u_min) / (u_max - u_min)

    return uncertainty.cpu().numpy()
