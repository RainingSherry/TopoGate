"""v5 per-sample adaptive mask ratio (SBAM-style).

v5 mask_noise_v5 uses a single global `mask_ratio` for all samples in a batch.
SBAM [arXiv:2404.08327, 2024] shows that per-sample adaptive mask ratio
significantly improves masked image modeling by tailoring the mask density to
the per-sample complexity (token salience).

For tabular TopoGate, the analogous notion is **per-sample feature complexity**:
  salience_i = mean(cosine distance to k nearest neighbors)

A high-salience sample has unstable neighbors and benefits from aggressive
masking (the model is forced to learn the hard neighbourhood). A low-salience
sample is in a homogeneous neighbourhood and benefits from light masking
(less information loss from the swap).

The implementation:
  - Compute salience_i from the kNN graph (CPU, once per epoch).
  - Per-row mask ratio = mask_base + mask_scale * salience_i (clipped to
    [mask_ratio_min, mask_ratio_max]).
  - Apply Gumbel-Sigmoid + STE per-row (broadcast logit across feature dim).

The two new learnable parameters (mask_base, mask_scale) get gradient signal
through the per-row y_soft mask via mask_ratio_reg_loss (computed below).

The fix is intended to be a drop-in replacement for apply_mask_noise_v5_ste
in scripts/learnable_gate/run_v5_separate.py: callers should pass the
(B,) tensor of mask ratios per row instead of a scalar.
"""
from __future__ import annotations

import torch


def compute_sample_salience(
    x: torch.Tensor,
    precomputed: "torch.Tensor | None" = None,
    k: int = 10,
) -> torch.Tensor:
    """Per-sample salience score in [0, 1].

    Args:
        x: (B, D) feature tensor on any device.
        precomputed: optional (B,) pre-computed salience (e.g. from the kNN
            graph used to build edge reliability).  If None, falls back to
            a quick kNN in feature space.

    Returns:
        (B,) tensor of salience in [0, 1].
    """
    if precomputed is not None:
        sal = precomputed.detach().clone()
    else:
        # Quick kNN-based salience: mean distance to k nearest neighbours.
        # We use cosine distance which is consistent with the build_pca_knn_graph
        # metric.
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        x_np = x.detach().cpu().numpy()
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(x_np)
        d, _ = nn.kneighbors(x_np)
        # Skip self (column 0 = 0 distance)
        d = d[:, 1:].mean(axis=1)
        sal = torch.as_tensor(d, device=x.device, dtype=x.dtype)
    s_min = sal.min()
    s_max = sal.max()
    span = (s_max - s_min).clamp(min=1e-8)
    return ((sal - s_min) / span).detach()


def apply_mask_noise_v5_per_sample(
    x: torch.Tensor,
    mask_ratio_per_sample: torch.Tensor,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """v5 Gumbel-Sigmoid + STE with per-row mask ratio.

    Args:
        x: (B, D) input tensor.
        mask_ratio_per_sample: (B,) tensor with one mask ratio per row.  Can
            be a leaf tensor with gradient (mask_base + mask_scale * salience).
        temperature: Gumbel temperature.

    Returns:
        (noisy_x, mask_hard, y_soft):
          - noisy_x uses hard mask (forward only).
          - mask_hard is the binary mask (B, D).
          - y_soft is the soft mask (B, D), used for mask_ratio_reg_loss.
    """
    assert mask_ratio_per_sample.ndim == 1, \
        f"mask_ratio_per_sample must be (B,), got {tuple(mask_ratio_per_sample.shape)}"
    assert mask_ratio_per_sample.shape[0] == x.shape[0], \
        f"mask_ratio_per_sample batch dim {mask_ratio_per_sample.shape[0]} != x batch dim {x.shape[0]}"
    p = torch.clamp(mask_ratio_per_sample, min=1e-5, max=1.0 - 1e-5)
    logit = torch.log(p / (1.0 - p))  # (B,)
    logit_b = logit.unsqueeze(1).expand(x.shape[0], x.shape[1])  # (B, D)
    if generator is not None:
        u = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    else:
        u = torch.rand(x.shape, device=x.device, dtype=x.dtype)
    gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logit_b + gumbel) / temperature)
    y_hard = (y_soft >= 0.5).to(x.dtype)
    # Forward path uses y_hard only.
    noisy_x = torch.where(y_hard > 0.5, x[torch.randperm(x.size(0))].to(x.device), x)
    return noisy_x, y_hard, y_soft


def per_sample_mask_ratio_reg_loss(
    y_soft: torch.Tensor,
    mask_ratio_per_sample: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Auxiliary loss that aligns per-row y_soft means with the row mask ratios.

    For each row i: mean(y_soft[i, :]) should track mask_ratio_per_sample[i].
    This provides gradient signal to mask_base and mask_scale.

    Returns scalar (mean squared discrepancy across rows).
    """
    if not isinstance(mask_ratio_per_sample, torch.Tensor):
        return torch.zeros((), device=y_soft.device if isinstance(y_soft, torch.Tensor) else "cpu")
    expected_per_row = y_soft.mean(dim=1)  # (B,)
    return weight * (expected_per_row - mask_ratio_per_sample.detach()).pow(2).mean()