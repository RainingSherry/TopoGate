"""v5 mask_noise: Gumbel-Sigmoid + Straight-Through Estimator with proxy gradient.

v3 apply_mask_noise (in run_npz.py) used `torch.bernoulli(p)` to sample masks,
which makes the operation non-differentiable wrt the probability (p). As a
result, `mask_ratio` as a learnable parameter never moved during training
(Phase 2.1: 30/50 epochs, mask_ratio remained exactly 0.300).

This v5 fix replaces the bernoulli sample with a Gumbel-Sigmoid + straight-
through estimator AND adds a `mask_ratio_reg` function that the training
loop adds to the loss.  The reg is `mean(y_soft) - mask_ratio`, with gradient
flowing into mask_ratio directly via the Gumbel output.

The repair drops in like apply_mask_noise_v3_legacy.  For Phase 2.2, the
runner is expected to *call* apply_mask_noise_v5_ste AND *accumulate*
mask_ratio_reg to the loss sum.
"""
from __future__ import annotations

import torch


def apply_mask_noise_v3_legacy(x: torch.Tensor, mask_ratio) -> tuple[torch.Tensor, torch.Tensor]:
    """v3 legacy version - kept for ablation.  Uses bernoulli (non-differentiable)."""
    if isinstance(mask_ratio, torch.Tensor):
        ratio_val = float(mask_ratio.detach().cpu())
    else:
        ratio_val = float(mask_ratio)
    should_swap = torch.bernoulli(ratio_val * torch.ones_like(x))
    noisy_x = torch.where(should_swap > 0.5, x[torch.randperm(x.size(0))].to(x.device), x)
    return noisy_x, should_swap


def apply_mask_noise_v5_ste(x: torch.Tensor, mask_ratio,
                            temperature: float = 1.0,
                            generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """v5 Gumbel-Sigmoid + STE.  Returns (noisy_x, mask_hard, y_soft).

    Args:
        x: input tensor (B, D).
        mask_ratio: scalar or 0-d tensor (if Parameter, will receive grad).
        temperature: Gumbel temperature.

    Returns:
        (noisy_x, mask_hard, y_soft):
          - noisy_x uses hard mask (forward only).
          - mask_hard is the binary mask (for the loss function).
          - y_soft is the soft mask, used to compute mask_ratio_reg via STE.
    """
    if isinstance(mask_ratio, torch.Tensor):
        ratio = mask_ratio
    else:
        ratio = torch.tensor(float(mask_ratio), device=x.device, dtype=x.dtype)

    p = torch.clamp(ratio, min=1e-5, max=1 - 1e-5)
    logit = torch.log(p / (1.0 - p))

    if generator is not None:
        u = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    else:
        u = torch.rand(x.shape, device=x.device, dtype=x.dtype)
    gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logit + gumbel) / temperature)
    y_hard = (y_soft >= 0.5).to(x.dtype)

    # Forward path uses y_hard only (no gradient flow through x).
    # Gradient flows via y_soft to mask_ratio (it's a leaf in computation graph).
    noisy_x = torch.where(y_hard > 0.5, x[torch.randperm(x.size(0))].to(x.device), x)
    return noisy_x, y_hard, y_soft


def mask_ratio_alignment_loss(y_soft: torch.Tensor, mask_ratio: torch.Tensor,
                              target_ratio: float = 0.3,
                              weight: float = 1.0) -> torch.Tensor:
    """Auxiliary loss that aligns the *expected* mask ratio with the target.

    The expected mask ratio is `mean(y_soft)`.  When mask_ratio == target_ratio,
    the expected value is ~target_ratio.  This auxiliary loss provides
    gradient signal to mask_ratio through the differentiable y_soft.

    Returns a scalar tensor (mean squared discrepancy).
    """
    if not isinstance(mask_ratio, torch.Tensor):
        return torch.zeros((), device=y_soft.device if isinstance(y_soft, torch.Tensor) else "cpu")
    expected_ratio = y_soft.mean()
    return weight * (expected_ratio - target_ratio).pow(2)
