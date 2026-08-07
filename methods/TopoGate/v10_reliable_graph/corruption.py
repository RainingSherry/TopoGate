"""Mask corruption kernels with explicit intervention masks."""

from __future__ import annotations

from typing import Literal

import torch


CorruptionStrategy = Literal["zero", "feature_shuffle"]


def apply_mask_corruption(
    x: torch.Tensor,
    ratio: float | torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    strategy: CorruptionStrategy = "zero",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt independently sampled entries and return the sampled mask.

    The returned mask is the Bernoulli intervention ``should_swap`` itself.  It
    therefore remains correct even when a replacement happens to equal the
    original value, a frequent event in sparse count matrices.

    Parameters
    ----------
    x:
        Input matrix with shape ``[batch_size, num_features]``.
    ratio:
        Scalar or broadcastable tensor of masking probabilities in ``[0, 1]``.
    generator:
        Optional device-compatible PyTorch random generator.
    strategy:
        ``"zero"`` performs standard zero masking.  ``"feature_shuffle"``
        independently draws each replacement from the same feature's empirical
        batch marginal; it never copies an unrelated sample as a whole row.
    """

    if x.ndim != 2:
        raise ValueError(f"x must be 2D [batch, features], got {tuple(x.shape)}.")
    probability = torch.as_tensor(ratio, dtype=x.dtype, device=x.device)
    if torch.any((probability < 0) | (probability > 1)):
        raise ValueError("ratio must contain probabilities in [0, 1].")
    random_values = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    should_swap_bool = random_values < probability

    if strategy == "zero":
        replacement = torch.zeros((), dtype=x.dtype, device=x.device)
        corrupted = torch.where(should_swap_bool, replacement, x)
    elif strategy == "feature_shuffle":
        batch_size, num_features = x.shape
        if batch_size < 2:
            corrupted = torch.where(should_swap_bool, torch.zeros_like(x), x)
        else:
            source_rows = torch.randint(
                low=0,
                high=batch_size,
                size=(batch_size, num_features),
                device=x.device,
                generator=generator,
            )
            feature_ids = torch.arange(num_features, device=x.device).expand(batch_size, -1)
            replacement = x[source_rows, feature_ids]
            corrupted = torch.where(should_swap_bool, replacement, x)
    else:
        raise ValueError(f"Unknown corruption strategy: {strategy!r}.")

    return corrupted, should_swap_bool.to(dtype=x.dtype)
