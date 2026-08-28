from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskDictionary:
    masks: np.ndarray
    donor_offsets: np.ndarray
    mask_seed: int
    donor_seed: int
    mask_ratio: float

    def validate(self, n_samples: int) -> None:
        if self.masks.ndim != 2 or self.masks.dtype != np.bool_:
            raise ValueError("masks must be a boolean [T, d] array")
        if self.donor_offsets.shape != (self.masks.shape[0],):
            raise ValueError("donor_offsets must have one entry per mask")
        if n_samples <= 1:
            if bool(np.any(self.donor_offsets != 0)):
                raise ValueError("single-sample profiles require zero donor offsets")
        elif bool(np.any((self.donor_offsets <= 0) | (self.donor_offsets >= n_samples))):
            raise ValueError("donor offsets must avoid self donors")


def build_mask_dictionary(
    *,
    n_samples: int,
    n_features: int,
    n_masks: int,
    mask_ratio: float,
    mask_seed: int,
    donor_seed: int,
) -> MaskDictionary:
    if min(n_samples, n_features, n_masks) <= 0:
        raise ValueError("mask dictionary dimensions must be positive")
    if not 0.0 < float(mask_ratio) < 1.0:
        raise ValueError("mask_ratio must be in (0, 1)")
    budget = max(1, min(n_features - 1 if n_features > 1 else 1, int(round(mask_ratio * n_features))))
    rng = np.random.default_rng(int(mask_seed))
    usage = np.zeros(n_features, dtype=np.int64)
    masks = np.zeros((n_masks, n_features), dtype=np.bool_)
    for row in range(n_masks):
        jitter = rng.random(n_features)
        order = np.lexsort((jitter, usage))
        selected = order[:budget]
        masks[row, selected] = True
        usage[selected] += 1
    donor_rng = np.random.default_rng(int(donor_seed))
    offsets = (
        np.zeros(n_masks, dtype=np.int64)
        if n_samples <= 1
        else donor_rng.integers(1, n_samples, size=n_masks, endpoint=False, dtype=np.int64)
    )
    result = MaskDictionary(
        masks=masks,
        donor_offsets=offsets,
        mask_seed=int(mask_seed),
        donor_seed=int(donor_seed),
        mask_ratio=float(mask_ratio),
    )
    result.validate(n_samples)
    return result


def corrupt_semantic(
    clean: np.ndarray,
    feature_mask: np.ndarray,
    *,
    donor_offset: int,
    corruption_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(clean, dtype=np.float32)
    feature_mask = np.asarray(feature_mask, dtype=np.bool_)
    if clean.ndim != 2 or feature_mask.shape != (clean.shape[1],):
        raise ValueError("feature mask does not match semantic matrix")
    corrupted = clean.copy()
    if corruption_mode == "donor_swap":
        if clean.shape[0] <= 1:
            donor = clean
        else:
            if donor_offset <= 0 or donor_offset >= clean.shape[0]:
                raise ValueError("invalid donor offset")
            donor = np.roll(clean, shift=int(donor_offset), axis=0)
        corrupted[:, feature_mask] = donor[:, feature_mask]
    elif corruption_mode == "zero":
        corrupted[:, feature_mask] = 0.0
    else:
        raise ValueError(f"unsupported corruption mode: {corruption_mode}")
    effective = (corrupted != clean) & feature_mask[None, :]
    return corrupted, effective
