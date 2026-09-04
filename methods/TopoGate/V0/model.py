"""Shared scMAE backbone and pseudo-view loss for TopoGate V0."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as functional

from methods.NeighborMix_scMAE.model import AutoEncoder as _ScMAEAutoEncoder


class WeightedAutoEncoder(_ScMAEAutoEncoder):
    """The stable scMAE backbone with optional per-sample loss weighting.

    With ``sample_weight=None`` and the default reconstruction normalization,
    this is algebraically the same objective as the historical F
    ``AutoEncoder.loss_mask``.  Supplying gate-derived weights selects the T
    pseudo-view objective without duplicating the network or training loop.

    中文说明：F 与 T 只改变 corruption/gate 以及 pseudo loss 的权重，二者
    共享这里继承的 scMAE encoder、decoder 和 mask predictor；本类没有新增
    一套 F/T 专属网络参数。
    """

    def loss_mask_weighted(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        mask_loss_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # 先统一检查三种张量的二维表达矩阵形状，避免后面的广播把错误
        # 的 batch/feature 维度静默掩盖掉。
        self._check_expression_shape(x, "x")
        self._check_expression_shape(target, "target")
        self._check_expression_shape(mask, "mask")
        if x.shape != target.shape or x.shape != mask.shape:
            raise ValueError("x, target, and mask must have identical shapes")
        if float(mask_loss_scale) < 0.0:
            raise ValueError("mask_loss_scale must be non-negative")

        mask = mask.to(dtype=x.dtype, device=x.device)
        target = target.to(dtype=x.dtype, device=x.device)
        # forward_mask 同时返回 latent、mask 预测和重构结果；F/T 的差异
        # 在输入 pseudo view 和 sample_weight 中体现，而不在 backbone 中分叉。
        latent, mask_logits, reconstruction = self.forward_mask(x)

        raw_mse = functional.mse_loss(reconstruction, target, reduction="none")
        # masked/unmasked feature 使用不同重构权重，保留历史 scMAE 的目标定义。
        feature_weights = mask * self.masked_data_weight + (1.0 - mask) * (
            1.0 - self.masked_data_weight
        )
        weighted_mse = feature_weights * raw_mse
        if self.normalize_reconstruction_by_weight:
            # 按实际 feature 权重归一化，避免 mask 比例改变 loss 的整体尺度。
            rec_per_sample = weighted_mse.sum(dim=1) / feature_weights.sum(dim=1).clamp_min(1e-8)
        else:
            rec_per_sample = weighted_mse.mean(dim=1)
        rec_per_sample = (1.0 - self.mask_loss_weight) * rec_per_sample

        # mask predictor 的 BCE 与重构项按 backbone 的历史比例混合。
        mask_per_sample = functional.binary_cross_entropy_with_logits(
            mask_logits, mask, reduction="none"
        ).mean(dim=1)
        mask_per_sample = self.mask_loss_weight * mask_per_sample
        total_per_sample = rec_per_sample + float(mask_loss_scale) * mask_per_sample

        if sample_weight is None:
            loss = total_per_sample.mean()
        else:
            # T 的 topology gate 只作为 batch 内 pseudo 样本的非负权重；
            # 加权平均仍对权重和归一化，避免 batch 中 gate 总量影响学习率尺度。
            weight = sample_weight.to(dtype=x.dtype, device=x.device).reshape(-1)
            if weight.shape[0] != x.shape[0]:
                raise ValueError(
                    f"sample_weight must have {x.shape[0]} entries, got {weight.shape[0]}"
                )
            if not torch.all(torch.isfinite(weight)) or torch.any(weight < 0.0):
                raise ValueError("sample_weight must be finite and non-negative")
            loss = (total_per_sample * weight).sum() / weight.sum().clamp_min(1e-8)

        # 诊断项显式 detach，写入 history 时不会保留反向图。
        parts = {
            "reconstruction_loss": rec_per_sample.mean().detach(),
            "mask_loss": mask_per_sample.mean().detach(),
            "total_loss": loss.detach(),
            "mask_positive_rate": mask.mean().detach(),
            "per_sample_loss": total_per_sample.detach(),
        }
        return latent, loss, parts


# Explicit aliases make the public V0 API easy to discover while preserving the
# old class name used by downstream scripts.
AutoEncoder = WeightedAutoEncoder
ScVICARAutoEncoder = WeightedAutoEncoder


__all__ = ["AutoEncoder", "ScVICARAutoEncoder", "WeightedAutoEncoder"]
