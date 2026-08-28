from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse


LossReturn = Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
]


class AutoEncoder(nn.Module):
    """
    scMAE 风格的掩码自编码器。

    数据流：
        被扰动的基因表达
            -> 编码器
            -> 潜在表示

        潜在表示
            -> 掩码预测器
            -> 掩码 logits

        [潜在表示, 掩码 logits]
            -> 解码器
            -> 重建后的基因表达

    说明
    -----
    1. 掩码预测器输出的是 logits，而不是概率，
       因此这里使用 BCEWithLogitsLoss。

    2. 默认情况下，解码器接收原始的掩码 logits，
       以保持与原始 scMAE 风格实现兼容。

    3. 该模块本身不负责生成扰动掩码。
       调用方应先构造：
           x_corrupted, mask = apply_scmae_noise(x, mask_ratio)

       然后再调用：
           latent, loss = model.loss_mask(x_corrupted, target, mask)

       其中：
           - 在 vanilla scMAE 中，target 通常是原始干净表达 x
           - 在 NeighborMix_scMAE 中，target 也可以是混合后的表达，
             具体取决于 target_mode
    """

    def __init__(
        self,
        num_genes: int,
        hidden_size: int = 128,
        dropout: float = 0.0,
        masked_data_weight: float = 0.75,
        mask_loss_weight: float = 0.7,
        decoder_use_sigmoid_mask: bool = False,
        detach_decoder_mask: bool = False,
        normalize_reconstruction_by_weight: bool = False,
    ):
        """
        参数说明：
            num_genes:
                基因数量，同时也是输入与输出的特征维度。

            hidden_size:
                潜在表示的维度。

            dropout:
                施加在编码器输入端的 dropout 比例。
                默认值为 0.0。将 dropout 与 scMAE 的输入扰动同时使用时需要注意，
                因为二者都会对输入进行扰动。

            masked_data_weight:
                分配给被扰动/被掩码位置的重建损失权重。
                可见位置的权重为 1 - masked_data_weight。

                示例：
                    masked_data_weight = 0.75
                    mask = 1 的位置权重为 0.75
                    mask = 0 的位置权重为 0.25

                这不是掩码比例。掩码比例由模型外部控制，
                通常通过 apply_scmae_noise(..., mask_ratio) 指定。

            mask_loss_weight:
                掩码预测 BCE 损失的权重。
                重建损失的权重为 1 - mask_loss_weight。

            decoder_use_sigmoid_mask:
                若为 False，解码器接收原始 mask logits。
                若为 True，解码器接收 sigmoid(mask_logits)。

                默认值 False 保持原始行为不变。

            detach_decoder_mask:
                若为 True，传入解码器的掩码特征会先 detach。
                这样可以阻止重建损失反向更新掩码预测器。

                默认值 False 保持原始行为不变。

            normalize_reconstruction_by_weight:
                若为 False，使用原始行为：
                    weighted_mse.mean()

                若为 True，使用按权重归一化后的加权 MSE：
                    weighted_mse.sum() / weights.sum()

                默认值 False 保持原始行为不变。
        """
        super().__init__()

        if num_genes <= 0:
            raise ValueError(f"num_genes must be positive, got {num_genes}.")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if not 0.0 <= float(masked_data_weight) <= 1.0:
            raise ValueError(
                f"masked_data_weight must be in [0, 1], got {masked_data_weight}."
            )
        if not 0.0 <= float(mask_loss_weight) <= 1.0:
            raise ValueError(
                f"mask_loss_weight must be in [0, 1], got {mask_loss_weight}."
            )

        self.num_genes = int(num_genes)
        self.hidden_size = int(hidden_size)
        self.masked_data_weight = float(masked_data_weight)
        self.mask_loss_weight = float(mask_loss_weight)

        self.decoder_use_sigmoid_mask = bool(decoder_use_sigmoid_mask)
        self.detach_decoder_mask = bool(detach_decoder_mask)
        self.normalize_reconstruction_by_weight = bool(normalize_reconstruction_by_weight)

        self.encoder_width = max(256, self.hidden_size * 2)

        self.encoder = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.num_genes, self.encoder_width),
            nn.LayerNorm(self.encoder_width),
            nn.Mish(inplace=True),
            nn.Linear(self.encoder_width, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Mish(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.mask_predictor = nn.Linear(self.hidden_size, self.num_genes)

        self.decoder = nn.Linear(
            in_features=self.hidden_size + self.num_genes,
            out_features=self.num_genes,
        )

    def _check_expression_shape(self, x: torch.Tensor, name: str) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D tensor with shape [batch_size, num_genes], "
                f"got shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.num_genes:
            raise ValueError(
                f"{name}.shape[1] must equal num_genes={self.num_genes}, "
                f"got {x.shape[1]}."
            )

    def forward_mask(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        用于掩码重建的前向传播。

        参数：
            x:
                被扰动后的输入表达矩阵，形状为 [batch_size, num_genes]。

        返回：
            latent:
                编码器输出的潜在表示，形状为 [batch_size, hidden_size]。

            mask_logits:
                掩码预测的原始 logits，形状为 [batch_size, num_genes]。
                注意这不是概率值。

            reconstruction:
                重建后的表达矩阵，形状为 [batch_size, num_genes]。
        """
        self._check_expression_shape(x, "x")

        latent = self.encoder(x)
        mask_logits = self.mask_predictor(latent)

        decoder_mask_feature = mask_logits
        if self.decoder_use_sigmoid_mask:
            decoder_mask_feature = torch.sigmoid(decoder_mask_feature)
        if self.detach_decoder_mask:
            decoder_mask_feature = decoder_mask_feature.detach()

        decoder_input = torch.cat([latent, decoder_mask_feature], dim=1)
        reconstruction = self.decoder(decoder_input)

        return latent, mask_logits, reconstruction

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        标准的 PyTorch forward 方法。

        这里直接调用 forward_mask(x)，
        以保持现有的 scMAE 风格行为不变。
        """
        return self.forward_mask(x)

    def _reconstruction_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算加权后的重建损失。

        默认情况下，这里保持原始 scMAE 风格行为：
            mean(weights * mse)

        若 normalize_reconstruction_by_weight=True：
            sum(weights * mse) / sum(weights)

        返回值还会再乘以 (1 - mask_loss_weight)，
        作为总损失中的重建项权重。
        """
        raw_mse = mse(reconstruction, target, reduction="none")

        weights = (
            mask * self.masked_data_weight
            + (1.0 - mask) * (1.0 - self.masked_data_weight)
        )

        weighted_mse = weights * raw_mse

        if self.normalize_reconstruction_by_weight:
            denominator = weights.sum().clamp_min(1e-8)
            rec_loss = weighted_mse.sum() / denominator
        else:
            rec_loss = weighted_mse.mean()

        return (1.0 - self.mask_loss_weight) * rec_loss

    def loss_mask(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        return_parts: bool = False,
    ) -> LossReturn:
        """
        计算 scMAE 掩码自编码器的损失。

        参数：
            x:
                经过 scMAE 掩码扰动后的输入表达矩阵。
                形状为 [batch_size, num_genes]。

            y:
                重建目标。
                对于 vanilla scMAE，通常为原始干净表达。
                对于 NeighborMix_scMAE，也可以是混合后的表达，
                具体取决于 target_mode。
                形状为 [batch_size, num_genes]。

            mask:
                二值扰动掩码。
                1 表示该位置被扰动/被掩码。
                0 表示该位置是可见的。
                通常期望其取值属于 {0, 1}。
                形状为 [batch_size, num_genes]。

            return_parts:
                若为 False，返回：
                    latent, loss

                若为 True，返回：
                    latent, loss, parts

                其中 parts 包含已 detach 的诊断张量：
                    reconstruction_loss
                    mask_loss
                    total_loss
                    reconstruction_mse_unweighted
                    mask_positive_rate

        返回：
            latent:
                编码器输出的潜在表示。

            loss:
                总训练损失。

            parts:
                可选的诊断信息字典，包含各个损失分量。
        """
        self._check_expression_shape(x, "x")
        self._check_expression_shape(y, "y")
        self._check_expression_shape(mask, "mask")

        if x.shape != y.shape or x.shape != mask.shape:
            raise ValueError(
                "x, y, and mask must have identical shapes. "
                f"Got x={tuple(x.shape)}, y={tuple(y.shape)}, mask={tuple(mask.shape)}."
            )

        mask = mask.to(dtype=x.dtype, device=x.device)
        y = y.to(dtype=x.dtype, device=x.device)

        latent, mask_logits, reconstruction = self.forward_mask(x)

        rec_loss = self._reconstruction_loss(
            reconstruction=reconstruction,
            target=y,
            mask=mask,
        )

        mask_loss = self.mask_loss_weight * bce_logits(
            mask_logits,
            mask,
            reduction="mean",
        )

        loss = rec_loss + mask_loss

        if return_parts:
            with torch.no_grad():
                parts = {
                    "reconstruction_loss": rec_loss.detach(),
                    "mask_loss": mask_loss.detach(),
                    "total_loss": loss.detach(),
                    "reconstruction_mse_unweighted": mse(
                        reconstruction,
                        y,
                        reduction="mean",
                    ).detach(),
                    "mask_positive_rate": mask.mean().detach(),
                }
            return latent, loss, parts

        return latent, loss

    @torch.no_grad()
    def feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取编码器的潜在表示作为特征表征。

        参数：
            x:
                干净的或经过预处理的表达矩阵，
                形状为 [batch_size, num_genes]。

        返回：
            形状为 [batch_size, hidden_size] 的潜在表示。
        """
        self._check_expression_shape(x, "x")
        return self.encoder(x)