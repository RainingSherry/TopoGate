"""v7 CrossAttnMixer — multi-neighbor cross-attention residual mixer.

设计动机
--------
v3_full/v6 都沿用 latent 线性混合：

    z_mixed = (1 - gate) * z_anchor + gate * z_neighbor_mean

其中 ``z_neighbor_mean`` 是 m 个邻居 latent 的均值。这有两个根因让"邻居参与"过弱：

1. **均值化损失信息**：m 个邻居 latent 被压成 1 个，再与 z_anchor 线性混合。
   当邻居质量参差时，gate 学不到"哪个邻居有用"——只能一刀切。
2. **B-side mask 与 v6 input mix 一致**：B 同样被 mask 后再编码，结果 attention 无法
   比较 B 的"完整指纹"vs"被破坏的 anchor"。Decoder 仍然能"靠 anchor 自己补自己"。

v7 改动（仅这两点，其它全部沿用 v6/v3）：

1. **每个邻居独立 encode**：anchor i 拥有 k_neighbor 个完整 latent，组成 Z_B[i] ∈ R^{k, hidden}
2. **B 不 mask**：B 喂 attention 的是完整 fingerprint（mask_b_anchor=False 默认）
3. **Cross-Attention Residual**：

       q = z_a.unsqueeze(1)                          # (bsz, 1, hidden)
       out, attn_w = cross_attn(q, Z_B, Z_B)          # (bsz, 1, hidden), (bsz, 1, k)
       z_combined = z_a + alpha[i] * out.squeeze(1)   # residual + gate

   ``alpha`` 仍由复用 ``LearnableGate`` 计算（与 v3/v6 接口完全一致）。
   alpha 现在控制**attention 残差幅度**：alpha 大 ⇒ 多用邻居，alpha 小 ⇒ 偏向 anchor。

复用的不变量
-----------
- ``LearnableGate``：4β / 6β / learnable_gate_max / enhanced_stats 完全一致
- Schedule：(1-t) * static_gate + t * gate_dyn，与 v6 一致
- Output keys (mean_node_gate / effective_gate_max / schedule_t) 与 LatentMixer 对齐

接口严格与 LatentMixer 对齐
---------------------------
``forward(z_anchor, z_neighbors_padded, stats, static_gate, schedule_t)``

但 ``z_neighbors_padded`` 形状为 (batch, k, hidden) 而非 (batch, hidden)。
注意：v6 接收 (batch, hidden) 的"均值"z_n——v7 完全不沿用，shape 不兼容。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from methods.TopoGate.learnable_gate.learnable_gate import LearnableGate


class CrossAttnMixer(nn.Module):
    """v7 cross-attention mixer with residual gate.

    Args:
        gate_min, gate_max: Output range for the gate (default (0, 0.5)).
        init_beta_*: Initial values for the four learnable coefficients, all 0.
        learnable_gate_max, gate_max_min, gate_max_max: as in LearnableGate.
        enhanced_stats: 4 (default) or 6.
        attn_heads: number of heads in the cross-attention (default 1).
        residual_dropout: dropout on the attention residual (default 0.0).
    """

    def __init__(
        self,
        gate_min: float = 0.0,
        gate_max: float = 0.15,
        init_beta_mutual: float = 0.0,
        init_beta_snn: float = 0.0,
        init_beta_perturb: float = 0.0,
        init_beta_uncertainty: float = 0.0,
        learnable_gate_max: bool = False,
        gate_max_min: float = 0.05,
        gate_max_max: float = 1.0,
        enhanced_stats: int = 4,
        attn_heads: int = 1,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.gate = LearnableGate(
            gate_min=self.gate_min,
            gate_max=self.gate_max,
            init_beta_mutual=init_beta_mutual,
            init_beta_snn=init_beta_snn,
            init_beta_perturb=init_beta_perturb,
            init_beta_uncertainty=init_beta_uncertainty,
            learnable_gate_max=bool(learnable_gate_max),
            gate_max_min=float(gate_max_min),
            gate_max_max=float(gate_max_max),
            enhanced_stats=enhanced_stats,
        )

        # Cross-attention：q=z_anchor, k=v=Z_B
        # 关键：query 来自 anchor（被 mask），key/value 来自完整邻居指纹（B 不 mask）。
        # 这迫使 attention 学到"哪些邻居有用"——而 v6 的 linear mix 无法学到。
        self.heads = max(1, int(attn_heads))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=1,            # placeholder
            num_heads=self.heads,
            dropout=float(residual_dropout),
            batch_first=True,
        )
        # Hold embed_dim as a buffer string for delayed init (set in first forward)
        self._hidden_size = None

    def _ensure_cross_attn(self, hidden_size: int) -> None:
        """初始化 MultiheadAttention（必须知道 hidden_size）。"""
        if self._hidden_size == hidden_size:
            return
        self._hidden_size = hidden_size
        # Recreate layer with proper embed_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.heads,
            dropout=self.cross_attn.dropout,
            batch_first=True,
        )
        # Move to correct device
        self.cross_attn = self.cross_attn.to(next(self.parameters()).device)

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_neighbors_padded: torch.Tensor,
        stats: torch.Tensor,
        static_gate: "torch.Tensor | None" = None,
        schedule_t: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """v7 cross-attention mix.

        Args:
            z_anchor: (batch, hidden) anchor latents (post-mask encoding).
            z_neighbors_padded: (batch, k, hidden) — 每个 anchor 的 k 邻居完整 latent
                                 （B-side 不 mask）。Padding 值（邻居数 < k 的 cell）
                                 会被 mask 屏蔽（不参与 attention 计算）。
            stats: (batch, enhanced_stats) per-node topology stats.
            static_gate: optional (batch,) v1-style gate for schedule fallback.
            schedule_t: interpolation scalar in [0, 1].

        Returns:
            z_combined: (batch, hidden) anchor + alpha * attention_output.
            mix_summary: dict with mean_node_gate, effective_gate_max, schedule_t.
        """
        bsz, hidden = z_anchor.shape
        self._ensure_cross_attn(hidden)

        if z_neighbors_padded.shape[0] != bsz:
            raise ValueError(
                f"batch size mismatch: anchor={bsz}, neighbors={z_neighbors_padded.shape[0]}"
            )
        # k_neighbor from second dim of z_neighbors_padded
        k = z_neighbors_padded.shape[1]

        # 计算 per-node gate α (复用 LearnableGate) — 与 v6 相同路径
        gate_dyn = self.gate(stats).view(-1, 1).to(dtype=z_anchor.dtype)
        t = float(max(0.0, min(1.0, schedule_t)))
        if t < 1.0 and static_gate is not None:
            sg = static_gate.to(dtype=gate_dyn.dtype, device=gate_dyn.device).view(-1, 1)
            gate = (1.0 - t) * sg + t * gate_dyn
        else:
            gate = gate_dyn  # (bsz, 1)

        # Attention mask: kp_mask=True 表示该邻居位置是 padding
        # 与 run_npz.py 中 padding=0 的约定一致
        # 如果没传 mask，假设全部有效
        # Cross-attention mask 形状 (N, S_k) where True = ignore
        # 这里我们用 key_padding_mask: (bsz, k), True = ignore
        key_padding_mask = (z_neighbors_padded.abs().sum(dim=2) == 0)  # (bsz, k) True=padding

        # Cross-attention: q=(bsz, 1, hidden), k=v=(bsz, k, hidden)
        q = z_anchor.unsqueeze(1)
        attn_out, _ = self.cross_attn(
            query=q,
            key=z_neighbors_padded,
            value=z_neighbors_padded,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        # attn_out: (bsz, 1, hidden)
        attn_out = attn_out.squeeze(1)  # (bsz, hidden)

        # Residual + gate: z_combined = z_a + alpha * attn_out
        # 注意：gate 在 v6 是乘 z_n 的因子；v7 是乘 attn residual 的幅度。
        z_combined = z_anchor + gate * attn_out

        mix_summary = {
            "mean_node_gate": float(gate.mean().detach().cpu()),
            "min_node_gate": float(gate.min().detach().cpu()),
            "max_node_gate": float(gate.max().detach().cpu()),
            "effective_gate_max": float(self.gate.effective_gate_max().detach().cpu()),
            "schedule_t": t,
            "k_neighbor": int(k),
        }
        return z_combined, mix_summary

    def beta_snapshot(self) -> dict:
        return self.gate.beta_snapshot()
