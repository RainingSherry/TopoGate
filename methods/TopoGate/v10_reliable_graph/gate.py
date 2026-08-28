"""Differentiable edge-level reliability gate for TopoGate V10."""

from __future__ import annotations

import math

import torch
from torch import nn


class EdgeGate(nn.Module):
    """Map non-redundant edge evidence to a reliability value per edge.

    The last feature dimension is expected to follow
    ``[similarity, mutual, SNN, density compatibility, stability]``.  Leading
    dimensions are preserved, so both dense ``[n, k, 5]`` and flattened
    ``[num_edges, 5]`` representations are supported.
    """

    def __init__(
        self,
        feature_dim: int = 5,
        hidden_dim: int = 32,
        gate_min: float = 0.0,
        gate_max: float = 1.0,
        init_gate: float = 0.25,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive.")
        if not 0.0 <= float(gate_min) < float(gate_max) <= 1.0:
            raise ValueError("Require 0 <= gate_min < gate_max <= 1.")
        if not float(gate_min) < float(init_gate) < float(gate_max):
            raise ValueError("init_gate must lie strictly between gate_min and gate_max.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if float(temperature) <= 0:
            raise ValueError("temperature must be positive.")

        self.feature_dim = int(feature_dim)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.temperature = float(temperature)
        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        final_layer = self.network[-1]
        assert isinstance(final_layer, nn.Linear)
        # A small non-zero output projection keeps the requested initial mean
        # close to ``init_gate`` while allowing gradients to reach the first
        # MLP layer on the very first backward pass.
        nn.init.normal_(final_layer.weight, mean=0.0, std=1e-2)
        normalized_init = (float(init_gate) - self.gate_min) / (self.gate_max - self.gate_min)
        nn.init.constant_(final_layer.bias, math.log(normalized_init / (1.0 - normalized_init)))

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return edge gates with invalid/padded edges set exactly to zero."""

        if features.ndim < 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must end in feature_dim={self.feature_dim}, got {tuple(features.shape)}."
            )
        logits = self.network(features).squeeze(-1) / self.temperature
        gates = self.gate_min + (self.gate_max - self.gate_min) * torch.sigmoid(logits)
        if valid_mask is not None:
            if valid_mask.shape != gates.shape:
                raise ValueError(
                    f"valid_mask must have shape {tuple(gates.shape)}, got {tuple(valid_mask.shape)}."
                )
            gates = gates * valid_mask.to(dtype=gates.dtype)
        return gates

    @staticmethod
    def statistics(
        gates: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return differentiable gate diagnostics over valid edges."""

        if valid_mask is None:
            values = gates.reshape(-1)
        else:
            if valid_mask.shape != gates.shape:
                raise ValueError("valid_mask and gates must have identical shapes.")
            values = gates[valid_mask]
        if values.numel() == 0:
            zero = gates.sum() * 0.0
            return {"mean": zero, "minimum": zero, "maximum": zero, "open_fraction": zero}
        return {
            "mean": values.mean(),
            "minimum": values.min(),
            "maximum": values.max(),
            "open_fraction": (values >= 0.5).to(dtype=values.dtype).mean(),
        }
