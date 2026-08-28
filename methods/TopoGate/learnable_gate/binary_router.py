"""BinaryRouter: differentiable hard routing between anchor and topology-aware mixed embedding.

Problem with the continuous gate (LearnableGate):
  mixed = (1-g)*anchor + g*neighbor,  g = sigmoid(beta·stats) ∈ (0, gate_max]
  - Even when g→0, the gradient vanishes if anchor≈neighbor
  - enron: full(g=0.075)=0.768 vs nomix=0.875, Δ=0.107 — even minimum mixing hurts
  - The gate can only suppress, never hard-reset

Solution: BinaryRouter
  r = GumbelSoftmax(logit), logit = beta·stats
  r ∈ {0, 1} (hard during inference, differentiable during training)
  x' = r * mixed + (1-r) * anchor
    = anchor                   if r=0  (topology says neighbor is bad)
    = mixed(anchor,neighbor)  if r=1  (topology says neighbor is good)

The beta parameters are shared with LearnableGate for convenience (same topology
features → same coefficients), but the output head is a CLASSIFICATION rather
than a REGRESSION — the model learns which topology patterns imply "use neighbor".

The Gumbel-Softmax uses a temperature schedule:
  - epochs 1..warmup:    temperature = init_temp (high = soft, ≈ v1 behaviour)
  - after ramp:          temperature → 0.01 (hard samples)
During INFERENCE (epoch=args.epochs+1): argmax over logits (pure hard routing).

Why this works when LearnableGate fails:
  1. r is genuinely binary — when r=0, x'=anchor exactly (not anchor*(1-g)+neighbor*g)
  2. Gradient flows through the logit → beta path even when anchor≈neighbor,
     because the routing decision itself is what matters, not the interpolation weight
  3. The model can learn "for this node's topology, the neighbor is always wrong"
     and hard-route to anchor without any residual mixing

The sample_weight in the pseudo-loss is simply r (the routing probability
during soft phase, or 1.0 during hard phase), so nodes that route to anchor
contribute zero pseudo-loss.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryRouter(nn.Module):
    """Differentiable binary router via Gumbel-Softmax.

    Args:
        temperature_init: Initial Gumbel-Softmax temperature (higher = softer).
            Default 5.0 — at init all betas=0, logits≈0, so with temp=5 the
            router is very soft, close to v1's continuous gate.
        temperature_min: Floor temperature after ramp. Default 0.01.
        warmup_epochs: Number of epochs with high temperature (soft routing).
        ramp_epochs: Linear cool-down from temperature_init to temperature_min.
        enhanced_stats: Same as LearnableGate. Default 4, or 6 for degree/cluster.
        init_beta_*: Beta initialisation. Defaults 0.0 (uniform prior).
    """

    def __init__(
        self,
        temperature_init: float = 5.0,
        temperature_min: float = 0.01,
        warmup_epochs: int = 20,
        ramp_epochs: int = 10,
        enhanced_stats: int = 4,
        init_beta_mutual: float = 0.0,
        init_beta_snn: float = 0.0,
        init_beta_perturb: float = 0.0,
        init_beta_uncertainty: float = 0.0,
        init_beta_degree: float = 0.0,
        init_beta_cluster: float = 0.0,
    ) -> None:
        super().__init__()
        self.temperature_init = float(temperature_init)
        self.temperature_min = float(temperature_min)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.enhanced_stats = int(enhanced_stats)

        self.beta_mutual = nn.Parameter(torch.tensor(float(init_beta_mutual)))
        self.beta_snn = nn.Parameter(torch.tensor(float(init_beta_snn)))
        self.beta_perturb = nn.Parameter(torch.tensor(float(init_beta_perturb)))
        self.beta_uncertainty = nn.Parameter(torch.tensor(float(init_beta_uncertainty)))
        if enhanced_stats == 6:
            self.beta_degree = nn.Parameter(torch.tensor(float(init_beta_degree)))
            self.beta_cluster = nn.Parameter(torch.tensor(float(init_beta_cluster)))

    def _compute_logits(self, stats: torch.Tensor) -> torch.Tensor:
        """logits = beta · stats  (higher → route to mixed/USE_NEIGHBOR)."""
        logits = (
            self.beta_mutual * stats[:, 0]
            + self.beta_snn * stats[:, 1]
            - self.beta_perturb * stats[:, 2]
            - self.beta_uncertainty * stats[:, 3]
        )
        if self.enhanced_stats == 6:
            logits = logits + self.beta_degree * stats[:, 4] - self.beta_cluster * stats[:, 5]
        return logits

    def _temperature(self, epoch: int) -> float:
        """Linear schedule from temperature_init → temperature_min."""
        if epoch <= self.warmup_epochs:
            return self.temperature_init
        t = min(1.0, (epoch - self.warmup_epochs) / max(1, self.ramp_epochs))
        return self.temperature_init + t * (self.temperature_min - self.temperature_init)

    def forward(
        self,
        stats: torch.Tensor,
        epoch: int,
        hard: bool = False,
    ) -> torch.Tensor:
        """Sample routing decision.

        Args:
            stats: (batch, 4) or (batch, 6) topology features.
            epoch: Current training epoch (for temperature schedule).
            hard: If True, use argmax (pure hard routing, no Gumbel noise).
                  If False, use Gumbel-Softmax (differentiable).

        Returns:
            (batch,) tensor of routing decisions:
              1.0 = USE_MIXED (neighbor topology is trusted)
              0.0 = USE_ANCHOR (self-reconstruction)
            Values are either {0.0, 1.0} (hard=True or very low temp)
            or soft probabilities in (0, 1) (high temperature).
        """
        logits = self._compute_logits(stats)

        if hard or not self.training:
            # Inference: pure hard argmax
            return (logits > 0).float()

        temperature = self._temperature(epoch)
        # Gumbel-Softmax: sample from Gumbel - log(-log(Uniform))
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        gumbel_logits = (logits + gumbel_noise) / temperature
        return torch.sigmoid(gumbel_logits)

    def routing_probability(self, stats: torch.Tensor) -> torch.Tensor:
        """Pure routing probability (no Gumbel noise). For analysis only."""
        return torch.sigmoid(self._compute_logits(stats))

    def beta_snapshot(self) -> dict:
        snap = {
            "beta_mutual": float(self.beta_mutual.detach().cpu()),
            "beta_snn": float(self.beta_snn.detach().cpu()),
            "beta_perturb": float(self.beta_perturb.detach().cpu()),
            "beta_uncertainty": float(self.beta_uncertainty.detach().cpu()),
        }
        if self.enhanced_stats == 6:
            snap["beta_degree"] = float(self.beta_degree.detach().cpu())
            snap["beta_cluster"] = float(self.beta_cluster.detach().cpu())
        return snap
