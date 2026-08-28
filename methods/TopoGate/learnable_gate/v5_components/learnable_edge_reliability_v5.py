"""v5 Edge reliability: simplified single-gamma learnable version.

v3 LearnableEdgeReliability had 4 learnable gammas (sim/mutual/snn/distance).
Phase 2.1 diagnosis showed: across 5 datasets × 3 seeds, all 4 gammas
converged to EXACTLY the same value (std=0.000000). This happens because:

  d(loss)/d(γ_k) ∝ feature_k(d/d_edge_weight)
where each feature_k has the same scale ~[0,1] and roughly same distribution,
so the 4 gradients are similar, and the 4 gammas drift together.

This v5 module fixes the issue in two ways (controlled by mode):

- mode='all_params_4f': 4 learnable γ (legacy v3 behaviour, kept for ablation)
- mode='one_param_scalar': a single learnable γ scalar used for all 4 signals
  (since they correlate, this single-γ form is mathematically equivalent and
  more stable).
- mode='one_fixed_one_learnable': γ_sim fixed=1.0, all others = single γ
  (different initialisation per feature)
- mode='one_param_per_learnable_lr': 4 γ, each with its own lr multiplier
  (so gradient magnitudes are no longer determined solely by feature scale).

The component is isolated from the rest of the pipeline by replicating the
public surface of `LearnableEdgeReliability` from v3.  To enable: set the
v5 runner flag --learnable_edge_reliability_v5 and --v5_gamma_mode <mode>.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from learnable_gate.neighbor_graph import NeighborGraph




VALID_MODES = (
    "all_params_4f",        # legacy v3: 4 separate gammas
    "one_param_scalar",     # v5 default: single scalar γ for all 4 features
    "one_fixed_one_learnable",  # γ_sim=1.0 fixed, others share 1 γ
    "one_param_per_learnable_lr",  # 4 γ, each with own lr multiplier
)


class LearnableEdgeReliabilityV5(nn.Module):
    """v5 edge reliability: simplified single-γ learnable variant.

    Modes:
      - 'all_params_4f': legacy v3 behaviour.
      - 'one_param_scalar': a single γ used as: rel = exp(γ*sim) * (1+γ*mutual)
        * (1+γ*snn) * exp(-γ*distance).  Mathematically equivalent to v3 when
        all four gammas collapse to the same value (which they always do).
      - 'one_fixed_one_learnable': γ_sim = 1.0 fixed, rest = γ (single learnable).
        Tests whether freezing sim helps learning.
      - 'one_param_per_learnable_lr': 4 γ, each gradient multiplied by a
        'learnable lr' per-feature (allows rebalancing gradient magnitudes).
    """

    def __init__(
        self,
        mode: str = "one_param_scalar",
        init_gamma: float = 1.0,
        # v3-compat kwargs (ignored if not applicable to mode)
        init_gamma_sim: float | None = None,
        init_gamma_mutual: float | None = None,
        init_gamma_snn: float | None = None,
        init_gamma_distance: float | None = None,
        # per-γ lr multipliers (mode='one_param_per_learnable_lr')
        init_lr_mul_sim: float = 1.0,
        init_lr_mul_mutual: float = 1.0,
        init_lr_mul_snn: float = 1.0,
        init_lr_mul_distance: float = 1.0,
        reg_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        # v3 legacy modes translated to v5 equivalent
        _MODE_TRANSLATION = {
            "sim": "all_params_4f",
            "sim_mutual": "all_params_4f",
            "sim_mutual_snn": "all_params_4f",
            "sim_mutual_snn_distance": "all_params_4f",
        }
        mode = _MODE_TRANSLATION.get(mode, mode)
        if mode not in VALID_MODES:
            raise ValueError(f"mode {mode!r} not in {VALID_MODES}")
        self.mode = mode
        self.reg_weight = float(reg_weight)

        # resolve init values, prefer v3-style if provided
        if init_gamma_sim is not None:
            init_gamma = init_gamma_sim

        if mode == "all_params_4f":
            self.gamma_sim = nn.Parameter(torch.tensor(float(init_gamma_sim or init_gamma)))
            self.gamma_mutual = nn.Parameter(torch.tensor(float(init_gamma_mutual or init_gamma)))
            self.gamma_snn = nn.Parameter(torch.tensor(float(init_gamma_snn or init_gamma)))
            self.gamma_distance = nn.Parameter(torch.tensor(float(init_gamma_distance or init_gamma)))
        elif mode == "one_param_scalar":
            self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        elif mode == "one_fixed_one_learnable":
            self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))
        elif mode == "one_param_per_learnable_lr":
            self.gamma_sim = nn.Parameter(torch.tensor(float(init_lr_mul_sim)))
            self.gamma_mutual = nn.Parameter(torch.tensor(float(init_lr_mul_mutual)))
            self.gamma_snn = nn.Parameter(torch.tensor(float(init_lr_mul_snn)))
            self.gamma_distance = nn.Parameter(torch.tensor(float(init_lr_mul_distance)))

    def effective_gammas(self) -> tuple[float, float, float, float]:
        """Return (sim, mutual, snn, distance) gammas for logging."""
        if self.mode == "all_params_4f":
            return (
                float(self.gamma_sim.detach().cpu()),
                float(self.gamma_mutual.detach().cpu()),
                float(self.gamma_snn.detach().cpu()),
                float(self.gamma_distance.detach().cpu()),
            )
        if self.mode == "one_param_scalar":
            g = float(self.gamma.detach().cpu())
            return (g, g, g, g)
        if self.mode == "one_fixed_one_learnable":
            g = float(self.gamma.detach().cpu())
            return (1.0, g, g, g)
        # one_param_per_learnable_lr
        return (
            float(self.gamma_sim.detach().cpu()),
            float(self.gamma_mutual.detach().cpu()),
            float(self.gamma_snn.detach().cpu()),
            float(self.gamma_distance.detach().cpu()),
        )

    def gamma_snapshot(self) -> dict:
        sim, mutual, snn, dist = self.effective_gammas()
        return {
            "gamma_sim": sim,
            "gamma_mutual": mutual,
            "gamma_snn": snn,
            "gamma_distance": dist,
        }

    def regularization_loss(self) -> torch.Tensor:
        if self.reg_weight <= 0:
            return torch.zeros((), device=next(self.parameters()).device)
        if self.mode == "all_params_4f":
            sq = (
                self.gamma_sim ** 2
                + self.gamma_mutual ** 2
                + self.gamma_snn ** 2
                + self.gamma_distance ** 2
            )
        elif self.mode == "one_param_scalar":
            sq = self.gamma ** 2 * 4
        elif self.mode == "one_fixed_one_learnable":
            sq = self.gamma ** 2 * 3
        else:
            sq = (
                self.gamma_sim ** 2
                + self.gamma_mutual ** 2
                + self.gamma_snn ** 2
                + self.gamma_distance ** 2
            )
        return self.reg_weight * sq

    def forward(self, graph: NeighborGraph) -> tuple[torch.Tensor, torch.Tensor]:
        if graph.indices.shape[1] == 0 or self.mode == "none":
            rel = torch.ones_like(graph.probs, dtype=torch.float32)
            return rel, rel.clone()

        device = next(self.parameters()).device
        sim_t = torch.as_tensor(graph.similarity, dtype=torch.float32, device=device)
        mutual_t = torch.as_tensor(graph.mutual.astype(np.float32), dtype=torch.float32, device=device)
        snn_t = torch.as_tensor(graph.snn, dtype=torch.float32, device=device)
        distance_t = torch.as_tensor(graph.distance, dtype=torch.float32, device=device)
        probs_t = torch.as_tensor(graph.probs, dtype=torch.float32, device=device)

        rel = torch.ones(graph.similarity.shape, dtype=torch.float32, device=device)
        if self.mode == "all_params_4f":
            rel = rel * torch.exp(self.gamma_sim * sim_t)
            rel = rel * (1.0 + self.gamma_mutual * mutual_t)
            rel = rel * (1.0 + self.gamma_snn * snn_t)
            rel = rel * torch.exp(-self.gamma_distance * distance_t)
        elif self.mode == "one_param_scalar":
            g = self.gamma
            rel = rel * torch.exp(g * sim_t)
            rel = rel * (1.0 + g * mutual_t)
            rel = rel * (1.0 + g * snn_t)
            rel = rel * torch.exp(-g * distance_t)
        elif self.mode == "one_fixed_one_learnable":
            g = self.gamma
            rel = rel * torch.exp(1.0 * sim_t)  # γ_sim fixed at 1.0
            rel = rel * (1.0 + g * mutual_t)
            rel = rel * (1.0 + g * snn_t)
            rel = rel * torch.exp(-g * distance_t)
        elif self.mode == "one_param_per_learnable_lr":
            rel = rel * torch.exp(self.gamma_sim * sim_t)
            rel = rel * (1.0 + self.gamma_mutual * mutual_t)
            rel = rel * (1.0 + self.gamma_snn * snn_t)
            rel = rel * torch.exp(-self.gamma_distance * distance_t)

        rel = torch.clamp(rel, min=1e-6, max=1e6)
        weights = probs_t * rel
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
        return rel, weights


def summarize_edge_weights_torch(weights_t: torch.Tensor) -> dict:
    if weights_t.numel() == 0:
        return {
            "edge_weight_entropy": 0.0,
            "effective_neighbor_count": 0.0,
            "max_edge_weight_mean": 0.0,
            "max_edge_weight_p95": 0.0,
            "fraction_effective_neighbors_lt_2": 1.0,
        }
    entropy = -torch.sum(weights_t * torch.log(torch.clamp(weights_t, min=1e-12)), dim=1)
    effective = torch.exp(entropy)
    max_w = torch.max(weights_t, dim=1).values
    return {
        "edge_weight_entropy": float(entropy.mean().detach().cpu()),
        "effective_neighbor_count": float(effective.mean().detach().cpu()),
        "max_edge_weight_mean": float(max_w.mean().detach().cpu()),
        "max_edge_weight_p95": float(torch.quantile(max_w, 0.95).detach().cpu()),
        "fraction_effective_neighbors_lt_2": float((effective < 2.0).float().mean().detach().cpu()),
    }
