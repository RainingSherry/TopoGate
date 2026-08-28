"""LearnableEdgeReliability: promote 4 gamma coefficients to nn.Parameter.

The original `compute_edge_reliability` in neighbor_graph.py takes 4 gamma
coefficients as argparse-fixed constants (gamma_sim=1.0, gamma_mutual=1.0,
gamma_snn=1.0, gamma_distance=1.0).  The 90-run multiseed analysis shows that
the ablation with all gammas fixed (gate_only - full = -0.0009) does NOT improve
ARI on 5 datasets — these coefficients are dead parameters from the gradient's
perspective.

This module wraps the 4 gamma into nn.Parameter so they can be learned via the
MAE + pseudo reconstruction loss.  The weights are computed from a 2D embedding
(numpy → torch tensor) so that autograd flows back.

Design choices:
- Parameters are kept raw (not softplus'd) so the initial value 1.0 corresponds
  to the original v1 default.
- To prevent numerical explosion (very large gamma_mutual can make rel = inf),
  we add a soft L2 regularisation term that the training loop accumulates.
  The actual loss term is exposed as `regularization_loss()`.
- The forward() returns torch tensors so the rest of the pipeline (which
  converts to numpy for kNN sampling) gets the gradient-tracking version.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .neighbor_graph import NeighborGraph, summarize_edge_weights


class LearnableEdgeReliability(nn.Module):
    """Per-edge reliability as a learnable affine combination of 4 signals.

    Args:
        mode: 'sim', 'sim_mutual', 'sim_mutual_snn', 'sim_mutual_snn_distance'.
              When 'none' or empty graph, falls back to graph.probs unchanged.
        init_gamma_sim, init_gamma_mutual, init_gamma_snn, init_gamma_distance:
              Initial values for the four learnable gammas.  Default 1.0 matches
              the v1 argparse defaults.
        reg_weight: weight for the soft L2 regularisation loss (default 1e-4).
                    Prevents the gammas from drifting to extremes.
    """

    def __init__(
        self,
        mode: str = "sim_mutual_snn_distance",
        init_gamma_sim: float = 1.0,
        init_gamma_mutual: float = 1.0,
        init_gamma_snn: float = 1.0,
        init_gamma_distance: float = 1.0,
        reg_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.reg_weight = float(reg_weight)
        self.gamma_sim = nn.Parameter(torch.tensor(float(init_gamma_sim)))
        self.gamma_mutual = nn.Parameter(torch.tensor(float(init_gamma_mutual)))
        self.gamma_snn = nn.Parameter(torch.tensor(float(init_gamma_snn)))
        self.gamma_distance = nn.Parameter(torch.tensor(float(init_gamma_distance)))

    def gamma_snapshot(self) -> dict:
        return {
            "gamma_sim": float(self.gamma_sim.detach().cpu()),
            "gamma_mutual": float(self.gamma_mutual.detach().cpu()),
            "gamma_snn": float(self.gamma_snn.detach().cpu()),
            "gamma_distance": float(self.gamma_distance.detach().cpu()),
        }

    def regularization_loss(self) -> torch.Tensor:
        """L2 penalty on gammas to keep them from drifting to extremes."""
        if self.reg_weight <= 0:
            return torch.zeros((), device=self.gamma_sim.device)
        sq = (
            self.gamma_sim ** 2
            + self.gamma_mutual ** 2
            + self.gamma_snn ** 2
            + self.gamma_distance ** 2
        )
        return self.reg_weight * sq

    def forward(self, graph: NeighborGraph) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (reliability, weights) as torch tensors with gradient flow.

        Args:
            graph: NeighborGraph object holding similarity / mutual / snn / distance
                   as numpy arrays.

        Returns:
            (rel, weights): both (n_cells, k) torch tensors.  rel has gradient
            flowing back to the 4 gamma params; weights are the row-normalised
            version.  Both have requires_grad=False numpy equivalents for logging.
        """
        if graph.indices.shape[1] == 0 or self.mode == "none":
            rel = torch.ones_like(graph.probs, dtype=torch.float32)
            weights = rel.clone()
            return rel, weights

        rel = torch.ones(graph.similarity.shape, dtype=torch.float32,
                         device=self.gamma_sim.device)
        sim_t = torch.as_tensor(graph.similarity, dtype=torch.float32,
                                device=self.gamma_sim.device)
        mutual_t = torch.as_tensor(graph.mutual.astype(np.float32), dtype=torch.float32,
                                   device=self.gamma_sim.device)
        snn_t = torch.as_tensor(graph.snn, dtype=torch.float32,
                                device=self.gamma_sim.device)
        distance_t = torch.as_tensor(graph.distance, dtype=torch.float32,
                                     device=self.gamma_sim.device)
        probs_t = torch.as_tensor(graph.probs, dtype=torch.float32,
                                  device=self.gamma_sim.device)
        if self.mode in {"sim", "sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
            rel = rel * torch.exp(self.gamma_sim * sim_t)
        if self.mode in {"sim_mutual", "sim_mutual_snn", "sim_mutual_snn_distance"}:
            rel = rel * (1.0 + self.gamma_mutual * mutual_t)
        if self.mode in {"sim_mutual_snn", "sim_mutual_snn_distance"}:
            rel = rel * (1.0 + self.gamma_snn * snn_t)
        if self.mode == "sim_mutual_snn_distance":
            rel = rel * torch.exp(-self.gamma_distance * distance_t)
        rel = torch.clamp(rel, min=1e-6, max=1e6)
        weights = probs_t * rel
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-12)
        return rel, weights


# Helper to convert a torch (n, k) tensor back to numpy for downstream
# numpy-only mix_mode code paths.  The caller is responsible for tracking
# gradients BEFORE this conversion (the conversion is only done for paths
# that don't go through `make_pseudo_batch(..., gate_tensor=...)`).
def edge_weights_to_numpy(weights_t: torch.Tensor) -> "np.ndarray":
    import numpy as np
    return weights_t.detach().cpu().numpy().astype(np.float32)


def summarize_edge_weights_torch(weights_t: torch.Tensor) -> dict:
    """Like summarize_edge_weights but for torch tensors (used in summary logging)."""
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
