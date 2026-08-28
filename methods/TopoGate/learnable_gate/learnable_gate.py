"""LearnableGate: per-node gate as a learnable affine transform on topology stats.

TopoGate static_gate used 4 hand-tuned beta coefficients that were never exposed to the
gradient path.  LearnableGate promotes them to nn.Parameter so the MAE loss can
shape them per-dataset.  Initializing all betas to zero gives:

    sigmoid(0) = 0.5
    gate       = gate_min + (gate_max - gate_min) * 0.5

which lands at half of the (gate_min, gate_max) range and is numerically close
to the v1 mean gate (~0.079 on a 50-dim synthetic).  Schedule t in [0, 1]
interpolates a precomputed v1-style gate (numpy) toward the live LearnableGate
output so the first warmup_epochs reproduce the static behaviour and the
remaining epochs gradually turn the knobs over to the model.

v3 enhancements:
- learnable_gate_max: the gate_max itself is a learnable parameter (initialised
  at the user-supplied gate_max).  This is the v3 upgrade that solves the
  gate-saturation problem.
- enhanced_stats (default 6): when set to 6 the stats tensor includes
  [mutual, snn, perturb, uncertainty, degree_norm, clustering_coeff] and 6
  betas are learned (beta_degree + beta_cluster).  This expands the topology
  features beyond mutual/snn which the ablation showed were nearly useless.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LearnableGate(nn.Module):
    """Compute per-node gate via sigmoid(beta . stats) with optional schedule.

    Args:
        gate_min, gate_max: Output range for the gate.
        init_beta_mutual, init_beta_snn, init_beta_perturb, init_beta_uncertainty:
            Initial values for the four learnable coefficients.  All four default
            to 0, which yields sigmoid(0) = 0.5 -> gate = mid-point of the
            (gate_min, gate_max) range.  Set these to the v1 defaults
            (1.0, 1.0, 2.0, 1.0) if exact v1 reproduction at schedule=0 is
            required.
        learnable_gate_max: if True, the gate_max is also a learnable parameter
            (initialised at the user-supplied gate_max).  This is the v3 upgrade
            that solves the gate-saturation problem (beta grows but actual gate
            output stays below 0.11 when gate_max=0.15 is fixed).  When True,
            the upper bound of the gate range is gate_max_min + softplus(raw),
            where gate_max_min is the smallest legal value.
        gate_max_min: floor for the learnable gate_max (default 0.05; prevents
            the model from collapsing to a zero-mixing regime).
        gate_max_max: ceiling for the learnable gate_max (default 1.0; prevents
            numerical instability from extreme mixing).
        enhanced_stats: int, default 4.  If set to 6 the stats tensor expects
            [mutual, snn, perturb, uncertainty, degree_norm, clustering_coeff]
            and 6 betas are learned.
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
        init_beta_degree: float = 0.0,
        init_beta_cluster: float = 0.0,
    ) -> None:
        super().__init__()
        if enhanced_stats not in (4, 6):
            raise ValueError(f"enhanced_stats must be 4 or 6, got {enhanced_stats}")
        self.enhanced_stats = int(enhanced_stats)
        self.gate_min = float(gate_min)
        self.gate_max_initial = float(gate_max)
        self.gate_max_min = float(gate_max_min)
        self.gate_max_max = float(gate_max_max)
        self.learnable_gate_max = bool(learnable_gate_max)
        self.beta_mutual = nn.Parameter(torch.tensor(float(init_beta_mutual)))
        self.beta_snn = nn.Parameter(torch.tensor(float(init_beta_snn)))
        self.beta_perturb = nn.Parameter(torch.tensor(float(init_beta_perturb)))
        self.beta_uncertainty = nn.Parameter(torch.tensor(float(init_beta_uncertainty)))
        if self.enhanced_stats == 6:
            self.beta_degree = nn.Parameter(torch.tensor(float(init_beta_degree)))
            self.beta_cluster = nn.Parameter(torch.tensor(float(init_beta_cluster)))
        if self.learnable_gate_max:
            init_raw = self._inverse_softplus(self.gate_max_initial)
            self.gate_max_raw = nn.Parameter(torch.tensor(float(init_raw)))
        else:
            self.register_buffer("gate_max_raw", torch.tensor(0.0))
        # beta_scale: external scalar (no gradient). Controlled by run_npz.py for
        # the legacy nomix-warmup experiment. beta_scale=0 also blocks the beta
        # gradient from this branch; it is not a learn-while-closed mechanism.
        self.register_buffer("beta_scale", torch.tensor(1.0))

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x)

    def _inverse_softplus(self, value: float) -> float:
        # value is the target effective gate_max value (already in [gate_max_min, gate_max_max]).
        # y = gate_max_min + span * sigmoid(raw)  =>  raw = logit((y - min) / span)
        y = float(value)
        span = max(float(self.gate_max_max - self.gate_max_min), 1e-6)
        p = (y - self.gate_max_min) / span
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        return float(np.log(p / (1.0 - p)))

    def effective_gate_max(self) -> torch.Tensor:
        if self.learnable_gate_max:
            raw = self.gate_max_raw
            span = max(float(self.gate_max_max - self.gate_max_min), 1e-6)
            return self.gate_max_min + span * torch.sigmoid(raw)
        return torch.tensor(self.gate_max_initial)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """Compute per-sample gate.

        Args:
            stats: (batch, 4) or (batch, 6) tensor.
                   4: [mutual, snn, perturb, uncertainty]
                   6: [mutual, snn, perturb, uncertainty, degree_norm, clustering_coeff]

        Returns:
            (batch,) gate tensor in [gate_min, effective_gate_max] with gradients
            flowing back to all betas AND gate_max when learnable_gate_max=True.
        """
        expected = self.enhanced_stats
        if stats.ndim != 2 or stats.shape[1] != expected:
            raise ValueError(f"stats must be (batch, {expected}), got {tuple(stats.shape)}")
        logits = (
            self.beta_mutual * stats[:, 0]
            + self.beta_snn * stats[:, 1]
            - self.beta_perturb * stats[:, 2]
            - self.beta_uncertainty * stats[:, 3]
        )
        if expected == 6:
            logits = logits + self.beta_degree * stats[:, 4] - self.beta_cluster * stats[:, 5]
        sig = torch.sigmoid(logits)
        gate_max_t = self.effective_gate_max()
        # beta_scale=0 yields NoMix and zero beta gradient from this branch;
        # beta_scale=1 restores the normal learned gate.
        return self.gate_min + (gate_max_t - self.gate_min) * sig * self.beta_scale

    def beta_snapshot(self) -> dict:
        snap = {
            "beta_mutual": float(self.beta_mutual.detach().cpu()),
            "beta_snn": float(self.beta_snn.detach().cpu()),
            "beta_perturb": float(self.beta_perturb.detach().cpu()),
            "beta_uncertainty": float(self.beta_uncertainty.detach().cpu()),
            "effective_gate_max": float(self.effective_gate_max().detach().cpu()),
            "beta_scale": float(self.beta_scale.detach().cpu()),
        }
        if self.enhanced_stats == 6:
            snap["beta_degree"] = float(self.beta_degree.detach().cpu())
            snap["beta_cluster"] = float(self.beta_cluster.detach().cpu())
        return snap


def build_gate_stats_tensor(
    graph_indices: "np.ndarray",
    graph_mutual: "np.ndarray",
    graph_snn: "np.ndarray",
    graph_probs: "np.ndarray",
    graph_similarity: "np.ndarray",
    uncertainty: "np.ndarray | None" = None,
    device: "torch.device | str | None" = None,
    enhanced_stats: int = 4,
) -> torch.Tensor:
    """Stack the per-node stats into a (n_cells, n_stats) tensor.

    Args:
        enhanced_stats: 4 (default) or 6.  When 6, appends degree_norm and
            clustering_coeff to the per-node vector.

    Ordering (4 stats): [mutual, snn, perturb, uncertainty] — MUST match
        LearnableGate.forward.
    Ordering (6 stats): [mutual, snn, perturb, uncertainty, degree_norm,
        clustering_coeff].
    """
    import numpy as np
    n_cells = int(graph_indices.shape[0])
    mutual_ratio = np.asarray(graph_mutual.mean(axis=1), dtype=np.float32)
    snn_avg = np.asarray(graph_snn.mean(axis=1), dtype=np.float32)
    perturb = (1.0 - np.sum(graph_probs * graph_similarity, axis=1)).astype(np.float32)
    if uncertainty is None:
        unc = np.zeros(n_cells, dtype=np.float32)
    else:
        unc = np.asarray(uncertainty, dtype=np.float32).reshape(-1)
        if unc.shape != (n_cells,):
            raise ValueError(f"uncertainty must have shape ({n_cells},), got {unc.shape}")
    cols = [mutual_ratio, snn_avg, perturb, unc]
    if enhanced_stats == 6:
        degree_norm = np.full(n_cells, float(graph_indices.shape[1]) / max(float(n_cells), 1.0),
                              dtype=np.float32)
        # Approximate local clustering coefficient for each node = (# edges
        # among node's neighbours) / (k * (k-1)).
        # - n ≤ 5000: exact O(n²) computation.
        # - n > 5000: sampled approximation (up to 2000 nodes), then broadcast
        #   the global mean to all nodes so beta_cluster always receives a valid signal.
        n = n_cells
        k = int(graph_indices.shape[1])
        cluster = np.zeros(n, dtype=np.float32)
        if k >= 2 and n > 0:
            if n <= 5000:
                # Small datasets: exact O(n²) computation
                rows = np.repeat(np.arange(n, dtype=np.int64), k)
                cols_idx = graph_indices.ravel()
                adj = np.zeros((n, n), dtype=bool)
                adj[rows, cols_idx] = True
                np.fill_diagonal(adj, False)
                mat = adj.astype(np.int32)
                mat2 = mat @ mat  # (n, n)
                triangles = (mat * mat2).sum(axis=1)
                local_edges = triangles // 2
                cluster = local_edges.astype(np.float32) / float(k * (k - 1))
            else:
                # Large datasets: sampled approximation
                # Sample up to 2000 nodes, compute their local clustering coefficients,
                # then broadcast the global mean as a constant estimate for all nodes.
                m = min(2000, n)
                sample_idx = np.random.choice(n, size=m, replace=False)
                ratios = np.zeros(m, dtype=np.float32)
                for si, i in enumerate(sample_idx):
                    neighbors_i = set(graph_indices[i].tolist())
                    if len(neighbors_i) < 2:
                        ratios[si] = 0.0
                        continue
                    triangles = 0
                    for j in neighbors_i:
                        neighbors_j = set(graph_indices[j].tolist())
                        triangles += len(neighbors_i & neighbors_j)
                    ratios[si] = triangles / float(len(neighbors_i) * (len(neighbors_i) - 1))
                global_cluster = np.mean(ratios)
                cluster[:] = global_cluster  # fill all nodes with global estimate
        cols.append(degree_norm)
        cols.append(cluster)
    elif enhanced_stats != 4:
        raise ValueError(f"enhanced_stats must be 4 or 6, got {enhanced_stats}")
    stacked = np.stack(cols, axis=1)
    return torch.as_tensor(stacked, device=device)
