"""v6 LatentMixer — replaces learnable_gate.mixing.make_pseudo_batch.

Design
------
The input-space variant mixes anchors and their neighbour means in *input* space:

    x_prime = (1 - gate) * x_anchor + gate * x_neighbor_mean       (input space)
    target  = x_anchor                                              (input space)

This is the path that makes the MAE task "reconstruct the clean anchor from
a noisy mix of two expression vectors" — a non-trivial loss channel that is
empirically *worse* than not mixing at all (-0.015 ARI over 15 datasets).

The v6 alternative mixes the two encoder outputs in *latent* space and asks
the decoder to reconstruct the anchor:

    z_a     = encoder(mask(anchor))           (encoder has seen the manifold)
    z_n     = encoder(mask(neighbour))
    z_mixed = (1 - gate) * z_a + gate * z_n
    recon   = decoder_from_latent(z_mixed)    (decoder reads from the manifold)
    loss    = MAE(recon, x_anchor)            (target still in input space)

The motivation is that encoder has already projected both inputs onto the
data manifold, so even a "wrong" neighbour sits at a sensible manifold
location rather than at a raw L2-noise location.  This trades "mix noise in
the worst possible place" (input) for "mix noise inside the encoder's
own representation space" (latent).

Parameter parity with LearnableGate
-----------------------------------
The gate is computed by `LearnableGate.forward(stats)`, so the per-node gate
dynamics are **identical** to the learnable_gate variant for the same
node_stats.  This isolates the experiment to "where to mix", with the gate
function held constant.

Auxiliary loss
--------------
`latent_consistency_weight` is exposed but defaults to 0 in Phase 1 so that
Phase 5.1 (smoke test) measures the pure "move mix to latent" effect.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from methods.TopoGate.learnable_gate.learnable_gate import LearnableGate


class LatentMixer(nn.Module):
    """v6 latent-space mix module.

    Holds a `LearnableGate` for gate calculation (parameter parity with the
    input-space variant) and applies the mix in latent space.

    Args:
        gate_min, gate_max: Output range for the gate (default (0, 0.5)).
        init_beta_mutual, init_beta_snn, init_beta_perturb, init_beta_uncertainty:
            Initial values for the four learnable coefficients.  All four
            default to 0 → sigmoid(0)=0.5 → gate = mid of (gate_min, gate_max).
            Set to v1 defaults (1.0, 1.0, 2.0, 1.0) for exact v1 reproduction
            at the start of training.
        learnable_gate_max: if True, `LearnableGate.gate_max` itself is a
            learnable scalar (initialised at `gate_max`).  This is the v3
            upgrade that prevents gate-saturation when β grows.
        gate_max_min, gate_max_max: lower / upper bound for the learnable
            `gate_max` parameter (only relevant when `learnable_gate_max=True`).
        enhanced_stats: 4 (default) or 6.
        latent_consistency_weight: weight for the auxiliary latent-consistency
            loss ||z_mixed - z_anchor||^2.  Phase 1 default 0 (off).

    Schedule (parity with run_npz.py)
    ---------------------------------
    When called with `schedule_t=<1.0` and `static_gate=<numpy array>`, the
    effective per-node gate is

        gate_eff = (1 - t) * static_gate + t * gate_dyn

    so that the first `warmup_epochs` epochs reproduce the v1 static behaviour
    exactly (β gradients only flow during `t > 0`) and the next `ramp_epochs`
    interpolate toward the live LearnableGate output.  Once `t = 1.0` the
    static anchor is dropped and `gate_eff = gate_dyn` (matching run_npz.py).
    """

    def __init__(
        self,
        gate_min: float = 0.0,
        gate_max: float = 0.5,
        init_beta_mutual: float = 0.0,
        init_beta_snn: float = 0.0,
        init_beta_perturb: float = 0.0,
        init_beta_uncertainty: float = 0.0,
        learnable_gate_max: bool = False,
        gate_max_min: float = 0.05,
        gate_max_max: float = 1.0,
        enhanced_stats: int = 4,
        latent_consistency_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_consistency_weight = float(latent_consistency_weight)
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

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_neighbor: torch.Tensor,
        stats: torch.Tensor,
        static_gate: "torch.Tensor | None" = None,
        schedule_t: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Apply latent-space mix.

        Args:
            z_anchor:   (batch, hidden) latent for the anchor batch.
            z_neighbor: (batch, hidden) latent for the sampled neighbour batch.
            stats:      (batch, enhanced_stats) per-node stats (same shape
                        consumed by LearnableGate).
            static_gate: optional (batch,) tensor of pre-computed v1-style
                        gates (numpy→torch).  When provided together with
                        `schedule_t < 1.0`, the effective gate becomes
                        `(1 - t) * static_gate + t * gate_dyn`.  Set to None
                        (the default) to skip the schedule and use `gate_dyn`
                        directly, matching run_npz.py behaviour at `t = 1`.
            schedule_t: interpolation scalar in [0, 1].  0 = pure static gate,
                        1 = pure learnable gate.  Outside [0, 1] is clipped.

        Returns:
            z_mixed:    (batch, hidden) mixed latent.
            mix_summary: dict with keys
                - mean_node_gate  (after schedule interpolation)
                - min_node_gate
                - max_node_gate
                - effective_gate_max
                - schedule_t
                - latent_consistency_loss (only if weight > 0)
        """
        if z_anchor.shape != z_neighbor.shape:
            raise ValueError(
                f"z_anchor and z_neighbor must have identical shape, got "
                f"{tuple(z_anchor.shape)} and {tuple(z_neighbor.shape)}."
            )
        if z_anchor.shape[0] != stats.shape[0]:
            raise ValueError(
                f"batch size mismatch: z has {z_anchor.shape[0]} but stats "
                f"has {stats.shape[0]}."
            )
        gate_dyn = self.gate(stats).view(-1, 1).to(dtype=z_anchor.dtype)
        t = float(max(0.0, min(1.0, schedule_t)))
        if t < 1.0 and static_gate is not None:
            sg = static_gate.to(dtype=gate_dyn.dtype, device=gate_dyn.device).view(-1, 1)
            gate = (1.0 - t) * sg + t * gate_dyn
        else:
            gate = gate_dyn

        z_mixed = (1.0 - gate) * z_anchor + gate * z_neighbor

        latent_consistency_loss = z_mixed.new_zeros(())
        if self.latent_consistency_weight > 0.0:
            latent_consistency_loss = ((z_mixed - z_anchor.detach()) ** 2).mean()

        mix_summary = {
            "mean_node_gate": float(gate.mean().detach().cpu()),
            "min_node_gate": float(gate.min().detach().cpu()),
            "max_node_gate": float(gate.max().detach().cpu()),
            "effective_gate_max": float(self.gate.effective_gate_max().detach().cpu()),
            "schedule_t": t,
            "latent_consistency_loss": float(latent_consistency_loss.detach().cpu()),
        }
        return z_mixed, mix_summary

    def beta_snapshot(self) -> dict:
        return self.gate.beta_snapshot()


def build_latent_mixer(
    gate_min: float = 0.0,
    gate_max: float = 0.5,
    init_beta_mutual: float = 0.0,
    init_beta_snn: float = 0.0,
    init_beta_perturb: float = 0.0,
    init_beta_uncertainty: float = 0.0,
    learnable_gate_max: bool = False,
    gate_max_min: float = 0.05,
    gate_max_max: float = 1.0,
    enhanced_stats: int = 4,
    latent_consistency_weight: float = 0.0,
) -> LatentMixer:
    """Convenience builder."""
    return LatentMixer(
        gate_min=gate_min,
        gate_max=gate_max,
        init_beta_mutual=init_beta_mutual,
        init_beta_snn=init_beta_snn,
        init_beta_perturb=init_beta_perturb,
        init_beta_uncertainty=init_beta_uncertainty,
        learnable_gate_max=learnable_gate_max,
        gate_max_min=gate_max_min,
        gate_max_max=gate_max_max,
        enhanced_stats=enhanced_stats,
        latent_consistency_weight=latent_consistency_weight,
    )