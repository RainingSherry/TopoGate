"""v6 latent space mix variant — isolated module for TopoGate.

This package implements a **separately isolated** TopoGate variant that mixes
z_anchor and z_neighbor in the encoder's latent space, instead of mixing x_anchor
and neighbor_mean in the input space as `learnable_gate/mixing.py` does.

Design constraints (per project rules):
- No edits to learnable_gate/, static_gate/, NeighborMix_scMAE/model.py, baseline/.
- Reuse `LearnableGate` for the gate calculation, so the gate dynamics are
  directly comparable between the input-space variant and the latent-space one.
- Reuse neighbor selection logic from learnable_gate/mixing.py *only* as a
  helper, not as the primary entry point (no re-implementation of mix logic).

Status: smoke-test prototype.  Phase 1 keeps latent_consistency_weight=0 to
isolate the "where to mix" effect from the "auxiliary loss" effect.
"""
from __future__ import annotations

from .latent_mixer import LatentMixer, build_latent_mixer
from .micro_encoder import MicroMAEEncoder

__all__ = ["LatentMixer", "build_latent_mixer", "MicroMAEEncoder"]