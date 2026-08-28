"""TopoGate V12: latent-space topology alignment.

This package is intentionally independent from the historical V9
``learnable_gate`` runner. It keeps topology out of the reconstruction input
and uses a differentiable edge gate only for a latent alignment objective.
"""

from .learnable_gate import (
    EDGE_FEATURE_NAMES,
    LearnableGate,
    build_gate_stats_tensor,
    topology_alignment_loss,
)
from .model import AutoEncoder

__all__ = [
    "AutoEncoder",
    "EDGE_FEATURE_NAMES",
    "LearnableGate",
    "build_gate_stats_tensor",
    "topology_alignment_loss",
]
