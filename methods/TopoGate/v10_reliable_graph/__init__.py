"""TopoGate V10: dynamic reliable-edge graph clustering core.

The package is isolated from historical V1--V9 implementations.  It exposes
only label-free graph construction, differentiable edge gating, deterministic
aggregation, a scalable masked autoencoder, and clustering-aligned objectives.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .corruption import CorruptionStrategy, apply_mask_corruption
from .gate import EdgeGate
from .graph import (
    EDGE_FEATURE_NAMES,
    GraphState,
    KNNGraph,
    build_consensus_graph,
    build_knn_graph,
    compute_edge_features,
    edge_recurrence_against,
    edge_features_tensor,
)
from .losses import (
    V10LossWeights,
    V10Objective,
    combine_v10_losses,
    edge_assignment_js_loss,
    entropy_balance_loss,
    gate_budget_loss,
    gate_regularization,
    gate_stability_loss,
    masked_reconstruction_loss,
    view_consistency_loss,
)
from .mixing import (
    aggregate_neighbors,
    full_neighbor_aggregate,
    gather_edge_assignments,
    mix_with_reliable_neighbors,
)
from .model import V10AutoEncoder


def __getattr__(name: str) -> Any:
    """Load the training entry points without making core imports eager."""

    if name in {"run_v10", "train_v10"}:
        value = getattr(import_module("methods.TopoGate.v10_reliable_graph.run"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CorruptionStrategy",
    "EDGE_FEATURE_NAMES",
    "EdgeGate",
    "GraphState",
    "KNNGraph",
    "V10AutoEncoder",
    "V10LossWeights",
    "V10Objective",
    "aggregate_neighbors",
    "apply_mask_corruption",
    "build_consensus_graph",
    "build_knn_graph",
    "compute_edge_features",
    "combine_v10_losses",
    "edge_assignment_js_loss",
    "edge_features_tensor",
    "edge_recurrence_against",
    "entropy_balance_loss",
    "full_neighbor_aggregate",
    "gate_budget_loss",
    "gate_regularization",
    "gate_stability_loss",
    "gather_edge_assignments",
    "masked_reconstruction_loss",
    "mix_with_reliable_neighbors",
    "run_v10",
    "train_v10",
    "view_consistency_loss",
]
