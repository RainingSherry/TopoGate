"""TopoGate V0: one scVICAR model with fixed (F) and topology (T) settings."""

from __future__ import annotations

from .config import V0Config, load_config, normalize_parameterization
from .corruption import apply_scmae_noise, compute_node_gate, make_pseudo_batch
from .graph import NeighborGraph, build_pca_knn_graph, compute_edge_reliability
from .model import AutoEncoder, ScVICARAutoEncoder, WeightedAutoEncoder
from .trainer import fit_predict, resolve_device

__all__ = [
    "AutoEncoder",
    "NeighborGraph",
    "ScVICARAutoEncoder",
    "V0Config",
    "WeightedAutoEncoder",
    "apply_scmae_noise",
    "build_pca_knn_graph",
    "compute_edge_reliability",
    "compute_node_gate",
    "fit_predict",
    "load_config",
    "make_pseudo_batch",
    "normalize_parameterization",
    "resolve_device",
]
