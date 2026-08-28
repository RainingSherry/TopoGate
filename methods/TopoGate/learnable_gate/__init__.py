"""LearnableGate package (formerly "v2").

The current mainline implementation where the 4 gate coefficients (β_mutual,
β_snn, β_perturb, β_uncertainty) are learnable nn.Parameter.
"""
from methods.TopoGate.learnable_gate.model import AutoEncoder
from methods.TopoGate.learnable_gate.mixing import compute_node_gate, make_pseudo_batch
from methods.TopoGate.learnable_gate.neighbor_graph import (
    NeighborGraph,
    build_pca_knn_graph,
    build_random_neighbors,
    build_far_neighbors,
    compute_edge_reliability,
)
from methods.TopoGate.learnable_gate.diagnostics import (
    embedding_geometry,
    mapped_predictions,
    per_cell_type_metrics,
)
from methods.TopoGate.learnable_gate.learnable_gate import LearnableGate, build_gate_stats_tensor
from methods.TopoGate.learnable_gate.run_npz import run_topogate, main as run_npz_main

__all__ = [
    "AutoEncoder",
    "NeighborGraph",
    "build_pca_knn_graph",
    "build_random_neighbors",
    "build_far_neighbors",
    "compute_edge_reliability",
    "compute_node_gate",
    "make_pseudo_batch",
    "embedding_geometry",
    "mapped_predictions",
    "per_cell_type_metrics",
    "LearnableGate",
    "build_gate_stats_tensor",
    "run_topogate",
    "run_npz_main",
]
