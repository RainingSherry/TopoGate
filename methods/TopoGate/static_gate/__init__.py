"""StaticGate package.

This directory contains the v1-phase code (read-only).  The original
`methods.TopoGate.*` imports are rewritten to point at v1-local copies.

If you want the current mainline implementation, use:
    from methods.TopoGate.learnable_gate.run_npz import run_topogate
"""
from methods.TopoGate.static_gate.model import AutoEncoder
from methods.TopoGate.static_gate.mixing import compute_node_gate, make_pseudo_batch
from methods.TopoGate.static_gate.neighbor_graph import (
    NeighborGraph,
    build_pca_knn_graph,
    build_random_neighbors,
    build_far_neighbors,
    compute_edge_reliability,
)
from methods.TopoGate.static_gate.diagnostics import (
    embedding_geometry,
    mapped_predictions,
    per_cell_type_metrics,
)
from methods.TopoGate.static_gate.run import main as run_v1_main

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
    "run_v1_main",
]
