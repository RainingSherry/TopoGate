"""TopoGate package facade with lazy legacy exports.

Importing a specific TopoGate variant must not eagerly import every historical
implementation.  In particular, the static-gate runner pulls in the optional
``scanpy`` stack, which is unrelated to the generic tabular V10 path.  The
legacy top-level symbols remain available through :func:`__getattr__`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_MODULE_EXPORTS = {
    "static_gate": "methods.TopoGate.static_gate",
    "learnable_gate": "methods.TopoGate.learnable_gate",
    "v10_reliable_graph": "methods.TopoGate.v10_reliable_graph",
    "V11": "methods.TopoGate.V11",
    "V12_latent_topology": "methods.TopoGate.V12_latent_topology",
    "V17_topology_native": "methods.TopoGate.V17_topology_native",
}

_ATTRIBUTE_EXPORTS = {
    "AutoEncoder": ("methods.TopoGate.learnable_gate.model", "AutoEncoder"),
    "NeighborGraph": ("methods.TopoGate.learnable_gate.neighbor_graph", "NeighborGraph"),
    "build_pca_knn_graph": ("methods.TopoGate.learnable_gate.neighbor_graph", "build_pca_knn_graph"),
    "build_random_neighbors": ("methods.TopoGate.learnable_gate.neighbor_graph", "build_random_neighbors"),
    "build_far_neighbors": ("methods.TopoGate.learnable_gate.neighbor_graph", "build_far_neighbors"),
    "compute_edge_reliability": ("methods.TopoGate.learnable_gate.neighbor_graph", "compute_edge_reliability"),
    "compute_node_gate": ("methods.TopoGate.learnable_gate.mixing", "compute_node_gate"),
    "make_pseudo_batch": ("methods.TopoGate.learnable_gate.mixing", "make_pseudo_batch"),
    "embedding_geometry": ("methods.TopoGate.learnable_gate.diagnostics", "embedding_geometry"),
    "mapped_predictions": ("methods.TopoGate.learnable_gate.diagnostics", "mapped_predictions"),
    "per_cell_type_metrics": ("methods.TopoGate.learnable_gate.diagnostics", "per_cell_type_metrics"),
    "LearnableGate": ("methods.TopoGate.learnable_gate.learnable_gate", "LearnableGate"),
    "build_gate_stats_tensor": ("methods.TopoGate.learnable_gate.learnable_gate", "build_gate_stats_tensor"),
    "run_topogate": ("methods.TopoGate.learnable_gate.run_npz", "run_topogate"),
    "run_npz_main": ("methods.TopoGate.learnable_gate.run_npz", "main"),
}


def __getattr__(name: str) -> Any:
    """Load historical modules and symbols only when they are requested."""

    if name in _MODULE_EXPORTS:
        value = import_module(_MODULE_EXPORTS[name])
    elif name in _ATTRIBUTE_EXPORTS:
        module_name, attribute_name = _ATTRIBUTE_EXPORTS[name]
        value = getattr(import_module(module_name), attribute_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    *_ATTRIBUTE_EXPORTS,
    *_MODULE_EXPORTS,
]
