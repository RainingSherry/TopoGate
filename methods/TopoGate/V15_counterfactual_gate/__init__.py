"""TopoGate V15: counterfactual utility gating for sparse high-dimensional data."""

from .config import V15Config, load_config
from .graph import CandidateGraph, build_candidate_graph, replace_candidate_edges
from .model import V15Model, abstaining_sparsemax, sparsemax

__all__ = [
    "CandidateGraph",
    "V15Config",
    "V15Model",
    "build_candidate_graph",
    "replace_candidate_edges",
    "fit_v15",
    "load_config",
    "run_v15",
    "sparsemax",
    "abstaining_sparsemax",
]


def __getattr__(name: str):
    if name in {"fit_v15", "run_v15"}:
        from .run import fit_v15, run_v15

        return {"fit_v15": fit_v15, "run_v15": run_v15}[name]
    raise AttributeError(name)
