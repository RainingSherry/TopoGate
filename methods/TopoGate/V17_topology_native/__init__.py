"""TopoGate V17 topology-native reference implementation."""

from .candidate import CandidateSet, build_candidate_union, shuffle_candidate_donors
from .config import V17Config, load_config
from .input_adapter import PreparedInput, ProjectionViews, build_projection_views, prepare_input
from .model import TopologyState, fit_topology, readout_topology
from .relation import RelationResult, affinity_from_coefficients, soft_threshold, solve_candidate_self_expression
from .spectral import SpectralResult, normalized_spectral_readout


def fit_v17(*args, **kwargs):
    """Lazily import the artifact-writing API so ``python -m ...run`` stays clean."""
    from .run import fit_v17 as _fit_v17

    return _fit_v17(*args, **kwargs)

__all__ = [
    "CandidateSet",
    "PreparedInput",
    "ProjectionViews",
    "RelationResult",
    "SpectralResult",
    "TopologyState",
    "V17Config",
    "affinity_from_coefficients",
    "build_candidate_union",
    "build_projection_views",
    "fit_topology",
    "fit_v17",
    "load_config",
    "normalized_spectral_readout",
    "prepare_input",
    "readout_topology",
    "shuffle_candidate_donors",
    "soft_threshold",
    "solve_candidate_self_expression",
]
