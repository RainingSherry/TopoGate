from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .candidate import CandidateSet, build_candidate_union
from .config import V17Config
from .input_adapter import build_projection_views, prepare_input
from .relation import RelationResult, affinity_from_coefficients, solve_candidate_self_expression
from .spectral import SpectralResult, normalized_spectral_readout


@dataclass(frozen=True)
class TopologyState:
    candidates: CandidateSet
    relation: RelationResult
    affinity: sp.csr_matrix
    profile: dict[str, Any]


def fit_topology(X: np.ndarray | sp.spmatrix, config: V17Config | None = None) -> TopologyState:
    """Fit input geometry, candidates, gate, and affinity without K or labels."""
    config = config or V17Config()
    prepared = prepare_input(X, input_mode=config.input_mode)
    projections = build_projection_views(
        prepared,
        n_views=config.projection_views,
        projection_dim=config.projection_dim,
        density=config.projection_density,
        seed=config.seed,
    )
    candidates = build_candidate_union(
        projections.values,
        k_per_view=config.candidate_k,
        union_k=config.candidate_union_k,
        block_size=config.candidate_block_size,
    )
    relation = solve_candidate_self_expression(
        projections.values,
        candidates,
        lambda_l1=config.lambda_l1,
        lambda_l2=config.lambda_l2,
        lambda_outlier=config.lambda_outlier,
        max_iter=config.solver_max_iter,
        tolerance=config.solver_tol,
        coefficient_epsilon=config.coefficient_epsilon,
    )
    affinity = affinity_from_coefficients(relation.coefficients)
    profile = {
        "input": prepared.profile,
        "projection": projections.profile,
        "candidate": candidates.profile,
        "relation": relation.profile,
        "labels_used": False,
        "K_used": False,
        "affinity_definition": "abs(C)+abs(C.T)",
    }
    return TopologyState(candidates, relation, affinity, profile)


def readout_topology(
    topology: TopologyState,
    n_clusters: int,
    config: V17Config | None = None,
) -> SpectralResult:
    """Use K only to discretize the already-fitted topology."""
    config = config or V17Config()
    return normalized_spectral_readout(
        topology.affinity,
        int(n_clusters),
        seed=config.seed,
        n_init=config.spectral_n_init,
        degree_epsilon=config.degree_epsilon,
    )
