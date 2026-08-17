from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.representation_consumer_probe.protocol import (
    CONSUMER_CONTRACTS,
    CONFIG,
    CandidatePool,
    audit_adapter_semantics,
    build_candidate_pool,
    build_h0,
    build_oracle_full_graph,
    build_oracle_pool_graph,
    build_random_graph,
    build_ungated_graph,
    budget_profile,
    directed_graph_from_rows,
    graph_budget_audit,
    gview,
    laplacians,
    numerical_loss_contract,
    spectral_embedding_with_audit,
    spectral_predict_with_audit,
    synthetic_apparatus_sanity,
)


def test_frozen_contract_has_new_budget_and_zero_eigenspace() -> None:
    assert CONFIG.neighbor_k == 20
    assert CONFIG.retention_ratio == 0.4
    assert CONFIG.budget_cap == 8
    assert CONFIG.primary_budget == CONFIG.budget_cap  # compatibility spelling only
    assert CONFIG.budget_sensitivity == (4, 8, 12)
    assert CONFIG.svd_random_state == 0
    assert CONFIG.spectral_drop_first is False
    assert CONFIG.batch_mode == "full_graph"
    assert CONFIG.labels_vector_used_in_fit is False


def test_h0_is_deterministic_and_candidate_pool_is_positive() -> None:
    rng = np.random.default_rng(7)
    x = np.abs(rng.normal(size=(40, 18))).astype(np.float32)
    h0_a, profile_a = build_h0(x)
    h0_b, profile_b = build_h0(x)
    np.testing.assert_array_equal(h0_a, h0_b)
    assert profile_a == profile_b
    pool = build_candidate_pool(h0_a)
    assert pool.indices.shape == (40, 20)
    assert np.all(pool.cosine[(pool.indices >= 0) & (pool.cosine != 0)] > 0)
    mask = np.zeros_like(pool.indices, dtype=bool)
    for row in range(mask.shape[0]):
        positive = np.flatnonzero((pool.indices[row] >= 0) & (pool.cosine[row] > 0))
        b_i = min(CONFIG.budget_cap, positive.size)
        mask[row, positive[:b_i]] = True
    graph = directed_graph_from_rows(pool, mask)
    assert graph.nnz == int(np.minimum(CONFIG.budget_cap, pool.positive_counts).sum())
    assert graph.diagonal().sum() == 0


def test_empty_candidate_pool_still_records_zero_budget_contract() -> None:
    pool = build_candidate_pool(np.ones((1, 4), dtype=np.float32))
    assert pool.indices.shape == (1, 0)
    assert np.array_equal(pool.effective_budget, np.array([0]))
    assert pool.profile["budget_cap"] == CONFIG.budget_cap
    assert pool.profile["fraction_budget_zero"] == 1.0


def test_laplacian_and_gview_are_finite_with_isolates() -> None:
    graph = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32))
    h0 = np.eye(3, 2, dtype=np.float32)
    ops = laplacians(graph)
    assert ops["isolated_nodes"] == 1
    assert np.isfinite(ops["L_sym"].data).all()
    assert np.isfinite(gview(h0, graph)).all()
    assert numerical_loss_contract(h0, graph)["finite"]


def test_current_v21_boundary_is_not_a_sample_edge_adapter() -> None:
    audit = audit_adapter_semantics(".")
    assert audit["status"] == "adapter_not_estimable"
    assert audit["semantic_fidelity_required"] is True
    assert audit["existing_edge_membership_symbol"] is False


def test_synthetic_apparatus_sanity() -> None:
    result = synthetic_apparatus_sanity()
    assert result["purpose"] == "apparatus_sanity_only"
    assert all(result["direction_checks"].values())
    assert result["graph_numerical_sanity"]["loss_finite"]
    assert result["spectral_recovery_sanity"]["clean_ari"] >= 0.95


def _small_pool() -> CandidatePool:
    indices = np.array(
        [
            [2, 1, -1],
            [0, 2, -1],
            [0, 1, 3],
            [2, -1, -1],
        ],
        dtype=np.int64,
    )
    cosine = np.array(
        [
            [0.99, 0.70, 0.0],
            [0.95, 0.60, 0.0],
            [0.90, 0.80, 0.75],
            [0.65, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    positive_counts = np.sum(cosine > 0, axis=1).astype(np.int64)
    return CandidatePool(indices, cosine, positive_counts, {})


def _row_counts(graph: sp.spmatrix) -> np.ndarray:
    return np.diff(graph.tocsr().indptr)


def test_row_specific_budget_is_shared_by_random_and_oracles() -> None:
    pool = _small_pool()
    expected = np.minimum(CONFIG.budget_cap, pool.positive_counts)
    assert np.array_equal(pool.effective_budget, expected)
    profile = budget_profile(pool)
    assert profile["budget_cap"] == 8
    assert profile["effective_budget_hash"] == pool.budget_hash
    random_graph = build_random_graph(pool, seed=42)
    labels = np.array([0, 0, 1, 1])
    pool_oracle = build_oracle_pool_graph(pool, labels)
    # Keep the diagnostic pool and full-space H0 consistent; all four rows
    # have enough positive full-space neighbours to honour the pool budget.
    h0 = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.7, 0.7], [0.6, 0.8]], dtype=np.float32
    )
    full_oracle = build_oracle_full_graph(h0, labels, pool=pool)
    for graph in (random_graph, pool_oracle, full_oracle):
        assert np.array_equal(_row_counts(graph), expected)
        assert graph_budget_audit(graph, pool)["effective_budget_hash"] == pool.budget_hash
    # Row 0 has one same-class candidate (node 1); the pool oracle must keep it.
    row0_neighbors = set(pool_oracle.getrow(0).indices.tolist())
    assert 1 in row0_neighbors


def test_random_builder_has_no_label_input_and_is_seed_reproducible() -> None:
    pool = _small_pool()
    first = build_random_graph(pool, seed=123)
    second = build_random_graph(pool, seed=123)
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    # This small fixture has no surplus positive slots; the seed is still part
    # of the API and is exercised by the without-replacement implementation.
    assert build_random_graph(pool, seed=7).shape == first.shape


def test_ungated_builder_keeps_all_positive_candidate_rows() -> None:
    pool = _small_pool()
    graph = build_ungated_graph(pool)
    expected = pool.positive_counts
    assert np.array_equal(_row_counts(graph), expected)
    assert graph.diagonal().sum() == 0


def test_spectral_consumer_uses_active_subgraph_and_zero_isolates() -> None:
    graph = sp.csr_matrix(
        (
            np.array([1.0, 0.8, 1.1, 0.9], dtype=np.float32),
            (np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0])),
        ),
        shape=(6, 6),
    )
    predictions, embedding, metadata = spectral_predict_with_audit(graph, 2, seed=42)
    assert predictions.shape == (6,)
    assert embedding.shape == (6, 2)
    assert np.all(embedding[4:] == 0.0)
    assert metadata["active_nodes"] == 4
    assert metadata["isolated_nodes"] == 2
    assert metadata["K_used_in_representation"] is True


def test_k_usage_is_consumer_level_not_global() -> None:
    assert CONSUMER_CONTRACTS["Spectral"] == {
        "K_used_in_representation": True,
        "K_used_in_readout": True,
        "labels_vector_used_in_fit": False,
    }
    for consumer in ("SimpleCut", "F"):
        assert CONSUMER_CONTRACTS[consumer]["K_used_in_representation"] is False
        assert CONSUMER_CONTRACTS[consumer]["K_used_in_readout"] is True
        assert CONSUMER_CONTRACTS[consumer]["labels_vector_used_in_fit"] is False
