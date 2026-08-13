from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V16_predictive_graph_gate.gate import (
    abstaining_sparsemax,
    assignment_readout,
    predictive_support,
    shuffle_support,
    summarize_gate,
)
from methods.TopoGate.V16_predictive_graph_gate.graph import build_candidate_graph
from methods.TopoGate.V16_predictive_graph_gate.sparse import (
    assess_count_domain,
    load_npz_matrix,
    prepare_counts,
    split_counts,
)
from scripts.V16.stress import apply_compound_stress


def _counts(seed: int = 3) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    values = rng.poisson(1.0, size=(18, 40)).astype(np.int64)
    values[rng.random(values.shape) < 0.86] = 0
    return sp.csr_matrix(values)


def test_count_thinning_is_nonnegative_and_conservative() -> None:
    counts = _counts()
    first, second = split_counts(counts, 0.5, 7)
    assert first.data.min(initial=0) >= 0
    assert second.data.min(initial=0) >= 0
    assert np.array_equal((first + second).toarray(), counts.toarray())


def test_sparse_candidate_graph_has_no_dense_distance_contract() -> None:
    graph = build_candidate_graph(_counts(), k=4)
    assert graph.indices.shape == (18, 4)
    assert graph.similarity.shape == graph.indices.shape
    assert graph.profile["storage"] == "sparse_cosine_knn"


def test_nonpositive_support_is_exact_null() -> None:
    scores = np.asarray([[-2.0, 0.0, -0.1], [0.3, -0.2, 0.0]], dtype=np.float32)
    valid = np.ones_like(scores, dtype=bool)
    pi = abstaining_sparsemax(scores, valid)
    assert np.array_equal(pi[0], np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.isclose(pi[1].sum(), 1.0)
    assert pi[1, 1] > 0.0
    q = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
    candidates = np.asarray([[1, 0, 1], [0, 0, 1]], dtype=np.int64)
    candidate_valid = np.ones_like(candidates, dtype=bool)
    q_out, _, _ = assignment_readout(
        q,
        candidates,
        candidate_valid,
        np.asarray([[-2.0, -0.1, 0.0], [-1.0, 0.0, -0.2]], dtype=np.float32),
        variant="V16_predictive_gate",
        temperature=0.5,
        seed=1,
    )
    assert np.array_equal(q_out, q)


def test_predictive_support_does_not_accept_labels() -> None:
    first, second = split_counts(_counts(), 0.5, 11)
    graph = build_candidate_graph(first, k=3)
    support, profile = predictive_support([second], graph.indices, graph.valid)
    assert support.shape == graph.indices.shape
    assert "labels" not in profile


def test_predictive_support_matches_manual_risk_difference() -> None:
    counts = sp.csr_matrix(np.asarray([[2, 0, 1], [1, 1, 0]], dtype=np.int64))
    candidates = np.asarray([[1], [0]], dtype=np.int64)
    valid = np.ones_like(candidates, dtype=bool)
    smoothing = 0.5
    support, _ = predictive_support([counts], candidates, valid, smoothing=smoothing)
    global_counts = np.asarray(counts.sum(axis=0)).ravel().astype(float)
    p0 = (global_counts + smoothing) / (global_counts.sum() + smoothing * counts.shape[1])
    donor = np.asarray(counts.getrow(1).toarray()).ravel()
    p1 = (donor + smoothing) / (donor.sum() + smoothing * counts.shape[1])
    expected = -np.sum(np.asarray([2.0, 0.0, 1.0]) * np.log(p0)) + np.sum(
        np.asarray([2.0, 0.0, 1.0]) * np.log(p1)
    )
    assert np.isclose(float(support[0, 0]), expected, atol=1e-6)


def test_effective_neighbors_is_conditional_on_edge_mass() -> None:
    summary = summarize_gate(np.asarray([[1.0, 0.0, 0.0], [0.2, 0.4, 0.4]], dtype=np.float32))
    assert np.isclose(summary["null_mass"], 0.6)
    assert np.isclose(summary["edge_mass"], 0.4)
    assert np.isclose(summary["effective_neighbors"], 1.5)
    assert np.isclose(summary["conditional_edge_entropy"], np.log(2.0) / 2.0)


def test_output_disabled_matches_self_only() -> None:
    rng = np.random.default_rng(2)
    q = rng.random((5, 3)).astype(np.float32)
    q /= q.sum(axis=1, keepdims=True)
    candidates = np.asarray([[1, 2], [0, 3], [4, 0], [2, 1], [3, 2]])
    valid = np.ones_like(candidates, dtype=bool)
    support = rng.normal(size=candidates.shape).astype(np.float32)
    q_self, pi_self, _ = assignment_readout(q, candidates, valid, support, variant="self_only", temperature=0.5, seed=1)
    q_disabled, pi_disabled, _ = assignment_readout(q, candidates, valid, support, variant="output_disabled", temperature=0.5, seed=1)
    assert np.array_equal(q_self, q_disabled)
    assert np.array_equal(pi_self, pi_disabled)
    q_gate, pi_gate, _ = assignment_readout(q, candidates, valid, support, variant="V16_predictive_gate", temperature=0.5, seed=1)
    assert np.allclose(q_gate.sum(axis=1), 1.0)
    assert np.allclose(pi_gate.sum(axis=1), 1.0)


def test_shuffled_support_preserves_values_but_changes_edge_assignment() -> None:
    scores = np.asarray([[0.1, 2.0, 0.5]], dtype=np.float32)
    valid = np.ones_like(scores, dtype=bool)
    shuffled = shuffle_support(scores, valid, seed=0)
    assert np.array_equal(np.sort(shuffled[0]), np.sort(scores[0]))
    assert not np.array_equal(shuffled, scores)


def test_log1p_domain_is_recovered_without_labels() -> None:
    counts = _counts().toarray().astype(np.float64)
    matrix, profile = assess_count_domain(np.log1p(counts), min_feature_dim=1)
    assert profile["count_semantics"] == "log1p_integer"
    assert np.array_equal(matrix.toarray(), counts)
    prepared = prepare_counts(np.log1p(counts), enforce_domain=False, min_feature_dim=1)
    assert prepared.counts.dtype == np.dtype(np.int64)


def test_npz_loader_uses_chunked_sparse_path(tmp_path) -> None:
    values = np.asarray([[1, 0, 2], [0, 3, 0]], dtype=np.int64)
    path = tmp_path / "tiny.npz"
    np.savez(path, x=values, y=np.asarray([0, 1]))
    loaded, storage = load_npz_matrix(path, chunk_rows=1)
    assert sp.issparse(loaded)
    assert storage == "sparse_npz_chunked"
    assert np.array_equal(loaded.toarray(), values)


def test_domain_certificate_records_dense_input_and_full_value_failure() -> None:
    _, dense_profile = assess_count_domain(
        np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int64),
        min_feature_dim=1,
        storage_override="dense_npz",
    )
    assert "dense_input_not_supported" in dense_profile["domain_reasons"]
    malformed = sp.csr_matrix(np.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=np.float64))
    _, malformed_profile = assess_count_domain(malformed, min_feature_dim=1)
    assert malformed_profile["count_semantics"] == "unsupported"


def test_compound_stress_preserves_count_domain_shape() -> None:
    stressed, metadata, mask = apply_compound_stress(_counts(), seed=9)
    assert stressed.shape == _counts().shape
    assert stressed.data.min(initial=0) >= 0
    assert np.allclose(stressed.data, np.rint(stressed.data))
    assert metadata["mode"] == "compound"
    assert mask.shape == (stressed.shape[0],)
