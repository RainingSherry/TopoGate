from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from methods.TopoGate.V16_1_predictive_graph_gate.config import V16_1Config
from methods.TopoGate.V16_1_predictive_graph_gate.gate import (
    assignment_readout,
    cross_fitted_predictive_support,
    summarize_gate,
    shuffle_support,
)
from methods.TopoGate.V16_1_predictive_graph_gate.graph import CandidateGraph, consensus_graph
from methods.TopoGate.V16_1_predictive_graph_gate.run import fit_v16_1
from methods.TopoGate.V16_1_predictive_graph_gate.sparse import (
    DenseNPZReference,
    TheoryDomainError,
    assess_count_domain,
    load_npz_matrix,
    prepare_counts,
    repeated_splits,
    split_counts,
    summarize_split_views,
)
from scripts.V16_1.dataset_registry import load_registry, resolve_metadata
from methods.TopoGate.V16_1_predictive_graph_gate.trainer import DeviceUnavailableError, resolve_device


def _small_counts() -> sp.csr_matrix:
    return sp.csr_matrix(
        np.asarray(
            [
                [3, 0, 0, 0],
                [0, 4, 0, 0],
                [2, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )


def test_bundled_count_registry_declares_local_source() -> None:
    registry = load_registry()
    metadata = resolve_metadata("Bach", registry)
    assert metadata["count_semantics"] == "raw_count"
    assert metadata["source_state"] == "available_local"
    assert metadata["source_version"] == "local_snapshot_unversioned"


def test_registered_count_semantics_keeps_dotted_dataset_name() -> None:
    metadata = resolve_metadata("tr45.wc")
    assert metadata["count_semantics"] == "word_count"
    assert metadata["semantics_source"] == "registered local word-count source"


def test_formal_device_does_not_silently_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError):
        resolve_device(V16_1Config(no_cuda=False, gpu=1))


def test_count_split_conserves_every_entry() -> None:
    first, second = split_counts(_small_counts(), fraction=0.5, seed=7)
    assert np.array_equal((first + second).toarray(), _small_counts().toarray())


def test_count_semantics_requires_source_declaration() -> None:
    values = sp.csr_matrix(np.asarray([[2, 0, 0], [0, 3, 0]], dtype=np.int64))
    _, unverified = assess_count_domain(values, min_feature_dim=1, min_zero_fraction=0.0)
    _, verified = assess_count_domain(
        values,
        count_semantics="raw_count",
        semantics_source="unit-test",
        min_feature_dim=1,
        min_zero_fraction=0.0,
        min_median_nnz=0.0,
    )
    assert unverified["theory_domain"] == "theory_domain_not_supported"
    assert verified["theory_domain"] == "candidate"


def test_dense_input_certificate_is_rejected_by_core_api() -> None:
    values = np.asarray([[2, 0, 0], [0, 3, 0]], dtype=np.int64)
    _, profile = assess_count_domain(
        values,
        count_semantics="raw_count",
        semantics_source="unit-test",
        min_feature_dim=1,
        min_zero_fraction=0.0,
        min_median_nnz=0.0,
        storage_override="dense_npz",
    )
    assert profile["theory_domain"] == "theory_domain_not_supported"
    assert "dense_input_not_supported" in profile["domain_reasons"]
    with pytest.raises(TheoryDomainError):
        prepare_counts(
            values,
            count_semantics="raw_count",
            semantics_source="unit-test",
            min_feature_dim=1,
            min_zero_fraction=0.0,
            min_median_nnz=0.0,
            input_storage="dense_npz",
        )


def test_expanded_count_policy_keeps_bonus_metrics_soft() -> None:
    values = sp.csr_matrix(np.asarray([[2, 0, 0], [0, 3, 0]], dtype=np.int64))
    _, profile = assess_count_domain(
        values,
        count_semantics="raw_count",
        semantics_source="unit-test",
        min_feature_dim=2000,
        min_zero_fraction=0.80,
        min_median_nnz=5.0,
        max_empty_fraction=0.10,
        input_policy="expanded_count",
        storage_override="sparse",
    )
    assert profile["theory_domain"] == "candidate"
    assert profile["domain_tier"] == "count_control"
    assert profile["bonus_feature_count"] == 1
    assert profile["domain_reasons"] == []


def test_csr_npz_bundle_is_loaded_without_dense_x_member(tmp_path) -> None:
    matrix = sp.csr_matrix(np.asarray([[2, 0, 0], [0, 3, 4]], dtype=np.int64))
    path = tmp_path / "counts.npz"
    sp.save_npz(path, matrix)
    loaded, storage = load_npz_matrix(path)
    assert storage == "sparse_npz_csr"
    assert sp.issparse(loaded)
    assert np.array_equal(loaded.toarray(), matrix.toarray())


def test_compressed_dense_npz_returns_header_reference(tmp_path) -> None:
    path = tmp_path / "dense.npz"
    np.savez_compressed(path, x=np.zeros((3, 7), dtype=np.float32))
    loaded, storage = load_npz_matrix(path)
    assert storage == "dense_npz"
    assert isinstance(loaded, DenseNPZReference)
    assert loaded.shape == (3, 7)
    with pytest.raises(TheoryDomainError):
        prepare_counts(
            loaded,
            count_semantics="raw_count",
            semantics_source="unit-test",
            input_storage=storage,
            input_policy="expanded_count",
        )


def test_split_profile_records_nonempty_heldout_views() -> None:
    profile = summarize_split_views(repeated_splits(_small_counts(), 0.5, 3, 42))
    assert profile["repeats"] == 3
    assert profile["has_nonempty_heldout"] is True
    assert 0.0 <= profile["joint_nonempty_row_fraction"] <= 1.0


def test_cross_fitted_support_uses_donor_view_a() -> None:
    view_a = sp.csr_matrix(np.asarray([[5, 0, 0], [0, 5, 0]], dtype=np.int64))
    view_b = sp.csr_matrix(np.asarray([[5, 0, 0], [5, 0, 0]], dtype=np.int64))
    indices = np.asarray([[1], [0]], dtype=np.int64)
    valid = np.ones_like(indices, dtype=bool)
    support, profile = cross_fitted_predictive_support(
        [(view_a, view_b)], indices, valid, smoothing=0.5, exchange_views=False
    )
    assert support[0, 0] < 0.0
    assert profile["donor_profile_view"] == "A"
    assert profile["anchor_evaluation_view"] == "B"


def test_per_token_support_matches_log_ratio_definition() -> None:
    view_a = sp.csr_matrix(np.asarray([[4, 0], [0, 2]], dtype=np.int64))
    view_b = sp.csr_matrix(np.asarray([[2, 0], [0, 2]], dtype=np.int64))
    indices = np.asarray([[0], [1]], dtype=np.int64)
    valid = np.ones_like(indices, dtype=bool)
    smoothing = 1.0
    support, _ = cross_fitted_predictive_support(
        [(view_a, view_b)], indices, valid, smoothing=smoothing, exchange_views=False
    )
    global_counts = np.asarray(view_a.sum(axis=0)).ravel().astype(float)
    p0 = (global_counts + smoothing) / (global_counts.sum() + smoothing * 2)
    donor0 = (np.asarray(view_a.getrow(0).toarray()).ravel() + smoothing * p0) / (4 + smoothing)
    expected = float(np.log(donor0[0]) - np.log(p0[0]))
    assert np.isclose(float(support[0, 0]), expected, atol=1e-6)


def test_cross_fitted_support_exchanges_thinning_roles() -> None:
    view_a = sp.csr_matrix(np.asarray([[4, 0], [0, 2]], dtype=np.int64))
    view_b = sp.csr_matrix(np.asarray([[2, 0], [0, 2]], dtype=np.int64))
    indices = np.asarray([[0], [1]], dtype=np.int64)
    valid = np.ones_like(indices, dtype=bool)
    _, profile = cross_fitted_predictive_support(
        [(view_a, view_b)], indices, valid, smoothing=1.0
    )
    assert profile["support_repeats"] == 1
    assert profile["support_evaluations"] == 2
    assert profile["view_exchange"] is True
    assert profile["donor_profile_view"] == "A_then_B"


def test_cross_fitted_support_separates_sparse_poisson_cluster_edges() -> None:
    rng = np.random.default_rng(123)
    rates = np.asarray(
        [[8, 8, 8, 8, 1, 1, 1, 1], [1, 1, 1, 1, 8, 8, 8, 8]],
        dtype=np.float64,
    )
    rows = []
    labels = []
    for cluster in range(2):
        for _ in range(8):
            rows.append(rng.poisson(rates[cluster]))
            labels.append(cluster)
    labels = np.asarray(labels, dtype=np.int64)
    counts = sp.csr_matrix(np.asarray(rows, dtype=np.int64))
    # Labels define the controlled fixture's same/cross candidate edges only;
    # the support function receives no labels.
    candidates = np.empty((labels.size, 2), dtype=np.int64)
    for row, label in enumerate(labels):
        same = np.flatnonzero(labels == label)
        same = same[same != row]
        cross = np.flatnonzero(labels != label)
        candidates[row] = (same[0], cross[0])
    valid = np.ones_like(candidates, dtype=bool)
    support, _ = cross_fitted_predictive_support(
        repeated_splits(counts, fraction=0.5, repeats=3, seed=42),
        candidates,
        valid,
    )
    assert float(np.mean(support[:, 0])) > float(np.mean(support[:, 1]))


def test_all_negative_support_is_exact_self_fallback() -> None:
    q_self = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
    indices = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    valid = np.ones_like(indices, dtype=bool)
    support = np.full_like(indices, -1.0, dtype=np.float32)
    q_out, pi, _ = assignment_readout(
        q_self,
        indices,
        valid,
        support,
        variant="V16_1_predictive_gate",
        temperature=0.5,
        seed=42,
    )
    assert np.array_equal(q_out, q_self)
    assert np.array_equal(pi[:, 0], np.ones(2, dtype=np.float32))
    assert np.array_equal(pi[:, 1:], np.zeros((2, 2), dtype=np.float32))


def test_output_disabled_is_the_self_only_readout() -> None:
    q_self = np.asarray([[0.6, 0.4], [0.2, 0.8]], dtype=np.float32)
    indices = np.asarray([[1], [0]], dtype=np.int64)
    valid = np.ones_like(indices, dtype=bool)
    support = np.asarray([[3.0], [3.0]], dtype=np.float32)
    q_out, pi, scores = assignment_readout(
        q_self,
        indices,
        valid,
        support,
        variant="output_disabled",
        temperature=0.5,
        seed=42,
    )
    assert np.array_equal(q_out, q_self)
    assert np.array_equal(pi[:, 0], np.ones(2, dtype=np.float32))
    assert np.array_equal(scores, np.zeros_like(support))


def test_consensus_keeps_only_recurrent_edges_and_caps_width() -> None:
    graph_a = CandidateGraph(
        np.asarray([[1, 2], [0, 2]], dtype=np.int64),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
        {},
    )
    graph_b = CandidateGraph(
        np.asarray([[1, 0], [0, 1]], dtype=np.int64),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
        {},
    )
    graph_c = CandidateGraph(
        np.asarray([[1, 2], [0, 1]], dtype=np.int64),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
        {},
    )
    consensus = consensus_graph([graph_a, graph_b, graph_c], k=1, min_repeats=2)
    assert consensus.indices.shape == (2, 1)
    assert consensus.valid.all()
    assert consensus.indices[0, 0] == 1
    assert np.isclose(consensus.profile["stable_edge_rate"], 6.0 / 12.0)


def test_shuffled_support_preserves_rowwise_values() -> None:
    scores = np.asarray([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    valid = np.asarray([[True, True, False], [True, True, True]])
    shuffled = shuffle_support(scores, valid, seed=1)
    for row in range(scores.shape[0]):
        assert np.array_equal(np.sort(shuffled[row, valid[row]]), np.sort(scores[row, valid[row]]))


def test_shuffled_support_changes_assignment_readout() -> None:
    q_self = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]], dtype=np.float32)
    candidates = np.asarray([[1, 2, 0], [0, 2, 1], [0, 1, 2]], dtype=np.int64)
    valid = np.ones_like(candidates, dtype=bool)
    support = np.tile(np.asarray([3.0, 2.0, -1.0], dtype=np.float32), (3, 1))
    q_gate, _, _ = assignment_readout(
        q_self,
        candidates,
        valid,
        support,
        variant="V16_1_predictive_gate",
        temperature=0.5,
        seed=0,
    )
    q_shuffled, _, _ = assignment_readout(
        q_self,
        candidates,
        valid,
        support,
        variant="shuffled_support",
        temperature=0.5,
        seed=0,
    )
    assert not np.array_equal(q_gate, q_shuffled)


def test_effective_neighbors_is_conditional_on_edge_mass() -> None:
    summary = summarize_gate(np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=np.float32))
    assert np.isclose(summary["null_mass"], 0.5)
    assert np.isclose(summary["edge_mass"], 0.5)
    assert np.isclose(summary["conditional_edge_entropy"], np.log(2.0) / 2.0)
    assert np.isclose(summary["effective_neighbors"], 1.5)


def test_fit_writes_assignment_and_latent_outputs(tmp_path) -> None:
    rng = np.random.default_rng(2)
    dense = np.zeros((8, 2000), dtype=np.int64)
    for row in range(8):
        columns = rng.choice(2000, size=8, replace=False)
        dense[row, columns] = rng.integers(2, 5, size=8)
    predictions, summary = fit_v16_1(
        sp.csr_matrix(dense),
        2,
        y=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        config=V16_1Config(
            hidden_dim=8,
            latent_dim=4,
            epochs=1,
            batch_size=4,
            no_cuda=True,
            min_feature_dim=2000,
            min_zero_fraction=0.80,
            min_median_nnz=5.0,
        ),
        save_dir=tmp_path,
        count_semantics="raw_count",
        semantics_source="unit-test",
        input_storage="sparse",
        k_protocol="benchmark_oracle_from_y",
    )
    assert predictions.shape == (8,)
    assert summary["labels_used_during_fit"] is False
    assert summary["benchmark_oracle_from_y"] is True
    cluster_probabilities = np.load(tmp_path / "cluster_probabilities.npy")
    embedding_self = np.load(tmp_path / "embedding_self.npy")
    embedding_final = np.load(tmp_path / "embedding_final.npy")
    assert cluster_probabilities.shape == (8, 2)
    assert embedding_final.shape == (8, 4)
    assert np.array_equal(embedding_final, embedding_self)
    with np.load(tmp_path / "gate_diagnostics.npz") as diagnostics:
        assert np.array_equal(cluster_probabilities, diagnostics["probabilities_final"])


def test_insufficient_samples_for_requested_clusters_is_domain_error() -> None:
    values = sp.csr_matrix(np.asarray([[2, 0, 0], [0, 3, 0]], dtype=np.int64))
    config = V16_1Config(
        no_cuda=True,
        min_feature_dim=1,
        min_zero_fraction=0.0,
        min_median_nnz=0.0,
    )
    with pytest.raises(TheoryDomainError) as error:
        fit_v16_1(
            values,
            3,
            config=config,
            count_semantics="raw_count",
            semantics_source="unit-test",
            input_storage="sparse",
        )
    assert "insufficient_samples_for_clusters" in error.value.profile["domain_reasons"]
