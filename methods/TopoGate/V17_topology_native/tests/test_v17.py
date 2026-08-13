from __future__ import annotations

import inspect

import numpy as np
import scipy.sparse as sp

from methods.TopoGate.V17_topology_native.candidate import (
    CandidateSet,
    build_candidate_union,
    shuffle_candidate_donors,
)
from methods.TopoGate.V17_topology_native.config import V17Config
from methods.TopoGate.V17_topology_native.input_adapter import (
    build_projection_views,
    load_sparse_npz,
    prepare_input,
)
from methods.TopoGate.V17_topology_native.model import fit_topology, readout_topology
from methods.TopoGate.V17_topology_native.relation import (
    affinity_from_coefficients,
    soft_threshold,
    solve_candidate_self_expression,
)
from methods.TopoGate.V17_topology_native.run import fit_v17
from methods.TopoGate.V17_topology_native.spectral import normalized_spectral_readout


def _manual_candidates() -> CandidateSet:
    indices = np.asarray([[1, 2], [0, 3], [3, 0], [2, 1]], dtype=np.int64)
    valid = np.ones_like(indices, dtype=bool)
    similarity = np.ones_like(indices, dtype=np.float32)
    return CandidateSet(indices, similarity, valid, valid.astype(np.int16), {"kind": "test"})


def _small_count_matrix() -> sp.csr_matrix:
    values = np.asarray(
        [
            [9, 6, 1, 0, 0, 0],
            [8, 7, 0, 0, 0, 0],
            [7, 8, 1, 0, 0, 0],
            [6, 9, 0, 0, 0, 0],
            [0, 0, 0, 8, 7, 1],
            [0, 0, 0, 9, 6, 0],
            [0, 0, 1, 7, 8, 0],
            [0, 0, 0, 6, 9, 1],
        ],
        dtype=np.float32,
    )
    return sp.csr_matrix(values)


def _small_config() -> V17Config:
    return V17Config(
        seed=11,
        input_mode="count",
        projection_views=2,
        projection_dim=4,
        candidate_k=3,
        candidate_union_k=5,
        candidate_block_size=3,
        lambda_l1=0.0,
        lambda_l2=1e-3,
        lambda_outlier=10.0,
        solver_max_iter=60,
        solver_tol=1e-7,
    )


def test_csr_field_npz_bundle_is_loaded_without_dense_matrix(tmp_path) -> None:
    matrix = _small_count_matrix()
    path = tmp_path / "bundle.npz"
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
    )
    loaded = load_sparse_npz(str(path))
    assert sp.isspmatrix_csr(loaded)
    np.testing.assert_array_equal(loaded.toarray(), matrix.toarray())


def test_sparse_input_and_candidate_path_do_not_form_full_pairwise_matrix() -> None:
    prepared = prepare_input(_small_count_matrix(), input_mode="count")
    assert sp.isspmatrix_csr(prepared.matrix)
    projections = build_projection_views(prepared, n_views=2, projection_dim=4, density="auto", seed=7)
    candidates = build_candidate_union(
        projections.values,
        k_per_view=2,
        union_k=3,
        block_size=3,
    )
    assert candidates.profile["full_pairwise_matrix_materialized"] is False
    assert candidates.indices.shape == (prepared.n_samples, 3)
    assert candidates.width < prepared.n_samples


def test_auto_mode_does_not_claim_count_semantics_from_integer_values() -> None:
    prepared = prepare_input(_small_count_matrix(), input_mode="auto")
    assert prepared.profile["input_mode_resolved"] == "nonnegative"
    assert prepared.profile["transform"] == "row_l2"


def test_soft_threshold_creates_exact_zeros() -> None:
    values = np.asarray([-0.5, -0.1, 0.0, 0.1, 0.5], dtype=np.float64)
    gated = soft_threshold(values, 0.1)
    np.testing.assert_array_equal(gated[1:4], np.zeros(3, dtype=np.float64))
    np.testing.assert_allclose(gated[[0, 4]], [-0.4, 0.4])


def test_relation_never_leaves_candidate_support_and_has_zero_diagonal() -> None:
    view = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    candidates = _manual_candidates()
    result = solve_candidate_self_expression(
        [view],
        candidates,
        lambda_l1=0.001,
        lambda_l2=0.001,
        lambda_outlier=10.0,
        max_iter=100,
        tolerance=1e-8,
        coefficient_epsilon=1e-10,
    )
    rows, cols = result.coefficients.nonzero()
    allowed = {
        (anchor, int(candidates.indices[anchor, position]))
        for anchor, position in zip(*np.where(candidates.valid), strict=True)
    }
    assert set(zip(rows.tolist(), cols.tolist(), strict=True)) <= allowed
    np.testing.assert_array_equal(result.coefficients.diagonal(), np.zeros(4))
    assert result.profile["coefficient_nnz"] > 0
    assert result.profile["optimizer"] == "fista_proximal_gradient"


def test_affinity_is_exactly_abs_c_plus_abs_transpose() -> None:
    coefficient = sp.csr_matrix(
        np.asarray([[0.0, -0.4, 0.0], [0.2, 0.0, 0.3], [0.0, 0.0, 0.0]], dtype=np.float32)
    )
    affinity = affinity_from_coefficients(coefficient)
    expected = np.abs(coefficient.toarray()) + np.abs(coefficient.toarray().T)
    np.testing.assert_allclose(affinity.toarray(), expected)
    np.testing.assert_allclose(affinity.toarray(), affinity.toarray().T)
    assert np.min(affinity.data) >= 0.0


def test_degree_zero_nodes_have_explicit_abstention_semantics() -> None:
    affinity = sp.csr_matrix(
        np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    result = normalized_spectral_readout(affinity, 2, seed=3, n_init=5, degree_epsilon=1e-12)
    assert result.labels[-1] == -1
    assert result.abstained[-1]
    np.testing.assert_array_equal(result.embedding[-1], np.zeros(2, dtype=np.float32))
    assert result.profile["status"] == "partial_abstention"


def test_all_zero_topology_does_not_invent_a_feature_space_partition() -> None:
    result = normalized_spectral_readout(sp.csr_matrix((4, 4)), 2, seed=3, n_init=5, degree_epsilon=1e-12)
    np.testing.assert_array_equal(result.labels, np.full(4, -1, dtype=np.int64))
    assert result.profile["status"] == "all_abstained"


def test_shuffled_candidate_control_is_label_free_and_excludes_self() -> None:
    signature = inspect.signature(shuffle_candidate_donors)
    assert "y" not in signature.parameters and "labels" not in signature.parameters
    shuffled = shuffle_candidate_donors(_manual_candidates(), seed=5)
    for anchor in range(shuffled.n_nodes):
        assert anchor not in shuffled.indices[anchor, shuffled.valid[anchor]]


def test_k_and_labels_are_absent_from_topology_fit() -> None:
    signature = inspect.signature(fit_topology)
    assert "n_clusters" not in signature.parameters
    assert "y" not in signature.parameters
    topology = fit_topology(_small_count_matrix(), _small_config())
    before = topology.relation.coefficients.copy()
    readout_topology(topology, 2, _small_config())
    readout_topology(topology, 3, _small_config())
    difference = topology.relation.coefficients - before
    assert difference.nnz == 0


def test_labels_change_metrics_only_and_outputs_have_declared_semantics(tmp_path) -> None:
    X = _small_count_matrix()
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_labels, first = fit_v17(X, 2, y, config=_small_config(), save_dir=first_dir)
    second_labels, second = fit_v17(X, 2, y[::-1], config=_small_config(), save_dir=second_dir)
    np.testing.assert_array_equal(first_labels, second_labels)
    first_c = sp.load_npz(first_dir / "coefficient_matrix.npz")
    second_c = sp.load_npz(second_dir / "coefficient_matrix.npz")
    assert (first_c - second_c).nnz == 0
    np.testing.assert_array_equal(np.load(first_dir / "predictions.npy"), first_labels)
    assert np.load(first_dir / "embedding_final.npy").shape == (X.shape[0], 2)
    assert first["labels_used_during_fit"] is False
    assert first["K_used_in_relation_solver"] is False
    assert second["topology"]["K_used"] is False
