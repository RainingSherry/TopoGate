from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.sparse as sp

from methods.TopoGate.V18_scmae_latent_gate.config import V18Config
from methods.TopoGate.V18_scmae_latent_gate.graph import build_candidate_graph, shuffle_candidate_graph
from methods.TopoGate.V18_scmae_latent_gate.model import fit_v18
from methods.TopoGate.V18_scmae_latent_gate.relation import EdgeGate, SparseRelation
from methods.TopoGate.V18_scmae_latent_gate.run import run_one
from methods.TopoGate.V18_scmae_latent_gate.scmae import masked_view
from methods.TopoGate.V18_scmae_latent_gate.spectral import normalized_spectral_readout


def _views() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(4)
    first = rng.normal(size=(18, 6)).astype(np.float32)
    return first, (first + 0.02 * rng.normal(size=first.shape)).astype(np.float32)


def _small_config() -> V18Config:
    return V18Config(seed=3, input_mode="continuous", hidden_size=8, mask_ratio=0.3,
                     n_views=2, candidate_k=3, candidate_width=5, batch_size=6,
                     epochs_mae=1, epochs_gate=1, epochs_joint=1, solver_max_iter=5,
                     spectral_n_init=2, device="cpu")


def test_candidate_graph_has_fixed_sparse_support_and_no_self_loops() -> None:
    graph = build_candidate_graph(_views(), k=3, width=5)
    assert graph.features.shape == (18, graph.width, 5)
    assert graph.profile["full_pairwise_matrix_materialized"] is False
    assert not np.any(graph.indices[graph.valid] == np.repeat(np.arange(18), graph.valid.sum(axis=1)))


def test_shuffled_control_recomputes_features_for_new_edges() -> None:
    views = _views()
    graph = build_candidate_graph(views, k=3, width=5)
    shuffled = shuffle_candidate_graph(graph, seed=19, views=views)
    assert shuffled.profile["features_recomputed_for_edges"] is True
    for i, slot in zip(*np.where(shuffled.valid), strict=True):
        j = int(shuffled.indices[i, slot])
        expected = float(np.mean([
            np.dot(view[i] / max(np.linalg.norm(view[i]), 1e-12),
                   view[j] / max(np.linalg.norm(view[j]), 1e-12))
            for view in views
        ]))
        assert np.isclose(shuffled.features[i, slot, 0], expected, atol=1e-5)


def test_fit_signature_has_no_labels_or_k_in_core_fit() -> None:
    signature = inspect.signature(fit_v18)
    assert "y" not in signature.parameters
    assert "labels" not in signature.parameters
    assert "n_clusters" in signature.parameters


def test_sparse_relation_preserves_exact_gate_support() -> None:
    graph = build_candidate_graph(_views(), k=2, width=3)
    initial = np.ones((graph.n_nodes, graph.width), dtype=np.float32)
    relation = SparseRelation(graph, initial)
    gate = np.zeros_like(initial)
    gate[graph.valid] = 1.0
    gate[0, 0] = 0.0
    coefficients = relation.coefficients(__import__("torch").as_tensor(gate)).detach().numpy()
    assert np.all(coefficients[~graph.valid] == 0.0)
    assert coefficients[0, 0] == 0.0


def test_gate_initialization_is_not_hard_open_saturated() -> None:
    gate = EdgeGate(5, init_bias=-2.0)
    features = __import__("torch").zeros((12, 5))
    _, expected, _ = gate(features, temperature=0.7, sample=False)
    assert float(expected.mean().detach()) < 0.8


def test_masked_view_reports_effective_value_changes() -> None:
    import torch

    values = torch.zeros((4, 5), dtype=torch.float32)
    corrupted, mask = masked_view(values, 0.5, torch.Generator().manual_seed(11))
    np.testing.assert_array_equal(corrupted.numpy(), values.numpy())
    np.testing.assert_array_equal(mask.numpy(), np.zeros_like(mask.numpy()))


def test_full_v18_smoke_writes_same_c_affinity_and_keeps_labels_out(tmp_path) -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(18, 7)).astype(np.float32)
    result = fit_v18(X, 2, config=_small_config(), variant="v18_full", save_dir=tmp_path)
    assert result.summary["labels_used_during_fit"] is False
    assert result.coefficients is not None and result.affinity is not None
    coefficient = sp.csr_matrix(result.coefficients)
    expected = abs(coefficient) + abs(coefficient.T)
    assert (expected != result.affinity).nnz == 0
    assert (tmp_path / "predictions.npy").exists()
    assert (tmp_path / "summary.json").exists()


def test_leiden_readout_does_not_require_k(tmp_path) -> None:
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    rng = np.random.default_rng(13)
    X = rng.normal(size=(16, 6)).astype(np.float32)
    result = fit_v18(X, None, config=_small_config(), variant="v18_leiden", save_dir=tmp_path)
    assert result.summary["n_clusters"] is None
    assert result.summary["K_used_only_in_readout"] is False
    assert result.summary["readout"]["K_used_only_in_readout"] is False


def test_leiden_runner_accepts_unlabeled_input_without_k(tmp_path) -> None:
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    rng = np.random.default_rng(14)
    data_path = tmp_path / "unlabeled.npy"
    np.save(data_path, rng.normal(size=(16, 6)).astype(np.float32))
    summary = run_one(data_path, tmp_path / "run", config=_small_config(), variant="v18_leiden",
                      n_clusters=None)
    assert summary["K_source"] == "not_applicable_leiden"
    assert summary["benchmark_oracle_from_y"] is False
    assert summary["n_clusters"] is None


def test_all_zero_readout_is_explicit_abstention() -> None:
    result = normalized_spectral_readout(sp.csr_matrix((5, 5)), 2, seed=1, n_init=2, degree_epsilon=1e-12)
    np.testing.assert_array_equal(result.labels, np.full(5, -1, dtype=np.int64))
    assert result.profile["status"] == "all_abstained"
