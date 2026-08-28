from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pytest
import torch

from methods.TopoGate.v10_reliable_graph.corruption import apply_mask_corruption
from methods.TopoGate.v10_reliable_graph.gate import EdgeGate
from methods.TopoGate.v10_reliable_graph.graph import (
    GraphState,
    KNNGraph,
    build_knn_graph,
    compute_edge_features,
    edge_recurrence_against,
    edge_features_tensor,
)
from methods.TopoGate.v10_reliable_graph.losses import (
    V10Objective,
    entropy_balance_loss,
    gate_budget_loss,
)
from methods.TopoGate.v10_reliable_graph.mixing import aggregate_neighbors


def test_corruption_mask_is_intervention_mask_when_values_are_unchanged() -> None:
    x = torch.zeros(3, 4)
    corrupted, mask = apply_mask_corruption(x, ratio=1.0, strategy="zero")

    assert torch.equal(corrupted, x)
    assert torch.equal(mask, torch.ones_like(x))


def test_edge_gate_all_parameters_receive_finite_nonzero_gradient() -> None:
    torch.manual_seed(7)
    gate = EdgeGate(feature_dim=5, hidden_dim=8, dropout=0.0)
    features = torch.randn(12, 5)
    output = gate(features)
    loss = (output * torch.linspace(0.5, 1.5, output.numel())).sum()
    loss.backward()

    for name, parameter in gate.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
        assert torch.any(parameter.grad != 0), f"zero gradient for {name}"


def _small_graph() -> KNNGraph:
    indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    shape = indices.shape
    return KNNGraph(
        indices=indices,
        similarity=np.full(shape, 0.8, dtype=np.float32),
        mutual=np.full(shape, 0.25, dtype=np.float32),
        snn=np.full(shape, 0.5, dtype=np.float32),
        density=np.full(shape, 0.35, dtype=np.float32),
        stability=np.full(shape, 0.75, dtype=np.float32),
        embedding=np.eye(3, dtype=np.float32),
        valid_mask=np.ones(shape, dtype=bool),
    )


def test_edge_features_have_canonical_five_fields_without_distance_redundancy() -> None:
    graph = _small_graph()
    assert GraphState is KNNGraph
    fields = tuple(dataclasses.fields(KNNGraph))
    field_names = tuple(field.name for field in fields)
    assert field_names[:7] == (
        "indices",
        "similarity",
        "mutual",
        "snn",
        "density",
        "stability",
        "embedding",
    )
    assert "distance" not in field_names

    features = compute_edge_features(graph)
    assert features.shape == (3, 2, 5)
    np.testing.assert_allclose(features[..., 0], graph.similarity)
    np.testing.assert_allclose(features[..., 1], graph.mutual)
    np.testing.assert_allclose(features[..., 2], graph.snn)
    np.testing.assert_allclose(features[..., 3], graph.density)
    np.testing.assert_allclose(features[..., 4], graph.stability)
    assert not np.allclose(features[..., 3], 1.0 - features[..., 0])
    torch_features = edge_features_tensor(graph)
    assert torch_features.shape == (3, 2, 5)
    assert torch_features.dtype == torch.float32


def test_knn_explicitly_removes_self_when_duplicate_rows_tie() -> None:
    x = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    graph = build_knn_graph(x, k=2, pca_dim=None, seed=3)

    for node, neighbors in enumerate(graph.indices):
        assert node not in neighbors


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="optional FAISS backend unavailable")
def test_hnsw_backend_returns_a_self_free_fixed_width_graph() -> None:
    x = np.random.default_rng(5).normal(size=(32, 6)).astype(np.float32)
    graph = build_knn_graph(
        x,
        k=4,
        pca_dim=None,
        backend="faiss_hnsw",
        hnsw_m=8,
        hnsw_ef_search=24,
    )

    assert graph.profile["knn_backend"] == "faiss_hnsw"
    assert graph.indices.shape == (32, 4)
    for node, neighbors in enumerate(graph.indices):
        assert node not in neighbors


def test_temporal_recurrence_target_is_separate_from_current_edge_feature() -> None:
    graph = _small_graph()
    no_prior = edge_recurrence_against(graph, None)
    same_prior = edge_recurrence_against(graph, graph)

    assert np.count_nonzero(no_prior) == 0
    np.testing.assert_array_equal(same_prior, graph.valid_mask.astype(np.float32))


def test_gate_budget_is_an_upper_bound_so_graph_can_fully_abstain() -> None:
    closed = torch.zeros(4, 3)
    open_ = torch.ones(4, 3)

    assert gate_budget_loss(closed, target=0.5).item() == 0.0
    assert gate_budget_loss(open_, target=0.5).item() > 0.0


def test_entropy_balance_can_preserve_an_unlabeled_nonuniform_prior() -> None:
    assignments = torch.tensor([[0.9, 0.1]]).repeat(16, 1)
    matching = entropy_balance_loss(assignments, prior=torch.tensor([0.9, 0.1]))
    forced_uniform = entropy_balance_loss(assignments, prior=torch.tensor([0.5, 0.5]))

    assert matching < forced_uniform


def test_full_neighbor_aggregation_is_deterministic_and_falls_back_exactly() -> None:
    values = torch.tensor(
        [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]],
    )
    neighbor_indices = torch.tensor([[1, 2], [0, 2], [0, 3], [1, 2]])
    gates = torch.tensor([[1.0, 3.0], [2.0, 0.0], [0.25, 0.75], [0.0, 0.0]])

    aggregate, normalized = aggregate_neighbors(values, neighbor_indices, gates)
    expected = torch.tensor([[4.5, 45.0], [1.0, 10.0], [5.5, 55.0], [7.0, 70.0]])
    expected_weights = torch.tensor(
        [[0.25, 0.75], [1.0, 0.0], [0.25, 0.75], [0.0, 0.0]],
    )
    torch.testing.assert_close(aggregate, expected)
    torch.testing.assert_close(normalized, expected_weights)

    repeat_aggregate, repeat_weights = aggregate_neighbors(values, neighbor_indices, gates)
    assert torch.equal(aggregate, repeat_aggregate)
    assert torch.equal(normalized, repeat_weights)

    closed, closed_weights = aggregate_neighbors(
        values,
        neighbor_indices,
        torch.zeros_like(gates),
    )
    assert torch.equal(closed, values)
    assert torch.equal(closed_weights, torch.zeros_like(gates))


def test_v10_objective_terms_are_finite_and_support_backward() -> None:
    torch.manual_seed(11)
    batch_size, latent_dim, n_clusters, n_neighbors, n_features = 6, 4, 3, 2, 5
    reconstruction = torch.randn(batch_size, n_features, requires_grad=True)
    target = torch.randn(batch_size, n_features)
    mask = torch.randint(0, 2, (batch_size, n_features), dtype=torch.float32)
    z1 = torch.randn(batch_size, latent_dim, requires_grad=True)
    z2 = torch.randn(batch_size, latent_dim, requires_grad=True)
    assignment_logits = torch.randn(batch_size, n_clusters, requires_grad=True)
    assignments = torch.softmax(assignment_logits, dim=-1)
    neighbor_indices = torch.tensor([[1, 2], [0, 2], [0, 3], [1, 4], [2, 5], [3, 4]])
    gate_logits = torch.randn(batch_size, n_neighbors, requires_grad=True)
    gates = torch.sigmoid(gate_logits)
    stability = torch.tensor([[1.0, 0.5], [1.0, 0.5], [0.5, 1.0], [1.0, 0.5], [0.5, 1.0], [1.0, 0.5]])

    objective = V10Objective()
    total, parts = objective(
        reconstruction,
        target,
        mask,
        z1,
        z2,
        assignments,
        neighbor_indices,
        gates,
        stability,
    )

    assert set(parts) == {
        "reconstruction",
        "view_consistency",
        "edge_assignment",
        "entropy_balance",
        "gate_budget",
        "gate_temporal",
        "total",
    }
    for name, value in parts.items():
        assert value.ndim == 0, f"objective part {name} is not scalar"
        assert torch.isfinite(value), f"objective part {name} is non-finite"
    assert torch.isfinite(total)
    total.backward()

    for name, tensor in {
        "reconstruction": reconstruction,
        "z1": z1,
        "z2": z2,
        "assignment_logits": assignment_logits,
        "gate_logits": gate_logits,
    }.items():
        assert tensor.grad is not None, f"missing objective gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"non-finite objective gradient for {name}"
