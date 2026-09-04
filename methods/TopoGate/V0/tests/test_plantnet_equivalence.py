"""Small reference checks against the frozen PlantNet F/T operators.

These tests compare the values consumed by the training loop, not just the
shared V0 API shape.  The PlantNet checkout is an optional test dependency;
when it is absent the normal V0 test suite remains runnable and these checks
are skipped with an explicit reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from methods.TopoGate.V0.corruption import (
    apply_scmae_noise,
    compute_node_gate,
    make_pseudo_batch,
)
from methods.TopoGate.V0.graph import build_pca_knn_graph, compute_edge_reliability


PLANTNET_ROOT = Path("/home/luolie/biopipeline/dimension-reduction/plantnet")


def _plantnet_modules():
    """Import the frozen F/T modules without making them V0 dependencies."""

    if not PLANTNET_ROOT.is_dir():
        pytest.skip(f"PlantNet checkout is not available: {PLANTNET_ROOT}")
    root_text = str(PLANTNET_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        from experimental_retired_models.NeighborMix_scMAE import run as fixed
        from experimental_retired_models.RG_NeighborMix_scMAE import mixing as topology_mixing
        from experimental_retired_models.RG_NeighborMix_scMAE import neighbor_graph as topology_graph
    except Exception as exc:  # pragma: no cover - environment-dependent optional stack
        pytest.skip(f"PlantNet reference modules are unavailable: {exc}")
    return fixed, topology_graph, topology_mixing


def _data() -> np.ndarray:
    return np.asarray(np.random.default_rng(902).normal(size=(24, 11)), dtype=np.float32)


def test_fixed_operator_matches_frozen_plantnet_on_toy_data() -> None:
    fixed, _topology_graph, _topology_mixing = _plantnet_modules()
    data = _data()

    old_indices, old_probs, _ = fixed.build_knn_distribution(
        data, k=5, pca_dim=7, tau=0.2, seed=42
    )
    new_graph = build_pca_knn_graph(data, k=5, pca_dim=7, tau=0.2, seed=42)
    np.testing.assert_array_equal(new_graph.indices, old_indices)
    np.testing.assert_array_equal(new_graph.probs, old_probs)

    batch_indices = np.array([0, 2, 8, 19], dtype=np.int64)
    batch = torch.as_tensor(data[batch_indices])
    old_view = fixed.sample_mix(
        data,
        batch_indices,
        batch,
        alpha=0.9,
        mix_neighbors=4,
        rng=np.random.default_rng(777),
        neighbor_indices=old_indices,
        neighbor_probs=old_probs,
    )
    gate, _sample_weight, _summary = compute_node_gate(
        new_graph, parameterization="fixed", alpha=0.9
    )
    new_view, new_sample_weight, _info = make_pseudo_batch(
        data,
        batch_indices,
        batch,
        parameterization="fixed",
        graph=new_graph,
        edge_weights=new_graph.probs,
        node_gate=gate,
        mix_neighbors=4,
        alpha=0.9,
        rng=np.random.default_rng(777),
        neighbor_estimator="current",
        legacy_plantnet=True,
    )
    torch.testing.assert_close(new_view, old_view, rtol=0, atol=0)
    torch.testing.assert_close(
        new_sample_weight, torch.ones(batch_indices.size, dtype=batch.dtype), rtol=0, atol=0
    )


def test_topology_operator_matches_frozen_plantnet_on_toy_data() -> None:
    _fixed, topology_graph, topology_mixing = _plantnet_modules()
    data = _data()

    old_graph = topology_graph.build_pca_knn_graph(data, k=5, pca_dim=7, tau=0.2, seed=42)
    new_graph = build_pca_knn_graph(data, k=5, pca_dim=7, tau=0.2, seed=42)
    for field in ("indices", "probs", "similarity", "distance", "mutual", "snn"):
        np.testing.assert_array_equal(getattr(new_graph, field), getattr(old_graph, field))

    old_reliability, old_weights, _ = topology_graph.compute_edge_reliability(
        old_graph, "sim_mutual_snn_distance", 1.0, 1.0, 1.0, 1.0
    )
    new_reliability, new_weights, _ = compute_edge_reliability(
        new_graph, "sim_mutual_snn_distance", 1.0, 1.0, 1.0, 1.0
    )
    np.testing.assert_array_equal(new_reliability, old_reliability)
    np.testing.assert_array_equal(new_weights, old_weights)

    old_gate, _old_diagnostic_weight, _ = topology_mixing.compute_node_gate(
        old_graph,
        old_weights,
        "topology",
        0.0,
        0.15,
        1.0,
        1.0,
        2.0,
        1.0,
    )
    new_gate, _new_diagnostic_weight, _ = compute_node_gate(
        new_graph,
        parameterization="topology",
        gate_min=0.0,
        gate_max=0.15,
        beta_mutual=1.0,
        beta_snn=1.0,
        beta_perturb=2.0,
        beta_uncertainty=1.0,
    )
    np.testing.assert_array_equal(new_gate, old_gate)

    batch_indices = np.array([0, 2, 8, 19], dtype=np.int64)
    batch = torch.as_tensor(data[batch_indices])
    old_view, old_sample_weight, _ = topology_mixing.make_pseudo_batch(
        data,
        batch_indices,
        batch,
        "reliability",
        old_graph,
        old_weights,
        old_gate,
        4,
        np.random.default_rng(777),
        neighbor_estimator="current",
    )
    new_view, new_sample_weight, _ = make_pseudo_batch(
        data,
        batch_indices,
        batch,
        parameterization="topology",
        graph=new_graph,
        edge_weights=new_weights,
        node_gate=new_gate,
        mix_neighbors=4,
        alpha=0.9,
        rng=np.random.default_rng(777),
        neighbor_estimator="current",
        legacy_plantnet=True,
    )
    torch.testing.assert_close(new_view, old_view, rtol=0, atol=0)
    torch.testing.assert_close(new_sample_weight, old_sample_weight, rtol=0, atol=0)


def test_legacy_row_swap_noise_matches_plantnet() -> None:
    fixed, _topology_graph, _topology_mixing = _plantnet_modules()
    values = torch.as_tensor(_data()[:7, :6])
    torch.manual_seed(31415)
    old_corrupted, old_mask = fixed.apply_scmae_noise(values, 0.4)
    torch.manual_seed(31415)
    new_corrupted, new_mask = apply_scmae_noise(
        values, 0.4, legacy_plantnet=True
    )
    torch.testing.assert_close(new_corrupted, old_corrupted, rtol=0, atol=0)
    torch.testing.assert_close(new_mask, old_mask, rtol=0, atol=0)
