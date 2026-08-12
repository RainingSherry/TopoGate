from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from methods.TopoGate.V22_topology_discriminator_hard_mask.config import V22Config
from methods.TopoGate.V22_topology_discriminator_hard_mask.graph import (
    build_svd_knn_graph,
    compute_topology_statistics,
)
from methods.TopoGate.V22_topology_discriminator_hard_mask.input_adapter import prepare_dual_input
from methods.TopoGate.V22_topology_discriminator_hard_mask.model import (
    CoordinateDiscriminator,
    CoordinateGate,
    random_topk_mask,
    straight_through_topk,
)
from methods.TopoGate.V22_topology_discriminator_hard_mask.trainer import (
    fit_v22,
    resolve_device,
    seed_all,
)


def _toy() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 1.0, 2.0, 0.0, 3.0],
            [1.0, 0.0, 2.0, 1.0, 0.0],
            [5.0, 4.0, 0.0, 0.0, 1.0],
            [4.0, 5.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 1.0, 3.0],
            [5.0, 4.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_discriminator_api_has_no_mask_or_hint_shortcut() -> None:
    names = set(inspect.signature(CoordinateDiscriminator.forward).parameters)
    assert {"context", "feature_indices", "topology_context", "values"} <= names
    assert "mask" not in names
    assert "hint" not in names


def test_coordinate_discriminator_pairs_are_shape_checked() -> None:
    model = CoordinateDiscriminator(5, 4, hidden_size=8, coordinate_embedding_dim=3)
    context = torch.randn(7, 4)
    indices = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    topology = torch.randn(7, 4)
    values = torch.randn(7)
    output = model(context, indices, topology, values)
    assert output.shape == (7,)
    with pytest.raises(ValueError, match="one scalar"):
        model(context, indices, topology, values[:2])


def test_topk_has_exact_budget_and_gate_gradient() -> None:
    logits = torch.randn(3, 6, requires_grad=True)
    eligible = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    mask_st, hard, budgets = straight_through_topk(
        logits,
        eligible,
        0.4,
        generator=torch.Generator().manual_seed(7),
    )
    assert budgets.tolist() == [2, 2, 0]
    assert hard.sum(dim=1).tolist() == [2.0, 2.0, 0.0]
    assert torch.all(hard[~eligible] == 0.0)
    mask_st.square().mean().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_always_visible_control_has_zero_random_mask() -> None:
    mask = random_topk_mask(
        (3, 6),
        0.0,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.count_nonzero(mask).item() == 0
    config = V22Config(variant="scmae_always_visible", random_mask_ratio=0.0)
    config.validate()
    assert config.uses_discriminator is False


def test_topology_statistics_include_stability_without_labels() -> None:
    X = _toy()
    sparse = sp.csr_matrix(X)
    graph = build_svd_knn_graph(
        sparse,
        neighbor_k=2,
        svd_target=0.95,
        svd_min_dim=2,
        svd_max_dim=3,
        seed=42,
    )
    stats, profile = compute_topology_statistics(X, graph, support_matrix=sparse, block_size=3)
    assert stats.shape == (6, 5, 4)
    assert profile["support_is_label_free"] is True
    assert np.isfinite(np.asarray(stats)).all()
    assert np.all((np.asarray(stats)[:, :, 3] >= 0.0) & (np.asarray(stats)[:, :, 3] <= 1.0))


def test_input_adapter_caps_features_without_labels() -> None:
    X = sp.csr_matrix(np.arange(6 * 9, dtype=np.float32).reshape(6, 9))
    prepared = prepare_dual_input(X, dataset_name="toy", input_protocol="shared_text", feature_cap=4)
    assert prepared.X_model.shape == (6, 4)
    assert prepared.X_graph.shape == (6, 4)
    assert prepared.X_support.shape == (6, 4)
    assert prepared.profile["feature_selection"] == "top_variance_label_free"


def test_v22_fit_is_label_free_and_gate_updates_on_cpu(tmp_path) -> None:
    X = _toy()
    sparse = sp.csr_matrix(X)
    config = V22Config(
        variant="v22_topology_discriminator_hard_gate",
        epochs=2,
        batch_size=3,
        hidden_size=8,
        discriminator_hidden=8,
        gate_hidden=8,
        coordinate_embedding_dim=4,
        feature_cap=5,
        graph_svd_min_dim=2,
        graph_svd_max_dim=3,
        neighbor_k=2,
        discriminator_coordinates_per_row=2,
    )
    embedding, diagnostics = fit_v22(
        X,
        sparse,
        sparse,
        config=config,
        seed=42,
        device="cpu",
        stats_cache_dir=tmp_path / "stats",
    )
    assert embedding.shape == (6, 8)
    assert diagnostics["labels_used_during_fit"] is False
    assert diagnostics["discriminator_steps"] > 0
    assert diagnostics["gate_updates"] > 0
    assert diagnostics["gate_nonzero_update_rate"] > 0.0
    assert max(float(row["gate_grad_norm"]) for row in diagnostics["history"]) > 0.0
    assert diagnostics["random_mask_profile"]["selected_total"] > 0
    assert diagnostics["gate_mask_profile"]["unique_feature_count"] > 0
    assert "discriminator_scmae_fake_accuracy" in diagnostics["history"][0]
    assert all(np.isfinite(list(row.values())).all() for row in diagnostics["history"])


def test_v22_fit_reuses_exact_topology_cache(tmp_path) -> None:
    X = _toy()
    sparse = sp.csr_matrix(X)
    config = V22Config(
        variant="v22_topology_discriminator_hard_gate",
        epochs=1,
        batch_size=3,
        hidden_size=8,
        discriminator_hidden=8,
        gate_hidden=8,
        coordinate_embedding_dim=4,
        feature_cap=5,
        graph_svd_min_dim=2,
        graph_svd_max_dim=3,
        neighbor_k=2,
        discriminator_coordinates_per_row=2,
    )
    cache = tmp_path / "stats"
    fit_v22(X, sparse, sparse, config=config, seed=42, device="cpu", stats_cache_dir=cache)
    _embedding, diagnostics = fit_v22(
        X,
        sparse,
        sparse,
        config=config,
        seed=42,
        device="cpu",
        stats_cache_dir=cache,
        reuse_topology_cache=True,
    )
    assert diagnostics["stats_profile"]["cache_reused"] is True
    assert diagnostics["graph_profile"]["graph_profile_available"] is False


def test_cooperative_keep_gate_uses_complementary_mask_and_updates(tmp_path) -> None:
    X = _toy()
    sparse = sp.csr_matrix(X)
    config = V22Config(
        protocol_id="v22_topology_discriminator_cooperative_keep_gate_v1",
        variant="v22_topology_discriminator_cooperative_keep_gate",
        epochs=1,
        batch_size=3,
        hidden_size=8,
        discriminator_hidden=8,
        gate_hidden=8,
        coordinate_embedding_dim=4,
        feature_cap=5,
        graph_svd_min_dim=2,
        graph_svd_max_dim=3,
        neighbor_k=2,
        discriminator_coordinates_per_row=2,
    )
    embedding, diagnostics = fit_v22(
        X,
        sparse,
        sparse,
        config=config,
        seed=42,
        device="cpu",
        stats_cache_dir=tmp_path / "stats",
    )
    assert embedding.shape == (6, 8)
    assert diagnostics["variant_contract"]["gate_semantics"] == "cooperative_keep_complementary_mask"
    assert diagnostics["variant_contract"]["gate_reward_mode"] == "cooperative_keep"
    assert diagnostics["gate_updates"] > 0
    assert diagnostics["gate_nonzero_update_rate"] > 0.0
    assert diagnostics["gate_keep_profile"]["selected_total"] > 0
    assert diagnostics["gate_effective_mask_profile"]["selected_total"] >= 0
    assert max(float(row["gate_grad_reconstruction_norm"]) for row in diagnostics["history"]) > 0.0
    assert max(float(row["gate_grad_discriminator_norm"]) for row in diagnostics["history"]) > 0.0
    assert all("discriminator_value_matched_accuracy" in row for row in diagnostics["history"])


def test_cpu_seed_does_not_touch_all_cuda_devices(monkeypatch) -> None:
    def fail_manual_seed_all(_seed: int) -> None:
        raise AssertionError("CPU V22 must not seed every visible CUDA device")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", fail_manual_seed_all)
    seed_all(42, torch.device("cpu"))


def test_cuda_resolution_requires_allowed_physical_gpu() -> None:
    with pytest.raises(ValueError, match="explicit physical"):
        resolve_device("cuda", None)
    with pytest.raises(ValueError, match="forbidden"):
        resolve_device("cuda", 0)
    with pytest.raises(ValueError, match="forbidden"):
        resolve_device("cuda", 7)
