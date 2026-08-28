from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from methods.TopoGate.V12_latent_topology.learnable_gate import (
    LearnableGate,
    build_gate_stats_tensor,
    rank_alignment_loss,
    topology_alignment_loss,
)
from methods.TopoGate.V12_latent_topology.model import AutoEncoder


def test_edge_stats_keep_neighbor_dimension() -> None:
    features = [torch.randn(4, 3) for _ in range(4)]
    stats = build_gate_stats_tensor(*features)
    assert stats.shape == (4, 3, 4)


def test_self_null_and_edge_only_weights_are_normalized() -> None:
    torch.manual_seed(7)
    gate = LearnableGate(feature_dim=4, hidden_dim=8, self_init_weight=0.8)
    stats = torch.randn(5, 3, 4)
    self_weight, edge_weights = gate(stats, topology_mode="self_null")
    assert self_weight.shape == (5,)
    assert edge_weights.shape == (5, 3)
    torch.testing.assert_close(self_weight + edge_weights.sum(dim=1), torch.ones(5))
    assert float(self_weight.mean()) > 0.7

    edge_self, edge_only = gate(stats, topology_mode="edge_only")
    torch.testing.assert_close(edge_self, torch.zeros(5))
    torch.testing.assert_close(edge_only.sum(dim=1), torch.ones(5))


def test_gate_and_alignment_are_differentiable_without_target_gradients() -> None:
    torch.manual_seed(7)
    gate = LearnableGate(feature_dim=4, hidden_dim=8)
    stats = torch.randn(5, 3, 4)
    anchor = torch.randn(5, 6, requires_grad=True)
    clean_self = torch.randn(5, 6, requires_grad=True)
    clean_neighbors = torch.randn(5, 3, 6, requires_grad=True)
    self_weight, edge_weights = gate(stats)
    loss, target = topology_alignment_loss(
        anchor,
        clean_neighbors,
        edge_weights,
        self_weight=self_weight,
        z_self=clean_self,
    )
    assert target.shape == anchor.shape
    loss.backward()
    assert clean_self.grad is None
    assert clean_neighbors.grad is None
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    gate_gradients = [p.grad for p in gate.parameters() if p.grad is not None]
    assert gate_gradients
    assert all(torch.isfinite(g).all() for g in gate_gradients)
    assert any(torch.any(g != 0) for g in gate_gradients)


def test_v12_autoencoder_additive_mask_loss_and_latent_interface() -> None:
    torch.manual_seed(11)
    model = AutoEncoder(num_genes=7, hidden_size=5, mask_loss_mode="additive")
    assert model.mask_loss_weight == 0.1
    x = torch.randn(4, 7)
    target = torch.randn(4, 7)
    mask = torch.randint(0, 2, (4, 7), dtype=torch.float32)
    latent, mask_logits, reconstruction = model.forward_mask(x)
    assert latent.shape == (4, 5)
    assert mask_logits.shape == reconstruction.shape == (4, 7)
    _, loss, parts = model.loss_mask_weighted(x, target, mask)
    expected = parts["raw_reconstruction_loss"] + 0.1 * parts["raw_mask_loss"]
    torch.testing.assert_close(loss.detach(), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        parts["mask_loss"], 0.1 * parts["raw_mask_loss"], rtol=1e-5, atol=1e-6
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_legacy_weighted_mask_loss_remains_explicit_ablation() -> None:
    model = AutoEncoder(num_genes=7, hidden_size=5, mask_loss_mode="legacy_weighted")
    x = torch.randn(4, 7)
    mask = torch.randint(0, 2, (4, 7), dtype=torch.float32)
    _, loss, parts = model.loss_mask_weighted(x, x, mask)
    expected = 0.9 * parts["raw_reconstruction_loss"] + 0.1 * parts["raw_mask_loss"]
    torch.testing.assert_close(loss.detach(), expected, rtol=1e-5, atol=1e-6)


def test_v12_default_decoder_preserves_mask_conditioned_contract() -> None:
    model = AutoEncoder(num_genes=7, hidden_size=5)
    assert model.decoder_mode == "legacy_mask_conditioned"
    assert model.decoder.in_features == 12
    latent, mask_logits, reconstruction = model.forward_mask(torch.randn(3, 7))
    assert latent.shape == (3, 5)
    assert mask_logits.shape == reconstruction.shape == (3, 7)


def test_nomix_does_not_construct_a_graph(tmp_path, monkeypatch) -> None:
    import methods.TopoGate.V12_latent_topology.run_npz as runner

    def fail_graph(*_args, **_kwargs):
        raise AssertionError("NoMix must not construct a graph")

    monkeypatch.setattr(runner, "build_pca_knn_graph", fail_graph)
    source = tmp_path / "tiny.npz"
    np.savez(source, x=np.random.default_rng(3).normal(size=(12, 3)).astype(np.float32), y=np.arange(12) % 2)
    args = Namespace(
        data_path=str(source),
        save_dir=str(tmp_path / "out"),
        dataset_name="tiny",
        method_name="TopoGate",
        variant_name="topogate_v12_nomix_test",
        seed=42,
        n_clusters=None,
        gpu=1,
        no_cuda=True,
        scale_input=True,
        input_mode="raw",
        hidden_size=8,
        dropout=0.0,
        decoder_mode="legacy_mask_conditioned",
        mask_ratio=0.2,
        masked_data_weight=0.75,
        mask_loss_weight=0.1,
        mask_loss_mode="additive",
        epochs=1,
        batch_size=12,
        lr=1e-3,
        weight_decay=1e-5,
        topology_enabled=False,
        topology_mode="self_null",
        lambda_topology=0.0,
        topology_warmup_epochs=20,
        topology_ramp_epochs=10,
        rank_loss_weight=0.0,
        rank_margin=0.1,
        self_init_weight=0.8,
        gate_hidden_size=8,
        gate_temperature=1.0,
        neighbor_k=3,
        knn_pca_dim=3,
        tau=0.2,
        log_interval=1,
    )
    summary = runner.train_and_evaluate(args)
    assert summary["topology_enabled"] is False
    assert (tmp_path / "out" / "summary.json").exists()


def test_rank_alignment_loss_rewards_top_similarity_edge() -> None:
    """The rank loss must penalise a gate that inverts the reliability ranking.

    ``good`` puts the highest weight on the most reliable edge; ``bad`` flips
    that order. ``rank_alignment_loss(good)`` must be strictly less than
    ``rank_alignment_loss(bad)``. ``edge_reliability`` must remain detached so
    it never enters a learnable graph, while the gate receives a non-zero
    gradient.
    """

    torch.manual_seed(13)
    reliability = torch.tensor(
        [
            [1.00, 0.50, 0.10],
            [0.90, 0.40, 0.05],
            [0.80, 0.30, 0.20],
            [0.70, 0.25, 0.15],
        ],
        dtype=torch.float32,
    )
    good_weights = torch.tensor(
        [
            [0.80, 0.15, 0.05],
            [0.70, 0.20, 0.10],
            [0.55, 0.30, 0.15],
            [0.60, 0.25, 0.15],
        ],
        dtype=torch.float32,
    )
    bad_weights = torch.tensor(
        [
            [0.05, 0.15, 0.80],
            [0.10, 0.20, 0.70],
            [0.15, 0.30, 0.55],
            [0.15, 0.25, 0.60],
        ],
        dtype=torch.float32,
    )
    good_loss = rank_alignment_loss(good_weights, reliability, margin=0.1)
    bad_loss = rank_alignment_loss(bad_weights, reliability, margin=0.1)
    assert torch.isfinite(good_loss).item()
    assert torch.isfinite(bad_loss).item()
    assert good_loss.item() < bad_loss.item()


def test_rank_alignment_loss_detaches_reliability_and_backprops_to_gate() -> None:
    """Reliability must stay grad-free; the rank loss must reach gate params."""

    torch.manual_seed(17)
    gate = LearnableGate(feature_dim=4, hidden_dim=8)
    stats = torch.randn(6, 4, 4)
    reliability = torch.randn(6, 4)
    self_weight, edge_weights = gate(stats)
    loss = rank_alignment_loss(edge_weights, reliability, margin=0.1)
    loss.backward()
    assert reliability.grad is None
    gate_grads = [p.grad for p in gate.parameters() if p.grad is not None]
    assert gate_grads
    assert any(torch.any(g != 0) for g in gate_grads)
    assert all(torch.isfinite(g).all() for g in gate_grads)


def test_rank_alignment_loss_is_zero_when_reliability_is_constant() -> None:
    """If every neighbour is equally reliable the gate is allowed to break ties."""

    torch.manual_seed(19)
    weights = torch.softmax(torch.randn(5, 3), dim=-1)
    reliability = torch.zeros(5, 3)
    loss = rank_alignment_loss(weights, reliability, margin=0.1)
    assert torch.isfinite(loss).item()
    assert float(loss.item()) == 0.0
