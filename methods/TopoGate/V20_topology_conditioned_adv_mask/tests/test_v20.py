from __future__ import annotations

import inspect

import numpy as np
import scipy.sparse as sp
import torch

from methods.TopoGate.V20_topology_conditioned_adv_mask.graph import build_svd_knn_graph, compute_topology_statistics
from methods.TopoGate.V20_topology_conditioned_adv_mask.config import V20Config
from methods.TopoGate.V20_topology_conditioned_adv_mask.model import FeatureGate, V20AutoEncoder, straight_through_topk
from methods.TopoGate.V20_topology_conditioned_adv_mask.trainer import fit_full, fit_scmae_only


def test_gate_has_shared_257_parameters() -> None:
    gate = FeatureGate(64)
    assert sum(parameter.numel() for parameter in gate.parameters()) == 257


def test_st_topk_has_exact_forward_budget_and_gradient() -> None:
    logits = torch.randn(3, 11, requires_grad=True)
    generator = torch.Generator().manual_seed(7)
    mask_st, hard = straight_through_topk(logits, 4, generator=generator, gumbel_scale=1.0, tau_ste=0.5)
    assert torch.all(hard.sum(dim=1) == 4)
    mask_st.square().mean().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_sparse_graph_and_statistics_are_finite() -> None:
    X = sp.csr_matrix(np.array([[1, 0, 0, 2], [0, 1, 0, 1], [0, 0, 2, 0], [1, 1, 0, 0]], dtype=np.float32))
    graph = build_svd_knn_graph(X, neighbor_k=2, svd_target=0.95, svd_min_dim=2, svd_max_dim=3, seed=42)
    stats, profile = compute_topology_statistics(X.toarray(), graph, block_size=2)
    assert stats.shape == (4, 4, 2)
    assert np.isfinite(np.asarray(stats)).all()
    assert profile["storage"] == "memory"
    assert graph.profile["self_edges"] == 0


def test_fit_signature_is_label_free() -> None:
    assert "y" not in inspect.signature(fit_full).parameters


def test_effective_mask_is_used_as_explicit_loss_target() -> None:
    model = V20AutoEncoder(num_genes=3, hidden_size=4)
    corrupted = torch.zeros(2, 3)
    target = torch.ones(2, 3)
    effective = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    _, parts = model.loss_encoder(corrupted, target, effective)
    assert float(parts["mask_positive_rate"]) == float(effective.mean())


def test_scmae_only_disables_graph_and_gate() -> None:
    config = V20Config(
        protocol_id="test",
        variant="scmae_only",
        epochs=2,
        batch_size=3,
        warmup_epochs=2,
        mask_target_mode="effective",
        random_mask_mode="bernoulli",
    )
    embedding, diagnostics = fit_scmae_only(
        np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 0.0]], dtype=np.float32),
        config=config,
        seed=42,
        device="cpu",
    )
    assert embedding.shape == (4, 128)
    assert diagnostics["gate"] is None
    assert diagnostics["graph_profile"]["enabled"] is False
    assert diagnostics["mask_target_mode"] == "effective"
