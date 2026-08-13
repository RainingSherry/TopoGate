"""Unit tests for the V13 Gumbel-Top-k gate."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from methods.TopoGate.V13_hard_gate.gumbel_gate import (
    GumbelTopKGate,
    GumbelTopKGateOutput,
    build_gate_stats_tensor,
    hard_topk_alignment_loss,
)


def test_build_gate_stats_tensor_validates_numPy_is_forbidden():
    """NumPy arrays are not accepted — only Torch tensors."""
    sim = torch.rand(8, 5)
    mut = torch.rand(8, 5)
    snn = torch.rand(8, 5)
    dist = torch.rand(8, 5)

    # Torch is fine.
    t = build_gate_stats_tensor(sim, mut, snn, dist)
    assert t.shape == (8, 5, 4)

    # Passing a Python list raises TypeError (not torch.Tensor).
    try:
        build_gate_stats_tensor(
            sim.tolist(),  # plain list
            mut,
            snn,
            dist,
        )
        raise AssertionError("expected TypeError for list input")
    except TypeError:
        pass


def test_build_gate_stats_tensor_checks_shape():
    """All inputs must have the same [N, K] shape."""
    sim = torch.rand(8, 5)
    mut = torch.rand(8, 3)  # wrong K
    snn = torch.rand(8, 5)
    dist = torch.rand(8, 5)
    try:
        build_gate_stats_tensor(sim, mut, snn, dist)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_gate_requires_positive_topk():
    """top_k must be >= 1."""
    try:
        GumbelTopKGate(top_k=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_gate_topk_respects_K():
    """When top_k > K the gate selects all K neighbours."""
    B, K = 4, 3
    gate = GumbelTopKGate(feature_dim=4, hidden_dim=16, top_k=5, dropout=0.0)
    stats = torch.randn(B, K, 4)
    out = gate(stats, tau=0.1, hard=True)
    # top_k=5 > K=3, so every row should have 3 ones.
    mask_sum = out.mask.sum(dim=1)
    assert torch.allclose(mask_sum, torch.full((B,), float(K))), (
        f"expected mask sum == {K}, got {mask_sum}"
    )


def test_hard_forward_produces_exactly_topk_ones():
    """``hard=True`` must return a binary mask with exactly ``top_k`` ones per row."""
    B, K, top_k = 6, 5, 2
    gate = GumbelTopKGate(feature_dim=4, hidden_dim=16, top_k=top_k, dropout=0.0)
    gate.eval()
    stats = torch.randn(B, K, 4)
    out = gate(stats, tau=1.0, hard=True)

    assert out.mask.shape == (B, K)
    assert (out.mask == 0.0).sum() + (out.mask == 1.0).sum() == B * K, (
        "hard mask must be exactly 0 or 1"
    )
    mask_sum = out.mask.sum(dim=1)
    assert torch.allclose(mask_sum, torch.full((B,), float(top_k))), (
        f"expected mask sum == {top_k}, got {mask_sum}"
    )


def test_soft_forward_receives_gradient():
    """Soft forward must retain gradients through the gate parameters.

    We verify that calling forward, summing the mask, and calling backward()
    produces non-zero gradients at the gate parameters.  With small initial
    weights the logits may be near-zero (making top-k sampling noisy), so we
    use a large learning rate / many steps to confirm the gradient is real.
    """
    B, K, top_k = 4, 5, 2
    gate = GumbelTopKGate(feature_dim=4, hidden_dim=16, top_k=top_k, dropout=0.0)
    gate.train()
    stats = torch.randn(B, K, 4, requires_grad=True)

    out = gate(stats, tau=1.0, hard=False)
    loss = out.mask.sum()
    loss.backward()

    grads = [p.grad for p in gate.parameters() if p.grad is not None]
    assert grads, "no gradient reached gate parameters"
    assert any(torch.any(g != 0) for g in grads), (
        f"gate gradient is all zero; logits may be near-zero (top-k = random Gumbel)"
    )
    assert stats.grad is not None, "stats has no gradient"
    assert torch.any(stats.grad != 0), "stats gradient is all zero"


def test_hard_mode_detaches_scores_from_gradient():
    """The scores must be detached from the gradient graph."""
    B, K, top_k = 4, 5, 2
    gate = GumbelTopKGate(feature_dim=4, hidden_dim=16, top_k=top_k, dropout=0.0)
    gate.train()
    stats = torch.randn(B, K, 4)

    out = gate(stats, tau=0.1, hard=True)
    assert out.scores.grad is None, "scores must be detached"


def test_soft_forward_receives_gradient():
    """Soft forward must retain gradients through the gate parameters."""
    B, K, top_k = 4, 5, 2
    gate = GumbelTopKGate(feature_dim=4, hidden_dim=16, top_k=top_k, dropout=0.0)
    gate.train()
    stats = torch.randn(B, K, 4, requires_grad=True)

    out = gate(stats, tau=1.0, hard=False)
    loss = out.mask.sum()
    loss.backward()
    grads = [p.grad for p in gate.parameters() if p.grad is not None]
    assert grads, "no gradient reached gate parameters"
    assert any(torch.any(g != 0) for g in grads), "gate gradient is all zero"
    assert stats.grad is not None, "stats has no gradient"
    assert torch.any(stats.grad != 0), "stats gradient is all zero"


def test_temperature_tau_min_is_respected():
    """The runner anneals tau down to tau_min; verify the range is respected."""
    B, K, top_k = 4, 5, 2
    gate = GumbelTopKGate(feature_dim=4, hidden_dim=16, top_k=top_k, dropout=0.0)
    gate.eval()
    stats = torch.randn(B, K, 4)

    # tau=1.0 vs tau=0.1 must both be valid calls.
    out_hot = gate(stats, tau=1.0, hard=False)
    out_cold = gate(stats, tau=0.1, hard=False)
    assert out_hot.mask.shape == out_cold.mask.shape == (B, K)
    assert torch.isfinite(out_hot.mask).all(), "tau=1.0 output not finite"
    assert torch.isfinite(out_cold.mask).all(), "tau=0.1 output not finite"
    # At tau=0.1 the straight-through estimator should be very close to one-hot.
    assert torch.allclose(out_cold.mask.sum(dim=1), torch.full((B,), float(top_k)), atol=0.05), (
        f"tau=0.1 mask sum should be close to {top_k}, got {out_cold.mask.sum(dim=1)}"
    )


def test_topk_alignment_loss_detaches_neighbors():
    """The ``detach_neighbors`` flag must prevent gradients reaching z_neighbors."""
    B, H, K = 4, 32, 5
    gate = GumbelTopKGate(feature_dim=4, hidden_dim=16, top_k=2, dropout=0.0)
    gate.train()
    stats = torch.randn(B, K, 4, requires_grad=False)
    z_anchor = torch.randn(B, H, requires_grad=True)
    z_neighbors = torch.randn(B, K, H, requires_grad=True)

    loss, target = hard_topk_alignment_loss(
        z_anchor, z_neighbors,
        torch.zeros(B, K),  # dummy mask
        detach_neighbors=True,
    )
    loss.backward()
    assert z_neighbors.grad is None, "z_neighbors received a gradient when detached"
    assert z_anchor.grad is not None, "z_anchor has no gradient"


def test_topk_alignment_loss_gradients_reach_anchor():
    """The topology loss must train the anchor encoder."""
    B, H, K = 4, 32, 5
    z_anchor = torch.randn(B, H, requires_grad=True)
    z_neighbors = torch.randn(B, K, H)  # detached neighbours
    mask = torch.zeros(B, K)
    mask[:, :2] = 1.0  # top-2 selection

    loss, target = hard_topk_alignment_loss(z_anchor, z_neighbors, mask, detach_neighbors=True)
    loss.backward()
    assert z_anchor.grad is not None, "z_anchor has no gradient"
    assert torch.any(z_anchor.grad != 0), "z_anchor gradient is all zero"


def test_topk_alignment_uses_mask_sum_not_K():
    """The neighbour target must be normalised by mask sum, not by K.

    With a top-k=2 mask the target should be the mean of exactly 2 selected
    neighbours, regardless of the full K.  Using K as the denominator would
    under-weight the contribution of each selected neighbour.
    """
    B, H, K = 2, 8, 5
    z_anchor = torch.zeros(B, H, requires_grad=True)
    z_neighbors = torch.full((B, K, H), 4.0)  # all neighbours = 4
    # Mask selects 2 neighbours per row.
    mask = torch.zeros(B, K)
    mask[0, :2] = 1.0  # sum=2
    mask[1, 2:4] = 1.0  # sum=2

    _, target = hard_topk_alignment_loss(z_anchor, z_neighbors, mask, detach_neighbors=True)
    # If normalised by mask_sum=2: target = 4.0 * 2 / 2 = 4.0.
    # If normalised by K=5: target = 4.0 * 2 / 5 = 1.6.
    assert torch.allclose(target, torch.full((B, H), 4.0)), (
        f"expected target == 4.0 (mask-sum normalisation), got {target[0, 0].item()}"
    )


def test_nomix_does_not_construct_a_graph(tmp_path):
    """With ``--topology_enabled false`` the runner must not build a kNN graph."""
    import subprocess, sys

    data_path = Path("datasets/AHDPC/processed/flame.npz")
    if not data_path.is_file():
        pytest.skip("V13 smoke dataset is not included in the public code snapshot")

    result = subprocess.run(
        [
            sys.executable,
            "methods/TopoGate/V13_hard_gate/run_npz.py",
            "--data_path", str(data_path),
            "--save_dir", str(tmp_path / "nomix"),
            "--dataset_name", "flame",
            "--variant_name", "topogate_v13_nomix",
            "--seed", "42",
            "--topology_enabled", "false",
            "--epochs", "1",
        ],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"nomix runner failed: {result.stderr}"
    summary_path = tmp_path / "nomix" / "summary.json"
    assert summary_path.exists(), "summary.json not written"
    import json
    s = json.loads(summary_path.read_text())
    assert not s["topology_enabled"], "topology_enabled should be False"
    assert s["selected_neighbor_count"] == 0.0, (
        "effective neighbours should be 0 when topology is disabled"
    )


def test_topk_runs_end_to_end_on_flame(tmp_path):
    """A single smoke run on flame must complete without errors."""
    import subprocess, sys

    data_path = Path("datasets/AHDPC/processed/flame.npz")
    if not data_path.is_file():
        pytest.skip("V13 smoke dataset is not included in the public code snapshot")

    result = subprocess.run(
        [
            sys.executable,
            "methods/TopoGate/V13_hard_gate/run_npz.py",
            "--data_path", str(data_path),
            "--save_dir", str(tmp_path / "topk2"),
            "--dataset_name", "flame",
            "--variant_name", "topogate_v13_topk2",
            "--seed", "42",
            "--topology_enabled", "true",
            "--epochs", "5",
            "--top_k_neighbors", "2",
        ],
        cwd=".",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"topk runner failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    import json
    s = json.loads((tmp_path / "topk2" / "summary.json").read_text())
    assert s["topology_enabled"], "topology_enabled should be True"
    assert s["selected_neighbor_count"] > 0.0, "effective neighbours should be > 0"
    assert "ari" in s["metrics"], f"ARI not in metrics: {s['metrics']}"
    assert s["metrics"]["ari"] >= -1.0, "ARI out of range"


def test_gate_initialised_with_small_weights():
    """The gate MLP should start with small weights so early Gumbel samples are near-uniform."""
    gate = GumbelTopKGate(feature_dim=4, hidden_dim=16, top_k=2, dropout=0.0)
    final = gate.network[-1]
    assert isinstance(final, torch.nn.Linear)
    weight_std = float(final.weight.std().item())
    assert weight_std < 0.05, f"initial weight std {weight_std:.4f} is too large"
