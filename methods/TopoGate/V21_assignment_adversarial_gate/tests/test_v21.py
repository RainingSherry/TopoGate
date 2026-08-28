from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from methods.TopoGate.V21_assignment_adversarial_gate.config import V21Config
from methods.TopoGate.V21_assignment_adversarial_gate.graph import build_svd_knn_graph
from methods.TopoGate.V21_assignment_adversarial_gate.model import (
    FeatureGate,
    StudentTClusterHead,
    information_maximization_loss,
    jensen_shannon_divergence,
    straight_through_changeable_topk,
    theoretical_js_upper_bound,
)
from methods.TopoGate.V21_assignment_adversarial_gate.readout import select_readout
from methods.TopoGate.V21_assignment_adversarial_gate.trainer import (
    fit_scmae_only,
    fit_v21,
    resolve_device,
    seed_all,
)


ROOT = Path(__file__).resolve().parents[4]


def _toy_matrix() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 4.0, 3.0, 2.0],
            [6.0, 5.0, 4.0, 3.0],
            [0.5, 1.5, 2.5, 3.5],
            [5.5, 4.5, 3.5, 2.5],
        ],
        dtype=np.float32,
    )


def test_gate_has_shared_257_parameters() -> None:
    gate = FeatureGate(64)
    assert sum(parameter.numel() for parameter in gate.parameters()) == 257


def test_changeable_topk_has_exact_budget_support_and_gradient() -> None:
    logits = torch.randn(3, 7, requires_grad=True)
    eligible = torch.tensor(
        [
            [1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    generator = torch.Generator().manual_seed(7)
    mask_st, hard, budgets = straight_through_changeable_topk(
        logits,
        eligible,
        0.4,
        generator=generator,
        gumbel_scale=1.0,
        tau_ste=0.5,
    )
    assert budgets.tolist() == [2, 2, 0]
    assert hard.sum(dim=1).tolist() == [2.0, 2.0, 0.0]
    assert torch.all(hard[~eligible] == 0.0)
    mask_st.square().mean().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_assignment_corruption_changes_every_selected_value() -> None:
    anchor = torch.tensor([[0.0, 0.0, 2.0, 0.0], [1.0, 0.0, 0.0, 3.0]])
    donor = torch.flip(anchor, dims=(0,))
    eligible = donor != anchor
    logits = torch.zeros_like(anchor, requires_grad=True)
    generator = torch.Generator().manual_seed(11)
    _mask_st, hard, _budgets = straight_through_changeable_topk(
        logits,
        eligible,
        0.5,
        generator=generator,
        gumbel_scale=0.0,
        tau_ste=0.5,
    )
    corrupted = anchor + hard * (donor - anchor)
    selected = hard.bool()
    assert bool(selected.any())
    assert torch.all(corrupted[selected] != anchor[selected])


def test_js_is_bounded_and_zero_for_identical_assignments() -> None:
    p = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    q = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    same = jensen_shannon_divergence(p, p)
    different = jensen_shannon_divergence(p, q)
    assert float(same) < 1e-7
    assert 0.0 < float(different) <= theoretical_js_upper_bound() + 1e-6


def test_infomax_prefers_balanced_confident_assignments() -> None:
    balanced = torch.tensor([[0.99, 0.01], [0.01, 0.99]])
    collapsed = torch.tensor([[0.99, 0.01], [0.99, 0.01]])
    uniform = torch.full((2, 2), 0.5)
    assert float(information_maximization_loss(balanced)) < float(information_maximization_loss(collapsed))
    assert float(information_maximization_loss(balanced)) < float(information_maximization_loss(uniform))


def test_student_t_head_initialises_without_labels() -> None:
    embedding = np.asarray([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]], dtype=np.float32)
    head = StudentTClusterHead(n_clusters=2, latent_dim=2)
    head.initialise(embedding, seed=42, n_init=2)
    probabilities = head(torch.from_numpy(embedding))
    assert bool(head.initialised)
    assert probabilities.shape == (4, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-6)


def test_standard_student_t_sum_is_more_confident_than_legacy_mean() -> None:
    embedding = np.asarray([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]], dtype=np.float32)
    legacy = StudentTClusterHead(n_clusters=2, latent_dim=2, distance_reduction="mean")
    standard = StudentTClusterHead(n_clusters=2, latent_dim=2, distance_reduction="sum")
    legacy.initialise(embedding, seed=42, n_init=2)
    standard.load_state_dict(legacy.state_dict())
    values = torch.from_numpy(embedding)
    legacy_q = legacy(values)
    standard_q = standard(values)
    assert torch.equal(legacy_q.argmax(dim=1), standard_q.argmax(dim=1))
    assert standard_q.max(dim=1).values.mean().item() > legacy_q.max(dim=1).values.mean().item()


def test_kmeans_readout_repairs_empty_training_head_without_labels() -> None:
    embedding = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]],
        dtype=np.float32,
    )
    collapsed = np.asarray(
        [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
        dtype=np.float32,
    )
    predictions, head_predictions, profile = select_readout(
        embedding,
        collapsed,
        n_clusters=2,
        mode="kmeans_embedding",
        kmeans_n_init=5,
        seed=42,
    )
    assert head_predictions is not None
    assert np.unique(head_predictions).size == 1
    assert np.unique(predictions).size == 2
    assert profile["labels_used_for_readout"] is False
    assert profile["primary"]["empty_clusters"] == 0
    assert profile["student_t_training_head"]["empty_clusters"] == 1


def test_legacy_student_t_readout_remains_available() -> None:
    embedding = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    probabilities = np.asarray(
        [[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]],
        dtype=np.float32,
    )
    predictions, head_predictions, profile = select_readout(
        embedding,
        probabilities,
        n_clusters=2,
        mode="student_t_head",
        kmeans_n_init=2,
        seed=42,
    )
    assert head_predictions is not None
    assert np.array_equal(predictions, head_predictions)
    assert profile["effective_mode"] == "student_t_head"


def test_legacy_scmae_readout_name_is_preserved() -> None:
    predictions, head_predictions, profile = select_readout(
        np.asarray([[0.0], [0.1], [4.0], [4.1]], dtype=np.float32),
        None,
        n_clusters=2,
        mode="student_t_head",
        kmeans_n_init=2,
        seed=42,
    )
    assert head_predictions is None
    assert np.unique(predictions).size == 2
    assert profile["primary_method"] == "kmeans_known_k"


def test_fit_signature_is_label_free() -> None:
    assert "y" not in inspect.signature(fit_v21).parameters


def test_extended_job_builder_accepts_unfiltered_manifest(tmp_path: Path) -> None:
    import importlib.util

    script = ROOT / "scripts/V21/run_extended_matrix.py"
    spec = importlib.util.spec_from_file_location("v21_extended_runner", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = {
        "manifest_id": "test",
        "datasets": [
            {
                "dataset_id": "one",
                "name": "one",
                "source_path": str(tmp_path / "one.npz"),
                "input_protocol": "clubench_bridge",
                "profile": {"n_samples": 4, "n_features": 2},
            }
        ],
    }
    jobs = module.build_jobs(manifest, ("scmae_only",), (42,), set(), tmp_path / "outputs")
    assert len(jobs) == 1
    assert jobs[0]["run_key"] == "test::one::scmae_only::seed42"


def test_cpu_seed_does_not_touch_all_cuda_devices(monkeypatch) -> None:
    def fail_manual_seed_all(_seed: int) -> None:
        raise AssertionError("CPU V21 must not seed every visible CUDA device")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", fail_manual_seed_all)
    seed_all(42, torch.device("cpu"))


def test_cuda_device_resolution_requires_an_allowed_physical_gpu() -> None:
    with pytest.raises(ValueError, match="explicit physical"):
        resolve_device("cuda", None)
    with pytest.raises(ValueError, match="forbidden"):
        resolve_device("cuda", 0)
    with pytest.raises(ValueError, match="forbidden"):
        resolve_device("cuda", 7)


def test_knn_graph_filters_self_after_duplicate_ties() -> None:
    X = sp.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    graph = build_svd_knn_graph(
        X,
        neighbor_k=2,
        svd_target=0.95,
        svd_min_dim=2,
        svd_max_dim=3,
        seed=42,
    )
    assert all(row not in graph.indices[row] for row in range(X.shape[0]))
    assert graph.profile["self_edges"] == 0


def test_scmae_only_disables_graph_head_and_gate() -> None:
    config = V21Config(
        protocol_id="test",
        variant="scmae_only",
        hidden_size=4,
        epochs=1,
        batch_size=3,
        warmup_epochs=1,
        cluster_n_init=2,
    )
    embedding, diagnostics = fit_scmae_only(_toy_matrix(), config=config, seed=42, device="cpu")
    assert embedding.shape == (6, 4)
    assert diagnostics["cluster_head"] is None
    assert diagnostics["gate"] is None
    assert diagnostics["graph_profile"]["enabled"] is False
    assert diagnostics["K_used_during_fit"] is False


def test_random_assignment_control_has_head_without_graph_or_gate() -> None:
    config = V21Config(
        protocol_id="test",
        variant="random_assignment_control",
        hidden_size=4,
        epochs=2,
        batch_size=6,
        warmup_epochs=1,
        cluster_n_init=2,
    )
    embedding, probabilities, diagnostics = fit_v21(
        _toy_matrix(),
        None,
        n_clusters=2,
        config=config,
        seed=42,
        device="cpu",
    )
    assert embedding.shape == (6, 4)
    assert probabilities is not None and probabilities.shape == (6, 2)
    assert diagnostics["cluster_head"] is not None
    assert diagnostics["gate"] is None
    assert diagnostics["graph_profile"]["enabled"] is False
    assert diagnostics["K_used_during_fit"] is True
    assert diagnostics["history"][-1]["assignment_effective_given_selected"] == 1.0


def test_topology_assignment_variant_builds_gate_and_updates_it() -> None:
    X = _toy_matrix()
    config = V21Config(
        protocol_id="test",
        variant="topology_assignment_adversarial",
        hidden_size=4,
        epochs=2,
        batch_size=6,
        warmup_epochs=1,
        cluster_n_init=2,
        graph_svd_min_dim=2,
        graph_svd_max_dim=3,
        neighbor_k=2,
        stats_block_size=2,
        gate_update_every=1,
    )
    embedding, probabilities, diagnostics = fit_v21(
        X,
        sp.csr_matrix(X),
        n_clusters=2,
        config=config,
        seed=42,
        device="cpu",
    )
    assert embedding.shape == (6, 4)
    assert probabilities is not None and probabilities.shape == (6, 2)
    assert diagnostics["gate"] is not None
    assert diagnostics["graph_profile"]["enabled"] is True
    assert diagnostics["gate_updates"] > 0
    assert diagnostics["history"][-1]["assignment_budget_fill"] == 1.0
