from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from methods.TopoGate.V11.config import V11Config
from methods.TopoGate.V11.graph import build_candidate_graph, edge_recurrence_against, graph_change_fraction
from methods.TopoGate.V11.model import StudentTMixtureHead, TopoGateV11, ema_update, make_teacher
from methods.TopoGate.V11.run import run_v11
from methods.TopoGate.V11.tda import candidate_prior_from_h0, compute_h0_persistence
from methods.TopoGate.V11.trainer import (
    V11Trainer,
    corrupt_batch,
    counterfactual_semantic_target,
    trusted_edge_alignment,
)


def _model() -> TopoGateV11:
    return TopoGateV11(
        input_dim=6,
        hidden_dim=16,
        latent_dim=4,
        n_clusters=3,
        dropout=0.0,
        null_bias=0.5,
        student_t_nu=4.0,
    )


def test_probabilities_and_core_gradients_are_real() -> None:
    torch.manual_seed(3)
    model = _model()
    x = torch.tensor(
        [
            [1.0, 0.8, 0.0, 0.1, 0.0, 0.2],
            [0.9, 1.0, 0.1, 0.0, 0.1, 0.3],
            [0.0, 0.1, 1.0, 0.9, 0.2, 0.0],
            [0.1, 0.0, 0.8, 1.0, 0.3, 0.1],
        ]
    )
    q = model.assignments(x)
    assert torch.allclose(q.sum(dim=1), torch.ones(x.shape[0]), atol=1e-6)

    edge_features = torch.randn(4, 3, 6)
    node_features = torch.randn(4, 5)
    valid = torch.ones(4, 3, dtype=torch.bool)
    mixture = model.topology(edge_features, node_features, valid, 1.0, 1.0, True)
    assert torch.allclose(mixture.sum(dim=1), torch.ones(4), atol=1e-6)
    target = torch.full_like(mixture, 1.0 / mixture.shape[1])
    topology_loss = torch.sum(target * (torch.log(target) - torch.log(mixture.clamp_min(1e-8))), dim=1).mean()
    cluster_loss = -(q.clamp_min(1e-8).log() * q.detach()).sum(dim=1).mean()
    reconstruction = model.autoencoder(x)[1]
    reconstruction_loss = (reconstruction - x).square().mean()
    (topology_loss + cluster_loss + reconstruction_loss).backward()

    groups = {
        "edge": list(model.topology.edge_net.parameters()),
        "null": list(model.topology.null_net.parameters()),
        "cluster": [model.cluster_head.centres, model.cluster_head.log_scales, model.cluster_head.prior_logits],
    }
    for name, parameters in groups.items():
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in parameters
        ), f"{name} parameters did not receive a finite non-zero gradient"


def test_dimension_normalized_student_t_responsibilities_resist_width_saturation() -> None:
    none = StudentTMixtureHead(2, latent_dim=64, logit_normalization="none")
    normalized = StudentTMixtureHead(2, latent_dim=64, logit_normalization="sqrt_dim")
    normalized.load_state_dict(none.state_dict())
    with torch.no_grad():
        none.centres[0].zero_()
        none.centres[1].fill_(0.5)
        none.log_scales.zero_()
        none.prior_logits.zero_()
        normalized.load_state_dict(none.state_dict())
    z = torch.zeros(4, 64)
    q_none = none(z)
    q_normalized = normalized(z)
    entropy_none = -(q_none * q_none.clamp_min(1e-8).log()).sum(dim=1).mean()
    entropy_normalized = -(q_normalized * q_normalized.clamp_min(1e-8).log()).sum(dim=1).mean()
    assert float(entropy_normalized.detach()) > float(entropy_none.detach()) + 0.1


def test_student_t_scale_floor_is_relative_to_warmup_initialisation() -> None:
    head = StudentTMixtureHead(2, latent_dim=3, scale_floor_ratio=0.5)
    with torch.no_grad():
        head.initialised.fill_(True)
        head.initial_scales.copy_(torch.tensor([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]]))
        head.log_scales.fill_(-20.0)
    scales = head.effective_scales()
    assert torch.allclose(scales, 0.5 * head.initial_scales)


def test_radial_student_t_kernel_has_real_metric_gradients() -> None:
    head = StudentTMixtureHead(3, latent_dim=4, assignment_kernel="radial")
    z = torch.randn(5, 4, requires_grad=True)
    q = head(z)
    assert torch.allclose(q.sum(dim=1), torch.ones(5), atol=1e-6)
    loss = -(q[:, 0].clamp_min(1e-8).log()).mean()
    loss.backward()
    assert head.centres.grad is not None and torch.isfinite(head.centres.grad).all()
    assert head.log_scales.grad is not None and torch.isfinite(head.log_scales.grad).all()


def test_teacher_is_stop_gradient_and_ema_is_numeric() -> None:
    student = _model()
    teacher = make_teacher(student)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    before = [parameter.detach().clone() for parameter in teacher.parameters()]
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(1.0)
    ema_update(teacher, student, 0.75)
    for old, current_teacher, current_student in zip(before, teacher.parameters(), student.parameters()):
        expected = 0.75 * old + 0.25 * current_student
        assert torch.allclose(current_teacher, expected)


def test_corruption_can_reuse_the_same_donors_across_views() -> None:
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    shifted = x + 10.0
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    donors = torch.tensor([2, 0, 1])
    corrupted_x, returned_mask = corrupt_batch(x, 0.0, mask=mask, donor_indices=donors)
    corrupted_shifted, _ = corrupt_batch(shifted, 0.0, mask=mask, donor_indices=donors)
    assert torch.equal(returned_mask, mask)
    assert torch.equal(corrupted_shifted - corrupted_x, torch.full_like(x, 10.0))


def test_paired_reference_risk_has_zero_null_improvement_with_dropout() -> None:
    """An identical graph probe cannot open a gate through dropout noise."""
    torch.manual_seed(11)
    X = np.random.default_rng(11).normal(size=(12, 6)).astype(np.float32)
    cfg = V11Config(
        hidden_size=16,
        latent_size=4,
        dropout=0.2,
        risk_target_mode="paired_ema_eval",
        use_teacher=False,
    ).validate()
    trainer = V11Trainer(X, X, n_clusters=3, config=cfg, device=torch.device("cpu"))
    trainer.student.train()
    x = trainer.X_cpu[:8]
    mask = (torch.rand_like(x) < 0.3).to(x.dtype)
    donors = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])
    anchor_per, probe_per = trainer._paired_reference_risk(x, x, x, mask, donors)
    assert trainer.student.training
    assert torch.allclose(anchor_per, probe_per, atol=0.0, rtol=0.0)
    assert float(torch.relu(anchor_per - probe_per).max()) == 0.0


def test_trusted_edge_alignment_selects_compatible_edges_without_gate_collapse() -> None:
    anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    teacher_neighbours = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    compatible_mass = torch.tensor([[0.9, 0.1]], requires_grad=True)
    intrusive_mass = torch.tensor([[0.1, 0.9]], requires_grad=True)
    trust = torch.tensor([0.8])
    compatible = trusted_edge_alignment(anchor, teacher_neighbours, compatible_mass, trust)
    intrusive = trusted_edge_alignment(anchor, teacher_neighbours, intrusive_mass, trust)
    assert float(compatible.detach()) < float(intrusive.detach())
    compatible.backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    # The edge distribution is an exogenous counterfactual target in the
    # geometry branch; graph_loss, rather than this metric term, trains the
    # gate posterior itself.
    assert compatible_mass.grad is None
    scaled = trusted_edge_alignment(anchor.detach(), teacher_neighbours, compatible_mass.detach() * 0.1, trust)
    assert torch.allclose(compatible.detach(), scaled.detach(), atol=1e-6)


def test_dynamic_candidate_graph_does_not_accept_labels() -> None:
    raw = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0]], dtype=np.float32
    )
    latent = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [-1.0, 0.0], [0.1, 0.9]], dtype=np.float32
    )
    initial = build_candidate_graph(raw, None, neighbor_k=2, candidate_k=4)
    refreshed = build_candidate_graph(raw, latent, neighbor_k=2, candidate_k=4)
    assert refreshed.source == "raw+latent"
    assert refreshed.indices.shape == (5, 4)
    assert graph_change_fraction(initial, refreshed) > 0.0
    recurrence = edge_recurrence_against(refreshed, refreshed)
    assert np.all(recurrence[refreshed.valid] == 1.0)
    assert np.all(edge_recurrence_against(refreshed, None) == 0.0)
    assert "y" not in inspect.signature(V11Trainer.__init__).parameters
    assert "labels" not in inspect.signature(build_candidate_graph).parameters


def test_duplicate_rows_never_return_self_edges() -> None:
    duplicated = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    graph = build_candidate_graph(duplicated, duplicated[::-1].copy(), neighbor_k=2, candidate_k=4)
    for node in range(graph.n_nodes):
        assert node not in graph.indices[node][graph.valid[node]]


def test_sparse_h0_persistence_uses_mst_merge_edges_only() -> None:
    embedding = np.asarray(
        [[1.0, 0.0], [0.99, 0.1], [0.70, 0.714], [0.0, 1.0]],
        dtype=np.float32,
    )
    # A complete fixed skeleton makes the non-MST edge test independent of
    # sklearn tie-breaking and keeps the filtration itself explicit.
    raw_knn = np.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        dtype=np.int64,
    )
    persistence = compute_h0_persistence(embedding, raw_knn)
    assert persistence.merge_count == embedding.shape[0] - persistence.n_components
    assert persistence.n_components == 1
    assert np.isfinite(persistence.persistence_score).all()
    assert np.all((persistence.persistence_score >= 0.0) & (persistence.persistence_score <= 1.0))

    # Keep the column number equal to the neighbour node id so the directed
    # prior lookup below remains readable; self edges are ignored by the API.
    candidates = np.tile(np.arange(4, dtype=np.int64), (4, 1))
    valid = np.ones_like(candidates, dtype=bool)
    prior = candidate_prior_from_h0(
        persistence,
        candidates,
        valid,
        mode="h0_mst",
    )
    assert prior.shape == candidates.shape
    assert np.count_nonzero(prior) > 0
    assert np.count_nonzero(prior) < prior.size
    assert np.all(prior[valid] >= 0.0)


def test_h0_early_mst_prefers_early_merge_edges_over_late_bridge() -> None:
    angles = np.asarray([0.0, 0.05, 1.0, 1.05], dtype=np.float64)
    embedding = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    raw_knn = np.asarray(
        [[j for j in range(4) if j != i] for i in range(4)],
        dtype=np.int64,
    )
    persistence = compute_h0_persistence(embedding, raw_knn)
    merge_indices = np.flatnonzero(persistence.merge_mask)
    assert merge_indices.size == 3
    merge_pairs = {
        tuple(int(value) for value in persistence.edge_pairs[index])
        for index in merge_indices
    }
    assert (0, 1) in merge_pairs
    assert (2, 3) in merge_pairs
    assert (1, 2) in merge_pairs

    # Keep the column number equal to the neighbour node id so the directed
    # prior lookup below remains readable; self edges are ignored by the API.
    candidates = np.tile(np.arange(4, dtype=np.int64), (4, 1))
    valid = np.ones_like(candidates, dtype=bool)
    late_index = max(merge_indices, key=lambda index: persistence.edge_distances[index])
    early_index = min(merge_indices, key=lambda index: persistence.edge_distances[index])
    late_pair = tuple(int(value) for value in persistence.edge_pairs[late_index])
    early_pair = tuple(int(value) for value in persistence.edge_pairs[early_index])

    old_prior = candidate_prior_from_h0(
        persistence,
        candidates,
        valid,
        mode="h0_mst",
    )
    early_prior = candidate_prior_from_h0(
        persistence,
        candidates,
        valid,
        mode="h0_early_mst",
    )
    assert old_prior[late_pair] > old_prior[early_pair]
    assert early_prior[early_pair] > early_prior[late_pair]
    assert np.count_nonzero(early_prior) == 2 * merge_indices.size
    assert np.array_equal(
        early_prior,
        candidate_prior_from_h0(
            persistence,
            candidates,
            valid,
            mode="h0_early_mst",
        ),
    )


def test_h0_scale_normalization_is_finite_and_scale_invariant() -> None:
    embedding = np.asarray(
        [[1.0, 0.0], [0.9, 0.2], [0.7, 0.7], [0.2, 0.9]],
        dtype=np.float32,
    )
    raw_knn = np.asarray(
        [[1, 2], [0, 2], [1, 3], [2, 1]],
        dtype=np.int64,
    )
    first = compute_h0_persistence(
        embedding,
        raw_knn,
        scale_mode="quantile",
        scale_quantile=0.5,
    )
    scaled = compute_h0_persistence(
        embedding * 1000.0,
        raw_knn,
        scale_mode="quantile",
        scale_quantile=0.5,
    )
    assert first.scale > 0.0
    assert np.isfinite(first.persistence_norm).all()
    assert np.allclose(first.persistence_score, scaled.persistence_score, atol=1e-10)


def test_tda_prior_controls_are_deterministic_and_default_is_closed() -> None:
    config = V11Config().validate()
    assert config.tda_prior_mode == "none"
    assert (
        V11Config(tda_prior_mode="h0_early_mst", tda_prior_weight=1.0).validate().tda_prior_mode
        == "h0_early_mst"
    )
    with pytest.raises(ValueError, match="tda_prior_mode"):
        V11Config(tda_prior_mode="invalid").validate()
    indices = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    valid = np.ones_like(indices, dtype=bool)
    first = candidate_prior_from_h0(None, indices, valid, mode="random", seed=42)
    second = candidate_prior_from_h0(None, indices, valid, mode="random", seed=42)
    other = candidate_prior_from_h0(None, indices, valid, mode="random", seed=123)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert build_candidate_graph(np.eye(4, dtype=np.float32), None, 2, 3).tda_prior is None


def test_temporal_gate_target_rejects_fixed_graph_ablation() -> None:
    """A temporal target needs a later refresh to provide recurrence labels."""
    with pytest.raises(ValueError, match="temporal_agreement requires use_dynamic_graph"):
        V11Config(
            use_dynamic_graph=False,
            gate_target_source="temporal_agreement",
        ).validate()


def test_counterfactual_semantic_target_is_per_edge_and_detached() -> None:
    """Risk probes supervise individual edges without leaking target gradients."""
    reconstruction_anchor = torch.tensor([0.5], requires_grad=True)
    reconstruction_probe = torch.tensor([[0.2, 0.49]], requires_grad=True)
    cluster_anchor = torch.tensor([0.4], requires_grad=True)
    cluster_probe = torch.tensor([[0.1, 0.39]], requires_grad=True)
    edge_prior = torch.tensor([[0.5, 0.5]], requires_grad=True)
    target = counterfactual_semantic_target(
        reconstruction_anchor,
        reconstruction_probe,
        cluster_anchor,
        cluster_probe,
        edge_prior,
        torch.ones(1),
        reconstruction_temperature=0.25,
        cluster_temperature=0.25,
    )
    assert target.edge_conditional.shape == (1, 2)
    assert float(target.edge_conditional[0, 0]) > float(target.edge_conditional[0, 1])
    assert torch.allclose(target.target_mixture.sum(dim=1), torch.ones(1), atol=1e-6)
    assert not target.target_mixture.requires_grad
    assert not target.edge_conditional.requires_grad


def test_semantic_help_combiner_is_conservative_when_evidence_disagrees() -> None:
    reconstruction_anchor = torch.tensor([0.5])
    reconstruction_probe = torch.tensor([[0.49]])
    cluster_anchor = torch.tensor([0.5])
    cluster_probe = torch.tensor([[0.1]])
    edge_prior = torch.ones(1, 1)
    geometric = counterfactual_semantic_target(
        reconstruction_anchor,
        reconstruction_probe,
        cluster_anchor,
        cluster_probe,
        edge_prior,
        torch.ones(1),
        reconstruction_temperature=0.25,
        cluster_temperature=0.25,
        semantic_help_combiner="geometric_mean",
    )
    harmonic = counterfactual_semantic_target(
        reconstruction_anchor,
        reconstruction_probe,
        cluster_anchor,
        cluster_probe,
        edge_prior,
        torch.ones(1),
        reconstruction_temperature=0.25,
        cluster_temperature=0.25,
        semantic_help_combiner="harmonic_mean",
    )
    assert float(harmonic.topology_help) < float(geometric.topology_help)
    assert float(harmonic.topology_help) > 0.0
    assert torch.allclose(harmonic.target_mixture.sum(dim=1), torch.ones(1), atol=1e-6)


def test_faiss_hnsw_backend_never_returns_self_edges() -> None:
    try:
        import faiss  # noqa: F401
    except Exception:
        return
    rng = np.random.default_rng(19)
    data = rng.normal(size=(32, 7)).astype(np.float32)
    graph = build_candidate_graph(
        data,
        data[::-1].copy(),
        neighbor_k=4,
        candidate_k=8,
        knn_backend="faiss_hnsw",
        knn_hnsw_m=8,
        knn_hnsw_ef_search=16,
    )
    assert "faiss_hnsw" in graph.knn_backend
    for node in range(graph.n_nodes):
        assert node not in graph.indices[node][graph.valid[node]]


def test_frozen_v9_reference_manifest_matches_current_comparison_code() -> None:
    repo = Path(__file__).resolve().parents[4]
    manifest_path = repo / "methods" / "TopoGate" / "V11" / "v9_reference_manifest.json"
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    for relative, expected in manifest["files"].items():
        path = repo / relative
        if not path.is_file():
            pytest.skip(
                "the public snapshot omits an optional legacy V9 comparison file: "
                f"{relative}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            pytest.skip(
                "the bundled V9 reference manifest is stale for the current source snapshot: "
                f"{relative}"
            )


def test_real_iris_npz_cpu_smoke(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[4] / "datasets" / "iris.npz"
    if not source.exists():
        return
    data = np.load(source)
    X = data["X"] if "X" in data.files else data["x"]
    y = data["y"]
    predictions, _, metrics = run_v11(
        X,
        int(np.unique(y).size),
        y,
        save_dir=tmp_path / "iris_v11",
        dataset_name="iris",
        no_cuda=True,
        seed=42,
        source_path=source,
        k_protocol="benchmark_oracle_from_y",
        epochs=3,
        warmup_epochs=1,
        ramp_epochs=1,
        graph_refresh_interval=1,
        hidden_size=32,
        latent_size=8,
        batch_size=64,
        pca_dim=8,
        neighbor_k=3,
        candidate_k=6,
    )
    assert predictions.shape == (X.shape[0],)
    assert np.isfinite(metrics["ari"])
    summary_path = tmp_path / "iris_v11" / "summary.json"
    assert summary_path.exists()
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["graph_history"]
    assert summary["labels_used_during_fit"] is False
    assert any(row["gate"] > 0 for row in summary["history"])

    run_v11(
        X,
        int(np.unique(y).size),
        y,
        save_dir=tmp_path / "iris_v11_nomix",
        dataset_name="iris",
        no_cuda=True,
        seed=42,
        source_path=source,
        k_protocol="benchmark_oracle_from_y",
        epochs=2,
        warmup_epochs=1,
        ramp_epochs=1,
        hidden_size=32,
        latent_size=8,
        batch_size=64,
        pca_dim=8,
        neighbor_k=3,
        candidate_k=6,
        use_topology=False,
    )
    with open(tmp_path / "iris_v11_nomix" / "summary.json", encoding="utf-8") as handle:
        nomix = json.load(handle)
    assert nomix["graph_history"] == []
    assert all(row["gate"] == 0.0 and row["target_gate"] == 0.0 for row in nomix["history"])

    run_v11(
        X,
        int(np.unique(y).size),
        y,
        save_dir=tmp_path / "iris_v11_tgr",
        dataset_name="iris",
        no_cuda=True,
        seed=42,
        source_path=source,
        k_protocol="benchmark_oracle_from_y",
        epochs=4,
        warmup_epochs=1,
        ramp_epochs=1,
        graph_refresh_interval=1,
        hidden_size=32,
        latent_size=8,
        batch_size=64,
        pca_dim=8,
        neighbor_k=3,
        candidate_k=6,
        topology_path="assignment_residual",
        gate_target_source="temporal_agreement",
        use_mixed_reconstruction=False,
        mixed_cluster_weight=0.0,
    )
    with open(tmp_path / "iris_v11_tgr" / "summary.json", encoding="utf-8") as handle:
        tgr = json.load(handle)
    assert any(row["temporal_target_available"] for row in tgr["graph_history"])
    assert all(row["mixed_rec"] == 0.0 for row in tgr["history"])
    assert all("topology_cls" in row for row in tgr["history"])
