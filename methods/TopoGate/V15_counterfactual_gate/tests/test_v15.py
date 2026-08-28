from __future__ import annotations

import inspect
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import scipy.sparse as sp
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods.TopoGate.V15_counterfactual_gate.graph import (
    CandidateGraph,
    _build_union,
    build_candidate_graph,
    restrict_candidate_scope,
)
from methods.TopoGate.V15_counterfactual_gate.config import V15Config, load_config
from methods.TopoGate.V15_counterfactual_gate.model import SphericalPrototypeHead, V15Model, abstaining_sparsemax
from methods.TopoGate.V15_counterfactual_gate.run import run_v15
from methods.TopoGate.V15_counterfactual_gate.sparse import apply_mask, prepare_input
from methods.TopoGate.V15_counterfactual_gate.trainer import (
    V15Trainer,
    _shuffle_scores_within_valid,
    align_teacher_assignments,
    build_teacher_reference,
    clean_output_utility,
    counterfactual_utility,
    local_consensus_utility,
    operator_aligned_utility,
    stability_gain_utility,
    view_local_assignment_quality,
)
import methods.TopoGate.V15_counterfactual_gate.trainer as trainer_module
from scripts.V15.replay_gate_readouts import (
    abstaining_sparsemax_numpy,
    abstaining_top1_numpy,
    apply_assignment_readout,
    candidate_scope_mask,
    counterfactual_delta_numpy,
    replay_run,
    select_quality_reference,
)


_AUDIT_PATH = _PROJECT_ROOT / "scripts" / "V15" / "audit_stage1b_certificates.py"
_AUDIT_SPEC = importlib.util.spec_from_file_location("v15_stage1b_audit", _AUDIT_PATH)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
_posthoc_graph_metrics = _AUDIT_MODULE._posthoc_graph_metrics
_posthoc_oracle_ceiling = _AUDIT_MODULE._posthoc_oracle_ceiling


def test_non_positive_utility_is_exact_null_self() -> None:
    scores = torch.tensor([[-2.0, -0.1, 0.0], [0.2, -0.5, 0.1]])
    probabilities = abstaining_sparsemax(scores)
    assert torch.equal(probabilities[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert torch.equal(probabilities[1, 2], torch.tensor(0.0))
    assert float(probabilities[1, 3]) > 0.0


def test_spherical_prototype_initialisation_is_confident_on_separated_views() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0], [0.98, 0.05], [0.95, -0.08],
            [-1.0, 0.0], [-0.97, 0.08], [-0.95, -0.06],
            [0.0, 1.0], [0.06, 0.98], [-0.05, 0.96],
        ],
        dtype=np.float32,
    )
    head = SphericalPrototypeHead(3, 2, temperature=0.05)
    head.initialise(embeddings, seed=7, n_init=3)
    probabilities = head(torch.as_tensor(embeddings))
    assert probabilities.shape == (9, 3)
    assert torch.isfinite(probabilities).all()
    assert float(probabilities.max(dim=1).values.mean().detach()) > 0.8


def test_main_v15_configuration_keeps_student_trainable() -> None:
    config = load_config(
        _PROJECT_ROOT
        / "methods"
        / "TopoGate"
        / "V15_counterfactual_gate"
        / "configs"
        / "topogate_v15.yaml"
    )
    assert config.cluster_head == "spherical_prototype"
    assert config.freeze_backbone_after_teacher is False
    assert config.mask_strategy == "zero"
    assert config.teacher_reference_mode == "consensus"
    assert config.utility_target_mode == "operator_aligned"
    assert config.utility_reference_mode == "teacher"
    assert config.direct_utility_source == "clean_output"
    assert config.utility_lambda_rec == 0.0
    assert config.utility_relative_baseline is False
    assert config.utility_probe_pairs == 2
    assert config.utility_min_gain == 0.5
    assert config.gate_mode == "direct_counterfactual"
    assert config.gate_training_mode == "detached"
    assert config.gate_opportunity_mode == "none"
    assert config.raw_view_cluster_weight == 0.0
    assert (config.k_raw, config.k_latent, config.candidate_cap) == (10, 10, 20)
    assert config.counterfactual_distill_start_epoch == 80
    assert config.counterfactual_distill_weight == 0.0
    assert config.final_prediction_source == "gate_readout"
    assert config.output_mode == "assignment"
    assert config.output_alpha == 1.0


def test_learned_counterfactual_mode_is_distinct_from_exact_readout() -> None:
    exact = V15Config(
        gate_mode="direct_counterfactual",
        utility_target_mode="operator_aligned",
        gate_opportunity_mode="none",
    )
    learned = V15Config(
        gate_mode="counterfactual_learned",
        utility_target_mode="operator_aligned",
        lambda_gate=1.0,
        gate_opportunity_mode="none",
    )
    assert exact.gate_mode != learned.gate_mode
    assert exact.gate_mode == "direct_counterfactual"
    assert learned.gate_mode == "counterfactual_learned"
    assert learned.utility_target_mode == "operator_aligned"
    with pytest.raises(ValueError, match="positive lambda_gate"):
        V15Config(
            gate_mode="counterfactual_learned",
            utility_target_mode="operator_aligned",
            gate_opportunity_mode="none",
            lambda_gate=0.0,
        )


def test_posthoc_oracle_ceiling_detects_recoverable_assignment_errors(tmp_path: Path) -> None:
    labels = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int64)
    self_prediction = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    edge_prediction = np.asarray([[0], [0], [1], [1], [1], [0]], dtype=np.int64)
    np.save(tmp_path / "predictions.npy", self_prediction)
    np.savez_compressed(
        tmp_path / "gate_diagnostics.npz",
        final_probe_self_prediction=self_prediction,
        final_probe_edge_prediction=edge_prediction,
        final_gate_valid=np.ones_like(edge_prediction, dtype=bool),
    )
    certificate = _posthoc_oracle_ceiling(tmp_path, labels)
    assert certificate["status"] == "available"
    assert certificate["recoverable_error_fraction"] == 1.0
    assert certificate["oracle_gate"]["ari"] == 1.0
    assert certificate["oracle_delta_ari_vs_self"] > 0.0


def test_view_quality_is_permutation_invariant_and_rewards_local_agreement() -> None:
    probabilities = np.asarray(
        [[0.95, 0.05], [0.9, 0.1], [0.1, 0.9], [0.05, 0.95]], dtype=np.float32
    )
    neighbors = np.asarray([[1, 2], [0, 3], [3, 0], [2, 1]], dtype=np.int64)
    quality = view_local_assignment_quality(probabilities, neighbors)
    permuted = view_local_assignment_quality(probabilities[:, ::-1], neighbors)
    assert quality["local_agreement"] == permuted["local_agreement"]
    assert quality["score"] == permuted["score"]


def test_stability_gain_target_is_detached_and_zero_for_identical_views() -> None:
    q_self = torch.tensor([[0.8, 0.2]], requires_grad=True)
    q_edge = q_self[:, None, :].expand(-1, 2, -1).clone()
    rec_self = torch.tensor([0.2])
    rec_edge = rec_self[:, None].expand(-1, 2).clone()
    valid = torch.ones((1, 2), dtype=torch.bool)
    utility, semantic, damage = stability_gain_utility(
        q_self,
        q_edge,
        rec_self,
        rec_edge,
        q_self_second=q_self.detach(),
        q_edge_second=q_edge.detach(),
        rec_self_second=rec_self,
        rec_edge_second=rec_edge,
        lambda_rec=0.25,
        clip=4.0,
        valid=valid,
    )
    assert torch.equal(utility, torch.zeros_like(utility))
    assert torch.equal(semantic, torch.zeros_like(semantic))
    assert torch.equal(damage, torch.zeros_like(damage))
    assert not utility.requires_grad


def test_operator_aligned_utility_rewards_the_same_masked_edge_operator() -> None:
    reference = torch.tensor([[0.9, 0.1]])
    self_first = torch.tensor([[0.6, 0.4]], requires_grad=True)
    self_second = torch.tensor([[0.65, 0.35]], requires_grad=True)
    edge_first = torch.tensor([[[0.85, 0.15], [0.6, 0.4]]], requires_grad=True)
    edge_second = torch.tensor([[[0.84, 0.16], [0.65, 0.35]]], requires_grad=True)
    valid = torch.ones((1, 2), dtype=torch.bool)
    utility, semantic, damage = operator_aligned_utility(
        reference,
        self_first,
        edge_first,
        self_second,
        edge_second,
        torch.tensor([0.2]),
        torch.tensor([[0.2, 0.2]]),
        lambda_rec=0.25,
        stability_weight=1.0,
        relative_baseline=True,
        reference_mode="teacher",
        reference_temperature=0.05,
        clip=4.0,
        valid=valid,
    )
    assert utility[0, 0] > 0.0
    assert torch.isclose(utility[0, 1], torch.tensor(0.0), atol=1e-6)
    assert semantic[0, 0] > semantic[0, 1]
    assert torch.equal(damage, torch.zeros_like(damage))
    assert not utility.requires_grad


def test_operator_minimum_gain_closes_the_gate_exactly() -> None:
    reference = torch.tensor([[0.9, 0.1]])
    self_first = torch.tensor([[0.6, 0.4]])
    self_second = torch.tensor([[0.65, 0.35]])
    edge_first = torch.tensor([[[0.85, 0.15], [0.6, 0.4]]])
    edge_second = torch.tensor([[[0.84, 0.16], [0.65, 0.35]]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    utility, _, _ = operator_aligned_utility(
        reference,
        self_first,
        edge_first,
        self_second,
        edge_second,
        torch.tensor([0.2]),
        torch.tensor([[0.2, 0.2]]),
        lambda_rec=0.25,
        stability_weight=1.0,
        relative_baseline=True,
        reference_mode="teacher",
        reference_temperature=0.05,
        min_gain=4.0,
        clip=4.0,
        valid=valid,
    )
    pi = abstaining_sparsemax(utility, valid)
    assert torch.all(utility <= 0.0)
    assert torch.equal(pi, torch.tensor([[1.0, 0.0, 0.0]]))


def test_clean_output_utility_rewards_exact_reference_improvement() -> None:
    reference = torch.tensor([[0.9, 0.1]])
    self_assignment = torch.tensor([[0.6, 0.4]])
    edge_assignment = torch.tensor([[[0.85, 0.15], [0.4, 0.6]]])
    utility = clean_output_utility(
        reference,
        self_assignment,
        edge_assignment,
        min_gain=0.0,
        clip=4.0,
        valid=torch.ones((1, 2), dtype=torch.bool),
    )
    assert utility[0, 0] > 0.0
    assert utility[0, 1] < 0.0
    assert not utility.requires_grad


def test_teacher_reference_downweights_disagreement() -> None:
    latent = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    raw = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    reference, agreement, disagreement = build_teacher_reference(
        latent,
        raw,
        mode="consensus",
        raw_weight=0.5,
        temperature=0.25,
    )
    assert torch.allclose(reference.sum(dim=1), torch.ones(2))
    assert agreement[0] > agreement[1]
    assert disagreement[0] < disagreement[1]
    assert not reference.requires_grad


def test_teacher_reference_aligns_arbitrary_component_permutation() -> None:
    latent = torch.tensor([[0.95, 0.05], [0.10, 0.90], [0.85, 0.15]])
    raw_aligned = torch.tensor([[0.90, 0.10], [0.05, 0.95], [0.80, 0.20]])
    raw_permuted = raw_aligned[:, [1, 0]]
    reference_aligned, agreement_aligned, _ = build_teacher_reference(
        latent, raw_aligned, mode="consensus", raw_weight=0.5, temperature=0.25
    )
    reference_permuted, agreement_permuted, _ = build_teacher_reference(
        latent, raw_permuted, mode="consensus", raw_weight=0.5, temperature=0.25
    )
    assert torch.allclose(reference_aligned, reference_permuted, atol=1e-6)
    assert torch.allclose(agreement_aligned, agreement_permuted, atol=1e-6)


def test_logit_transport_stays_on_probability_simplex() -> None:
    model = V15Model(
        input_dim=5,
        hidden_dim=8,
        latent_dim=3,
        n_clusters=3,
        dropout=0.0,
        student_t_nu=4.0,
        cluster_normalize_latent=True,
        cluster_cosine_temperature=0.1,
    )
    z_self = torch.randn(4, 3)
    donors = torch.randn(4, 2, 3)
    edge_mass = torch.tensor([[0.6, 0.0], [0.2, 0.3], [0.0, 0.0], [1.0, 0.0]])
    probabilities = model.mix_assignments(z_self, donors, edge_mass, alpha=0.25)
    assert probabilities.shape == (4, 3)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-6)


def test_probability_transport_stays_on_simplex_and_respects_null() -> None:
    model = V15Model(
        input_dim=5,
        hidden_dim=8,
        latent_dim=3,
        n_clusters=3,
        dropout=0.0,
        student_t_nu=4.0,
        cluster_normalize_latent=True,
        cluster_cosine_temperature=0.1,
    )
    q_self = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    q_donor = torch.tensor([[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], [[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]]])
    edge_mass = torch.tensor([[0.0, 0.0], [0.4, 0.0]])
    probabilities = model.mix_probabilities(q_self, q_donor, edge_mass, alpha=0.25)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(probabilities[0], q_self[0], atol=1e-6)


def test_assignment_transport_has_exact_null_branch() -> None:
    model = V15Model(
        input_dim=5,
        hidden_dim=8,
        latent_dim=3,
        n_clusters=3,
        dropout=0.0,
        student_t_nu=4.0,
        cluster_normalize_latent=True,
        cluster_cosine_temperature=0.1,
    )
    q_self = torch.tensor([[0.7, 0.2, 0.1]])
    q_edge = torch.tensor([[[0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]])
    pi = torch.tensor([[1.0, 0.0, 0.0]])
    output = model.mix_assignment_output(q_self, q_edge, pi)
    assert torch.allclose(output, q_self, atol=1e-6)
    assert torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-6)


def test_assignment_transport_embedding_uses_the_same_null_and_edge_mass() -> None:
    z_self = torch.tensor([[1.0, 0.0]])
    z_edge = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])
    pi = torch.tensor([[0.25, 0.5, 0.25]])
    embedding = V15Model.mix_assignment_embedding(z_self, z_edge, pi)
    expected = 0.25 * z_self + 0.5 * z_edge[:, 0] + 0.25 * z_edge[:, 1]
    assert torch.allclose(embedding, expected, atol=1e-6)


def test_assignment_transport_embedding_matches_null_and_edge_mass() -> None:
    model = V15Model(
        input_dim=5,
        hidden_dim=8,
        latent_dim=3,
        n_clusters=3,
        dropout=0.0,
        student_t_nu=4.0,
        cluster_normalize_latent=True,
        cluster_cosine_temperature=0.1,
    )
    z_self = torch.tensor([[1.0, 0.0, 0.0]])
    z_edge = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    null_pi = torch.tensor([[1.0, 0.0, 0.0]])
    edge_pi = torch.tensor([[0.0, 1.0, 0.0]])
    null_embedding = model.mix_assignment_embedding(z_self, z_edge, null_pi)
    edge_embedding = model.mix_assignment_embedding(z_self, z_edge, edge_pi)
    assert torch.allclose(null_embedding, z_self, atol=1e-6)
    assert torch.allclose(edge_embedding, z_edge[:, 0], atol=1e-6)
    assert not torch.allclose(edge_embedding, z_self, atol=1e-6)


def test_invalid_candidates_have_zero_mass() -> None:
    scores = torch.tensor([[0.8, 0.4, 0.2]])
    valid = torch.tensor([[True, False, True]])
    probabilities = abstaining_sparsemax(scores, valid)
    assert torch.equal(probabilities[0, 2], torch.tensor(0.0))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))


def test_sparse_input_stays_csr_until_batch_fetch() -> None:
    raw = sp.csr_matrix(
        np.asarray([[0.0, 2.0, 0.0, 1.0], [0.0, 0.0, 3.0, 0.0]], dtype=np.float32)
    )
    prepared = prepare_input(raw, sparse_zero_threshold=0.0)
    assert prepared.sparse
    assert sp.isspmatrix_csr(prepared.matrix)
    assert prepared.profile["representation"] == "csr_log1p_row_normalized"
    batch = prepared.get(np.asarray([1]), torch.device("cpu"))
    assert batch.shape == (1, 4)
    assert torch.isfinite(batch).all()


def test_tfidf_sparse_transform_preserves_csr_and_row_norm() -> None:
    raw = sp.csr_matrix(np.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 2.0]], dtype=np.float32))
    prepared = prepare_input(raw, sparse_zero_threshold=0.0, sparse_transform="tfidf_l2")
    assert prepared.profile["representation"] == "csr_tfidf_l2"
    norms = np.sqrt(np.asarray(prepared.matrix.multiply(prepared.matrix).sum(axis=1)).ravel())
    assert np.allclose(norms, np.ones(2), atol=1e-5)


def test_row_swap_corruption_uses_observed_batch_values() -> None:
    x = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
    generator = torch.Generator().manual_seed(17)
    corrupted, mask = apply_mask(x, 1.0, generator=generator, strategy="row_swap")
    assert torch.equal(mask.bool(), corrupted.ne(x))
    for column in range(x.shape[1]):
        assert set(corrupted[:, column].tolist()).issubset(set(x[:, column].tolist()))


def test_zero_mask_never_marks_an_unchanged_sparse_zero() -> None:
    x = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    corrupted, mask = apply_mask(
        x,
        0.0,
        generator=torch.Generator().manual_seed(5),
        strategy="zero",
    )
    assert not mask[0].bool().any()
    assert torch.equal(mask[1].bool(), torch.tensor([False, True, False]))
    assert torch.equal(mask.bool(), corrupted.ne(x))


def test_candidate_graph_excludes_self_edges_for_duplicates() -> None:
    raw = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    prepared = prepare_input(raw, sparse_zero_threshold=0.0)
    latent = np.asarray(raw, dtype=np.float32)
    graph = build_candidate_graph(
        prepared,
        k_raw=3,
        k_latent=3,
        candidate_cap=5,
        raw_svd_dim=2,
        latent_embedding=latent,
        latent_graph_dim=2,
        seed=3,
    )
    for node in range(graph.n_nodes):
        assert node not in graph.indices[node][graph.valid[node]]


def test_candidate_graph_handles_single_zero_row_without_self_or_nan() -> None:
    raw = np.zeros((1, 4), dtype=np.float32)
    prepared = prepare_input(raw, sparse_zero_threshold=0.0)
    graph = build_candidate_graph(
        prepared,
        k_raw=3,
        k_latent=3,
        candidate_cap=5,
        raw_svd_dim=2,
        latent_embedding=np.zeros((1, 3), dtype=np.float32),
        latent_graph_dim=2,
        seed=4,
    )
    assert not graph.valid.any()
    assert np.isfinite(graph.features).all()


def test_union_prioritizes_consensus_then_raw_fallback() -> None:
    raw_indices = np.asarray([[1, 2], [0, 2], [0, 1], [0, 1]], dtype=np.int64)
    latent_indices = np.asarray([[3, 1], [3, 0], [3, 1], [2, 1]], dtype=np.int64)
    similarities = np.ones_like(raw_indices, dtype=np.float32)
    embedding = np.eye(4, dtype=np.float32)
    indices, features, valid, _ = _build_union(
        raw_indices,
        similarities,
        latent_indices,
        similarities,
        embedding,
        embedding,
        cap=2,
    )
    assert np.array_equal(indices[0, valid[0]], np.asarray([1, 2]))
    assert np.isclose(features[0, 0, 2], 0.0)
    assert np.isclose(features[0, 1, 2], -1.0)


def test_union_quota_keeps_latent_only_recall_when_capacity_permits() -> None:
    raw_indices = np.asarray([[1, 2], [0, 2], [0, 1], [0, 1]], dtype=np.int64)
    latent_indices = np.asarray([[1, 3], [0, 3], [1, 3], [1, 2]], dtype=np.int64)
    similarities = np.ones_like(raw_indices, dtype=np.float32)
    embedding = np.eye(4, dtype=np.float32)
    indices, features, valid, _ = _build_union(
        raw_indices,
        similarities,
        latent_indices,
        similarities,
        embedding,
        embedding,
        cap=3,
    )
    selected = indices[0, valid[0]]
    sources = features[0, valid[0], 2]
    assert set(selected.tolist()) == {1, 2, 3}
    assert set(sources.tolist()) == {0.0, -1.0, 1.0}


def test_complete_union_preserves_every_raw_and_latent_candidate() -> None:
    raw_indices = np.asarray([[1, 2], [0, 2], [0, 1], [0, 1]], dtype=np.int64)
    latent_indices = np.asarray([[3, 1], [3, 0], [3, 1], [2, 1]], dtype=np.int64)
    similarities = np.ones_like(raw_indices, dtype=np.float32)
    embedding = np.eye(4, dtype=np.float32)
    indices, _features, valid, profile = _build_union(
        raw_indices,
        similarities,
        latent_indices,
        similarities,
        embedding,
        embedding,
        cap=4,
    )
    for row in range(raw_indices.shape[0]):
        union = set(indices[row, valid[row]].tolist())
        assert set(raw_indices[row].tolist()).issubset(union)
        assert set(latent_indices[row].tolist()).issubset(union)
    assert profile["complete_union_requested"] is True


def test_counterfactual_target_is_detached_and_has_explicit_shape() -> None:
    q_teacher = torch.softmax(torch.randn(2, 3), dim=1)
    q_self = torch.softmax(torch.randn(2, 3), dim=1)
    q_edge = torch.softmax(torch.randn(2, 4, 3), dim=2).requires_grad_(True)
    rec_self = torch.rand(2)
    rec_edge = torch.rand(2, 4)
    valid = torch.tensor([[True, True, False, True], [True, False, True, True]])
    utility = counterfactual_utility(
        q_teacher,
        q_self,
        q_edge,
        rec_self,
        rec_edge,
        lambda_rec=0.25,
        clip=4.0,
        valid=valid,
    )
    assert utility.shape == (2, 4)
    assert not utility.requires_grad
    assert torch.all(utility[~valid] == -4.0)


def test_counterfactual_target_preserves_absolute_zero() -> None:
    q_teacher = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    q_self = q_teacher.clone()
    q_edge = q_teacher[:, None, :].expand(-1, 3, -1).clone()
    rec_self = torch.tensor([0.25, 0.5])
    rec_edge = rec_self[:, None].expand(-1, 3).clone()
    valid = torch.ones((2, 3), dtype=torch.bool)
    utility = counterfactual_utility(
        q_teacher,
        q_self,
        q_edge,
        rec_self,
        rec_edge,
        lambda_rec=0.25,
        clip=4.0,
        valid=valid,
    )
    assert torch.equal(utility, torch.zeros_like(utility))


def test_local_consensus_utility_is_detached_and_shape_preserving() -> None:
    q_self_first = torch.softmax(torch.randn(2, 3), dim=1)
    q_self_second = torch.softmax(torch.randn(2, 3), dim=1)
    q_edge_first = torch.softmax(torch.randn(2, 3, 3), dim=2)
    q_edge_second = torch.softmax(torch.randn(2, 3, 3), dim=2)
    q_donor = torch.softmax(torch.randn(2, 3, 3), dim=2)
    rec_self = torch.rand(2)
    rec_edge = torch.rand(2, 3)
    valid = torch.tensor([[True, True, False], [True, False, True]])
    utility, semantic, damage = local_consensus_utility(
        q_self_first,
        q_edge_first,
        q_self_second,
        q_edge_second,
        q_donor,
        rec_self,
        rec_edge,
        lambda_rec=0.25,
        stability_weight=1.0,
        confidence_weight=0.5,
        relative_baseline=False,
        clip=4.0,
        valid=valid,
    )
    assert utility.shape == semantic.shape == damage.shape == (2, 3)
    assert not utility.requires_grad
    assert torch.all(utility[~valid] == -4.0)


def test_local_consensus_mode_dispatches_through_target_for_batch(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_local_consensus(*args, **kwargs):
        calls["count"] += 1
        valid = kwargs["valid"]
        zeros = torch.zeros_like(valid, dtype=torch.float32)
        return zeros, zeros, zeros

    monkeypatch.setattr(trainer_module, "local_consensus_utility", fake_local_consensus)
    rng = np.random.default_rng(61)
    X = np.abs(rng.normal(size=(12, 6))).astype(np.float32)
    X[rng.random(X.shape) < 0.6] = 0.0
    data = prepare_input(X, sparse_zero_threshold=0.5, sparse_transform="log1p_row")
    config = V15Config(
        epochs=2,
        teacher_pretrain_epochs=1,
        batch_size=6,
        hidden_dim=8,
        latent_dim=3,
        dropout=0.0,
        raw_svd_dim=3,
        latent_graph_dim=3,
        k_raw=2,
        k_latent=2,
        candidate_cap=4,
        graph_refresh_interval=2,
        gate_mode="direct_counterfactual",
        utility_target_mode="local_consensus",
        utility_probe_pairs=1,
        output_mode="assignment",
        gate_opportunity_mode="none",
        cluster_frequency_weight=0.0,
        cluster_frequency_uniform_mix=0.0,
        raw_view_cluster_weight=0.0,
        n_init=1,
        no_cuda=True,
    )
    trainer = V15Trainer(data, 3, config, torch.device("cpu"))

    class ForbiddenScorer(torch.nn.Module):
        def forward(self, _: torch.Tensor) -> torch.Tensor:
            raise AssertionError("local-consensus exact path invoked the amortized scorer")

    trainer.model.utility_scorer = ForbiddenScorer()
    result = trainer.fit()
    assert calls["count"] > 0
    assert result.gate_diagnostics["final_utility_hat"].shape == (12, 4)


def test_stage1b_graph_metrics_are_posthoc_and_exclude_self_edges() -> None:
    indices = np.asarray([[1, 2], [0, 2], [3, 0], [2, 1]], dtype=np.int64)
    valid = np.ones_like(indices, dtype=bool)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    metrics = _posthoc_graph_metrics(indices, valid, labels)
    assert metrics["edge_purity"] == 0.5
    assert metrics["candidate_recall_budget_normalized"] == 1.0
    assert metrics["candidate_coverage_any_same_label"] == 1.0
    assert metrics["self_edge_count"] == 0
    assert metrics["invalid_index_count"] == 0


def test_stage1b_utility_audit_keeps_heldout_claim_separate(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    target = np.asarray([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    predicted = np.asarray([[0.8, -0.8], [0.2, -0.2]], dtype=np.float32)
    valid = np.ones_like(target, dtype=bool)
    np.savez(run / "gate_diagnostics.npz", utility_target=target, utility_hat=predicted, utility_valid=valid)
    certificate = _AUDIT_MODULE._utility_certificate(run, {"labels_used_during_fit": False})
    assert certificate["in_sample_scorer_fit"]["auroc"] is not None
    assert certificate["held_out_utility_prediction"]["status"] == "not_available"
    assert certificate["independent_view_counterfactual_gain"]["status"] == "not_available"


def test_trainer_does_not_accept_labels() -> None:
    assert "labels" not in inspect.signature(V15Trainer.__init__).parameters
    assert "y" not in inspect.signature(V15Trainer.__init__).parameters


def test_direct_counterfactual_fit_never_calls_utility_scorer() -> None:
    class ForbiddenScorer(torch.nn.Module):
        def forward(self, _: torch.Tensor) -> torch.Tensor:
            raise AssertionError("direct counterfactual path invoked the amortized scorer")

    rng = np.random.default_rng(19)
    X = np.abs(rng.normal(size=(15, 7))).astype(np.float32)
    X[rng.random(X.shape) < 0.65] = 0.0
    X[:5, 0] += 3.0
    X[5:10, 1] += 3.0
    X[10:, 2] += 3.0
    data = prepare_input(X, sparse_zero_threshold=0.5, sparse_transform="log1p_row")
    config = V15Config(
        epochs=2,
        teacher_pretrain_epochs=1,
        teacher_selection_warmup_epochs=0,
        batch_size=5,
        hidden_dim=10,
        latent_dim=4,
        dropout=0.0,
        raw_svd_dim=4,
        latent_graph_dim=4,
        k_raw=3,
        k_latent=3,
        candidate_cap=5,
        graph_refresh_interval=2,
        gate_mode="direct_counterfactual",
        utility_target_mode="operator_aligned",
        output_mode="assignment",
        gate_opportunity_mode="none",
        teacher_reference_mode="latent",
        raw_view_cluster_weight=0.0,
        lambda_gate=0.0,
        cluster_frequency_weight=0.0,
        cluster_frequency_uniform_mix=0.0,
        n_init=1,
        no_cuda=True,
    )
    trainer = V15Trainer(data, 3, config, torch.device("cpu"))
    trainer.model.utility_scorer = ForbiddenScorer()
    result = trainer.fit()
    assert result.probabilities.shape == (15, 3)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert result.gate_diagnostics["final_predicted_pi"].shape == (15, 6)
    assert np.array_equal(
        result.gate_diagnostics["final_probe_self_prediction"],
        result.gate_diagnostics["final_student_probabilities"].argmax(axis=1),
    )


def test_learned_counterfactual_fits_scorer_on_train_and_reports_holdout() -> None:
    rng = np.random.default_rng(29)
    X = np.abs(rng.normal(size=(18, 8))).astype(np.float32)
    X[rng.random(X.shape) < 0.65] = 0.0
    X[:6, 0] += 3.0
    X[6:12, 1] += 3.0
    X[12:, 2] += 3.0
    data = prepare_input(X, sparse_zero_threshold=0.5, sparse_transform="log1p_row")
    config = V15Config(
        epochs=3,
        teacher_pretrain_epochs=1,
        teacher_selection_warmup_epochs=0,
        batch_size=6,
        hidden_dim=10,
        latent_dim=4,
        dropout=0.0,
        raw_svd_dim=4,
        latent_graph_dim=4,
        k_raw=3,
        k_latent=3,
        candidate_cap=6,
        graph_refresh_interval=2,
        gate_mode="counterfactual_learned",
        utility_target_mode="operator_aligned",
        utility_reference_mode="cross_view",
        utility_probe_pairs=1,
        utility_min_gain=0.0,
        output_mode="assignment",
        gate_opportunity_mode="none",
        teacher_reference_mode="latent",
        raw_view_cluster_weight=0.0,
        lambda_gate=1.0,
        utility_sign_weight=0.0,
        cluster_frequency_weight=0.0,
        cluster_frequency_uniform_mix=0.0,
        warmup_epochs=0,
        utility_holdout_fraction=0.4,
        n_init=1,
        no_cuda=True,
    )
    trainer = V15Trainer(data, 3, config, torch.device("cpu"))
    initial = [parameter.detach().clone() for parameter in trainer.model.utility_scorer.parameters()]
    result = trainer.fit()
    joint = [row for row in result.history if row["phase"] == "joint"]
    assert any(row["loss_gate_regression"] > 0.0 for row in joint)
    assert any(row["loss_gate_holdout"] >= 0.0 for row in joint)
    assert any(
        not torch.allclose(before, after.detach())
        for before, after in zip(initial, trainer.model.utility_scorer.parameters())
    )
    assert result.gate_diagnostics["final_utility_hat"].shape == (18, 6)


def test_delayed_output_disabled_has_zero_topology_loss_and_exports_clean_student() -> None:
    rng = np.random.default_rng(23)
    X = np.abs(rng.normal(size=(18, 8))).astype(np.float32)
    X[rng.random(X.shape) < 0.7] = 0.0
    X[:6, 0] += 3.0
    X[6:12, 1] += 3.0
    X[12:, 2] += 3.0
    data = prepare_input(X, sparse_zero_threshold=0.5, sparse_transform="log1p_row")
    config = V15Config(
        epochs=4,
        teacher_pretrain_epochs=1,
        teacher_selection_warmup_epochs=0,
        batch_size=6,
        hidden_dim=10,
        latent_dim=4,
        dropout=0.0,
        raw_svd_dim=4,
        latent_graph_dim=4,
        k_raw=3,
        k_latent=3,
        candidate_cap=6,
        graph_refresh_interval=2,
        gate_mode="output_disabled",
        utility_target_mode="operator_aligned",
        utility_reference_mode="cross_view",
        output_mode="assignment",
        gate_opportunity_mode="none",
        teacher_reference_mode="latent",
        raw_view_cluster_weight=0.0,
        lambda_gate=0.0,
        cluster_frequency_weight=0.0,
        cluster_frequency_uniform_mix=0.0,
        counterfactual_distill_weight=0.1,
        counterfactual_distill_start_epoch=3,
        final_prediction_source="student_clean",
        warmup_epochs=0,
        n_init=1,
        no_cuda=True,
    )
    result = V15Trainer(data, 3, config, torch.device("cpu")).fit()
    joint = [row for row in result.history if row["phase"] == "joint"]
    assert [row["distill_active"] for row in joint] == [False, True, True]
    assert all(row["loss_counterfactual_distill"] == 0.0 for row in joint)
    assert all(row["distill_topology_mass"] == 0.0 for row in joint)
    assert np.allclose(
        result.probabilities,
        result.gate_diagnostics["final_student_probabilities"],
        atol=1e-6,
    )
    assert np.array_equal(result.predictions, result.probabilities.argmax(axis=1))


def test_distillation_rejects_non_counterfactual_gate_modes() -> None:
    with pytest.raises(ValueError, match="counterfactual distillation requires"):
        V15Config(
            gate_mode="union_uniform",
            counterfactual_distill_weight=0.1,
            counterfactual_distill_start_epoch=80,
        )


def test_shuffled_control_preserves_valid_score_multisets() -> None:
    torch.manual_seed(31)
    scores = torch.tensor([[1.0, 2.0, 3.0, -9.0], [4.0, 5.0, -8.0, -7.0]])
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    shuffled = _shuffle_scores_within_valid(scores, valid)
    for row in range(scores.shape[0]):
        assert torch.equal(
            torch.sort(shuffled[row, valid[row]]).values,
            torch.sort(scores[row, valid[row]]).values,
        )
        assert torch.equal(shuffled[row, ~valid[row]], scores[row, ~valid[row]])


def test_replay_top1_and_sparsemax_preserve_exact_null_contract() -> None:
    utility = np.asarray([[-1.0, 0.4, 0.2], [-0.2, -0.1, -0.3]], dtype=np.float32)
    valid = np.ones_like(utility, dtype=bool)
    top1 = abstaining_top1_numpy(utility, valid)
    sparse = abstaining_sparsemax_numpy(utility, valid)
    assert np.array_equal(top1[0], np.asarray([0.0, 0.0, 1.0, 0.0]))
    assert np.array_equal(top1[1], np.asarray([1.0, 0.0, 0.0, 0.0]))
    assert np.isclose(top1.sum(axis=1), 1.0).all()
    assert np.isclose(sparse[1], np.asarray([1.0, 0.0, 0.0, 0.0])).all()


def test_replay_assignment_operator_keeps_selected_edge_exact() -> None:
    q_self = np.asarray([[0.8, 0.2]], dtype=np.float32)
    q_edge = np.asarray([[[0.1, 0.9], [0.7, 0.3]]], dtype=np.float32)
    z_self = np.asarray([[1.0, 0.0]], dtype=np.float32)
    z_edge = np.asarray([[[0.0, 1.0], [2.0, 0.0]]], dtype=np.float32)
    pi = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    embedding, probabilities = apply_assignment_readout(q_self, q_edge, pi, z_self, z_edge)
    assert np.allclose(embedding, z_edge[:, 0])
    assert np.allclose(probabilities, q_edge[:, 0])


def test_replay_counterfactual_delta_has_the_expected_sign() -> None:
    reference = np.asarray([[0.9, 0.1]], dtype=np.float32)
    q_self = np.asarray([[0.6, 0.4]], dtype=np.float32)
    q_edge = np.asarray([[[0.85, 0.15], [0.4, 0.6]]], dtype=np.float32)
    delta = counterfactual_delta_numpy(reference, q_self, q_edge)
    assert delta[0, 0] > 0.0
    assert delta[0, 1] < 0.0


def test_replay_quality_selector_uses_label_free_local_coherence() -> None:
    neighbors = np.asarray([[1], [0], [3], [2]], dtype=np.int64)
    coherent = np.asarray(
        [[0.95, 0.05], [0.90, 0.10], [0.05, 0.95], [0.10, 0.90]], dtype=np.float32
    )
    fragmented = np.asarray(
        [[0.95, 0.05], [0.05, 0.95], [0.95, 0.05], [0.05, 0.95]], dtype=np.float32
    )
    selected, scores = select_quality_reference(
        {"coherent": coherent, "fragmented": fragmented}, neighbors
    )
    assert selected == "coherent"
    assert scores["coherent"]["score"] > scores["fragmented"]["score"]


def test_replay_candidate_scope_uses_graph_source_identity() -> None:
    features = np.zeros((1, 3, 3), dtype=np.float32)
    features[0, :, 2] = np.asarray([-1.0, 0.0, 1.0])
    valid = np.ones((1, 3), dtype=bool)
    assert np.array_equal(candidate_scope_mask(features, valid, "both_views"), [[False, True, False]])
    assert np.array_equal(candidate_scope_mask(features, valid, "raw_supported"), [[True, True, False]])
    assert np.array_equal(candidate_scope_mask(features, valid, "latent_supported"), [[False, True, True]])


def test_training_candidate_scope_masks_sources_without_rebuilding_graph() -> None:
    features = np.zeros((1, 3, 6), dtype=np.float32)
    features[0, :, 2] = np.asarray([-1.0, 0.0, 1.0])
    graph = CandidateGraph(
        indices=np.asarray([[1, 2, 3]], dtype=np.int64),
        features=features,
        valid=np.ones((1, 3), dtype=bool),
        raw_indices=np.asarray([[1]], dtype=np.int64),
        latent_indices=np.asarray([[2]], dtype=np.int64),
        raw_embedding=np.zeros((1, 2), dtype=np.float32),
        latent_embedding=np.zeros((1, 2), dtype=np.float32),
        profile={},
    )
    scoped = restrict_candidate_scope(graph, "raw_supported")
    assert np.array_equal(scoped.indices, graph.indices)
    assert np.array_equal(scoped.valid, [[True, True, False]])
    assert scoped.profile["candidate_scope"] == "raw_supported"
    assert scoped.profile["mean_valid_candidates"] == 2.0


def test_quality_auto_reference_can_select_augmented_view() -> None:
    config = V15Config(teacher_reference_mode="quality_auto")
    assert config.teacher_reference_mode == "quality_auto"


def test_replay_run_requires_full_edge_assignments(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    np.savez_compressed(run / "gate_diagnostics.npz", final_utility_hat=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="replay certificate arrays"):
        replay_run(run, tmp_path / "replay")


def test_tiny_realistic_smoke_writes_v15_contract(tmp_path) -> None:
    rng = np.random.default_rng(11)
    X = np.abs(rng.normal(size=(18, 8))).astype(np.float32)
    X[rng.random(X.shape) < 0.7] = 0.0
    X[:6, 0] += 4.0
    X[6:12, 1] += 4.0
    X[12:, 2] += 4.0
    y = np.repeat(np.arange(3), 6)
    output = tmp_path / "tiny"
    run_v15(
        X,
        3,
        y,
        save_dir=output,
        dataset_name="v15_tiny",
        seed=42,
        no_cuda=True,
        epochs=2,
        batch_size=9,
        hidden_dim=12,
        latent_dim=4,
        dropout=0.0,
        raw_svd_dim=4,
        latent_graph_dim=4,
        k_raw=3,
        k_latent=3,
        candidate_cap=6,
        graph_refresh_interval=2,
        teacher_pretrain_epochs=1,
        warmup_epochs=0,
        n_init=1,
        final_prediction_source="gate_readout",
    )
    for name in (
        "resolved_config.json",
        "metrics.json",
        "summary.json",
        "predictions.npy",
        "labels_true.npy",
        "embedding_final.npy",
        "embedding_self.npy",
        "embedding_transport.npy",
        "cluster_probabilities.npy",
        "gate_diagnostics.npz",
        "teacher_embedding.npy",
        "teacher_probabilities_clean.npy",
        "teacher_probabilities_augmented.npy",
        "teacher_probabilities_epoch0.npy",
        "teacher_probabilities_epoch_last.npy",
        "teacher_probabilities_shuffled.npy",
        "teacher_probabilities_raw.npy",
        "teacher_probabilities_raw_aligned.npy",
        "teacher_probabilities_reference.npy",
        "teacher_reference_agreement.npy",
        "teacher_reference_disagreement.npy",
        "teacher_selection.json",
    ):
        assert (output / name).exists(), name
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["labels_used_during_fit"] is False
    assert summary["K"] == 3
    assert summary["benchmark_oracle_from_y"] is False
    assert len(summary["source_files_sha256"]) == 6
    assert summary["output_files"]["teacher_probabilities_raw"] == "teacher_probabilities_raw.npy"
    raw_teacher = np.load(output / "teacher_probabilities_raw.npy")
    assert raw_teacher.shape == (X.shape[0], 3)
    raw_teacher_aligned = np.load(output / "teacher_probabilities_raw_aligned.npy")
    assert raw_teacher_aligned.shape == (X.shape[0], 3)
    assert summary["output_files"]["teacher_probabilities_reference"] == "teacher_probabilities_reference.npy"
    reference = np.load(output / "teacher_probabilities_reference.npy")
    assert reference.shape == (X.shape[0], 3)
    with np.load(output / "gate_diagnostics.npz") as diagnostics:
        assert np.array_equal(diagnostics["anchor_indices"], np.arange(X.shape[0]))
        assert diagnostics["utility_target"].shape[0] == X.shape[0]
        assert diagnostics["predicted_pi"].shape[1] == 7
        assert diagnostics["utility_features"].shape == (X.shape[0], 6, 6)
        assert diagnostics["raw_candidate_indices"].shape[0] == X.shape[0]
        assert diagnostics["latent_candidate_indices"].shape[0] == X.shape[0]
        assert diagnostics["utility_train_anchor"].shape == (X.shape[0],)
        assert diagnostics["gate_valid"].shape == (X.shape[0], 6)
        assert diagnostics["utility_independent_cluster_gain"].shape == (X.shape[0], 6)
        assert diagnostics["teacher_reference_agreement"].shape == (X.shape[0],)
        assert diagnostics["final_q_self"].shape == (X.shape[0], 3)
        assert diagnostics["final_q_edge"].shape == (X.shape[0], 6, 3)
        assert diagnostics["final_edge_embedding"].shape == (X.shape[0], 6, 4)
        assert diagnostics["teacher_reference_disagreement"].shape == (X.shape[0],)
        assert diagnostics["final_predicted_pi"].shape == (X.shape[0], 7)
        assert diagnostics["final_utility_hat"].shape == (X.shape[0], 6)
        assert diagnostics["final_gate_valid"].shape == (X.shape[0], 6)
        assert diagnostics["final_probe_self_prediction"].shape == (X.shape[0],)
        assert diagnostics["final_probe_edge_prediction"].shape == (X.shape[0], 6)
        assert diagnostics["final_embedding_self"].shape == (X.shape[0], 4)
        assert diagnostics["final_embedding_transport"].shape == (X.shape[0], 4)
        assert np.allclose(
            np.load(output / "embedding_final.npy"),
            np.load(output / "embedding_transport.npy"),
            atol=1e-6,
        )
        assert np.any(diagnostics["utility_train_anchor"])
        assert np.any(~diagnostics["utility_train_anchor"])
        assert np.isfinite(diagnostics["utility_features"]).all()
        edge_mass = diagnostics["final_predicted_pi"][:, 1:].sum(axis=1)
        if np.any(edge_mass > 1e-6):
            embedding_final = np.load(output / "embedding_final.npy")
            embedding_self = np.load(output / "embedding_self.npy")
            assert np.any(
                np.linalg.norm(embedding_final[edge_mass > 1e-6] - embedding_self[edge_mass > 1e-6], axis=1)
                > 1e-6
            )
    audit = _AUDIT_MODULE.audit_run(output)
    assert audit["certificates"]["teacher"]["status"] == "available"
    assert "raw_reference_assignment_jsd_mean" in audit["certificates"]["teacher"]["checks"]
    assert audit["certificates"]["utility"]["held_out_utility_prediction"]["status"] == "available"
    assert audit["certificates"]["utility"]["independent_view_counterfactual_gain"]["status"] == "available"
    assert audit["certificates"]["utility"]["posthoc_assignment_correction_gain"]["status"] == "available"
