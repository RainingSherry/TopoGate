from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse

from methods.TopoGate.V0.config import V0Config, load_config, normalize_parameterization
from methods.TopoGate.V0.corruption import compute_node_gate, make_pseudo_batch
from methods.TopoGate.V0.diagnostics import evaluate_unsupervised_views
from methods.TopoGate.V0.graph import build_pca_knn_graph, compute_edge_reliability
from methods.TopoGate.V0.model import WeightedAutoEncoder
from methods.TopoGate.V0.run import _load_npz, _prepare_array, resolve_runtime_device, run_one
from methods.TopoGate.V0.trainer import fit_predict, resolve_device


def _small_config(parameterization: str) -> V0Config:
    return V0Config(
        parameterization=parameterization,
        hidden_size=4,
        epochs=1,
        batch_size=6,
        neighbor_k=3,
        mix_neighbors=2,
        knn_pca_dim=4,
        n_top_features=7,
        kmeans_n_init=2,
    )


def _data() -> np.ndarray:
    return np.maximum(
        np.random.default_rng(17).normal(size=(16, 7)), 0.0
    ).astype(np.float32)


def test_parameterization_aliases_and_yaml_configs() -> None:
    assert normalize_parameterization("F") == "fixed"
    assert normalize_parameterization("-t") == "topology"
    fixed = load_config(Path(__file__).parents[1] / "configs/topogate_v0_fixed.yaml")
    topology = load_config(Path(__file__).parents[1] / "configs/topogate_v0_topology.yaml")
    assert fixed.parameterization == "fixed"
    assert topology.parameterization == "topology"
    assert fixed.resolved_dict()["effective_edge_reliability_mode"] == "base_probability"
    assert topology.resolved_dict()["effective_edge_reliability_mode"] == "sim_mutual_snn_distance"


def test_graph_probabilities_and_reliability_are_row_normalized() -> None:
    graph = build_pca_knn_graph(_data(), k=4, pca_dim=5, tau=0.2, seed=42)
    assert graph.indices.shape == (16, 4)
    np.testing.assert_allclose(graph.probs.sum(axis=1), np.ones(16), rtol=0, atol=2e-6)
    reliability, weights, summary = compute_edge_reliability(
        graph, "sim_mutual_snn_distance", 1.0, 1.0, 1.0, 1.0
    )
    assert reliability.shape == weights.shape == graph.probs.shape
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(16), rtol=0, atol=2e-6)
    assert summary["effective_neighbor_count"] > 0.0


def test_fixed_and_topology_corruption_keep_expected_convex_combination() -> None:
    data = _data()[:8]
    graph = build_pca_knn_graph(data, k=3, pca_dim=4, tau=0.2, seed=3)
    _, topology_weights, _ = compute_edge_reliability(
        graph, "sim_mutual_snn_distance", 1.0, 1.0, 1.0, 1.0
    )
    fixed_gate, fixed_sample_weight, _ = compute_node_gate(
        graph, parameterization="fixed", alpha=0.9
    )
    topology_gate, topology_sample_weight, topology_summary = compute_node_gate(
        graph, parameterization="topology", gate_min=0.0, gate_max=0.15
    )
    np.testing.assert_allclose(fixed_gate, np.full(8, 0.1, dtype=np.float32))
    np.testing.assert_allclose(fixed_sample_weight, np.ones(8, dtype=np.float32))
    assert np.all((topology_gate >= 0.0) & (topology_gate <= 0.15))
    assert np.all((topology_sample_weight >= 0.0) & (topology_sample_weight <= 1.0))
    np.testing.assert_allclose(
        topology_sample_weight,
        topology_gate / max(float(topology_gate.max()), 1e-8),
        atol=1e-7,
    )
    assert topology_summary["sample_weight_mode"] == "gate_over_empirical_max"

    batch_indices = np.array([0, 2, 5], dtype=np.int64)
    batch = torch.as_tensor(data[batch_indices])
    fixed_view, fixed_weights, _ = make_pseudo_batch(
        data,
        batch_indices,
        batch,
        parameterization="fixed",
        graph=graph,
        edge_weights=graph.probs,
        node_gate=fixed_gate,
        mix_neighbors=3,
        alpha=0.9,
        rng=np.random.default_rng(9),
        neighbor_estimator="full",
    )
    topology_view, topology_weights_out, _ = make_pseudo_batch(
        data,
        batch_indices,
        batch,
        parameterization="topology",
        graph=graph,
        edge_weights=topology_weights,
        node_gate=topology_gate,
        mix_neighbors=3,
        alpha=0.9,
        rng=np.random.default_rng(9),
        neighbor_estimator="full",
    )
    np.testing.assert_allclose(fixed_weights.numpy(), np.ones(3), atol=1e-7)
    assert not torch.equal(fixed_view, batch)
    assert not torch.equal(topology_view, batch)
    np.testing.assert_allclose(
        topology_weights_out.numpy(), topology_gate[batch_indices] / topology_gate.max(), atol=1e-6
    )


def test_weighted_loss_without_sample_weights_matches_base_loss() -> None:
    torch.manual_seed(33)
    base = WeightedAutoEncoder(num_genes=7, hidden_size=4)
    target = torch.randn(5, 7)
    corrupted = target.clone()
    corrupted[:, ::2] = torch.randn(5, 4)
    mask = (corrupted != target).float()
    _, expected = base.loss_mask(corrupted, target, mask)
    _, actual, _parts = base.loss_mask_weighted(corrupted, target, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("parameterization", ["fixed", "topology"])
def test_fit_predict_is_label_free_and_supports_both_parameterizations(parameterization: str) -> None:
    assert "y" not in inspect.signature(fit_predict).parameters
    predictions, embedding, diagnostics = fit_predict(
        _data(),
        n_clusters=3,
        config=_small_config(parameterization),
        seed=42,
        device="cpu",
    )
    assert predictions is not None and predictions.shape == (16,)
    assert embedding.shape == (16, 4)
    assert diagnostics["core_summary"]["labels_used_during_fit"] is False
    assert diagnostics["core_summary"]["parameterization"] == parameterization
    assert diagnostics["neighbor_indices"].shape == (16, 3)


def test_fit_predict_without_k_only_returns_representation() -> None:
    predictions, embedding, diagnostics = fit_predict(
        _data(),
        n_clusters=None,
        config=_small_config("fixed"),
        seed=7,
        device="cpu",
    )
    assert predictions is None
    assert embedding.shape == (16, 4)
    assert diagnostics["core_summary"]["readout_enabled"] is False


@pytest.mark.parametrize("parameterization", ["fixed", "topology"])
def test_fit_predict_is_bitwise_reproducible_for_each_parameterization(parameterization: str) -> None:
    config = _small_config(parameterization)
    first = fit_predict(_data(), n_clusters=3, config=config, seed=123, device="cpu")
    second = fit_predict(_data(), n_clusters=3, config=config, seed=123, device="cpu")
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_fit_predict_supports_graph_disabled_label_free_path() -> None:
    config = V0Config(
        parameterization="topology",
        hidden_size=4,
        epochs=1,
        batch_size=6,
        use_pseudo=False,
        n_top_features=7,
        kmeans_n_init=2,
    )
    predictions, embedding, diagnostics = fit_predict(
        _data(), n_clusters=3, config=config, seed=42, device="cpu"
    )
    assert predictions is not None and embedding.shape == (16, 4)
    assert diagnostics["core_summary"]["graph_enabled"] is False
    assert diagnostics["core_summary"]["pseudo_enabled"] is False
    assert diagnostics["neighbor_indices"].shape == (16, 0)


def test_prepare_array_records_label_free_modes_and_feature_selection() -> None:
    raw_counts = np.array(
        [[0, 1, 4, 2], [3, 0, 1, 5], [2, 2, 0, 1]], dtype=np.float32
    )
    raw_config = V0Config(
        input_mode="auto", n_top_features=2, scale_input=False, target_sum=10.0
    )
    prepared, profile = _prepare_array(raw_counts, config=raw_config, dataset_name="toy")
    assert prepared.shape == (3, 2)
    assert profile["input_mode_used"] == "raw"
    assert profile["feature_selection_strategy"] == "variance_top_features"
    continuous = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    _, continuous_profile = _prepare_array(
        continuous,
        config=V0Config(input_mode="auto", n_top_features=0, scale_input=False),
        dataset_name="continuous",
    )
    assert continuous_profile["input_mode_used"] == "log1p"
    with pytest.raises(ValueError, match="integer-like"):
        _prepare_array(
            continuous,
            config=V0Config(input_mode="raw", n_top_features=0, scale_input=False),
            dataset_name="continuous",
        )


def test_unsupervised_eval_loss_uses_both_views_once() -> None:
    class ConstantLossModel(torch.nn.Module):
        def loss_mask_weighted(self, x, target, mask):
            latent = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
            return latent, torch.tensor(2.0, dtype=x.dtype, device=x.device), {}

    data = _data()[:4, :3]
    graph = build_pca_knn_graph(data, k=0, pca_dim=3, tau=0.2, seed=1)
    result = evaluate_unsupervised_views(
        model=ConstantLossModel(),
        data_np=data,
        clean_embedding=np.zeros((4, 2), dtype=np.float32),
        graph=graph,
        batch_size=4,
        mask_ratio=0.4,
        seed=3,
        device=torch.device("cpu"),
    )
    assert result["eval_mask_loss"] == pytest.approx(2.0)


def test_trainer_device_respects_cpu_and_forbidden_physical_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="forbidden"):
        resolve_device("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(ValueError, match="forbidden"):
        resolve_device("cuda")


def test_sparse_npz_and_unlabelled_explicit_k_runner_contract(tmp_path: Path) -> None:
    dense = np.array(
        [[1, 0, 2, 0], [0, 3, 0, 1], [2, 0, 1, 0], [0, 1, 0, 4]],
        dtype=np.float32,
    )
    matrix = sparse.csr_matrix(dense)
    source = tmp_path / "sparse.npz"
    np.savez_compressed(
        source,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
    )
    loaded, labels, profile = _load_npz(source)
    assert labels is None
    assert profile["sparse_storage"] is True
    np.testing.assert_array_equal(loaded, dense)
    summary = run_one(
        source,
        tmp_path / "run",
        config=_small_config("fixed"),
        seed=42,
        device="cpu",
        n_clusters=2,
    )
    assert summary["K_source"] == "explicit_n_clusters"
    assert not (tmp_path / "run" / "labels_true.npy").exists()
    assert not (tmp_path / "run" / "label_mapping.json").exists()


def test_runner_writes_clear_output_contract_without_label_leakage(tmp_path: Path) -> None:
    X = _data()
    y = np.repeat(np.arange(2), 8)
    source = tmp_path / "toy.npz"
    np.savez_compressed(source, X=X, y=y)
    output = tmp_path / "run"
    summary = run_one(
        source,
        output,
        config=_small_config("topology"),
        seed=42,
        device="cpu",
        n_clusters=None,
    )
    required = {
        "resolved_config.json",
        "predictions.npy",
        "labels_true.npy",
        "predictions_mapped.npy",
        "label_mapping.json",
        "embedding_final.npy",
        "metrics.json",
        "summary.json",
        "status.json",
        "run_record.json",
        "node_gate.npy",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert summary["K_source"] == "benchmark_oracle_from_y"
    assert summary["labels_used_during_fit"] is False
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    assert json.loads((output / "label_mapping.json").read_text()) == {"0": "0", "1": "1"}


def test_forbidden_gpu_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="forbidden"):
        resolve_runtime_device("cuda", 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(ValueError, match="forbidden"):
        resolve_runtime_device("cuda", 7)
