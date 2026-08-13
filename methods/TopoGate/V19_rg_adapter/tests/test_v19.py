from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from methods.TopoGate.V19_rg_adapter.config import V19Config
from methods.TopoGate.V19_rg_adapter.graph import (
    build_pca_knn_graph,
    compute_edge_reliability,
)
from methods.TopoGate.V19_rg_adapter.input_adapter import (
    load_npz_matrix_only,
    prepare_input,
)
from methods.TopoGate.V19_rg_adapter.mixing import compute_node_gate, make_pseudo_batch
from methods.TopoGate.V19_rg_adapter.model import WeightedAutoEncoder
from methods.TopoGate.V19_rg_adapter.run import resolve_runtime_device, run_one
from methods.TopoGate.V19_rg_adapter.trainer import fit_predict
from scripts.V19.build_manifest import build_manifest


_PLANTNET_ROOT = os.environ.get("TOPOGATE_PLANTNET_ROOT")
PLANTNET_ROOT = None if not _PLANTNET_ROOT else Path(_PLANTNET_ROOT).expanduser()


def _original_rg_modules():
    if PLANTNET_ROOT is None or not PLANTNET_ROOT.is_dir():
        pytest.skip("original PlantNet RG source is unavailable; set TOPOGATE_PLANTNET_ROOT to enable fidelity checks")
    if str(PLANTNET_ROOT) not in sys.path:
        sys.path.insert(0, str(PLANTNET_ROOT))
    original_graph = importlib.import_module(
        "experimental_retired_models.RG_NeighborMix_scMAE.neighbor_graph"
    )
    original_mixing = importlib.import_module(
        "experimental_retired_models.RG_NeighborMix_scMAE.mixing"
    )
    return original_graph, original_mixing


def _small_config(variant: str) -> V19Config:
    return V19Config(
        variant=variant,
        hidden_size=8,
        epochs=1,
        batch_size=8,
        neighbor_k=3,
        mix_neighbors=2,
        knn_pca_dim=4,
        n_top_features=4,
        kmeans_n_init=2,
    )


def test_graph_reliability_and_gate_match_original_rg() -> None:
    original_graph, original_mixing = _original_rg_modules()
    rng = np.random.default_rng(18)
    X = rng.normal(size=(24, 9)).astype(np.float32)
    expected_graph = original_graph.build_pca_knn_graph(X, k=5, pca_dim=6, tau=0.2, seed=42)
    actual_graph = build_pca_knn_graph(X, k=5, pca_dim=6, tau=0.2, seed=42)
    np.testing.assert_array_equal(actual_graph.indices, expected_graph.indices)
    for name in ("probs", "similarity", "distance", "embedding", "snn"):
        np.testing.assert_allclose(getattr(actual_graph, name), getattr(expected_graph, name), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(actual_graph.mutual, expected_graph.mutual)

    expected_rel, expected_weights, _ = original_graph.compute_edge_reliability(
        expected_graph,
        mode="sim_mutual_snn_distance",
        gamma_sim=1.0,
        gamma_mutual=1.0,
        gamma_snn=1.0,
        gamma_distance=1.0,
    )
    actual_rel, actual_weights, _ = compute_edge_reliability(
        actual_graph,
        mode="sim_mutual_snn_distance",
        gamma_sim=1.0,
        gamma_mutual=1.0,
        gamma_snn=1.0,
        gamma_distance=1.0,
    )
    np.testing.assert_allclose(actual_rel, expected_rel, rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=0, atol=1e-7)

    expected_gate, expected_sample_weight, _ = original_mixing.compute_node_gate(
        expected_graph,
        expected_weights,
        "topology",
        0.0,
        0.15,
        1.0,
        1.0,
        2.0,
        1.0,
        uncertainty=None,
    )
    actual_gate, actual_sample_weight, _ = compute_node_gate(
        actual_graph,
        actual_weights,
        "topology",
        0.0,
        0.15,
        1.0,
        1.0,
        2.0,
        1.0,
        uncertainty=None,
    )
    np.testing.assert_allclose(actual_gate, expected_gate, rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual_sample_weight, expected_sample_weight, rtol=0, atol=1e-7)


def test_precomputed_graph_embedding_preserves_graph_result() -> None:
    rng = np.random.default_rng(27)
    X = rng.normal(size=(24, 9)).astype(np.float32)
    reference = build_pca_knn_graph(X, k=5, pca_dim=6, tau=0.2, seed=42)
    cached = build_pca_knn_graph(
        X,
        k=5,
        pca_dim=6,
        tau=0.2,
        seed=42,
        precomputed_embedding=reference.embedding,
    )
    np.testing.assert_array_equal(cached.indices, reference.indices)
    np.testing.assert_allclose(cached.probs, reference.probs, rtol=0, atol=1e-7)
    np.testing.assert_allclose(cached.snn, reference.snn, rtol=0, atol=1e-7)


def test_reliability_pseudo_batch_matches_original_rg() -> None:
    torch = pytest.importorskip("torch")
    original_graph, original_mixing = _original_rg_modules()
    data = np.random.default_rng(3).normal(size=(18, 7)).astype(np.float32)
    expected_graph = original_graph.build_pca_knn_graph(data, 4, 5, 0.2, 7)
    actual_graph = build_pca_knn_graph(data, 4, 5, 0.2, 7)
    _, expected_weights, _ = original_graph.compute_edge_reliability(
        expected_graph, "sim_mutual_snn_distance", 1.0, 1.0, 1.0, 1.0
    )
    _, actual_weights, _ = compute_edge_reliability(
        actual_graph, "sim_mutual_snn_distance", 1.0, 1.0, 1.0, 1.0
    )
    expected_gate, _, _ = original_mixing.compute_node_gate(
        expected_graph, expected_weights, "topology", 0.0, 0.15, 1.0, 1.0, 2.0, 1.0
    )
    actual_gate, _, _ = compute_node_gate(
        actual_graph, actual_weights, "topology", 0.0, 0.15, 1.0, 1.0, 2.0, 1.0
    )
    batch_indices = np.array([0, 2, 5, 9], dtype=np.int64)
    batch = torch.as_tensor(data[batch_indices])
    expected, expected_weight, _ = original_mixing.make_pseudo_batch(
        data,
        batch_indices,
        batch,
        "reliability",
        expected_graph,
        expected_weights,
        expected_gate,
        3,
        np.random.default_rng(99),
    )
    actual, actual_weight, _ = make_pseudo_batch(
        data,
        batch_indices,
        batch,
        actual_graph,
        actual_weights,
        actual_gate,
        3,
        np.random.default_rng(99),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(actual_weight, expected_weight, rtol=0, atol=1e-7)


def test_weighted_scmae_loss_matches_original_rg() -> None:
    torch = pytest.importorskip("torch")
    _original_graph, _original_mixing = _original_rg_modules()
    original_model_module = importlib.import_module(
        "experimental_retired_models.RG_NeighborMix_scMAE.model"
    )
    torch.random.default_generator.manual_seed(33)
    expected_model = original_model_module.AutoEncoder(
        num_genes=7,
        hidden_size=8,
        dropout=0.0,
        masked_data_weight=0.75,
        mask_loss_weight=0.7,
    )
    actual_model = WeightedAutoEncoder(
        num_genes=7,
        hidden_size=8,
        dropout=0.0,
        masked_data_weight=0.75,
        mask_loss_weight=0.7,
    )
    actual_model.load_state_dict(expected_model.state_dict())
    generator = torch.Generator().manual_seed(9)
    target = torch.randn((5, 7), generator=generator)
    corrupted = target.clone()
    corrupted[:, ::2] = torch.randn((5, 4), generator=generator)
    mask = (corrupted != target).float()
    sample_weight = torch.linspace(0.2, 1.0, 5)
    expected_latent, expected_loss, expected_parts = expected_model.loss_mask_weighted(
        corrupted, target, mask, sample_weight=sample_weight
    )
    actual_latent, actual_loss, actual_parts = actual_model.loss_mask_weighted(
        corrupted, target, mask, sample_weight=sample_weight
    )
    torch.testing.assert_close(actual_latent, expected_latent, rtol=0, atol=1e-7)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=1e-7)
    for key in ("reconstruction_loss", "mask_loss", "total_loss", "mask_positive_rate"):
        torch.testing.assert_close(actual_parts[key], expected_parts[key], rtol=0, atol=1e-7)


def test_preprocessing_protocols_are_distinct_and_label_free() -> None:
    counts = np.array(
        [[10, 0, 3, 0, 1], [0, 9, 0, 2, 1], [8, 0, 4, 0, 2], [0, 7, 0, 3, 2]],
        dtype=np.float32,
    )
    native = prepare_input(
        counts,
        dataset_name="Baron Human",
        input_protocol="rg_native",
        n_top_features=3,
    )
    bridge = prepare_input(
        counts,
        dataset_name="Baron Human",
        input_protocol="clubench_bridge",
        n_top_features=3,
    )
    text = prepare_input(
        counts / np.maximum(counts.max(axis=1, keepdims=True), 1.0),
        dataset_name="cnae9",
        input_protocol="shared_text",
        n_top_features=3,
    )
    assert native.X.shape == (4, 3)
    assert bridge.X.shape == (4, 5)
    assert text.X.shape == (4, 5)
    assert native.profile["normalization"].startswith("normalize_total")
    assert bridge.profile["hvg"]["strategy"] == "disabled"
    assert text.profile["input_kind"] == "sparse_text_features"
    assert all(profile.profile["labels_used"] is False for profile in (native, bridge, text))


def test_core_fit_signature_excludes_labels() -> None:
    signature = inspect.signature(fit_predict)
    assert "y" not in signature.parameters
    assert "labels" not in signature.parameters
    assert "n_clusters" in signature.parameters


def test_matrix_only_loader_does_not_return_benchmark_labels(tmp_path: Path) -> None:
    data_path = tmp_path / "matrix_only.npz"
    np.savez(data_path, x=np.ones((4, 3), dtype=np.float32), y=np.array([0, 1, 0, 1]))
    loaded = load_npz_matrix_only(data_path)
    assert loaded.labels is None
    assert loaded.profile["labels_accessed"] is False
    np.testing.assert_array_equal(loaded.X, np.ones((4, 3), dtype=np.float32))


def test_unsupervised_fit_skips_kmeans_and_writes_only_x_diagnostics() -> None:
    X = np.maximum(np.random.default_rng(23).normal(size=(24, 7)), 0.0).astype(np.float32)
    config = V19Config(
        variant="rg_full",
        hidden_size=8,
        epochs=1,
        batch_size=8,
        neighbor_k=3,
        mix_neighbors=2,
        knn_pca_dim=4,
        n_top_features=7,
        kmeans_n_init=2,
    )
    predictions, embedding, diagnostics = fit_predict(
        X,
        n_clusters=None,
        config=config,
        seed=42,
        device="cpu",
        evaluate_unsupervised=True,
    )
    assert predictions is None
    assert embedding.shape == (24, 8)
    assert set(diagnostics["unsupervised_diagnostics"]) == {
        "eval_mask_loss",
        "latent_view_cosine_mean",
        "latent_view_cosine_std",
        "input_neighbor_overlap",
        "latent_mean_feature_std",
    }
    assert diagnostics["core_summary"]["readout_enabled"] is False
    assert diagnostics["core_summary"]["n_clusters"] is None


def test_scmae_only_never_calls_graph_builder(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("scmae_only attempted to build a graph")

    monkeypatch.setattr(
        "methods.TopoGate.V19_rg_adapter.trainer.build_pca_knn_graph", forbidden
    )
    X = np.random.default_rng(12).normal(size=(18, 6)).astype(np.float32)
    predictions, embedding, diagnostics = fit_predict(
        X,
        n_clusters=3,
        config=_small_config("scmae_only"),
        seed=42,
        device="cpu",
    )
    assert predictions.shape == (18,)
    assert embedding.shape == (18, 8)
    assert diagnostics["neighbor_indices"].shape == (18, 0)
    assert diagnostics["core_summary"]["graph_enabled"] is False
    assert diagnostics["core_summary"]["pseudo_enabled"] is False


def test_runner_writes_auditable_output_contract(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    X = np.maximum(rng.normal(size=(20, 7)), 0.0).astype(np.float32)
    labels = np.repeat(np.arange(2), 10)
    data_path = tmp_path / "cnae9.npz"
    np.savez_compressed(data_path, x=X, y=labels)
    output = tmp_path / "run"
    summary = run_one(
        data_path,
        output,
        config=_small_config("scmae_only"),
        input_protocol="shared_text",
        seed=42,
        device="cpu",
        dataset_name="cnae9",
        dataset_id="cnae9__shared_text",
    )
    required = {
        "resolved_config.json",
        "dataset_profile.json",
        "preprocess_profile.json",
        "predictions.npy",
        "labels_true.npy",
        "embedding_final.npy",
        "metrics.json",
        "summary.json",
        "training_history.json",
        "neighbor_indices.npy",
        "edge_reliability.npy",
        "node_gate.npy",
        "status.json",
        "run_record.json",
        "launcher.log",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert summary["K_source"] == "benchmark_oracle_from_y"
    assert summary["labels_used_during_fit"] is False
    assert summary["labels_used_during_preprocessing"] is False
    assert np.load(output / "neighbor_indices.npy").shape == (20, 0)
    assert json.loads((output / "status.json").read_text())["status"] == "completed"


def test_manifest_has_fixed_eleven_strata_and_66_runs(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path / "manifest.json")
    assert len(manifest["datasets"]) == 11
    assert manifest["expected_runs_total"] == 66
    assert manifest["formal_seeds_in_order"] == [42, 123, 7]
    assert sum(row["input_protocol"] == "rg_native" for row in manifest["datasets"]) == 3
    assert sum(row["input_protocol"] == "clubench_bridge" for row in manifest["datasets"]) == 3
    assert sum(row["input_protocol"] == "shared_text" for row in manifest["datasets"]) == 5
    assert all(row["source_hash"] == "unavailable" for row in manifest["datasets"])
    assert all(
        row["comparison_scope"] == "archived_sota_bridge_eligible"
        for row in manifest["datasets"]
        if row["input_protocol"] in {"clubench_bridge", "shared_text"}
    )
    assert all(
        row["comparison_scope"] == "internal_rg_native_only"
        for row in manifest["datasets"]
        if row["input_protocol"] == "rg_native"
    )


def test_forbidden_gpu_validation(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="forbidden"):
        resolve_runtime_device("cuda", 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(ValueError, match="forbidden"):
        resolve_runtime_device("cuda", 7)
