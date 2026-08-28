from __future__ import annotations

import subprocess
import sys
import json

import numpy as np


def test_legacy_topogate_import_is_lazy() -> None:
    code = """
import sys
import methods.TopoGate
assert 'scanpy' not in sys.modules
from methods.TopoGate.learnable_gate.model import AutoEncoder
assert AutoEncoder is not None
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_v10_runner_export_is_lazy_and_callable() -> None:
    import methods.TopoGate.v10_reliable_graph as v10

    assert callable(v10.run_v10)


def test_run_v10_smoke_saves_predictions_separately_from_true_labels(tmp_path) -> None:
    data_path = "datasets/iris.npz"
    data = np.load(data_path)
    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"])
    expected_k = int(np.unique(y).size)

    from methods.TopoGate.v10_reliable_graph.run import run_v10

    predictions, elapsed, metrics = run_v10(
        x,
        y=y,
        gpu=1,
        seed=42,
        save_dir=tmp_path,
        return_metrics=True,
        no_cuda=True,
        epochs=2,
        warmup_epochs=0,
        ramp_epochs=1,
        batch_size=150,
        eval_batch_size=256,
        neighbor_k=5,
        edge_batch_size=32,
        latent_dim=8,
        hidden_dim=32,
        decoder_rank=8,
        gate_hidden_dim=8,
        refresh_interval=1,
    )

    assert elapsed >= 0.0
    assert predictions.shape == (x.shape[0],)
    assert predictions.dtype == np.int64
    assert metrics["ari"] == metrics["ari"]
    predictions_file = np.load(tmp_path / "predictions.npy")
    labels_true_file = np.load(tmp_path / "labels_true.npy")
    assert predictions_file.shape == (x.shape[0],)
    assert labels_true_file.shape == (x.shape[0],)
    assert np.array_equal(labels_true_file, np.unique(y, return_inverse=True)[1])
    assert not np.shares_memory(predictions_file, labels_true_file)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_clusters"] == expected_k
    assert summary["k_source"] == "labels_unique"
    assert summary["prototype_initialization_epoch"] == 1
    assert summary["prototype_initialization_method"].startswith("kmeans_on_normalized_ema")
    assert summary["output_contract"]["predictions"] == "predictions.npy"
    assert summary["output_contract"]["labels_true_encoded"] == "labels_true.npy"
    assert (tmp_path / "label_mapping.json").exists()
    final_edges = np.load(tmp_path / "final_graph_edges.npz")
    assert set(final_edges.files) == {
        "source",
        "target",
        "gate",
        "input_latent_stability",
        "temporal_target",
    }
    assert final_edges["source"].shape == final_edges["gate"].shape


def test_feature_only_does_not_save_untrained_prototype_diagnostics(tmp_path) -> None:
    data = np.load("datasets/iris.npz")
    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"])
    from methods.TopoGate.v10_reliable_graph.run import run_v10

    run_v10(
        x,
        y=y,
        gpu=1,
        seed=42,
        save_dir=tmp_path,
        no_cuda=True,
        epochs=1,
        batch_size=150,
        latent_dim=8,
        hidden_dim=32,
        decoder_rank=8,
        graph_enabled=False,
        dynamic_graph=False,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["prototype_initialization_epoch"] is None
    assert summary["prototype_metrics"] is None
    assert summary["output_contract"]["prototype_predictions"] is None
    assert not (tmp_path / "prototype_predictions.npy").exists()
    assert not (tmp_path / "cluster_probabilities.npy").exists()
