from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from methods.TopoGate.learnable_gate.run_npz import run_topogate
from scripts.v9_regime.run_matrix import _sample_data


def test_runner_semantic_outputs_and_label_invariance(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(36, 8)).astype(np.float64)
    y = np.repeat(np.arange(3), 12)
    y_permuted = y[::-1].copy()
    common = {
        "epochs": 2,
        "batch_size": 36,
        "hidden_size": 16,
        "neighbor_k": 5,
        "mix_neighbors": 4,
        "mask_ratio": 0.3,
        "no_cuda": True,
        "legacy_labels_output": False,
        "config_dir": str(Path("methods/TopoGate/learnable_gate/configs").resolve()),
        "dataset_name": "test_protocol",
    }
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    pred_a, _, _ = run_topogate(
        x, n_clusters=3, y=y, seed=42, gpu=1, return_metrics=True,
        save_dir=str(out_a), variant="learnable_gate_v9_adaptive", **common,
    )
    pred_b, _, _ = run_topogate(
        x, n_clusters=3, y=y_permuted, seed=42, gpu=1, return_metrics=True,
        save_dir=str(out_b), variant="learnable_gate_v9_adaptive", **common,
    )
    np.testing.assert_array_equal(pred_a, pred_b)
    np.testing.assert_array_equal(np.load(out_a / "embedding_final.npy"), np.load(out_b / "embedding_final.npy"))
    assert not (out_a / "labels.npy").exists()
    assert (out_a / "predictions.npy").exists()
    summary = json.loads((out_a / "summary.json").read_text(encoding="utf-8"))
    assert summary["labels_used_during_fit"] is False
    assert summary["legacy_labels_output"] is False


def test_matrix_sampling_uses_protocol_standardization(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    x = (rng.normal(size=(120, 5)) * np.array([2.0, 1.0, 0.5, 3.0, 4.0])).astype(np.float32)
    y = np.repeat(np.arange(3), 40)
    source = tmp_path / "toy.npz"
    np.savez(source, x=x, y=y)
    record = {
        "source_path": str(source),
        "n": 120,
        "n_clusters": 3,
    }
    standardized, returned_y, meta = _sample_data(record, seed=42, max_samples=0)
    np.testing.assert_array_equal(returned_y, y)
    np.testing.assert_allclose(np.mean(standardized, axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(standardized, axis=0), 1.0, atol=1e-5)
    assert meta["preprocessing"]["input_preprocessing"] == "nan_to_num_then_column_standard_scaler"
    assert meta["labels_used_during_fit"] is False

    sampled, sampled_y, sampled_meta = _sample_data(record, seed=42, max_samples=30)
    indices = np.sort(np.random.default_rng(42).choice(120, size=30, replace=False))
    expected = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    np.testing.assert_allclose(sampled, expected[indices], atol=1e-5)
    np.testing.assert_array_equal(sampled_y, y[indices])
    assert sampled_meta["row_sampling"] is True
