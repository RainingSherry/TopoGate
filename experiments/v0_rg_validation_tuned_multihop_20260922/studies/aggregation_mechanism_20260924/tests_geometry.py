"""Focused invariance, topology, and loader tests for the posthoc audit."""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from geometry_analysis import geometry_metrics, pair_merge_errors, run, sample_indices


def test_normalized_distances_are_invariant_to_uniform_scale():
    z = np.array([[0., 0.], [.1, .2], [1.5, 0.], [1.7, .1], [0., 2.], [.2, 2.1]])
    y = np.array([0, 0, 1, 1, 2, 2])
    _, classes_a, pairs_a = geometry_metrics(z, y, ["a", "b", "c"], k=1)
    _, classes_b, pairs_b = geometry_metrics(z * 17.0, y, ["a", "b", "c"], k=1)
    for a, b in zip(classes_a, classes_b):
        np.testing.assert_allclose(a["compactness_over_global_spread"], b["compactness_over_global_spread"])
        assert a["knn_cross_class_fraction"] == b["knn_cross_class_fraction"]
        assert a["same_class_components"] == b["same_class_components"]
    for a, b in zip(pairs_a, pairs_b):
        for field in ("centroid_distance_over_global_spread", "pooled_scatter_over_global_spread",
                      "centroid_distance_over_pooled_scatter"):
            np.testing.assert_allclose(a[field], b[field])


def test_global_knn_induced_graph_detects_fragmented_class():
    z = np.array([[0.], [.05], [10.], [10.05], [5.], [5.05]])
    y = np.array([0, 0, 0, 0, 1, 1])
    _, classes, _ = geometry_metrics(z, y, ["fragmented", "intact"], k=1)
    assert classes[0]["same_class_components"] == 2
    assert classes[0]["largest_component_fraction"] == .5
    assert classes[1]["same_class_components"] == 1
    assert classes[1]["largest_component_fraction"] == 1.0


def test_full_x_merge_overlay_and_fixed_label_free_sample():
    y = np.array([0, 0, 1, 1])
    pred = np.array([4, 5, 4, 5])
    pair = pair_merge_errors(y, pred, ["a", "b"])[0]
    assert pair["full_x_merge_pairs"] == 2
    assert pair["full_x_possible_cross_class_pairs"] == 4
    assert pair["full_x_merge_rate"] == .5
    np.testing.assert_array_equal(sample_indices(100, 10), sample_indices(100, 10))


def test_real_file_pathway_writes_complete_tables():
    prior_script = Path(__file__).resolve().parent.parent / "multihop_mechanism_20260924" / "step1_outcomes.py"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prior_root = root / "prior"
        inputs = prior_root / "inputs"
        inputs.mkdir(parents=True)
        shutil.copy2(prior_script, prior_root / "step1_outcomes.py")
        y = np.repeat(np.arange(3), 8)
        source = root / "source.npz"
        np.savez_compressed(source, y=y)
        row = {"dataset": "Tosches", "n_samples": len(y), "h1_config_id": "H1",
               "selected_config_id": "H8_B2", "feature_sha256": "feature", "validation_labels_sha256": "validation"}
        for arm, config in (("h1", "H1"), ("selected", "H8_B2")):
            output = root / arm
            output.mkdir()
            record = output / "record_seed_42.json"
            record.write_text(json.dumps({"status": "completed", "seed": 42,
                                          "selection_label_scope": "validation_only", "test_labels_loaded": False,
                                          "feature_sha256": "feature", "validation_labels_sha256": "validation",
                                          "labels_used_during_fit": False,
                                          "K_provenance": "benchmark_oracle_from_y",
                                          "config": {"hop_variant": config}}), encoding="utf-8")
            np.savez_compressed(output / "predictions_seed_42.npz", predictions=np.roll(y, 1))
            z = np.column_stack((y + np.linspace(0, .2, len(y)), np.sin(np.arange(len(y)))))
            np.savez_compressed(output / "embedding_seed_42.npz", embedding=z * (2 if arm == "selected" else 1))
            row["h1_records" if arm == "h1" else "selected_records"] = {"42": str(record)}
        (inputs / "selection_lock.json").write_text(json.dumps({"status": "LOCKED_BEFORE_TEST",
                                                               "selection_scope": "validation_only", "rows": [row]}), encoding="utf-8")
        (inputs / "validation_best_multihop_vs_h1_full_x_60.json").write_text(json.dumps({
            "candidate_selection_scope": "validation_only", "evaluation_scope": "all_rows_of_original_X",
            "rows": [{"dataset": "Tosches", "source_path": str(source), "multihop_config_id": "H8_B2",
                      "h1_config_id": "H1"}]}), encoding="utf-8")
        out = root / "geometry"
        summary = run(prior_root, out, "Tosches", 42, smoke=True)
        assert summary["status"] == "completed"
        assert summary["posthoc_full_x_labels_loaded"] is True
        assert summary["labels_used_for_model_fit"] is False
        assert summary["n_sampled"] == len(y)
        assert sum(1 for _ in (out / "per_class.csv").open(encoding="utf-8")) == 1 + 2 * 3
        assert sum(1 for _ in (out / "per_class_pair.csv").open(encoding="utf-8")) == 1 + 2 * 3
        assert (out / "geometry.pdf").stat().st_size > 0
