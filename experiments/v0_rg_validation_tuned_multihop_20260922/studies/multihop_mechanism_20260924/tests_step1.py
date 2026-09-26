"""Small exact checks for the contingency-based pair accounting."""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_outcomes import analyze_item, evaluate, pair_decomposition  # noqa: E402


def brute_force(y: np.ndarray, h1: np.ndarray, h8: np.ndarray) -> tuple[dict, dict]:
    pair = {"h1": {"split": 0, "merge": 0}, "h8": {"split": 0, "merge": 0}}
    transition = {key: 0 for key in (
        "corrected_split", "worsened_split", "stable_split",
        "corrected_merge", "worsened_merge", "stable_merge",
        "stable_correct_same_class", "stable_correct_different_class")}
    per_sample = {key: np.zeros(len(y), dtype=np.int64) for key in (
        "h1_split", "h1_merge", "h8_split", "h8_merge",
        "corrected_split", "worsened_split", "stable_split",
        "corrected_merge", "worsened_merge", "stable_merge")}
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            truth_same = y[i] == y[j]
            same1 = h1[i] == h1[j]
            same8 = h8[i] == h8[j]
            if truth_same:
                if not same1:
                    pair["h1"]["split"] += 1
                    per_sample["h1_split"][[i, j]] += 1
                if not same8:
                    pair["h8"]["split"] += 1
                    per_sample["h8_split"][[i, j]] += 1
                if not same1 and same8:
                    key = "corrected_split"
                elif same1 and not same8:
                    key = "worsened_split"
                elif not same1 and not same8:
                    key = "stable_split"
                else:
                    key = "stable_correct_same_class"
            else:
                if same1:
                    pair["h1"]["merge"] += 1
                    per_sample["h1_merge"][[i, j]] += 1
                if same8:
                    pair["h8"]["merge"] += 1
                    per_sample["h8_merge"][[i, j]] += 1
                if same1 and not same8:
                    key = "corrected_merge"
                elif not same1 and same8:
                    key = "worsened_merge"
                elif same1 and same8:
                    key = "stable_merge"
                else:
                    key = "stable_correct_different_class"
            transition[key] += 1
            if key in per_sample:
                per_sample[key][[i, j]] += 1
    for arm in ("h1", "h8"):
        pair[arm]["total_error"] = pair[arm]["split"] + pair[arm]["merge"]
    transition["net_error_reduction"] = (
        transition["corrected_split"] + transition["corrected_merge"]
        - transition["worsened_split"] - transition["worsened_merge"])
    per_sample["net_error_reduction"] = (
        per_sample["corrected_split"] + per_sample["corrected_merge"]
        - per_sample["worsened_split"] - per_sample["worsened_merge"])
    pair["transition"] = transition
    pair["n_samples"] = len(y)
    pair["total_pairs"] = len(y) * (len(y) - 1) // 2
    return pair, per_sample


def test_pair_contingency_matches_enumeration() -> None:
    rng = np.random.default_rng(20260924)
    for n in range(1, 35):
        for _ in range(8):
            y = rng.integers(0, 1 + n // 4, size=n)
            h1 = rng.integers(0, 2 + n // 5, size=n)
            h8 = rng.integers(0, 2 + n // 6, size=n)
            expected_pair, expected_per_sample = brute_force(y, h1, h8)
            actual_pair, actual_per_sample = pair_decomposition(y, h1, h8)
            assert actual_pair == expected_pair
            assert actual_per_sample.keys() == expected_per_sample.keys()
            for key in actual_per_sample:
                np.testing.assert_array_equal(actual_per_sample[key], expected_per_sample[key])


def test_hungarian_per_class_metrics() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    prediction = np.array([9, 9, 9, 4, 4, 9])
    metrics, classes, mapped = evaluate(y, prediction, ["A", "B"])
    assert metrics["acc"] == 5 / 6
    assert classes[0]["recall"] == 1
    assert classes[0]["precision"] == 3 / 4
    assert classes[1]["recall"] == 2 / 3
    assert classes[1]["precision"] == 1
    np.testing.assert_array_equal(mapped, [0, 0, 0, 1, 1, 0])


def test_fixed_saved_artifact_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        y = np.asarray(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
        encoded = np.repeat(np.arange(3), 3)
        source = root / "source.npz"
        np.savez_compressed(source, x=np.eye(9), y=y)
        arm_paths = {}
        predictions = {"h1": np.asarray([0, 0, 1, 1, 1, 1, 2, 2, 2]),
                       "h8": np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])}
        for arm, config_id in (("h1", "H1"), ("h8", "H8_B1_q1_a1p5")):
            folder = root / arm
            folder.mkdir()
            record = folder / "record_seed_42.json"
            payload = {"status": "completed", "seed": 42, "selection_label_scope": "validation_only",
                       "test_labels_loaded": False, "labels_used_during_fit": False,
                       "feature_sha256": "feature", "validation_labels_sha256": "validation",
                       "K_provenance": "benchmark_oracle_from_y", "n_clusters": 3,
                       "config": {"hop_variant": "H1" if arm == "h1" else "H8_B1",
                                  "high_hop_probability_multiplier": 1.0,
                                  "mix_amplitude_multiplier": 1.0 if arm == "h1" else 1.5}}
            record.write_text(json.dumps(payload), encoding="utf-8")
            np.savez_compressed(folder / "predictions_seed_42.npz", predictions=predictions[arm])
            embedding = np.column_stack((np.arange(9, dtype=float),
                                         np.arange(9, dtype=float) ** 2 / (10 if arm == "h1" else 8)))
            np.savez_compressed(folder / "embedding_seed_42.npz", embedding=embedding)
            arm_paths["h1_records" if arm == "h1" else "selected_records"] = {"42": str(record)}
        row = {"dataset": "synthetic", "h1_config_id": "H1", "selected_config_id": "H8_B1_q1_a1p5",
               "feature_sha256": "feature", "validation_labels_sha256": "validation", "n_samples": 9,
               **arm_paths}
        bg = {"dataset": "synthetic", "source_path": str(source), "n_samples": 9,
              "h1_config_id": "H1", "multihop_config_id": "H8_B1_q1_a1p5",
              "per_seed": {"42": {"h1": evaluate(encoded, predictions["h1"], ["A", "B", "C"])[0],
                                  "multihop": evaluate(encoded, predictions["h8"], ["A", "B", "C"])[0]}}}
        result = analyze_item("synthetic", 42, row, bg, source, root / "out",
                              {"sha256": "lock"}, {"sha256": "background"}, "script")
        assert result["status"] == "completed"
        assert result["posthoc_full_x_labels_loaded"] is True
        assert len(result["h1"]["readout_repeats"]) == 10
        assert len(result["readout_paired_delta"]) == 10
        assert result["pair_errors"]["h1"]["total_error"] > result["pair_errors"]["h8"]["total_error"]
        repeated = analyze_item("synthetic", 42, row, bg, source, root / "out",
                                {"sha256": "lock"}, {"sha256": "background"}, "script")
        assert repeated["config_hash"] == result["config_hash"]
