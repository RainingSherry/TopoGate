"""Semantic checks for the label-posthoc pair-rate accounting."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readout_analysis import STATES, pair_rates, run_case, select_exploratory  # noqa: E402


def brute_rates(y: np.ndarray, pred: np.ndarray, n_classes: int):
    split = {c: [0, 0] for c in range(n_classes)}
    merged = {(a, b): [0, 0] for a in range(n_classes) for b in range(a + 1, n_classes)}
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] == y[j]:
                slot = split[int(y[i])]
                slot[1] += 1
                slot[0] += int(pred[i] != pred[j])
            else:
                slot = merged[tuple(sorted((int(y[i]), int(y[j]))))]
                slot[1] += 1
                slot[0] += int(pred[i] == pred[j])
    return split, merged


def test_pair_rates_known_example():
    y = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([5, 5, 6, 5, 7, 8])
    classes, pairs = pair_rates(y, pred, ["a", "b", "c"])
    assert [(r["split_count"], r["true_pair_count"], r["split_rate"]) for r in classes] == [
        (2, 3, 2 / 3), (1, 1, 1.0), (0, 0, None)]
    assert [(r["merge_count"], r["true_pair_count"], r["merge_rate"]) for r in pairs] == [
        (2, 6, 1 / 3), (0, 3, 0.0), (0, 2, 0.0)]


def test_contingency_equals_brute_force_and_cluster_relabeling():
    for seed in (1, 7, 42):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 4, 23)
        pred = rng.choice([10, 30, 100, 200, 900], size=len(y))
        classes, pairs = pair_rates(y, pred, ["a", "b", "c", "d"])
        split, merged = brute_rates(y, pred, 4)
        for row in classes:
            assert (row["split_count"], row["true_pair_count"]) == tuple(split[row["class_id"]])
        for row in pairs:
            key = (row["class_a_id"], row["class_b_id"])
            assert (row["merge_count"], row["true_pair_count"]) == tuple(merged[key])
        renamed = np.array([{10: 3, 30: 1, 100: 7, 200: 2, 900: 5}[x] for x in pred])
        assert pair_rates(y, renamed, ["a", "b", "c", "d"]) == (classes, pairs)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_cross_seed_selection_uses_all_repeats_and_id_tiebreak(tmp_path):
    dirs = []
    for seed in (42, 123, 7):
        case = tmp_path / f"seed_{seed}"
        case.mkdir()
        (case / "summary.json").write_text(json.dumps({
            "status": "completed", "dataset": "Tosches", "seed": seed,
            "smoke": False, "readout_random_states": list(STATES),
            "inputs": {"lock_sha256": "locked"}}), encoding="utf-8")
        rows = []
        for state in STATES:
            for c in range(3):
                for arm in ("h1", "h8"):
                    # c=0 and c=1 tie exactly; deterministic ID chooses both.
                    h1 = 0.6 if c < 2 else 0.4
                    h8 = (0.4 if c < 2 else 0.35) + (0.03 if seed == 7 else 0)
                    rows.append({"dataset": "Tosches", "seed": seed, "arm": arm,
                                 "random_state": state, "class_id": c, "class_name": str(c),
                                 "support": 5, "true_pair_count": 10, "split_count": 0,
                                 "split_rate": h1 if arm == "h1" else h8})
        _write_csv(case / "class_split_rates.csv", rows)
        dirs.append(case)
    result = select_exploratory("Tosches", dirs, tmp_path / "aggregate")
    assert [int(row["class_id"]) for row in result["selected_top2"]] == [0, 1]
    assert all(row["n_seed_state_pairs"] == 60 for row in result["selected_top2"])
    assert (tmp_path / "aggregate" / "class_selection_complete.csv").exists()
    assert (tmp_path / "aggregate" / "selection_top2.png").exists()
    for case in dirs:
        with (case / "class_split_rates.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row["arm"] == "h8" and int(row["class_id"]) != 0:
                row["split_rate"] = "0.8"
        _write_csv(case / "class_split_rates.csv", rows)
    one_positive = select_exploratory("Tosches", dirs, tmp_path / "aggregate_one_positive")
    assert [int(row["class_id"]) for row in one_positive["selected_top2"]] == [0]


def test_selection_rejects_incomplete_or_smoke_cases(tmp_path):
    paths = []
    for seed in (42, 123, 7):
        path = tmp_path / str(seed)
        path.mkdir()
        (path / "summary.json").write_text(json.dumps({
            "status": "completed", "dataset": "Baron", "seed": seed,
            "smoke": seed == 7, "readout_random_states": list(STATES),
            "inputs": {"lock_sha256": "locked"}}), encoding="utf-8")
        paths.append(path)
    try:
        select_exploratory("Baron", paths, tmp_path / "aggregate")
    except ValueError as exc:
        assert "non-smoke" in str(exc)
    else:
        raise AssertionError("smoke selection was accepted")


def test_smoke_case_artifact_contract(tmp_path):
    previous = tmp_path / "previous"
    (previous / "inputs").mkdir(parents=True)
    rng = np.random.default_rng(4)
    y = np.repeat(np.arange(3), 20)
    source = tmp_path / "source.npz"
    np.savez(source, y=y)
    record_maps = {}
    for arm in ("h1", "h8"):
        run = tmp_path / arm
        run.mkdir()
        record = run / "record_seed_42.json"
        record.write_text(json.dumps({
            "status": "completed", "seed": 42, "selection_label_scope": "validation_only",
            "test_labels_loaded": False, "labels_used_during_fit": False,
            "feature_sha256": "feature", "K_provenance": "benchmark_oracle_from_y",
            "validation_labels_sha256": "validation", "n_clusters": 3,
            "config": {"hop_variant": "H1" if arm == "h1" else "H8_B2"}}), encoding="utf-8")
        np.savez(run / "embedding_seed_42.npz", embedding=rng.normal(size=(len(y), 4)))
        record_maps[arm] = {"42": str(record)}
    (previous / "inputs" / "selection_lock.json").write_text(json.dumps({
        "status": "LOCKED_BEFORE_TEST", "selection_scope": "validation_only",
        "rows": [{"dataset": "Tosches", "selected_config_id": "H8_B2", "h1_config_id": "H1",
                  "feature_sha256": "feature", "validation_labels_sha256": "validation",
                  "n_samples": len(y),
                  "h1_records": record_maps["h1"], "selected_records": record_maps["h8"]}]}), encoding="utf-8")
    (previous / "inputs" / "validation_best_multihop_vs_h1_full_x_60.json").write_text(json.dumps({
        "candidate_selection_scope": "validation_only",
        "rows": [{"dataset": "Tosches", "multihop_config_id": "H8_B2",
                  "h1_config_id": "H1", "source_path": str(source)}]}), encoding="utf-8")
    out = tmp_path / "out"
    result = run_case(previous, out, "Tosches", 42, smoke=True)
    assert result["status"] == "completed"
    assert result["posthoc_full_x_labels_loaded"] is True
    assert result["labels_used_for_model_fit"] is False
    with np.load(out / "readout_predictions.npz") as packed:
        assert set(packed.files) == {"h1", "h8"}
        assert packed["h1"].shape == packed["h8"].shape == (2, len(y))
    assert np.array_equal(np.load(out / "labels_true.npy"), y)
    with (out / "readout_repeats.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(0 < float(row["normalized_inertia"]) < 1 for row in rows)


class ReadoutSemanticTests(unittest.TestCase):
    def test_known_example(self):
        test_pair_rates_known_example()

    def test_brute_force(self):
        test_contingency_equals_brute_force_and_cluster_relabeling()

    def test_cross_seed_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            test_cross_seed_selection_uses_all_repeats_and_id_tiebreak(Path(directory))

    def test_incomplete_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            test_selection_rejects_incomplete_or_smoke_cases(Path(directory))

    def test_smoke_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            test_smoke_case_artifact_contract(Path(directory))


if __name__ == "__main__":
    unittest.main()
