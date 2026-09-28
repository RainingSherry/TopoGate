"""Small exact posthoc coverage/config/readout checks; never trains an encoder."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import depth_worker as d


def test_core(tmp: Path):
    assert d.expected_variant("Kwide", "H4_B1") == "H4_B1"
    assert d.expected_variant("H8", "H4_B1") == "H8_B1"
    manifest = json.loads((d.HERE / "REUSE_MANIFEST.json").read_text())
    student = manifest["cases"]["student_grade"]
    selected = d.materialize(student["expected_config_template"], d.expected_variant("Kwide", student["variant"]))
    assert selected.hop_variant == "H4_B1" and selected.high_hop_probability_multiplier == student["q"]
    assert selected.mix_amplitude_multiplier == student["amplitude"]
    communities = np.array([0, 0, 1, 2], dtype=np.int32)
    graph = np.array([[1], [0], [0], [0]], dtype=np.int32)
    near = np.zeros((4, 3), dtype=bool)
    near[np.arange(4)[:, None], communities[graph]] = True
    arrays = {"offsets": np.array([0, 2, 4, 6, 8]),
              "nodes": np.array([1, 2, 0, 3, 0, 3, 0, 2]),
              "probs": np.array([0.25000002, 0.75000002, 0.5, 0.5,
                                  0.5, 0.5, 0.5, 0.5]),
              "hops": np.array([1, 2, 1, 9, 2, 9, 1, 9]),
              "draw_count": np.array([1])}
    np.savez(tmp / "plan.npz", **arrays)
    loaded = d.load_saved_plan(tmp / "plan.npz", 4)
    rows = d.coverage(loaded, near, communities)
    assert [r["candidate_count"] for r in rows] == [2, 2, 2, 2]
    assert np.isclose(rows[0]["fraction_candidates_community_absent_from_h1"], 0.5)
    assert np.isclose(rows[0]["candidate_probability_mass_community_absent_from_h1"], 0.75)
    assert np.isclose(rows[1]["candidate_probability_mass_community_absent_from_h1"], 0.5)
    bad = dict(arrays); bad["nodes"] = arrays["nodes"].copy(); bad["nodes"][0] = 4
    np.savez(tmp / "bad.npz", **bad)
    try: d.load_saved_plan(tmp / "bad.npz", 4)
    except ValueError: pass
    else: raise AssertionError("invalid candidate index accepted")
    embedding = np.array([[0, 0], [0, .1], [5, 5], [5, 5.1]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    readouts, preds = d.readouts(embedding, labels, 2)
    assert len(readouts) == 10 and preds.shape == (10, 4)
    assert [r["readout_state"] for r in readouts] == list(range(10))
    print("PASS H4 Kwide selected reference, q/amplitude fixed, float32 row normalization, exact novelty, invalid index, ten embedding readouts")


def test_actual_results():
    root = d.HERE / "outputs/E2_E6"
    manifest = json.loads((d.HERE / "REUSE_MANIFEST.json").read_text())
    checked = 0
    for dataset in manifest["cases"]:
        folder = root / dataset
        if not (folder / "record.json").exists():
            continue
        record = json.loads((folder / "record.json").read_text())
        assert record["status"] == "completed" and record["completed_cells"] == 15
        assert record["completed_readouts"] == 150 and not record["missing_cells"]
        assert all(d.sha256(folder / name) == digest
                   for name, digest in record["artifact_sha256"].items())
        readouts = pd.read_csv(folder / "readouts.csv")
        per_seed = pd.read_csv(folder / "per_seed.csv")
        summary = pd.read_csv(folder / "summary.csv")
        coverage = pd.read_csv(folder / "coverage_by_anchor.csv")
        assert len(readouts) == 150 and len(per_seed) == 15
        n_anchors = len(coverage[(coverage.strategy == "H1") & (coverage.seed == 7)])
        with np.load(folder / "readout_predictions.npz", allow_pickle=False) as packed:
            assert len(packed.files) == 15
            assert all(packed[key].shape == (10, n_anchors)
                       for key in packed.files)
        assert set(readouts.readout_state) == set(range(10))
        assert (summary.n_seeds == 3).all()
        for row in per_seed.itertuples():
            group = readouts[(readouts.strategy == row.strategy) & (readouts.seed == row.seed)]
            assert len(group) == 10 and abs(group.ARI.mean() - row.ARI_mean) < 1e-12
        for row in summary.itertuples():
            group = per_seed[per_seed.strategy == row.strategy]
            assert abs(group.ARI_mean.mean() - row.ARI_mean) < 1e-12
            assert abs(group.ARI_mean.std(ddof=1) - row.ARI_sd) < 1e-12
        selected = manifest["cases"][dataset]["variant"].split("_")[0]
        for seed in d.SEEDS:
            a = coverage[(coverage.strategy == selected) & (coverage.seed == seed)].sort_values("anchor")
            b = coverage[(coverage.strategy == "Kwide") & (coverage.seed == seed)].sort_values("anchor")
            assert len(a) == len(b) and np.array_equal(a.candidate_count, b.candidate_count)
            h1 = coverage[(coverage.strategy == "H1") & (coverage.seed == seed)]
            assert np.max(np.abs(h1.candidate_probability_mass_community_absent_from_h1)) < 1e-12
        checked += 1
    print(f"PASS {checked} completed real cases: 15×10 readouts, three-seed aggregation, all artifact hashes, H1 zero novelty, exact per-anchor Kwide count matching")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="topogate-depth-review-") as dirname:
        test_core(Path(dirname))
    test_actual_results()
