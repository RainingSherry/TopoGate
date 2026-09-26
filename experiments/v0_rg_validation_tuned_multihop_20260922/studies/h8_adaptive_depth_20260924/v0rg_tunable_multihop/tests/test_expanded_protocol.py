from pathlib import Path
import json

import numpy as np

from v0rg_tunable_multihop.graph import NeighborGraph
from v0rg_tunable_multihop.hop_sampling import build_exact_hop_plan
from v0rg_tunable_multihop.config import V0_RGConfig
from v0rg_tunable_multihop.validation_protocol import prepare_isolated_split
from v0rg_tunable_multihop.select_validation import select_validation_records


def _cycle_graph() -> NeighborGraph:
    indices = np.asarray([[1], [2], [3], [4], [5], [0]], dtype=np.int64)
    zeros = np.zeros_like(indices, dtype=np.float32)
    return NeighborGraph(indices, np.ones_like(zeros), zeros, zeros, np.eye(6, dtype=np.float32),
                         np.zeros_like(indices, dtype=bool), zeros, {})


def test_expanded_aliases_match_legacy_plans() -> None:
    graph = _cycle_graph()
    weights = np.ones((6, 1), dtype=np.float64)
    for legacy, expanded in (("H12", "H2_B1"), ("H123", "H3_B1")):
        old = build_exact_hop_plan(graph, weights, legacy, mix_neighbors=1)
        new = build_exact_hop_plan(graph, weights, expanded, mix_neighbors=1)
        assert old.draw_count == new.draw_count
        for old_row, new_row, old_prob, new_prob, old_hop, new_hop in zip(
            old.indices, new.indices, old.probs, new.probs, old.hop_labels, new.hop_labels
        ):
            np.testing.assert_array_equal(old_row, new_row)
            np.testing.assert_array_equal(old_prob, new_prob)
            np.testing.assert_array_equal(old_hop, new_hop)


def test_h5_preserves_draw_count_and_exact_shells() -> None:
    graph = _cycle_graph()
    plan = build_exact_hop_plan(graph, np.ones((6, 1)), "H5_B4", mix_neighbors=3)
    assert plan.draw_count == 1
    assert plan.profile["max_hop"] == 5
    assert plan.profile["shell_width_multiplier"] == 4
    assert set(plan.hop_labels[0].tolist()) == {1, 2, 3, 4, 5}
    assert 0 not in plan.indices[0]


def test_h8_and_high_hop_probability_multiplier() -> None:
    graph = _cycle_graph()
    baseline = build_exact_hop_plan(graph, np.ones((6, 1)), "H8_B1", mix_neighbors=1)
    weighted = build_exact_hop_plan(
        graph, np.ones((6, 1)), "H8_B1", mix_neighbors=1,
        high_hop_probability_multiplier=2.0,
    )
    assert baseline.profile["max_hop"] == 8
    assert baseline.profile["high_hop_probability_multiplier"] == 1.0
    np.testing.assert_array_equal(baseline.indices[0], weighted.indices[0])
    one_hop = weighted.hop_labels[0] == 1
    assert weighted.probs[0][~one_hop].sum() > baseline.probs[0][~one_hop].sum()
    np.testing.assert_allclose(weighted.probs[0].sum(), 1.0)


def test_tuning_strength_parameters_are_validated() -> None:
    config = V0_RGConfig(
        hop_variant="H6_B2", high_hop_probability_multiplier=0.5,
        mix_amplitude_multiplier=1.5,
    )
    assert config.high_hop_probability_multiplier == 0.5
    assert config.mix_amplitude_multiplier == 1.5
    try:
        V0_RGConfig(hop_variant="H1", high_hop_probability_multiplier=2.0)
    except ValueError as exc:
        assert "H1" in str(exc)
    else:
        raise AssertionError("H1 accepted a high-hop probability multiplier")


def test_tuning_manifest_has_no_test_label_path(tmp_path: Path) -> None:
    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = np.repeat(np.arange(2), 10)
    source = tmp_path / "source.npz"
    np.savez_compressed(source, X=X, y=y)
    prepare_isolated_split(source, tmp_path / "split")
    tuning = json.loads((tmp_path / "split" / "tuning_manifest.json").read_text())
    assert "test_labels_path" not in tuning
    assert tuning["test_labels_loaded"] is False
    assert tuning["test_label_path_exposed"] is False
    assert set(np.load(tmp_path / "split" / "features_only.npz", allow_pickle=False).files) == {"X"}


def test_selection_rejects_test_label_records(tmp_path: Path) -> None:
    records = []
    for variant in ("H1", "H2_B1"):
        for seed in (42, 123, 7):
            record = {
                "seed": seed, "config": {"hop_variant": variant}, "test_labels_loaded": False,
                "selection_label_scope": "validation_only", "K_provenance": "benchmark_oracle_from_y",
                "validation_metrics": {"ari": 0.2 if variant == "H1" else 0.3},
            }
            path = tmp_path / f"{variant}_{seed}.json"
            path.write_text(json.dumps(record))
            records.append(path)
    selected = select_validation_records(records)
    assert selected["chosen_hop_variant"] == "H2_B1"
    bad = json.loads(records[-1].read_text())
    bad["test_labels_loaded"] = True
    records[-1].write_text(json.dumps(bad))
    try:
        select_validation_records(records)
    except ValueError as exc:
        assert "final-test" in str(exc)
    else:
        raise AssertionError("test-access record was accepted")
