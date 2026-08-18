from scripts.support_target_validation_probe.m1_preflight import evaluate
from scripts.support_target_validation_probe import protocol
from scripts.support_target_validation_probe.m1_matrix import _require_preflight_authorization

import json
import pytest


def _row(dataset, seed, estimable):
    return {
        "dataset": dataset,
        "seed": seed,
        "magnitude_match_estimable": estimable,
        "exact_changed_coordinate_budget": True,
        "support_change_rate_exact_zero": True,
        "row_value_multiset_mismatch_count": 0,
        "match_failure_count": 0,
        "dataset_total_relative_mismatch": 0.01 if estimable else 0.09,
        "median_row_relative_mismatch": 0.05,
    }


def test_non_estimable_preflight_blocks_gpu_without_negative_result():
    rows = [_row("Mouse_retina", 42, True), _row("Baron Human", 42, False)]
    decision = evaluate(rows)
    assert decision["status"] == "magnitude_match_not_estimable"
    assert decision["formal_m1_gpu_runs_authorized"] is False
    assert decision["gpu_runs_started"] == 0
    assert decision["non_estimable"] == [{"dataset": "Baron Human", "seed": 42, "dataset_total_relative_mismatch": 0.09, "median_row_relative_mismatch": 0.05}]


def test_gpu_launcher_rejects_non_estimable_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "RESULT_ROOT", tmp_path)
    target = tmp_path / "M1_preflight"
    target.mkdir()
    (target / "decision.json").write_text(json.dumps({"status": "magnitude_match_not_estimable", "formal_m1_gpu_runs_authorized": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="blocked"):
        _require_preflight_authorization()
