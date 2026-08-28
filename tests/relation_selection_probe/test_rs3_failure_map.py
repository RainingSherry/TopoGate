import pytest

from scripts.relation_selection_probe.rs3_failure_map import _selector_aggregate


def test_selector_aggregate_keeps_material_capture_and_three_seed_pairing():
    rows = [
        {"selector": "B0_cosine", "H_pool": 0.2, "Delta_S": 0.05, "Capture_S": 0.25},
        {"selector": "B0_cosine", "H_pool": 0.2, "Delta_S": 0.06, "Capture_S": 0.30},
        {"selector": "B0_cosine", "H_pool": 0.2, "Delta_S": 0.04, "Capture_S": 0.20},
    ]
    result = _selector_aggregate(rows, "B0_cosine")
    assert result["Delta_S_mean"] == pytest.approx(0.05)
    assert result["Capture_S_median"] == pytest.approx(0.25)
    assert result["material_opportunity"] is True
    assert result["material_positive_capture"] is True
