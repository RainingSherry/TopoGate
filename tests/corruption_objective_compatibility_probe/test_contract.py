from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.corruption_objective_compatibility_probe import analysis, e0_integrity, protocol
from scripts.corruption_objective_compatibility_probe.corrected_matching import greedy_matching


def test_protocol_freezes_new_project_and_gpu_firewall() -> None:
    protocol.validate_contract()
    config = protocol.resolved_config()
    assert config["project_id"] == "corruption_objective_compatibility_probe"
    assert config["legal_gpu_pool"] == [1, 2, 3, 4, 5, 6]
    assert config["forbidden_gpu_ids"] == [0, 7]
    assert config["e1"]["new_gpu_runs"] == 36
    assert config["e2"]["new_gpu_runs"] == 72
    assert config["e3"]["does_not_change_fit_or_gates"] is True
    assert "raw_X_zero_nonzero_descriptive_only" in config["e3"]["support_semantics"]


def test_e0_toy_contract_is_complete_and_support_line_stays_locked() -> None:
    checks = e0_integrity._toy_checks()
    assert all(checks.values())
    assert checks["exact_support_crossing"] is True
    assert checks["tie_sensitivity"] is True


def test_same_set_matching_is_globally_disjoint() -> None:
    pairs, _ = greedy_matching(
        np.asarray([1.0, 0.7, 0.4, 0.2], dtype=np.float32),
        np.asarray([1.0, 0.7, 0.4, 0.2], dtype=np.float32),
        2,
        mode="max",
        seed=42,
        row=0,
        same_set=True,
    )
    endpoints = pairs.reshape(-1)
    assert len(np.unique(endpoints)) == 4


def test_atomic_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "summary.json"
    analysis.write_json(path, {"finite": True, "values": [1, 2]})
    assert json.loads(path.read_text(encoding="utf-8"))["finite"] is True
    assert not list(path.parent.glob("*.tmp-*"))

