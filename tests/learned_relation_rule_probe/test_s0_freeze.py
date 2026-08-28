import json

import pytest

from scripts.learned_relation_rule_probe.s0_freeze import run


def test_s0_freeze_is_protocol_only(tmp_path):
    audit = run(tmp_path / "S0_freeze")
    assert audit["status"] == "completed_valid"
    assert audit["formal_performance_run_started"] is False
    assert audit["holdout_disjoint_from_current_panel"] is True
    decision = json.loads((tmp_path / "S0_freeze" / "decision.json").read_text())
    assert decision["next_stage_authorized"] is True
    assert decision["authorized_next_stage"] == "A1"
    resolved = json.loads((tmp_path / "S0_freeze" / "resolved_config.json").read_text())
    assert resolved["holdout"]["selected_count"] == 12
    assert resolved["holdout"]["used_by_this_project"] is False
    hashes = json.loads((tmp_path / "S0_freeze" / "artifact_hashes.json").read_text())
    assert hashes["raw_artifacts_included"] is False
    assert "audit.json" in hashes["files"]


def test_s0_refuses_preexisting_raw_artifact(tmp_path):
    output = tmp_path / "S0_freeze"
    output.mkdir()
    (output / "embedding.npy").write_bytes(b"not a result")
    with pytest.raises(RuntimeError, match="forbidden raw artifacts"):
        run(output)
