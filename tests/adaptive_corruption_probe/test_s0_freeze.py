import json

import pytest

from scripts.adaptive_corruption_probe.s0_freeze import run


def test_s0_freeze_authorizes_only_b1(tmp_path):
    audit = run(tmp_path / "S0_freeze")
    assert audit["status"] == "completed_valid"
    assert audit["formal_performance_run_started"] is False
    decision = json.loads((tmp_path / "S0_freeze" / "decision.json").read_text())
    assert decision["next_stage_authorized"] is True
    assert decision["authorized_next_stage"] == "B1"
    resolved = json.loads((tmp_path / "S0_freeze" / "resolved_config.json").read_text())
    assert resolved["holdout_status"].startswith("not_yet_selected")
    hashes = json.loads((tmp_path / "S0_freeze" / "artifact_hashes.json").read_text())
    assert hashes["raw_artifacts_included"] is False
    assert "audit.json" in hashes["files"]


def test_s0_refuses_preexisting_raw_artifact(tmp_path):
    output = tmp_path / "S0_freeze"
    output.mkdir()
    (output / "checkpoint.pt").write_bytes(b"not a result")
    with pytest.raises(RuntimeError, match="forbidden raw artifacts"):
        run(output)
