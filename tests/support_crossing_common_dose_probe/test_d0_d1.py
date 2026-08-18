import json
from pathlib import Path

import numpy as np

from scripts.support_crossing_common_dose_probe import d0_freeze, protocol


def test_d0_freeze_uses_existing_terminal_m1_without_gpu(tmp_path: Path, monkeypatch):
    h0_root = tmp_path / "h0"
    m1_root = tmp_path / "m1"
    m1_root.mkdir()
    for index, dataset in enumerate(protocol.DEVELOPMENT_PANEL):
        dataset_root = h0_root / dataset
        dataset_root.mkdir(parents=True)
        np.save(dataset_root / "H0.npy", np.full((2, 2), index + 1, dtype=np.float32))
    (m1_root / "decision.json").write_text(
        json.dumps({"status": "magnitude_match_not_estimable", "gpu_runs_started": 0}),
        encoding="utf-8",
    )
    (m1_root / "audit.json").write_text(
        json.dumps({"audit_ok": True, "model_training_started": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(protocol, "H0_ROOT", h0_root)
    monkeypatch.setattr(protocol, "M1_ROOT", m1_root)

    audit = d0_freeze.run_freeze(tmp_path)
    assert audit["audit_ok"] is True
    assert audit["d2_gpu_runs_started"] == 0
    assert audit["checks"]["m1_status_is_estimability_terminal"] is True
    assert (tmp_path / "audit.json").exists()
    assert protocol.RESULT_ROOT.name == "support_crossing_common_dose_probe"
