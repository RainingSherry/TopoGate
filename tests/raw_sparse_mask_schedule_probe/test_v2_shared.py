from pathlib import Path

from scripts import raw_sparse_mask_schedule_v2_shared as v2


def test_v2_shared_mode_clears_only_user_selected_resource_gates():
    assert v2.V2_ALLOWED_GPUS == (1, 2, 3, 4, 5, 6)
    assert set(v2.protocol.FORBIDDEN_GPU_IDS) == {0, 7}
    assert v2.V2_PROTOCOL_ID.endswith("_v2_shared")


def test_v2_cell_does_not_promote_failed_audit(monkeypatch, tmp_path: Path):
    main_root = tmp_path / "MAIN"
    monkeypatch.setattr(v2, "V2_MAIN_ROOT", main_root)
    protocol_path_fields = ("PROTOCOL_ID", "PLAN_VERSION", "MAIN_ROOT", "FREEZE_ROOT", "FIXED_ROOT", "REPR_ROOT", "COMPUTE_ROOT", "FINAL_ROOT")
    original_protocol_paths = {name: getattr(v2.protocol, name) for name in protocol_path_fields}

    def fake_run_one(*args, **kwargs):
        run_dir = main_root / "cnae9" / "CLEAN_AE" / "seed42"
        run_dir.mkdir(parents=True, exist_ok=True)
        v2.run_main.write_json_atomic(run_dir / "audit.json", {"audit_ok": False})
        return {
            "project_id": "raw_sparse_mask_schedule_probe",
            "protocol_id": v2.V2_PROTOCOL_ID,
            "dataset": "cnae9",
            "arm": "CLEAN_AE",
            "seed": 42,
            "status": "incomplete_compute",
            "audit_ok": False,
        }

    monkeypatch.setattr(v2.run_main, "_run_one", fake_run_one)
    try:
        summary = v2._run_cell("cnae9", "CLEAN_AE", 42, 2)
        assert summary["audit_ok"] is False
        assert summary["resource_mode"] == "shared_resource_allowed"
        assert (main_root / "cnae9" / "CLEAN_AE" / "seed42" / "audit.json").read_text().find('"audit_ok": false') >= 0
    finally:
        for name, value in original_protocol_paths.items():
            setattr(v2.protocol, name, value)
