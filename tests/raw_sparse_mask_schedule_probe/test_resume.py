from pathlib import Path

from scripts.raw_sparse_mask_schedule_probe.run_main import _existing_valid, write_json_atomic
from scripts.raw_sparse_mask_schedule_probe import provenance


def test_resume_accepts_exact_hash_match_and_rejects_drift(tmp_path: Path):
    run = tmp_path / "run"
    expected = {
        "project_id": "raw_sparse_mask_schedule_probe",
        "protocol_id": "raw_sparse_mask_schedule_probe_v1",
        "dataset": "toy",
        "arm": "CLEAN_AE",
        "seed": 42,
        "source_sha256": "source-a",
        "adapter_hash": "adapter-a",
        "scale_hash": "scale-a",
        "code_sha256": "code-a",
        "status": "completed_valid",
        "labels_loaded_during_fit": False,
    }
    write_json_atomic(run / "summary.json", expected)
    assert _existing_valid(run, expected)
    drifted = dict(expected, source_sha256="source-b")
    assert not _existing_valid(run, drifted)
    code_drift = dict(expected, code_sha256="code-b")
    assert not _existing_valid(run, code_drift)


def test_resume_rejects_incident_marked_summary(tmp_path: Path):
    run = tmp_path / "run"
    expected = {
        "project_id": "raw_sparse_mask_schedule_probe",
        "protocol_id": "raw_sparse_mask_schedule_probe_v1",
        "dataset": "toy",
        "arm": "CLEAN_AE",
        "seed": 42,
        "source_sha256": "source-a",
        "adapter_hash": "adapter-a",
        "scale_hash": "scale-a",
        "code_sha256": "code-a",
        "status": "incomplete_compute",
        "labels_loaded_during_fit": False,
        "audit_ok": False,
        "formal_validity": "excluded_due_to_dispatch_occupancy_guard_defect",
    }
    write_json_atomic(run / "summary.json", expected)
    expected["status"] = "completed_valid"
    assert not _existing_valid(run, expected)
