from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.V25.launch_e1_pilot import ARMS, PROTOCOL_ID, _load_previous, collapse_panels, pilot_audit_admits_confirmation


def _toy_manifest(phase: str, datasets: tuple[str, ...] = ("toy",)) -> dict:
    jobs = []
    for dataset in datasets:
        for seed in (42, 123, 7):
            panel_key = f"panel::{dataset}::{seed}"
            for arm in ARMS:
                jobs.append(
                    {
                        "phase": phase,
                        "panel_run_key": panel_key,
                        "dataset": dataset,
                        "seed": seed,
                        "arm": arm,
                    }
                )
    return {
        "manifest_id": "toy_manifest",
        "protocol_id": PROTOCOL_ID,
        "phases": {
            phase: {
                "expected_panel_jobs": len(datasets) * 3,
                "expected_arm_jobs": len(jobs),
                "jobs": jobs,
            }
        },
    }


def test_collapse_panels_requires_one_shared_nrt_panel(tmp_path: Path) -> None:
    source = tmp_path / "toy.npz"
    source.write_bytes(b"toy-source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = []
    for seed in (42, 123, 7):
        for arm in ARMS:
            rows.append(
                {
                    "phase": "pilot",
                    "panel_run_key": f"panel::toy::{seed}",
                    "arm": arm,
                    "arms": list(ARMS),
                    "dataset": "toy",
                    "input_protocol": "shared_text",
                    "source_path": str(source),
                    "source_sha256": digest,
                    "seed": seed,
                    "primary_readout": "clean_embedding_known_k_kmeans",
                    "K_source": "benchmark_oracle_from_y",
                }
            )
    manifest = _toy_manifest("pilot")
    panels = collapse_panels(manifest, rows, tmp_path / "out")
    assert len(panels) == 3
    assert all(panel["arms"] == list(ARMS) for panel in panels)
    assert {panel["seed"] for panel in panels} == {42, 123, 7}


def test_collapse_panels_rejects_missing_arm(tmp_path: Path) -> None:
    source = tmp_path / "toy.npz"
    source.write_bytes(b"toy-source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = [
        {
            "phase": "pilot",
            "panel_run_key": "panel::toy::42",
            "arm": arm,
            "arms": list(ARMS),
            "dataset": "toy",
            "input_protocol": "shared_text",
            "source_path": str(source),
            "source_sha256": digest,
            "seed": 42,
        }
        for arm in ("N", "R")
    ]
    with pytest.raises(ValueError, match="exactly N/R/T"):
        collapse_panels(_toy_manifest("pilot"), rows, tmp_path / "out")


def test_collapse_panels_supports_frozen_confirmation_phase(tmp_path: Path) -> None:
    source = tmp_path / "toy.npz"
    source.write_bytes(b"toy-source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = [
        {
            "phase": "confirmation",
            "panel_run_key": f"panel::toy::{seed}",
            "arm": arm,
            "arms": list(ARMS),
            "dataset": "toy",
            "input_protocol": "shared_text",
            "source_path": str(source),
            "source_sha256": digest,
            "seed": seed,
        }
        for seed in (42, 123, 7)
        for arm in ARMS
    ]
    panels = collapse_panels(
        _toy_manifest("confirmation"),
        rows,
        tmp_path / "out",
        "confirmation",
    )
    assert len(panels) == 3
    assert all(panel["phase"] == "confirmation" for panel in panels)


def test_launcher_protocol_constants_are_frozen() -> None:
    assert PROTOCOL_ID == "v25_e1_v21_matched_nrt_v1"
    assert ARMS == ("N", "R", "T")


def test_confirmation_admission_rejects_incomplete_or_wrong_manifest_coverage() -> None:
    manifest = _toy_manifest("pilot")
    expected = sorted({job["panel_run_key"] for job in manifest["phases"]["pilot"]["jobs"]})
    complete = {
        "manifest_id": manifest["manifest_id"],
        "phase": "pilot",
        "coverage_complete": True,
        "phase_gate": {"passes": True},
        "expected_panel_count": 3,
        "panel_count": 3,
        "audit_ok_count": 3,
        "expected_datasets": ["toy"],
        "expected_seeds": [7, 42, 123],
        "expected_panel_keys": expected,
        "observed_panel_keys": expected,
    }
    assert pilot_audit_admits_confirmation(complete, manifest)
    missing_seed = dict(complete, expected_seeds=[42, 123], observed_panel_keys=expected[:-1], coverage_complete=False)
    assert not pilot_audit_admits_confirmation(missing_seed, manifest)
    wrong_dataset = dict(complete, expected_datasets=["other"])
    assert not pilot_audit_admits_confirmation(wrong_dataset, manifest)


def test_stale_running_panel_is_requeued_without_duplicate_active_process(tmp_path: Path) -> None:
    state = tmp_path / "queue_state.json"
    state.write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "phase": "holdout",
                "panels": [{"panel_run_key": "panel::toy::42", "status": "running", "pid": 999999999, "attempts": 1}],
            }
        )
    )
    panels = [{"panel_run_key": "panel::toy::42", "status": "queued", "attempts": 0, "output_dir": str(tmp_path / "toy") }]
    _load_previous(state, panels, "holdout")
    assert panels[0]["status"] == "queued"
    assert panels[0]["error"] == "retry_after_stale_running_launcher"
    assert panels[0]["attempts"] == 1
