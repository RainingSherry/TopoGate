from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.V25.build_paper_figures import build_figures
from scripts.V25.build_paper_evidence import _e2_rows, build_bundle


ROOT = Path(__file__).resolve().parents[3]
V25_ROOT = ROOT / "result" / "V25_systematic_mechanism_study"


def test_paper_evidence_bundle_preserves_frozen_claim_scope(tmp_path: Path) -> None:
    summary = build_bundle(V25_ROOT, tmp_path / "PaperEvidence")
    audit = summary["claim_scope_audit"]
    assert audit["audit_ok"] is True
    assert audit["checks"]["a2_veto_and_no_e4_recorded"] is True
    assert audit["checks"]["claim_freeze_primary_endpoint_is_frozen"] is True
    assert summary["holdout"]["status"] == "inconclusive_not_completed"
    assert summary["holdout"]["primary_endpoint_evaluable"] is False
    assert (tmp_path / "PaperEvidence" / "a2_claim_evidence_matrix.csv").is_file()
    assert (tmp_path / "PaperEvidence" / "frozen_claim.json").is_file()


def test_paper_evidence_exports_dataset_seed_units_only(tmp_path: Path) -> None:
    build_bundle(V25_ROOT, tmp_path / "PaperEvidence")
    payload = json.loads((tmp_path / "PaperEvidence" / "paper_evidence_summary.json").read_text())
    assert payload["e2"]["coordinate_distributions_descriptive_only"] is True
    # Nine audited panels x three seeds x the frozen ten-metric semantic
    # schema.  Coordinate counts remain descriptive, not inferential units.
    assert payload["e2"]["dataset_seed_rows"] == 90
    assert payload["confirmation"]["panel_count"] == payload["confirmation"]["audit_ok_count"] == 9


def test_paper_evidence_records_legacy_optional_a1_artifacts_without_reconstruction(tmp_path: Path) -> None:
    # Exercise the compatibility path with an explicit legacy source tree;
    # the current formal A1 result intentionally contains these artifacts.
    legacy_root = tmp_path / "legacy_root"
    legacy_root.mkdir()
    for name in ("A0", "A2", "E1", "PhaseC", "PhaseD", "PhaseE"):
        (legacy_root / name).symlink_to(V25_ROOT / name, target_is_directory=True)
    shutil.copytree(V25_ROOT / "A1", legacy_root / "A1")
    for name in ("structural_opportunity_summary.csv", "e3_replay_summary.json"):
        (legacy_root / "A1" / name).unlink()
    build_bundle(legacy_root, tmp_path / "PaperEvidence")
    payload = json.loads((tmp_path / "PaperEvidence" / "paper_evidence_summary.json").read_text())
    missing = set(payload["missing_source_files"])
    assert "A1/structural_opportunity_summary.csv" in missing
    assert "A1/e3_replay_summary.json" in missing
    assert not (tmp_path / "PaperEvidence" / "structural_opportunity_summary.csv").exists()
    source_manifest = json.loads((tmp_path / "PaperEvidence" / "source_manifest.json").read_text())
    assert source_manifest["sources"]["A1/structural_opportunity_summary.csv"]["exists"] is False


def test_paper_figures_are_bound_to_evidence_inputs(tmp_path: Path) -> None:
    evidence = tmp_path / "PaperEvidence"
    build_bundle(V25_ROOT, evidence)
    manifest = build_figures(evidence, evidence)
    assert len(manifest["figures"]) == 5
    assert all((evidence / path).is_file() for path in manifest["figures"])
    assert manifest["figure_formats"] == ["png", "pdf", "svg"]
    assert len(manifest["figure_assets"]) == 15
    assert all((evidence / path).is_file() for path in manifest["figure_assets"])
    for figure_path in manifest["figures"]:
        figure_stem = (evidence / figure_path).with_suffix("")
        assert figure_stem.with_suffix(".pdf").is_file()
        assert figure_stem.with_suffix(".svg").is_file()
    assert manifest["claim_scope"]["holdout"] == "not represented; Phase D is inconclusive_not_completed"


def _e2_fixture() -> tuple[dict, dict]:
    expected = {
        "datasets": ["d1", "d2", "d3"],
        "seeds": [42, 123, 7],
        "expected_count": 9,
    }
    audits = [
        {
            "dataset_id": dataset,
            "seed": seed,
            "audit_ok": True,
            "metrics": {"model_variance": {"difference": 0.1}},
        }
        for dataset in expected["datasets"]
        for seed in expected["seeds"]
    ]
    return {"audits": audits}, expected


def test_e2_export_rejects_missing_dataset_seed_panel(tmp_path: Path) -> None:
    payload, expected = _e2_fixture()
    payload["audits"] = [audit for audit in payload["audits"] if audit["dataset_id"] != "d3"]
    with pytest.raises(ValueError, match="E2-A coverage mismatch"):
        _e2_rows(payload, expected)


def test_e2_export_rejects_duplicate_panel_key(tmp_path: Path) -> None:
    payload, expected = _e2_fixture()
    payload["audits"][-1] = dict(payload["audits"][0])
    with pytest.raises(ValueError, match="E2-A coverage mismatch"):
        _e2_rows(payload, expected)
