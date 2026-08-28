from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.V25.build_closure_artifacts import (
    ALLOWED_STAGES,
    E1_COLUMNS,
    GAP_COLUMNS,
    TAXONOMY_COLUMNS,
    build_closure_artifacts,
)


ROOT = Path(__file__).resolve().parents[3]
V25_ROOT = ROOT / "result" / "V25_systematic_mechanism_study"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_closure_artifacts_have_required_schema_and_are_weight_free(tmp_path: Path) -> None:
    manifest = build_closure_artifacts(V25_ROOT, tmp_path)
    gap_rows = _rows(tmp_path / "V25_GAP_MAP.csv")
    assert list(gap_rows[0]) == GAP_COLUMNS
    assert (tmp_path / "V25_GAP_MAP.md").is_file()
    assert (tmp_path / "V25_NEXT_SERIES_DECISION.md").is_file()
    assert manifest["weight_free"] is True
    assert not list(tmp_path.glob("*.pt"))
    assert not list(tmp_path.glob("*.npy"))


def test_taxonomy_covers_v1_v24_and_uses_frozen_stages(tmp_path: Path) -> None:
    build_closure_artifacts(V25_ROOT, tmp_path)
    rows = _rows(tmp_path / "failure_localization_taxonomy.csv")
    assert list(rows[0]) == TAXONOMY_COLUMNS
    assert {row["version"] for row in rows}.issuperset({f"V{i}" for i in range(1, 25)})
    assert all(row["primary_stage"] in ALLOWED_STAGES for row in rows)
    assert all(row["secondary_stage"] in ALLOWED_STAGES for row in rows)
    assert {row["version"] for row in rows} >= {"V23", "V24"}


def test_e1_summary_has_six_datasets_and_pilot_e2_is_deferred(tmp_path: Path) -> None:
    build_closure_artifacts(V25_ROOT, tmp_path)
    rows = _rows(tmp_path / "E1_MECHANISM_SUMMARY.csv")
    assert list(rows[0]) == E1_COLUMNS
    assert len(rows) == 6
    assert len({row["dataset"] for row in rows}) == 6
    assert {row["e2_status"] for row in rows if row["phase"] == "pilot"} == {"deferred"}
    assert {row["e2_status"] for row in rows if row["phase"] == "confirmation"} == {"confirmation_only"}
    confirmation = next(row for row in rows if row["dataset"] == "Baron Human")
    assert abs(float(confirmation["s_d"]) - 0.04461680986715221) < 1e-12


def test_holdout_is_marked_inconclusive_and_next_series_closes_v26(tmp_path: Path) -> None:
    build_closure_artifacts(V25_ROOT, tmp_path)
    manifest = json.loads((tmp_path / "V25_CLOSURE_ARTIFACTS.json").read_text(encoding="utf-8"))
    text = (tmp_path / "V25_NEXT_SERIES_DECISION.md").read_text(encoding="utf-8")
    assert manifest["holdout_status"] == "inconclusive_not_completed"
    assert manifest["audit"]["holdout_not_negative"] is True
    assert manifest["closure_decision"] == "close_without_v26"
    assert "V25 closure is not authorization to run V26" in text
    assert "not a negative" in text
