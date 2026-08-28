from pathlib import Path

from scripts.V25.audit_final_paper import audit_final_paper


ROOT = Path(__file__).resolve().parents[3]


def test_v25_final_paper_audit_passes() -> None:
    result = audit_final_paper(
        ROOT / "result/V25_systematic_mechanism_study",
        ROOT / "papers/V25_systematic_mechanism_study/paper",
    )
    assert result["status"] == "audit_ok"
    assert not result["failed_checks"]
