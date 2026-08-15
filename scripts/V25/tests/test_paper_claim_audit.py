from __future__ import annotations

from pathlib import Path

from scripts.V25.audit_paper_claims import audit_paper_claims, _markdown


ROOT = Path(__file__).resolve().parents[3]
V25_ROOT = ROOT / "result" / "V25_systematic_mechanism_study"


def test_paper_claim_audit_recomputes_frozen_states_and_scope() -> None:
    audit = audit_paper_claims(V25_ROOT, ROOT / "refine-logs" / "V25_MANUSCRIPT_WORKING_DRAFT.md")
    assert audit["status"] == "audit_ok"
    assert all(audit["checks"].values())
    assert audit["checks"]["e1_effect_states_recomputed"] is True
    assert audit["checks"]["holdout_firewall"] is True
    assert audit["checks"]["draft_numeric_anchors"] is True
    assert audit["checks"]["draft_direct_scope_firewall"] is True
    assert {row["claim_id"] for row in audit["claim_rows"]} == {"C1", "C2", "C3", "C4"}


def test_paper_claim_audit_markdown_keeps_scope_firewall() -> None:
    audit = audit_paper_claims(V25_ROOT)
    report = _markdown(audit)
    assert "not fully label-free fitting" in report
    assert "inconclusive_not_completed" in report
    assert "universal topology-superiority" in report
