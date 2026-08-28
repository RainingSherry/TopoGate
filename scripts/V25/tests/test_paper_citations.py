from pathlib import Path

from scripts.V25.audit_paper_citations import audit, audit_formal


ROOT = Path(__file__).resolve().parents[3]


def test_v25_local_citation_lifecycle() -> None:
    result = audit(
        ROOT / "refine-logs/V25_MANUSCRIPT_WORKING_DRAFT.md",
        ROOT / "papers/V25_systematic_mechanism_study/references.bib",
        ROOT / "papers/references/INDEX.md",
    )
    assert result["status"] == "audit_ok"
    assert all(result["checks"].values())


def test_v25_formal_latex_citation_lifecycle() -> None:
    result = audit_formal(
        ROOT / "papers/V25_systematic_mechanism_study/paper/main.tex",
        ROOT / "papers/V25_systematic_mechanism_study/paper/references.bib",
        ROOT / "papers/references/INDEX.md",
    )
    assert result["status"] == "audit_ok"
    assert all(result["checks"].values())
