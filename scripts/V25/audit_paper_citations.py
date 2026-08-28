#!/usr/bin/env python3
"""Check the local citation lifecycle for the V25 working manuscript.

This is intentionally a local provenance check. It verifies that cited entries
have a non-trivial PDF, an INDEX record, and a matching bibliography key. It
does not perform web lookups or certify semantic correctness of every sentence.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAFT = ROOT / "refine-logs" / "V25_MANUSCRIPT_WORKING_DRAFT.md"
DEFAULT_BIB = ROOT / "papers" / "V25_systematic_mechanism_study" / "references.bib"
DEFAULT_INDEX = ROOT / "papers" / "references" / "INDEX.md"
MIN_PDF_BYTES = 100_000

REFERENCES = {
    1: {
        "key": "xu2024sccdcg",
        "pdf": ROOT / "papers/references/pdf/13_sccdcg_arxiv2024.pdf",
        "index_marker": "### 12 | scCDCG",
    },
    2: {
        "key": "yan2026scmib",
        "pdf": ROOT / "papers/references/pdf/29_scdebcl_2026.pdf",
        "index_marker": "### 19 | scMIB",
    },
    3: {
        "key": "li2025dcboost",
        "pdf": ROOT / "papers/references/pdf/clustering_sota_2026/NeurIPS-2025-DCBoost.pdf",
        "index_marker": "### 37 | DCBoost",
    },
    4: {
        "key": "li2025lfss",
        "pdf": ROOT / "papers/references/pdf/clustering_sota_2026/ICML-2025-LFSS.pdf",
        "index_marker": "### 57 | LFSS",
    },
}


def _read_tex_with_inputs(path: Path, seen: set[Path] | None = None) -> str:
    """Read a LaTeX root and its local ``\\input{...}`` children."""
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen or not path.is_file():
        return ""
    seen.add(path)
    text = path.read_text(encoding="utf-8")
    parts = [text]
    for child in re.findall(r"\\input\{([^}]+)\}", text):
        child_path = Path(child)
        if child_path.suffix != ".tex":
            child_path = child_path.with_suffix(".tex")
        parts.append(_read_tex_with_inputs(path.parent / child_path, seen))
    return "\n".join(parts)


def audit(draft: Path = DEFAULT_DRAFT, bib: Path = DEFAULT_BIB, index: Path = DEFAULT_INDEX) -> dict[str, Any]:
    draft_text = draft.read_text(encoding="utf-8")
    bib_text = bib.read_text(encoding="utf-8")
    index_text = index.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for ref, entry in REFERENCES.items():
        pdf = Path(entry["pdf"])
        exists = pdf.is_file()
        size = pdf.stat().st_size if exists else 0
        row = {
            "ref": ref,
            "bib_key": entry["key"],
            "bib_key_present": f"{{{entry['key']}" in bib_text,
            "pdf": str(pdf.relative_to(ROOT)),
            "pdf_exists": exists,
            "pdf_size_bytes": size,
            "pdf_size_ok": size >= MIN_PDF_BYTES,
            "index_marker": entry["index_marker"],
            "index_present": entry["index_marker"] in index_text,
            "draft_marker_present": f"[{ref}]" in draft_text,
        }
        rows.append(row)
    checks = {
        "all_references_have_bib_keys": all(row["bib_key_present"] for row in rows),
        "all_references_have_pdf": all(row["pdf_exists"] and row["pdf_size_ok"] for row in rows),
        "all_references_have_index_entries": all(row["index_present"] for row in rows),
        "all_references_are_used_in_draft": all(row["draft_marker_present"] for row in rows),
        "no_citation_todo_marker": "[CITATION TODO" not in draft_text,
        "scmae_missing_pdf_boundary_declared": "scMAE" in draft_text and "PDF" in draft_text and "unavailable" in draft_text,
    }
    return {
        "protocol_id": "v25_local_citation_lifecycle_v1",
        "status": "audit_ok" if all(checks.values()) else "audit_failed",
        "semantic_web_review": "not_run",
        "checks": checks,
        "references": rows,
        "excluded_boundary": "scMAE local PDF unavailable; excluded from formal citation",
    }


def audit_formal(
    manuscript: Path,
    bib: Path = DEFAULT_BIB,
    index: Path = DEFAULT_INDEX,
) -> dict[str, Any]:
    """Audit the citation set actually used by the final LaTeX manuscript.

    The working-draft audit above checks numbered prose references.  LaTeX uses
    bibliography keys instead, so the formal audit extracts ``\\cite...`` keys
    from the manuscript and verifies only that cited set against the local PDF
    and INDEX lifecycle records.
    """
    manuscript_text = _read_tex_with_inputs(manuscript)
    bib_text = bib.read_text(encoding="utf-8")
    index_text = index.read_text(encoding="utf-8")
    cited_keys: list[str] = []
    for group in re.findall(r"\\cite[A-Za-z*]*\{([^}]*)\}", manuscript_text):
        cited_keys.extend(key.strip() for key in group.split(",") if key.strip())
    cited_keys = sorted(set(cited_keys))
    bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)", bib_text))
    by_key = {entry["key"]: entry for entry in REFERENCES.values()}
    rows: list[dict[str, Any]] = []
    for key in cited_keys:
        entry = by_key.get(key)
        if entry is None:
            rows.append(
                {
                    "bib_key": key,
                    "known_local_reference": False,
                    "bib_key_present": key in bib_keys,
                    "pdf_exists": False,
                    "pdf_size_ok": False,
                    "index_present": False,
                }
            )
            continue
        pdf = Path(entry["pdf"])
        size = pdf.stat().st_size if pdf.is_file() else 0
        rows.append(
            {
                "bib_key": key,
                "known_local_reference": True,
                "bib_key_present": key in bib_keys,
                "pdf": str(pdf.relative_to(ROOT)),
                "pdf_exists": pdf.is_file(),
                "pdf_size_bytes": size,
                "pdf_size_ok": size >= MIN_PDF_BYTES,
                "index_marker": entry["index_marker"],
                "index_present": entry["index_marker"] in index_text,
            }
        )
    checks = {
        "manuscript_exists": manuscript.is_file(),
        "nonempty_citation_set": bool(cited_keys),
        "all_citations_known_local": all(row["known_local_reference"] for row in rows),
        "all_citations_have_bib_keys": all(row["bib_key_present"] for row in rows),
        "all_citations_have_verified_pdf": all(row.get("pdf_exists") and row.get("pdf_size_ok") for row in rows),
        "all_citations_have_index_entries": all(row.get("index_present") for row in rows),
        "scmae_missing_pdf_boundary_declared": (
            "scMAE" in manuscript_text
            and "not cited" in manuscript_text
            and "PDF" in manuscript_text
        ),
    }
    return {
        "protocol_id": "v25_formal_latex_citation_audit_v1",
        "status": "audit_ok" if all(checks.values()) else "audit_failed",
        "manuscript": str(manuscript),
        "bib": str(bib),
        "index": str(index),
        "cited_keys": cited_keys,
        "checks": checks,
        "references": rows,
        "excluded_boundary": "scMAE local PDF unavailable; excluded from formal citation",
    }


def _formal_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# V25 Formal LaTeX Citation Audit",
        "",
        f"**Status:** `{result['status']}`  ",
        f"**Protocol:** `{result['protocol_id']}`  ",
        f"**Manuscript:** `{result['manuscript']}`",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in result["checks"].items())
    lines.extend(
        [
            "",
            "Citations are extracted from the formal LaTeX root and all local `\\input` sections.",
            "The scMAE PDF remains an explicit excluded boundary; this is not an external semantic review.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--manuscript", type=Path, default=None, help="audit citation keys in a formal LaTeX manuscript")
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit_formal(args.manuscript, args.bib, args.index) if args.manuscript else audit(args.draft, args.bib, args.index)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.manuscript:
            args.output.with_suffix(".md").write_text(_formal_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, ensure_ascii=False))
    return 0 if result["status"] == "audit_ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
