#!/usr/bin/env python3
"""Audit the compiled V25 manuscript against frozen evidence and assets.

This is a deterministic publication-boundary check.  It does not select a
result, rerun a model, or treat an incomplete holdout as evidence.  It verifies
the final PDF, LaTeX log, formal citation set, figure/table provenance, and the
numeric/scope firewall already checked by ``audit_paper_claims.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.V25.audit_paper_claims import DRAFT_ANCHORS, audit_paper_claims
from scripts.V25.audit_paper_citations import audit_formal


ROOT = _REPO_ROOT
DEFAULT_V25 = ROOT / "result" / "V25_systematic_mechanism_study"
DEFAULT_PAPER = ROOT / "papers" / "V25_systematic_mechanism_study" / "paper"
FIGURE_LABELS = {"fig:atlas", "fig:chain", "fig:e1", "fig:diagnostics", "fig:boundary"}
FIGURE_STEMS = {
    "V25_Figure1_failure_atlas",
    "V25_Figure2_mechanism_chain",
    "V25_Figure3_e1_selectivity",
    "V25_Figure4_diagnostics",
    "V25_Figure5_local_global_boundary",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_text(paper: Path) -> str:
    paths = [paper / "main.tex"]
    paths.extend(sorted((paper / "sections").glob("*.tex")))
    paths.extend(sorted((paper / "tables").glob("*.tex")))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths if path.is_file())


def _pdf_text(pdf: Path) -> tuple[str, str | None]:
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return "", str(exc)
    if proc.returncode != 0:
        return "", proc.stderr.strip() or f"pdftotext exited {proc.returncode}"
    return proc.stdout, None


def audit_final_paper(v25_root: Path = DEFAULT_V25, paper: Path = DEFAULT_PAPER) -> dict[str, Any]:
    pdf = paper / "main.pdf"
    log = paper / "main.log"
    source = _source_text(paper)
    extracted, pdf_error = _pdf_text(pdf) if pdf.is_file() else ("", "missing PDF")
    evidence = v25_root / "PaperEvidence"

    checks: dict[str, bool] = {
        "latex_source_exists": (paper / "main.tex").is_file(),
        "pdf_exists_and_nonempty": pdf.is_file() and pdf.stat().st_size > 0,
        "pdftotext_succeeded": pdf_error is None,
    }

    log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    checks["compile_log_exists"] = log.is_file()
    checks["no_undefined_citation_or_reference"] = not bool(
        re.search(r"undefined|There were undefined|Citation `[^`]+`", log_text, re.IGNORECASE)
    )
    checks["no_overfull_hbox_warning"] = "Overfull \\hbox" not in log_text
    checks["pdf_has_no_unresolved_markers"] = "??" not in extracted and "[?]" not in extracted

    figure_labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    figure_includes = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source))
    checks["five_figure_environments"] = len(re.findall(r"\\begin\{figure\}", source)) == 5
    checks["five_figure_labels"] = FIGURE_LABELS <= figure_labels
    checks["five_figure_includes"] = FIGURE_STEMS <= figure_includes
    checks["pdf_contains_five_figure_captions"] = sum(1 for label in ("Mechanism localization chain", "Failure Atlas", "Generic intervention effect", "E2 diagnostic geometry", "V23 local/global boundary evidence") if label in extracted) == 5

    figure_manifest_path = evidence / "figure_manifest.json"
    figure_manifest = _read_json(figure_manifest_path) if figure_manifest_path.is_file() else {}
    figure_assets = figure_manifest.get("figure_assets", [])
    paper_figure_dir = paper / "figures"
    asset_checks: list[dict[str, Any]] = []
    for relative in figure_assets:
        evidence_path = evidence / relative
        paper_path = paper_figure_dir / Path(relative).name
        asset_checks.append(
            {
                "asset": relative,
                "evidence_exists": evidence_path.is_file(),
                "paper_exists": paper_path.is_file(),
                "hash_equal": evidence_path.is_file() and paper_path.is_file() and _sha256(evidence_path) == _sha256(paper_path),
            }
        )
    checks["figure_manifest_has_15_assets"] = len(figure_assets) == 15
    checks["all_figure_assets_hash_match"] = bool(asset_checks) and all(
        row["evidence_exists"] and row["paper_exists"] and row["hash_equal"] for row in asset_checks
    )

    table_manifest_path = paper / "tables" / "latex_assets_manifest.json"
    table_manifest = _read_json(table_manifest_path) if table_manifest_path.is_file() else {}
    table_source_checks: list[dict[str, Any]] = []
    for name, record in table_manifest.get("sources", {}).items():
        path = Path(record["path"])
        table_source_checks.append(
            {"name": name, "exists": path.is_file(), "hash_equal": path.is_file() and _sha256(path) == record.get("sha256")}
        )
    checks["table_manifest_exists"] = table_manifest_path.is_file()
    checks["table_source_hashes_match"] = bool(table_source_checks) and all(
        row["exists"] and row["hash_equal"] for row in table_source_checks
    )

    claim_audit = audit_paper_claims(v25_root)
    checks["frozen_claim_audit_passes"] = claim_audit["status"] == "audit_ok"
    pdf_anchor_presence = {
        anchor: (
            anchor in extracted
            or (anchor == "0/6" and "zero of six" in extracted.lower())
            or (anchor == "inconclusive_not_completed" and "inconclusive not completed" in extracted.lower())
        )
        for anchor in DRAFT_ANCHORS
    }
    checks["pdf_numeric_anchors_present"] = all(pdf_anchor_presence.values())
    forbidden_patterns = {
        "incomplete replication": r"\bincomplete\s+replication\b",
        "established independent validation": r"\bindependent\s+(?:validation|replication)\s+(?:is|was)\s+established\b",
        "universal topology superiority": r"\buniversal\s+topology\s+superiority\b",
        "holdout negative result": r"\bholdout\s+(?:failed|is\s+a)\s+(?:negative|model-performance)\s+result\b",
    }
    forbidden_found = [name for name, pattern in forbidden_patterns.items() if re.search(pattern, source + "\n" + extracted, re.IGNORECASE)]
    checks["scope_forbidden_phrases_absent"] = not forbidden_found

    formal_citation = audit_formal(paper / "main.tex")
    checks["formal_citation_audit_passes"] = formal_citation["status"] == "audit_ok"
    status = "audit_ok" if all(checks.values()) else "audit_failed"
    return {
        "protocol_id": "v25_final_paper_audit_v1",
        "status": status,
        "paper": str(paper),
        "pdf": str(pdf),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "forbidden_phrases_found": forbidden_found,
        "figure_assets": asset_checks,
        "table_sources": table_source_checks,
        "claim_audit": {"status": claim_audit["status"], "checks": claim_audit["checks"]},
        "formal_citation_audit": formal_citation,
        "pdf_text_error": pdf_error,
        "pdf_anchor_presence": pdf_anchor_presence,
        "source_hashes": {
            "figure_manifest": _sha256(figure_manifest_path) if figure_manifest_path.is_file() else None,
            "latex_assets_manifest": _sha256(table_manifest_path) if table_manifest_path.is_file() else None,
        },
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# V25 Final Paper Audit",
        "",
        f"**Status:** `{result['status']}`",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in result["checks"].items())
    if result["failed_checks"]:
        lines.extend(["", "Failed checks: " + ", ".join(f"`{name}`" for name in result["failed_checks"])])
    lines.extend(["", "This audit is deterministic and same-workspace; it is not an external reviewer verdict.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v25-root", type=Path, default=DEFAULT_V25)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = audit_final_paper(args.v25_root, args.paper)
    output_dir = args.output_dir or args.paper
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "FINAL_PAPER_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "FINAL_PAPER_AUDIT.md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "failed_checks": result["failed_checks"]}, ensure_ascii=False))
    return 0 if result["status"] == "audit_ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
