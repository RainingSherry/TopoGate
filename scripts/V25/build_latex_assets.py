#!/usr/bin/env python3
"""Generate evidence-bound LaTeX tables and provenance for the V25 paper.

Only the frozen PaperEvidence bundle is read.  This script never reads raw
datasets, labels, checkpoints, or training logs, and never selects a result by
its value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = ROOT / "result" / "V25_systematic_mechanism_study" / "PaperEvidence"
DEFAULT_PAPER = ROOT / "papers" / "V25_systematic_mechanism_study" / "paper"
ROW_END = " " + chr(92) * 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _tex(value: Any) -> str:
    text = str(value)
    for source, target in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        text = text.replace(source, target)
    return text


def _signed(value: float) -> str:
    return f"{value:+.6f}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _e1_table(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], {})[row["metric"]] = row
    lines = [
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        "Dataset & $I_d$ & $S_d$ & $S_d$ state" + ROW_END,
        r"\midrule",
    ]
    for dataset in sorted(grouped):
        item = grouped[dataset]
        lines.append(
            f"{_tex(dataset)} & {_signed(float(item['I_d']['mean']))} & "
            f"{_signed(float(item['S_d']['mean']))} & {_tex(item['S_d']['state'])}" + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _atlas_table(summary: dict[str, Any]) -> str:
    a0 = summary["retrospective"]["a0"]
    a1 = summary["retrospective"]["a1"]
    rows = [
        ("V1--V22 registry rows", a0["v1_v22_rows"]),
        (r"Paired $\Delta$ARI rows", a0["v1_v22_paired_rows"]),
        ("Dataset/protocol/readout units", a0["v1_v22_units"]),
        ("Material positive rows", a1["positive_rows"]),
        ("Material negative rows", a1["negative_rows"]),
        ("Observed-small rows", a1["small_rows"]),
    ]
    lines = [r"\begin{tabular}{lr}", r"\toprule", "Quantity & Value" + ROW_END, r"\midrule"]
    lines.extend(f"{label} & {_tex(value)}" + ROW_END for label, value in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _layers_table() -> str:
    rows = [
        ("A0/A1", "V1--V22 registry and atlas", "dataset / protocol / readout", "observational"),
        ("A0 boundary", "V23/V24 No-Go records", "boundary record", "isolated historical boundary"),
        ("E1", "V21 matched N/R/T", "dataset; seeds repeated", "matched prospective case study"),
        ("E2", "feature and optimizer diagnostics", r"dataset $\times$ seed", "diagnostic/post-hoc"),
        ("E3", "local/global replay boundary", "artifact-complete case", "observational boundary"),
        ("Phase D", "predeclared holdout", "six panels", "incomplete compute"),
    ]
    lines = [
        r"{\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}p{0.13\linewidth}p{0.25\linewidth}p{0.22\linewidth}p{0.18\linewidth}@{}}",
        r"\toprule",
        "Layer & Evidence & Statistical unit & Causal status" + ROW_END,
        r"\midrule",
    ]
    lines.extend(" & ".join(_tex(value) for value in row) + ROW_END for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", ""])
    return "\n".join(lines)


def build_assets(evidence: Path, paper: Path) -> dict[str, Any]:
    required = {
        "e1_dataset_effects.csv": evidence / "e1_dataset_effects.csv",
        "paper_evidence_summary.json": evidence / "paper_evidence_summary.json",
        "source_manifest.json": evidence / "source_manifest.json",
        "references.bib": ROOT / "papers" / "V25_systematic_mechanism_study" / "references.bib",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen paper inputs: {missing}")

    paper.mkdir(parents=True, exist_ok=True)
    tables = paper / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write(tables / "e1_effects.tex", _e1_table(_read_csv(required["e1_dataset_effects.csv"])))
    _write(tables / "atlas_summary.tex", _atlas_table(_read_json(required["paper_evidence_summary.json"])))
    _write(tables / "evidence_layers.tex", _layers_table())
    figure_manifest = evidence / "figure_manifest.json"
    if figure_manifest.is_file():
        figure_dir = paper / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure_manifest, figure_dir / "figure_manifest.json")
    # The project-level bibliography keeps filesystem paths in ``note`` fields
    # for provenance.  Escape underscores in the paper copy so BibTeX can emit
    # those notes without treating a path component as math syntax.
    bib_text = required["references.bib"].read_text(encoding="utf-8").replace("_", r"\_")
    _write(paper / "references.bib", bib_text)

    manifest = {
        "protocol_id": "v25_latex_assets_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/V25/build_latex_assets.py",
        "evidence_dir": str(evidence.resolve()),
        "paper_dir": str(paper.resolve()),
        "statistical_unit": "dataset/protocol/readout for atlas; dataset x seed for E1 summaries",
        "sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in required.items()
        },
        "outputs": [
            "tables/e1_effects.tex",
            "tables/atlas_summary.tex",
            "tables/evidence_layers.tex",
            "references.bib",
        ],
        "claim_scope": {
            "atlas": "observational V1-V22",
            "e1": "conditional heterogeneous audited V21 case study",
            "holdout": "inconclusive_not_completed; not a negative result",
        },
    }
    if figure_manifest.is_file():
        manifest["sources"]["figure_manifest.json"] = {
            "path": str(figure_manifest.resolve()),
            "sha256": _sha256(figure_manifest),
        }
        manifest["outputs"].append("figures/figure_manifest.json")
    _write(tables / "latex_assets_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _write(
        paper / "PROVENANCE.md",
        "# V25 LaTeX Asset Provenance\n\n"
        "These tables, the figure manifest copy, and the bibliography copy are generated from the frozen "
        "PaperEvidence bundle. No model, label, checkpoint, or training log is "
        "read by the generator. See `tables/latex_assets_manifest.json` and "
        "`figures/figure_manifest.json` for "
        "source SHA256 values.\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    args = parser.parse_args()
    manifest = build_assets(args.evidence, args.paper)
    print(json.dumps({"protocol_id": manifest["protocol_id"], "outputs": manifest["outputs"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
