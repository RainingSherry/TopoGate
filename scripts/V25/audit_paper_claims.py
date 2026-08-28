#!/usr/bin/env python3
"""Audit paper-facing V25 claims against the frozen evidence bundle.

This audit is deliberately narrower than the experiment contract audit.  It does
not run a model or infer a population effect.  It re-reads the frozen A0/A1/A2,
E1, Phase C/E, and PaperEvidence artifacts, recomputes the E1 state labels from
the saved seed values, and emits a claim ledger with explicit scope boundaries.
The output is suitable for drafting and for a later zero-context paper-to-
evidence audit once a manuscript exists.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


DELTA = 0.03
REQUIRED_SEEDS = {42, 123, 7}
EXPECTED_A0 = {
    "v1_v22_rows": 2209,
    "v1_v22_paired_rows": 1637,
    "v1_v22_units": 431,
    "v23_v24_boundary_records": 2,
}
EXPECTED_A1 = {
    "paired_rows": 1637,
    "positive_rows": 194,
    "negative_rows": 680,
    "small_rows": 763,
}
EXPECTED_E1_DATASETS = {"Baron Human", "Campbell", "hate_speech"}
DRAFT_ANCHORS = (
    "2,209",
    "1,637",
    "431",
    "194",
    "680",
    "763",
    "+0.044617",
    "-0.065332",
    "-0.033410",
    "0/6",
    "inconclusive_not_completed",
)
DRAFT_FORBIDDEN_DIRECT = (
    "structural quality causes clustering harm",
    "topology consistently improves clustering",
    "the objective-conflict mechanism is proven",
    "local geometry improvement is sufficient in general",
    "holdout failed as a model-performance conclusion",
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    if value in (None, "", "NA", "None", "null"):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _expected_state(values: list[float], *, delta: float = DELTA) -> str:
    if len(values) != 3:
        return "Inconclusive"
    mean = sum(values) / len(values)
    same_sign_count = sum(value > 0 for value in values) if mean >= 0 else sum(value < 0 for value in values)
    if mean > delta and same_sign_count >= 2:
        return "Positive"
    if mean < -delta and same_sign_count >= 2:
        return "Negative"
    if abs(mean) <= delta and all(abs(value) < delta for value in values):
        return "Observed-Small"
    return "Inconclusive"


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_tex_with_inputs(path: Path, seen: set[Path] | None = None) -> str:
    """Read a LaTeX root plus local input files for source-level audits."""
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


def _anchor_present(text: str, anchor: str) -> bool:
    if anchor in text:
        return True
    if anchor == "0/6":
        return "zero of six" in text.lower()
    if anchor == "inconclusive_not_completed":
        return "inconclusive\\_not\\_completed" in text or "inconclusive not completed" in text.lower()
    return False


def _claim_rows(
    *,
    a0: dict[str, Any],
    a1: dict[str, Any],
    e1_rows: list[dict[str, str]],
    closure: dict[str, Any],
    source_root: Path,
) -> list[dict[str, Any]]:
    effects = {
        (row["dataset"], row["metric"]): row
        for row in e1_rows
    }
    e1_evidence = _relative(source_root, source_root / "PaperEvidence/e1_dataset_effects.csv")
    closure_evidence = _relative(source_root, source_root / "PhaseE/closure.json")
    return [
        {
            "claim_id": "C1",
            "claim": "V1-V22 form an observational atlas of heterogeneous structural-intervention outcomes.",
            "status": "supported_with_observational_scope",
            "evidence": _relative(source_root, source_root / "PaperEvidence/paper_evidence_summary.json"),
            "statistical_unit": "dataset/protocol/readout; seeds and variants are repeated records",
            "causal_status": "observational",
            "numeric_anchor": f"{a0['v1_v22_rows']} registry rows; {a0['v1_v22_paired_rows']} paired rows; {a0['v1_v22_units']} units; {a1['positive_rows']} positive, {a1['negative_rows']} negative, {a1['small_rows']} observed-small",
            "allowed_wording": "observational atlas of heterogeneous outcomes",
            "forbidden_wording": "structural quality causes harm; universal structural failure; pooled causal effect",
        },
        {
            "claim_id": "C2",
            "claim": "Topology-dependent selection has a conditional incremental effect in the audited V21 case study.",
            "status": "supported_with_case_study_scope",
            "evidence": e1_evidence,
            "statistical_unit": "dataset; three seeds are repeated measurements",
            "causal_status": "matched prospective case study",
            "numeric_anchor": "; ".join(
                f"{dataset}: S_d={effects[(dataset, 'S_d')]['mean']} ({effects[(dataset, 'S_d')]['state']})"
                for dataset in sorted(EXPECTED_E1_DATASETS)
            ),
            "allowed_wording": "dataset-conditional effect in the audited V21 case study",
            "forbidden_wording": "universal topology superiority; population effect; independent replication",
        },
        {
            "claim_id": "C3",
            "claim": "E2/E3 provide localization diagnostics, not a universal causal explanation.",
            "status": "diagnostic_only",
            "evidence": _relative(source_root, source_root / "PaperEvidence/paper_evidence_summary.json"),
            "statistical_unit": "dataset x seed for E2; isolated boundary rows for E3",
            "causal_status": "diagnostic/post-hoc",
            "numeric_anchor": "E2 coordinate distributions are descriptive; post-hoc label-aware metrics are not fit inputs",
            "allowed_wording": "diagnostic/localization evidence",
            "forbidden_wording": "proven objective-conflict law; proven local-to-global theorem",
        },
        {
            "claim_id": "C4",
            "claim": "Independent holdout replication was not established.",
            "status": "inconclusive_not_completed",
            "evidence": closure_evidence,
            "statistical_unit": "six predeclared holdout panels; zero evaluable panels",
            "causal_status": "incomplete_compute",
            "numeric_anchor": f"expected={closure['independent_holdout']['expected_panel_count']}; completed={closure['independent_holdout']['completed_panel_count']}; status={closure['independent_holdout']['status']}",
            "allowed_wording": "independent replication not established; holdout inconclusive_not_completed",
            "forbidden_wording": "holdout negative result; replicated on held-out datasets",
        },
    ]


def audit_paper_claims(v25_root: Path, draft_path: Path | None = None) -> dict[str, Any]:
    """Return an evidence-bound paper claim audit without writing files."""
    evidence = v25_root / "PaperEvidence"
    a0 = _read_json(v25_root / "A0/registry_summary.json")
    a1 = _read_json(v25_root / "A1/a1_summary.json")
    a2 = _read_json(v25_root / "A2/A2_decision.json")
    claim = _read_json(v25_root / "PhaseC/FROZEN_PAPER_CLAIM.json")
    phase = _read_json(v25_root / "E1/confirmation/Audit/phase_summary.json")
    closure = _read_json(v25_root / "PhaseE/closure.json")
    paper_summary = _read_json(evidence / "paper_evidence_summary.json")
    scope = _read_json(evidence / "claim_scope_audit.json")
    e1_rows = _read_csv(evidence / "e1_dataset_effects.csv")

    checks: dict[str, bool] = {}
    checks.update({f"a0_{key}": a0.get(key) == value for key, value in EXPECTED_A0.items()})
    checks.update({f"a1_{key}": a1.get(key) == value for key, value in EXPECTED_A1.items()})
    checks["a0_a1_summary_reconciles"] = (
        paper_summary.get("retrospective", {}).get("a0", {}).get("v1_v22_rows") == a0.get("v1_v22_rows")
        and paper_summary.get("retrospective", {}).get("a1", {}).get("paired_rows") == a1.get("paired_rows")
    )
    checks["a1_observational_boundary"] = a1.get("no_causal_claim") is True and a1.get("label_free_evaluation") is False
    checks["a2_retained_without_e4"] = a2.get("decision") == "retain_e1" and a2.get("no_new_e4") is True
    checks["claim_endpoint_frozen"] = (
        claim.get("claim_family") == "selection"
        and claim.get("primary_endpoint_key") == "S_full_ARI"
        and claim.get("activation_subset") == ["E1_NRT"]
        and claim.get("delta_threshold") == DELTA
    )
    checks["e1_complete_audited_phase"] = (
        phase.get("panel_count") == 9
        and phase.get("audit_ok_count") == 9
        and set(phase.get("datasets", {})) == EXPECTED_E1_DATASETS
        and phase.get("equivalence_claim") is False
    )
    checks["e1_effect_rows_complete"] = len(e1_rows) == 6 and {(row.get("dataset"), row.get("metric")) for row in e1_rows} == {
        (dataset, metric) for dataset in EXPECTED_E1_DATASETS for metric in ("I_d", "S_d")
    }
    state_checks: list[dict[str, Any]] = []
    for dataset, payload in sorted(phase.get("datasets", {}).items()):
        seeds = set(payload.get("seeds", []))
        for metric in ("I_d", "S_d"):
            effect = payload.get(metric, {})
            values = [_number(value) for value in effect.get("seed_values", [])]
            values = [value for value in values if value is not None]
            row = next(item for item in e1_rows if item.get("dataset") == dataset and item.get("metric") == metric)
            raw_mean = _number(row.get("mean"))
            state_checks.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "mean_matches_phase": raw_mean is not None and abs(raw_mean - float(effect.get("mean"))) < 1e-12,
                    "state_matches_seed_rule": row.get("state") == _expected_state(values),
                    "seed_set_complete": seeds == REQUIRED_SEEDS and len(values) == 3,
                }
            )
    checks["e1_effect_states_recomputed"] = all(
        item["mean_matches_phase"] and item["state_matches_seed_rule"] and item["seed_set_complete"]
        for item in state_checks
    )
    checks["holdout_firewall"] = (
        closure.get("independent_holdout", {}).get("status") == "inconclusive_not_completed"
        and closure.get("independent_holdout", {}).get("primary_endpoint_evaluable") is False
        and closure.get("independent_holdout", {}).get("performance_result") is False
        and scope.get("checks", {}).get("holdout_not_used_as_negative_result") is True
    )
    checks["paper_scope_audit_passes"] = scope.get("audit_ok") is True

    draft_audit: dict[str, Any] = {
        "path": str(draft_path) if draft_path is not None else None,
        "checked": draft_path is not None,
        "missing_anchors": [],
        "forbidden_direct_phrases": [],
    }
    if draft_path is not None:
        draft_text = _read_tex_with_inputs(draft_path) if draft_path.suffix == ".tex" else draft_path.read_text(encoding="utf-8")
        draft_audit["missing_anchors"] = [anchor for anchor in DRAFT_ANCHORS if not _anchor_present(draft_text, anchor)]
        draft_audit["forbidden_direct_phrases"] = [phrase for phrase in DRAFT_FORBIDDEN_DIRECT if phrase in draft_text]
        checks["draft_numeric_anchors"] = not draft_audit["missing_anchors"]
        checks["draft_direct_scope_firewall"] = not draft_audit["forbidden_direct_phrases"]

    source_paths = [
        v25_root / "A0/registry_summary.json",
        v25_root / "A1/a1_summary.json",
        v25_root / "A2/A2_decision.json",
        v25_root / "PhaseC/FROZEN_PAPER_CLAIM.json",
        v25_root / "E1/confirmation/Audit/phase_summary.json",
        evidence / "e1_dataset_effects.csv",
        evidence / "paper_evidence_summary.json",
        evidence / "claim_scope_audit.json",
        v25_root / "PhaseE/closure.json",
    ]
    sources = {
        str(path.relative_to(v25_root.parent.parent)): {"sha256": _sha256(path), "exists": path.is_file()}
        for path in source_paths
    }
    return {
        "protocol_id": "v25_paper_claim_audit_v1",
        "generated_at": _now(),
        "review_independence": "deterministic_same_workspace",
        "acceptance_status": "provisional",
        "status": "audit_ok" if all(checks.values()) else "audit_failed",
        "checks": checks,
        "state_checks": state_checks,
        "draft_audit": draft_audit,
        "claim_rows": _claim_rows(a0=a0, a1=a1, e1_rows=e1_rows, closure=closure, source_root=v25_root),
        "sources": sources,
        "scope_firewall": {
            "allowed": scope.get("allowed_claim_scope", []),
            "forbidden": scope.get("forbidden_claim_scope", []),
            "holdout_status": closure.get("independent_holdout", {}).get("status"),
            "known_k_boundary": "E1 is a real-ground-truth, benchmark-known-K evaluation; not fully label-free fitting.",
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(audit: dict[str, Any]) -> str:
    checks = audit["checks"]
    lines = [
        "# V25 Paper Claim Audit",
        "",
        f"**Status:** `{audit['status']}`  ",
        f"**Protocol:** `{audit['protocol_id']}`  ",
        f"**Generated:** `{audit['generated_at']}`  ",
        "**Scope:** deterministic recheck of frozen artifacts; not an external reviewer verdict.",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in checks.items())
    lines.extend(["", "## Claim Ledger", "", "| ID | Status | Evidence | Allowed wording |", "|---|---|---|---|"])
    for row in audit["claim_rows"]:
        lines.append(f"| {row['claim_id']} | `{row['status']}` | `{row['evidence']}` | {row['allowed_wording']} |")
    lines.extend(
        [
            "",
            "## Scope Firewall",
            "",
            "- E1 is a real-ground-truth, benchmark-known-K evaluation, not fully label-free fitting.",
            "- E2/E3 remain diagnostic/post-hoc evidence; coordinate counts are not inferential sample sizes.",
            "- The holdout is `inconclusive_not_completed`, not a negative performance result.",
            "- No universal topology-superiority or pooled historical causal claim is permitted.",
            "",
            "See `V25_PAPER_CLAIM_AUDIT.json` and `V25_PAPER_CLAIM_LEDGER.csv` for machine-readable details.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v25-root", type=Path, default=Path("result/V25_systematic_mechanism_study"))
    parser.add_argument("--output-dir", type=Path, default=Path("review-stage"))
    parser.add_argument("--draft", type=Path, default=None, help="optional manuscript draft to check for anchors and direct overclaims")
    args = parser.parse_args()
    audit = audit_paper_claims(args.v25_root, args.draft)
    _write_json(args.output_dir / "V25_PAPER_CLAIM_AUDIT.json", audit)
    _write_csv(args.output_dir / "V25_PAPER_CLAIM_LEDGER.csv", audit["claim_rows"])
    (args.output_dir / "V25_PAPER_CLAIM_AUDIT.md").write_text(_markdown(audit), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "checks": len(audit["checks"]), "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0 if audit["status"] == "audit_ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
