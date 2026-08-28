#!/usr/bin/env python3
"""Freeze one V25 paper claim and its primary holdout endpoint.

Claim selection is intentionally explicit.  This utility never chooses a claim
from the most favorable result and never edits the A2 measurement schema.  It
only records the author's selected, already-predeclared claim family together
with the evidence paths available at the time of the freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "result" / "V25_systematic_mechanism_study"

CLAIMS: dict[str, dict[str, Any]] = {
    "selection": {
        "claim_id": "selection_conditional_utility",
        "primary_endpoint": "S_full_ARI = ARI_T - ARI_R",
        "endpoint_key": "S_full_ARI",
        "activation_subset": ["E1_NRT"],
        "required_evidence": ["E1_full"],
        "falsifier": "No predeclared holdout dataset has a seed-stable material S_full_ARI effect, or the T/R matching audit fails.",
        "allowed_wording": "Topology-dependent selection has conditional incremental utility in the audited V21 case study; this is not a universal population claim.",
        "secondary_metrics": ["I_full_ARI", "S_1step_ARI", "NMI", "E2-A feature summaries", "E2-B gradient geometry"],
    },
    "generic_intervention": {
        "claim_id": "generic_intervention_effect",
        "primary_endpoint": "I_full_ARI = ARI_R - ARI_N",
        "endpoint_key": "I_full_ARI",
        "activation_subset": ["E1_NRT"],
        "required_evidence": ["E1_full"],
        "falsifier": "The predeclared holdout shows only observed-small I_full_ARI effects across all eligible datasets.",
        "allowed_wording": "The utility of the matched assignment intervention is separable from topology selection in the audited protocol; no universal causal claim is made.",
        "secondary_metrics": ["S_full_ARI", "I_1step_ARI", "NMI"],
    },
    "objective_compatibility": {
        "claim_id": "objective_compatibility",
        "primary_endpoint": "sign(S_1step_ARI) = sign(S_full_ARI)",
        "endpoint_key": "objective_sign_agreement",
        "activation_subset": ["E1_NRT", "E2-B", "E2-C"],
        "required_evidence": ["E1_full", "E2-B", "E2-C"],
        "falsifier": "The frozen holdout does not reproduce the predeclared one-step/full sign relationship.",
        "allowed_wording": "The observed selection effect is consistent with the frozen objective-compatibility diagnostic in this protocol; the one-step probe is not a causal proof.",
        "secondary_metrics": ["I_full_ARI", "S_full_ARI", "gradient cosine and norm ratios"],
    },
    "local_global": {
        "claim_id": "local_to_global_disconnect",
        "primary_endpoint": "1[delta_kNN_label_purity > 0 and delta_ARI <= 0]",
        "endpoint_key": "local_positive_and_global_nonpositive",
        "activation_subset": ["E3_frozen_matched_pair"],
        "required_evidence": ["E3"],
        "falsifier": "All eligible artifact-complete cases with positive local change also have positive global ARI change.",
        "allowed_wording": "Post-hoc local geometry improvement can fail to convert into global clustering gain in the audited cases.",
        "secondary_metrics": ["label-free neighborhood stability", "spectral gap", "NMI"],
    },
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


def _evidence_files(root: Path, confirmation_root: Path | None) -> list[dict[str, str]]:
    paths = [
        root / "A0" / "registry_summary.json",
        root / "A1" / "a1_summary.json",
        root / "A2" / "A2_decision.json",
        root / "A2" / "measurement_schema.json",
    ]
    if confirmation_root is not None:
        for candidate in (confirmation_root / "Audit" / "phase_summary.json", confirmation_root / "queue_state.json"):
            if candidate.is_file():
                paths.append(candidate)
    evidence: list[dict[str, str]] = []
    for path in paths:
        if path.is_file():
            evidence.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    return evidence


def freeze_claim(root: Path, claim_family: str, confirmation_root: Path | None, note: str = "") -> dict[str, Any]:
    if claim_family not in CLAIMS:
        raise ValueError(f"unknown claim family {claim_family!r}; choose one of {sorted(CLAIMS)}")
    a2 = _read_json(root / "A2" / "A2_decision.json")
    schema = _read_json(root / "A2" / "measurement_schema.json")
    if a2.get("decision") != "retain_e1":
        raise ValueError("claim freeze requires A2=retain_e1; no prospective claim may be frozen after a veto")
    claim = dict(CLAIMS[claim_family])
    schema_activation = schema.get("claim_activation", {})
    schema_key = {"selection": "selection", "generic_intervention": "generic_intervention", "objective_compatibility": "objective_compatibility", "local_global": "local_global"}[claim_family]
    expected = schema_activation.get(schema_key)
    if expected is None:
        raise ValueError(f"A2 schema has no activation entry for {schema_key}")
    if list(expected) != list(claim["activation_subset"]):
        raise ValueError(f"claim activation does not match A2 schema for {claim_family}")
    existing_path = root / "PhaseC" / "FROZEN_PAPER_CLAIM.json"
    if existing_path.is_file():
        existing = _read_json(existing_path)
        if existing.get("claim_family") != claim_family or existing.get("primary_endpoint_key") != claim["endpoint_key"]:
            raise ValueError(
                "a paper claim is already frozen; changing claim family or primary endpoint is forbidden"
            )
        # Idempotent re-entry is useful for audits, but it must not refresh the
        # timestamp or replace evidence with a more favorable post-hoc bundle.
        return existing
    frozen_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "protocol_id": "v25_claim_freeze_v1",
        "study": "V25_systematic_mechanism_study",
        "claim_family": claim_family,
        "claim": claim,
        "primary_endpoint": claim["primary_endpoint"],
        "primary_endpoint_key": claim["endpoint_key"],
        "delta_threshold": schema.get("delta_threshold"),
        "threshold_sensitivity": schema.get("threshold_sensitivity", []),
        "activation_subset": claim["activation_subset"],
        "secondary_metrics": claim["secondary_metrics"],
        "falsifier": claim["falsifier"],
        "allowed_wording": claim["allowed_wording"],
        "selection_mode": "explicit_author_choice_after_predeclared_A2_schema; no automatic_best_result_selection",
        "frozen_at": frozen_at,
        "author_note": note,
        "evidence": _evidence_files(root, confirmation_root),
        "holdout_rule": "activate exactly the listed subset and endpoint; do not substitute a secondary metric",
    }
    out_dir = root / "PhaseC"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "FROZEN_PAPER_CLAIM.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = [
        "# V25 Frozen Paper Claim",
        "",
        f"- Claim family: `{claim_family}`",
        f"- Primary endpoint: `{claim['primary_endpoint']}`",
        f"- Activation subset: `{', '.join(claim['activation_subset'])}`",
        f"- Delta threshold: `{schema.get('delta_threshold')}`; sensitivity: `{schema.get('threshold_sensitivity', [])}`",
        f"- Frozen at: `{frozen_at}`",
        "",
        "## Frozen Wording",
        "",
        claim["allowed_wording"],
        "",
        "## Falsifier",
        "",
        claim["falsifier"],
        "",
        "## Governance",
        "",
        "This file freezes one predeclared endpoint for holdout validation. Secondary metrics cannot replace it, and an unattractive result cannot reopen claim selection or create V26.",
        "",
    ]
    (out_dir / "FROZEN_PAPER_CLAIM.md").write_text("\n".join(markdown), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--claim-family", choices=sorted(CLAIMS), required=True)
    parser.add_argument("--confirmation-root", type=Path, default=None)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    payload = freeze_claim(args.root, args.claim_family, args.confirmation_root, args.note)
    print(json.dumps({"status": "frozen", "claim_family": payload["claim_family"], "primary_endpoint": payload["primary_endpoint"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
