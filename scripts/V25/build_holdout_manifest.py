#!/usr/bin/env python3
"""Build the claim-dependent V25 holdout manifest.

The frozen claim chooses the measurement subset, not the dataset outcome.  E1
claims use the shared N/R/T panel; the local-to-global claim uses a matched
pair and the frozen E3 metrics.  This command only writes a manifest and never
launches training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "result" / "V25_systematic_mechanism_study"
SEEDS = (42, 123, 7)

CLAIM_SCHEMAS: dict[str, dict[str, Any]] = {
    "selection": {
        "activation_subset": ["E1_NRT"],
        "arms": ["N", "R", "T"],
        "primary_endpoint": "S_full_ARI = ARI_T - ARI_R",
        "primary_endpoint_key": "S_full_ARI",
        "required_metrics": ["ARI"],
    },
    "generic_intervention": {
        "activation_subset": ["E1_NRT"],
        "arms": ["N", "R", "T"],
        "primary_endpoint": "I_full_ARI = ARI_R - ARI_N",
        "primary_endpoint_key": "I_full_ARI",
        "required_metrics": ["ARI"],
    },
    "objective_compatibility": {
        "activation_subset": ["E1_NRT", "E2-B", "E2-C"],
        "arms": ["N", "R", "T"],
        "primary_endpoint": "sign(S_1step_ARI) = sign(S_full_ARI)",
        "primary_endpoint_key": "objective_sign_agreement",
        "required_metrics": ["ARI", "gradient_geometry", "actual_Adam_one_step"],
    },
    "local_global": {
        "activation_subset": ["E3_frozen_matched_pair"],
        "arms": ["matched_pair"],
        "primary_endpoint": "1[delta_kNN_purity > 0 and delta_ARI <= 0]",
        "primary_endpoint_key": "local_positive_and_global_nonpositive",
        "required_metrics": ["frozen_local_metric", "ARI"],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_manifest(root: Path) -> dict[str, Any]:
    a2_path = root / "A2" / "A2_decision.json"
    if not a2_path.is_file():
        raise ValueError("A2/A2_decision.json is required before building a claim-dependent holdout manifest")
    a2 = _read(a2_path)
    if a2.get("decision") != "retain_e1":
        raise ValueError(f"claim-dependent holdout is vetoed unless A2 decision is retain_e1; got {a2.get('decision')!r}")
    claim_path = root / "PhaseC" / "FROZEN_PAPER_CLAIM.json"
    activation_path = root / "PhaseD" / "holdout_activation_manifest.json"
    if not claim_path.is_file() or not activation_path.is_file():
        raise ValueError("PhaseC claim freeze and PhaseD adapter preflight are required")
    claim = _read(claim_path)
    activation = _read(activation_path)
    claim_family = str(claim.get("claim_family"))
    schema = CLAIM_SCHEMAS.get(claim_family)
    if schema is None:
        raise ValueError(f"unsupported frozen claim family: {claim_family}")
    claim_hash = _sha256(claim_path)
    if activation.get("claim_freeze_sha256") != claim_hash:
        raise ValueError("adapter preflight is not bound to the current frozen claim")
    if list(claim.get("activation_subset", [])) != list(schema["activation_subset"]):
        raise ValueError("frozen claim activation subset differs from the predeclared schema")
    rows = activation.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("adapter preflight has no valid holdout datasets")

    jobs: list[dict[str, Any]] = []
    for row in rows:
        if row.get("preflight_status") != "valid" or row.get("adapter_valid") is not True:
            raise ValueError(f"invalid adapter-preflight row: {row}")
        dataset_id = str(row["dataset_id"])
        for seed in SEEDS:
            panel_key = f"v25_holdout::{claim_family}::{dataset_id}::{seed}"
            panel = {
                "panel_run_key": panel_key,
                "phase": "holdout",
                "claim_family": claim_family,
                "dataset": dataset_id,
                "domain": row.get("domain"),
                "input_protocol": row.get("input_protocol"),
                "input_adapter": row.get("input_adapter"),
                "feature_selection": row.get("feature_selection"),
                "normalization": row.get("normalization"),
                "max_features": row.get("max_features"),
                "graph_input": row.get("graph_input"),
                "model_input": row.get("model_input"),
                "source_path": row.get("source_path"),
                "source_sha256": row.get("current_source_sha256"),
                "seed": int(seed),
                "arms": list(schema["arms"]),
                "primary_endpoint": schema["primary_endpoint"],
                "primary_endpoint_key": schema["primary_endpoint_key"],
                "required_metrics": list(schema["required_metrics"]),
                "K_source": row.get("K_source"),
                "n_clusters": row.get("n_clusters"),
                "labels_used_during_fit": False,
                "selection_uses_labels_or_outcomes": False,
                "output_dir": str((root / "PhaseD" / "holdout" / claim_family / dataset_id.replace(" ", "_") / f"seed{seed}").resolve()),
                "status": "queued_manifest_only",
            }
            for arm in schema["arms"]:
                jobs.append(panel | {"run_key": f"{panel_key}::{arm}", "arm": arm, "execution_unit": "claim_dependent_holdout"})
    return {
        "manifest_id": "v25_holdout_activation_manifest_v1",
        "protocol_id": "v25_holdout_claim_dependent_v1",
        "phase": "holdout",
        "claim_freeze_sha256": claim_hash,
        "claim_family": claim_family,
        "primary_endpoint": schema["primary_endpoint"],
        "primary_endpoint_key": schema["primary_endpoint_key"],
        "activation_subset": list(schema["activation_subset"]),
        "arms": list(schema["arms"]),
        "measurement_schema": schema,
        "generated_without_holdout_outcomes": True,
        "selection_uses_labels_or_outcomes": False,
        "seeds": list(SEEDS),
        "expected_panel_jobs": len(rows) * len(SEEDS),
        "expected_arm_jobs": len(jobs),
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = build_manifest(args.root)
    output = args.out or (args.root / "PhaseD" / "holdout_activation_manifest_claim_dependent.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "claim_family": payload["claim_family"], "arms": payload["arms"], "panel_jobs": payload["expected_panel_jobs"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
