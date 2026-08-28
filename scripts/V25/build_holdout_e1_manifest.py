#!/usr/bin/env python3
"""Build a claim-dependent, outcome-independent E1 holdout manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "result" / "V25_systematic_mechanism_study"
SEEDS = (42, 123, 7)
ARMS = ("N", "R", "T")
PROTOCOL_ID = "v25_e1_v21_matched_nrt_v1"


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
        raise ValueError("A2/A2_decision.json is required before building holdout E1")
    a2 = _read(a2_path)
    if a2.get("decision") != "retain_e1":
        raise ValueError(f"holdout E1 is vetoed unless A2 decision is retain_e1; got {a2.get('decision')!r}")
    phase_manifest = root / "PhaseD" / "holdout_activation_manifest.json"
    claim_path = root / "PhaseC" / "FROZEN_PAPER_CLAIM.json"
    if not phase_manifest.is_file() or not claim_path.is_file():
        raise ValueError("PhaseC freeze and PhaseD preflight are required before building holdout E1")
    phase = _read(phase_manifest)
    claim = _read(claim_path)
    if phase.get("claim_freeze_sha256") != _sha256(claim_path):
        raise ValueError("PhaseD manifest does not match the current frozen claim")
    if phase.get("activation_subset") != ["E1_NRT"] and "E1_NRT" not in list(phase.get("activation_subset", [])):
        raise ValueError("the frozen claim does not activate E1_NRT; no three-arm holdout manifest is permitted")
    rows = phase.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("PhaseD preflight has no valid datasets")
    jobs: list[dict[str, Any]] = []
    for row in rows:
        if row.get("preflight_status") != "valid" or row.get("adapter_valid") is not True:
            raise ValueError(f"invalid PhaseD dataset row: {row}")
        for seed in SEEDS:
            panel_key = f"v25_e1_holdout::{row['dataset_id']}::{seed}"
            panel = {
                "run_key": panel_key,
                "panel_run_key": panel_key,
                "phase": "holdout",
                "dataset": str(row["dataset_id"]),
                "input_protocol": str(row["input_protocol"]),
                "input_adapter": row.get("input_adapter"),
                "feature_selection": row.get("feature_selection"),
                "normalization": row.get("normalization"),
                "max_features": row.get("max_features"),
                "graph_input": row.get("graph_input"),
                "model_input": row.get("model_input"),
                "source_path": str(row["source_path"]),
                "source_sha256": str(row["current_source_sha256"]),
                "seed": int(seed),
                "arms": list(ARMS),
                "primary_readout": "clean_embedding_known_k_kmeans",
                "K_source": row.get("K_source"),
                "n_clusters": row.get("n_clusters"),
                "labels_used_during_fit": False,
                "selection_uses_labels_or_outcomes": False,
                "output_dir": str((root / "PhaseD" / "E1" / str(row["dataset_id"]).replace(" ", "_") / f"seed{seed}").resolve()),
                "status": "queued_manifest_only",
            }
            for arm in ARMS:
                jobs.append(panel | {"run_key": f"{panel_key}::{arm}", "arm": arm, "execution_unit": "shared_three_arm_panel"})
    return {
        "manifest_id": "v25_holdout_e1_manifest_v1",
        "protocol_id": PROTOCOL_ID,
        "a2_decision": "retain_e1",
        "phase": "holdout",
        "claim_freeze_sha256": _sha256(claim_path),
        "claim_family": claim.get("claim_family"),
        "generated_without_e1_holdout_outcomes": True,
        "generated_without_e1_outcomes": True,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
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
    out = args.out or (args.root / "PhaseD" / "holdout_e1_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen", "manifest_id": payload["manifest_id"], "panel_jobs": payload["expected_panel_jobs"], "arm_jobs": payload["expected_arm_jobs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
