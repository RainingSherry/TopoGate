#!/usr/bin/env python3
"""Validate and freeze the claim-dependent V25 holdout activation manifest.

The preflight may inspect source shape, labels availability for the declared K
boundary, and the frozen adapter.  It never reads ARI, E1 outcomes, or any
performance artifact when choosing or validating a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "result" / "V25_systematic_mechanism_study"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _resolve(path: str, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _load_adapter(source: Path, dataset_id: str, protocol: str) -> tuple[dict[str, Any], Any]:
    # Imports are local so the manifest-only audit remains usable without
    # importing the training model.
    from methods.TopoGate.V21_assignment_adversarial_gate.input_adapter import load_npz, prepare_dual_input

    loaded = load_npz(source)
    prepared = prepare_dual_input(loaded.X, dataset_name=dataset_id, input_protocol=protocol)
    profile = dict(prepared.profile)
    profile.update(
        {
            "input_adapter": "prepare_dual_input",
            "feature_selection": "adapter_default_label_free",
            "normalization": "prepare_dual_input_frozen",
            "graph_input": "X_graph_from_prepare_dual_input",
            "model_input": "X_model_from_prepare_dual_input",
            "labels_used": False,
            "K_used": False,
        }
    )
    return profile, loaded


def preflight(root: Path, datasets: list[str], explicit_k: dict[str, int]) -> dict[str, Any]:
    a2_path = root / "A2" / "A2_decision.json"
    if not a2_path.is_file():
        raise ValueError("A2/A2_decision.json is required before holdout preflight")
    a2 = _read_json(a2_path)
    if a2.get("decision") != "retain_e1":
        raise ValueError(f"holdout preflight is vetoed unless A2 decision is retain_e1; got {a2.get('decision')!r}")
    claim_path = root / "PhaseC" / "FROZEN_PAPER_CLAIM.json"
    if not claim_path.is_file():
        raise ValueError("PhaseC/FROZEN_PAPER_CLAIM.json is required before holdout preflight")
    claim = _read_json(claim_path)
    candidates_payload = _read_json(root / "A2" / "holdout_candidate_manifest.json")
    candidates = {str(row["dataset_id"]): row for row in candidates_payload.get("candidates", []) if row.get("holdout_eligible") is True}
    if not datasets:
        raise ValueError("provide an explicit, predeclared --dataset list; the preflight never selects candidates automatically")
    unknown = sorted(set(datasets) - set(candidates))
    if unknown:
        raise ValueError(f"datasets are not eligible frozen candidates: {unknown}")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset_id in datasets:
        entry = candidates[dataset_id]
        source = _resolve(str(entry["source_path"]), ROOT)
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "domain": entry.get("domain"),
            "source_path": str(source.resolve()),
            "manifest_source_hash": entry.get("source_hash"),
            "input_protocol": entry.get("input_protocol"),
            "selection_uses_labels_or_outcomes": entry.get("outcome_selection_declared", False),
        }
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            current_hash = _sha256(source)
            if entry.get("source_hash") in (None, "", "unavailable"):
                raise ValueError("candidate has no auditable source hash")
            if current_hash != entry["source_hash"]:
                raise ValueError("source hash differs from the frozen A2 manifest")
            if entry.get("outcome_selection_declared"):
                raise ValueError("candidate manifest declares outcome-dependent selection")
            profile, loaded = _load_adapter(source, dataset_id, str(entry["input_protocol"]))
            if loaded.labels is not None:
                k = int(np.unique(np.asarray(loaded.labels)).size)
                k_source = "benchmark_oracle_from_y"
            elif dataset_id in explicit_k:
                k = int(explicit_k[dataset_id])
                k_source = "explicit_n_clusters"
            else:
                raise ValueError("unlabelled candidate requires an explicit K mapping")
            if k <= 1:
                raise ValueError("K must be greater than one")
            row.update(
                {
                    "current_source_sha256": current_hash,
                    "adapter_profile": profile,
                    "input_adapter": profile.get("input_adapter"),
                    "feature_selection": profile.get("feature_selection"),
                    "normalization": profile.get("normalization"),
                    "max_features": profile.get("max_features", entry.get("max_features", "adapter_default")),
                    "graph_input": profile.get("graph_input"),
                    "model_input": profile.get("model_input"),
                    "n_samples": int(profile["n_samples"]),
                    "n_features_original": int(profile["n_features_original"]),
                    "n_features_selected": int(profile["n_features_selected"]),
                    "n_clusters": k,
                    "K_source": k_source,
                    "labels_loaded_outer_only": loaded.labels is not None,
                    "adapter_valid": True,
                    "preflight_status": "valid",
                }
            )
            rows.append(row)
        except Exception as exc:  # retain a structured failure, not a silent exclusion
            row.update({"adapter_valid": False, "preflight_status": "invalid_design", "error": f"{type(exc).__name__}: {exc}"})
            failures.append(row)
    if failures:
        raise ValueError(json.dumps({"invalid_candidates": failures}, ensure_ascii=False))
    payload = {
        "protocol_id": "v25_holdout_preflight_v1",
        "claim_freeze_path": str(claim_path.resolve()),
        "claim_freeze_sha256": _sha256(claim_path),
        "claim_family": claim.get("claim_family"),
        "primary_endpoint": claim.get("primary_endpoint"),
        "activation_subset": claim.get("activation_subset"),
        "selection_policy": {
            "dataset_list_supplied_before_holdout_outcomes": True,
            "selection_uses_labels_or_outcomes": False,
            "candidate_selection": "explicit_dataset_ids_from_A2_frozen_manifest",
        },
        "measurement_schema_frozen": True,
        "adapter_contract_frozen": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": rows,
    }
    out = root / "PhaseD"
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_activation_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", action="append", required=True, help="eligible A2 dataset_id; repeat in frozen order")
    parser.add_argument("--n-clusters", action="append", default=[], metavar="DATASET_ID=K", help="required only for unlabelled candidates")
    args = parser.parse_args()
    explicit: dict[str, int] = {}
    for raw in args.n_clusters:
        dataset_id, value = raw.split("=", 1)
        if dataset_id in explicit:
            raise ValueError(f"duplicate explicit K for {dataset_id}")
        explicit[dataset_id] = int(value)
    payload = preflight(args.root, args.dataset, explicit)
    print(json.dumps({"status": "valid", "claim_family": payload["claim_family"], "datasets": [row["dataset_id"] for row in payload["datasets"]]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
