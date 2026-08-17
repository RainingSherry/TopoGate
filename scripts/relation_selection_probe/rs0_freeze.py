"""Create the immutable RS0 freeze artifact for relation_selection_probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .relation_features import (
    DATASETS,
    FEATURE_FAMILIES,
    HOLDOUT_SEEDS,
    MATERIALITY_DELTA,
    PILOT_SEEDS,
    PRIMARY_DATASETS,
    RS1_DELTA_AP,
    RS1_LIFT,
    RS2_CAPTURE,
    S0_ROOT,
    S1_ROOT,
    VIEW_DIM,
    VIEW_SEEDS,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_REPORT_ROOT = PROJECT_ROOT / "reports/representation_consumer_probe"
OLD_RESULT_ROOT = PROJECT_ROOT / "result/representation_consumer_probe"
OLD_HOLDOUT = OLD_REPORT_ROOT / "STAGE5_HOLDOUT_MANIFEST.json"
DEFAULT_OUTPUT = Path("result/relation_selection_probe/RS0_freeze")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (OLD_REPORT_ROOT / "CLOSED.md").exists():
        raise FileNotFoundError("representation_consumer_probe is not formally closed")
    if not OLD_HOLDOUT.exists():
        raise FileNotFoundError(f"missing inherited holdout manifest: {OLD_HOLDOUT}")
    inherited = output_dir / "inherited_holdout_manifest.json"
    shutil.copyfile(OLD_HOLDOUT, inherited)
    source_hash = _sha256(OLD_HOLDOUT)
    copied_hash = _sha256(inherited)
    if source_hash != copied_hash:
        raise RuntimeError("holdout copy hash mismatch")
    s0_hash_path = S0_ROOT / "artifact_hashes.json"
    s1_hash_path = S1_ROOT / "artifact_hashes.json"
    resolved = {
        "project_id": "relation_selection_probe",
        "protocol_id": "relation_selection_probe_rs0_v1",
        "authorized_stages": ["RS0", "RS1", "RS2", "RS3"],
        "locked_stages": [
            "RS4_learned_selector",
            "GNN",
            "Transformer",
            "TopoCut",
            "DCGC_transplantation",
            "new_reconstruction_objective",
            "hyperparameter_search",
        ],
        "datasets": list(DATASETS),
        "primary_datasets": list(PRIMARY_DATASETS),
        "pilot_seeds": list(PILOT_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "feature_families": {key: list(value) for key, value in FEATURE_FAMILIES.items()},
        "view_seeds": list(VIEW_SEEDS),
        "view_dim": VIEW_DIM,
        "neighbor_k": 20,
        "budget_cap": 8,
        "materiality_delta": MATERIALITY_DELTA,
        "rs1_delta_ap_threshold": RS1_DELTA_AP,
        "rs1_lift_threshold": RS1_LIFT,
        "rs2_capture_threshold": RS2_CAPTURE,
        "old_project": "representation_consumer_probe",
        "old_project_status": "CLOSED",
        "old_artifacts_modified": False,
        "inherited_pre_outcome_holdout": True,
        "holdout_membership_modified": False,
        "labels_used_in_feature_extraction": False,
        "labels_used_in_diagnostic_targets": True,
        "R_O_pool_reference_reuse": True,
        "source_artifacts": {
            "s0_artifact_hash_manifest": str(s0_hash_path),
            "s0_artifact_hash": _sha256(s0_hash_path),
            "s1_artifact_hash_manifest": str(s1_hash_path),
            "s1_artifact_hash": _sha256(s1_hash_path),
            "closed_report": str(OLD_REPORT_ROOT / "CLOSED.md"),
        },
    }
    write_json(output_dir / "resolved_config.json", resolved)
    manifest = {
        "project_id": "relation_selection_probe",
        "stage": "RS0_freeze",
        "protocol_id": "relation_selection_probe_rs0_v1",
        "status": "completed_valid",
        "inherited_holdout_manifest": "inherited_holdout_manifest.json",
        "inherited_holdout_source_sha256": source_hash,
        "inherited_holdout_copy_sha256": copied_hash,
        "membership_modified": False,
        "old_project_closed": True,
        "old_artifacts_modified": False,
        "labels_used_in_feature_extraction": False,
        "labels_used_in_diagnostic_targets": True,
    }
    write_json(output_dir / "rs0_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
