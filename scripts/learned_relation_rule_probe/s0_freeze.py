"""Write and audit the protocol-only S0 artifact for Track A."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from . import protocol


DEFAULT_OUTPUT = protocol.PROJECT_ROOT / "result/learned_relation_rule_probe/S0_freeze"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_head() -> tuple[str | None, str]:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(protocol.PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, "git_metadata_unavailable_or_non_git_source_tree"
    return head, "verified" if head == protocol.BASE_COMMIT else "head_differs_from_frozen_start"


def _holdout_summary() -> dict[str, Any]:
    path = protocol.HOLDOUT_SOURCE
    if not path.exists():
        raise FileNotFoundError(f"missing dormant holdout manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    selected = value.get("selected_dataset_ids", [])
    required = {
        "frozen_before_holdout_outcomes": True,
        "selection_basis": "label_free_dataset_characteristics_only",
        "outcome_features_used": False,
        "historical_ari_used": False,
        "pilot_outcome_used": False,
        "dataset_membership_frozen": True,
    }
    mismatches = {key: value.get(key) for key, expected in required.items() if value.get(key) != expected}
    if mismatches or value.get("status") != "dormant_due_to_adapter_not_estimable" or len(selected) != 12:
        raise ValueError(f"dormant holdout is not eligible for A5 reuse: {mismatches}")
    if set(selected) & set(protocol.ALL_PANEL_DATASETS):
        raise ValueError("A5 holdout overlaps the development/sentinel panel")
    return {
        "manifest_id": value.get("manifest_id"),
        "status": value.get("status"),
        "selected_dataset_ids": selected,
        "selected_count": len(selected),
        "source_sha256": sha256_file(path),
        "source_relpath": str(path.relative_to(protocol.PROJECT_ROOT)),
        "outcome_independent": True,
        "disjoint_from_current_panel": True,
        "future_track_b_disjointness_required": True,
        "used_by_this_project": False,
    }


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    protocol.validate_contract()
    old_report = protocol.PROJECT_ROOT / "reports/relation_selection_probe/DECISION.md"
    old_closed = protocol.PROJECT_ROOT / "reports/relation_selection_probe/README.md"
    if not old_report.exists() or not old_closed.exists():
        raise FileNotFoundError("relation_selection_probe terminal evidence is missing")
    holdout = _holdout_summary()
    head, head_status = _git_head()
    docs = [
        protocol.PROJECT_ROOT / "reports/learned_relation_rule_probe/README.md",
        protocol.PROJECT_ROOT / "reports/learned_relation_rule_probe/PROTOCOL.md",
        protocol.PROJECT_ROOT / "reports/learned_relation_rule_probe/PRE_REGISTRATION.md",
        protocol.PROJECT_ROOT / "reports/learned_relation_rule_probe/S0_FREEZE.md",
        protocol.PROJECT_ROOT / "reports/learned_relation_rule_probe/RESULTS.md",
        protocol.PROJECT_ROOT / "reports/learned_relation_rule_probe/PUBLISH_MANIFEST.md",
        protocol.PROJECT_ROOT / "scripts/learned_relation_rule_probe/protocol.py",
        protocol.PROJECT_ROOT / "scripts/learned_relation_rule_probe/s0_freeze.py",
    ]
    missing = [str(path.relative_to(protocol.PROJECT_ROOT)) for path in docs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"A0 contract files missing: {missing}")
    forbidden_suffixes = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
    preexisting_raw = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.glob("**/*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ) if output_dir.exists() else []
    if preexisting_raw:
        raise RuntimeError(f"S0 output already contains forbidden raw artifacts: {preexisting_raw}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = protocol.resolved_config()
    config["holdout"] = holdout
    config["git_head_at_audit"] = head
    config["git_head_status"] = head_status
    config["formal_performance_run_started"] = False
    write_json(output_dir / "resolved_config.json", config)
    source_manifest = {
        "base_commit": protocol.BASE_COMMIT,
        "git_head_at_audit": head,
        "git_head_status": head_status,
        "files": {
            str(path.relative_to(protocol.PROJECT_ROOT)): sha256_file(path) for path in docs
        },
        "old_project_read_only": True,
        "holdout_manifest": holdout,
    }
    write_json(output_dir / "source_manifest.json", source_manifest)
    audit = {
        "project_id": protocol.PROJECT_ID,
        "stage": "S0",
        "status": "completed_valid",
        "base_commit": protocol.BASE_COMMIT,
        "old_project_terminal_read_only": True,
        "labels_used_during_fit": False,
        "diagnostic_supervision_deployable": False,
        "development_roles_frozen": True,
        "holdout_outcome_independent": holdout["outcome_independent"],
        "holdout_disjoint_from_current_panel": holdout["disjoint_from_current_panel"],
        "future_track_b_disjointness_required": True,
        "burned_primary_datasets": list(protocol.DEVELOPMENT_DATASETS),
        "sentinel_datasets_not_exposed_to_A1_A4": list(protocol.SENTINEL_DATASETS),
        "gpu_pool_valid": True,
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "formal_performance_run_started": False,
        "raw_artifacts_published": False,
        "issues": [],
    }
    decision = {
        "stage": "S0",
        "status": "completed_valid",
        "primary_gate_pass": None,
        "next_stage_authorized": True,
        "authorized_next_stage": "A1",
        "terminal_reason": None,
        "formal_performance_run_started": False,
    }
    run_manifest = {
        "project_id": protocol.PROJECT_ID,
        "stage": "S0",
        "job_count": 0,
        "execution_class": "protocol_freeze_only",
        "status": "completed_valid",
        "incomplete_compute": False,
    }
    write_json(output_dir / "audit.json", audit)
    write_json(output_dir / "decision.json", decision)
    write_json(output_dir / "run_manifest.json", run_manifest)
    hashable = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_hashes.json")
    write_json(
        output_dir / "artifact_hashes.json",
        {
            "project_id": protocol.PROJECT_ID,
            "stage": "S0",
            "algorithm": "sha256_file",
            "files": {path.name: sha256_file(path) for path in hashable},
            "raw_artifacts_included": False,
        },
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
