"""Write and audit the protocol-only S0 artifact for Track B."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from . import protocol


DEFAULT_OUTPUT = protocol.PROJECT_ROOT / "result/adaptive_corruption_probe/S0_freeze"


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


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    protocol.validate_contract()
    required_paths = [
        protocol.PROJECT_ROOT / "reports/representation_consumer_probe/CLOSED.md",
        protocol.PROJECT_ROOT / "reports/relation_selection_probe/DECISION.md",
    ]
    missing_old = [str(path.relative_to(protocol.PROJECT_ROOT)) for path in required_paths if not path.exists()]
    if missing_old:
        raise FileNotFoundError(f"old terminal projects missing: {missing_old}")
    docs = [
        protocol.PROJECT_ROOT / "reports/adaptive_corruption_probe/README.md",
        protocol.PROJECT_ROOT / "reports/adaptive_corruption_probe/PROTOCOL.md",
        protocol.PROJECT_ROOT / "reports/adaptive_corruption_probe/PRE_REGISTRATION.md",
        protocol.PROJECT_ROOT / "reports/adaptive_corruption_probe/S0_FREEZE.md",
        protocol.PROJECT_ROOT / "reports/adaptive_corruption_probe/RESULTS.md",
        protocol.PROJECT_ROOT / "reports/adaptive_corruption_probe/PUBLISH_MANIFEST.md",
        protocol.PROJECT_ROOT / "scripts/adaptive_corruption_probe/protocol.py",
        protocol.PROJECT_ROOT / "scripts/adaptive_corruption_probe/s0_freeze.py",
    ]
    missing = [str(path.relative_to(protocol.PROJECT_ROOT)) for path in docs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"B0 contract files missing: {missing}")
    forbidden_suffixes = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
    preexisting_raw = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.glob("**/*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ) if output_dir.exists() else []
    if preexisting_raw:
        raise RuntimeError(f"S0 output already contains forbidden raw artifacts: {preexisting_raw}")
    head, head_status = _git_head()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = protocol.resolved_config()
    config.update(
        {
            "git_head_at_audit": head,
            "git_head_status": head_status,
            "formal_performance_run_started": False,
        }
    )
    write_json(output_dir / "resolved_config.json", config)
    source_manifest = {
        "base_commit": protocol.BASE_COMMIT,
        "git_head_at_audit": head,
        "git_head_status": head_status,
        "files": {
            str(path.relative_to(protocol.PROJECT_ROOT)): sha256_file(path) for path in docs
        },
        "old_projects_read_only": True,
        "holdout_selection_frozen": False,
        "development_overlap_allowed_but_audited": True,
        "final_holdout_overlap_forbidden": True,
        "holdout_selection_rule_frozen": "label_free_dataset_characteristics_only_before_B5",
    }
    write_json(output_dir / "source_manifest.json", source_manifest)
    audit = {
        "project_id": protocol.PROJECT_ID,
        "stage": "S0",
        "status": "completed_valid",
        "base_commit": protocol.BASE_COMMIT,
        "old_projects_terminal_read_only": True,
        "roles_frozen_before_outcomes": True,
        "corruption_semantics_frozen": True,
        "matching_fields_frozen": True,
        "backbone_frozen": True,
        "backbone": dict(protocol.BACKBONE_CONFIG),
        "h0_support_threshold_ratio": protocol.H0_SUPPORT_THRESHOLD_RATIO,
        "corruption_rate": protocol.CORRUPTION_RATE,
        "pair_budget_rule": protocol.BACKBONE_CONFIG["pair_budget_rule"],
        "labels_used_during_fit": False,
        "hardness_utility_decoupled": True,
        "positive_control_required_before_null": True,
        "decision_hierarchy_frozen": True,
        "level_1_contrast": "ARI(C0_MatchedRandom)-ARI(C_clean_no_corruption)",
        "level_1_library_contrast": "max_C Delta_clean(C)",
        "level_2_contrast": "Delta_random(C)=ARI(C)-ARI(C0_MatchedRandom)",
        "level_2_structured_arms": list(protocol.STRUCTURED_ARMS),
        "level_3_minimum_material_delta_random": protocol.MATERIAL_DELTA_ARI,
        "level_3_minimum_distinct_role_winners": 2,
        "simple_principle_minimum_material_datasets": protocol.SIMPLE_MIN_DATASET_COUNT,
        "random_corruption_sufficient_label": "random_corruption_sufficient",
        "cross_track_holdout_disjointness_required": True,
        "gpu_pool_valid": True,
        "forbidden_gpu_ids": list(protocol.FORBIDDEN_GPU_IDS),
        "holdout_selection_frozen": False,
        "formal_performance_run_started": False,
        "raw_artifacts_published": False,
        "issues": [],
    }
    decision = {
        "stage": "S0",
        "status": "completed_valid",
        "primary_gate_pass": None,
        "next_stage_authorized": True,
        "authorized_next_stage": "B1",
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
