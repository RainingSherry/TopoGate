"""Frozen protocol for the raw sparse mask schedule probe.

This project is deliberately independent from all V-series and previous
corruption/topology studies.  The only primary manipulation is how a fixed
mask budget is sampled from a raw zero-preserving matrix.  Labels are loaded
only after fitting, for the outer benchmark readout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "raw_sparse_mask_schedule_probe"
PROTOCOL_ID = "raw_sparse_mask_schedule_probe_v1"
PLAN_VERSION = "raw_sparse_mask_schedule_probe_overnight_v1"
RESULT_ROOT = PROJECT_ROOT / "result" / PROJECT_ID
REPORT_ROOT = PROJECT_ROOT / "reports" / PROJECT_ID
FREEZE_ROOT = RESULT_ROOT / "FREEZE"
MAIN_ROOT = RESULT_ROOT / "MAIN"
FIXED_ROOT = RESULT_ROOT / "FIXED_RATIO_ORACLE"
REPR_ROOT = RESULT_ROOT / "REPR_LOCALIZATION"
COMPUTE_ROOT = RESULT_ROOT / "COMPUTE"
FINAL_ROOT = RESULT_ROOT / "FINAL"

DATA_ROOT = PROJECT_ROOT / "datasets"
E3_SUMMARY = PROJECT_ROOT / "result/corruption_objective_compatibility_probe/E3_raw_audit/summary.json"

DATASETS = (
    "Mouse_retina",
    "Baron Human",
    "Campbell",
    "cnae9",
    "hate_speech",
    "sms_spam_collection",
)
BIOLOGICAL_DATASETS = ("Mouse_retina", "Baron Human", "Campbell")
NONBIOLOGICAL_DATASETS = ("cnae9", "hate_speech", "sms_spam_collection")
ROLE_BY_DATASET = {
    "Mouse_retina": "biological",
    "Baron Human": "biological",
    "Campbell": "biological",
    "cnae9": "nonbiological",
    "hate_speech": "nonbiological",
    "sms_spam_collection": "nonbiological",
}
SEEDS = (42, 123, 7)
ARMS = ("CLEAN_AE", "ALL_FIXED", "ACTIVE_FIXED", "ALL_VARIABLE", "ACTIVE_VARIABLE")
MASK_TARGETS = ("ALL", "ACTIVE")
MASK_SCHEDULES = ("FIXED", "VARIABLE")
FIXED_MASK_RATIO = 0.25
VARIABLE_MASK_LOW = 0.05
VARIABLE_MASK_HIGH = 0.45
MATERIAL_DELTA_ARI = 0.03
SVD_COMPONENTS = 32

LEGAL_GPU_POOL = (1, 2, 3, 4, 5, 6)
FORBIDDEN_GPU_IDS = (0, 7)
PER_RUN_TIMEOUT_SECONDS = 1800
HARD_WALL_SECONDS = 11 * 3600 + 30 * 60
NEW_LAUNCH_CUTOFF_SECONDS = 11 * 3600
FINALIZATION_RESERVE_SECONDS = 30 * 60
MAX_RETRIES = 1

BACKBONE = {
    "encoder": ["d", 64, 32],
    "decoder": [32, 64, "d"],
    "activation": "ReLU",
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "epochs": 30,
    "batch_size_candidates": [512, 256, 128, 64],
    "batch_size_rule": "first_outcome_independent_forward_backward_smoke",
    "readout": "clean_embedding_known_K_KMeans_outer_benchmark_only",
}

LABEL_FIREWALL = {
    "fit_inputs": ["X0", "arm", "seed", "epoch", "batch_order"],
    "forbidden_fit_inputs": ["y", "K", "ARI", "NMI", "ACC"],
    "labels_allowed": "post_fit_outer_metrics_only",
    "labels_loaded_after_fit": True,
}

LOCKED_ROUTES = (
    "new_gate",
    "topology_selector",
    "neighbormix_rescue",
    "accg_rescue",
    "support_crossing_matcher_optimization",
    "GAN",
    "learned_corruption_generator",
    "attention_transformer_sweep",
    "corruption_rate_tuning",
    "feature_importance_selector",
    "residual_hard_selector",
    "geometry_hard_selector",
    "holdout_claim",
    "post_hoc_dataset_removal",
)

TERMINAL_DECISIONS = (
    "INCOMPLETE_COMPUTE",
    "STOP_RAW_SUPPORT_MASKING",
    "STOP_MULTISCALE_MASKING",
    "MECHANISM_SENSITIVITY_NO_METHOD_NECESSITY",
    "RAW_SPARSE_MASK_PRINCIPLE_CANDIDATE",
    "REPRESENTATION_SPACE_AUGMENTATION_CANDIDATE",
    "AUGMENTATION_BRANCH_CLOSED",
)


def resolved_config() -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "protocol_id": PROTOCOL_ID,
        "plan_version": PLAN_VERSION,
        "execution_mode": "local-only; no automatic GitHub push",
        "datasets": list(DATASETS),
        "biological_datasets": list(BIOLOGICAL_DATASETS),
        "nonbiological_datasets": list(NONBIOLOGICAL_DATASETS),
        "role_by_dataset": dict(ROLE_BY_DATASET),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "mask_target_space": list(MASK_TARGETS),
        "mask_schedule": list(MASK_SCHEDULES),
        "fixed_mask_ratio": FIXED_MASK_RATIO,
        "variable_mask_ratio": [VARIABLE_MASK_LOW, VARIABLE_MASK_HIGH],
        "material_delta_ari": MATERIAL_DELTA_ARI,
        "svd_components": SVD_COMPONENTS,
        "backbone": dict(BACKBONE),
        "legal_gpu_pool": list(LEGAL_GPU_POOL),
        "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
        "per_run_timeout_seconds": PER_RUN_TIMEOUT_SECONDS,
        "hard_wall_seconds": HARD_WALL_SECONDS,
        "new_launch_cutoff_seconds": NEW_LAUNCH_CUTOFF_SECONDS,
        "label_firewall": dict(LABEL_FIREWALL),
        "locked_routes": list(LOCKED_ROUTES),
        "formal_matrix": {
            "main_runs": len(DATASETS) * len(ARMS) * len(SEEDS),
            "svd_runs": len(DATASETS) * len(SEEDS),
            "fixed_oracle_max_new_runs": len(DATASETS) * 4,
            "repr_localization_runs": len(DATASETS) * 2 * len(SEEDS),
        },
        "publication_exclusions": [
            "raw_data", "labels", "scale_arrays", "masks", "embeddings",
            "predictions", "weights", "checkpoints", "worker_logs", "caches",
        ],
        "terminal_decisions": list(TERMINAL_DECISIONS),
    }


def validate_contract() -> None:
    if PROJECT_ID != "raw_sparse_mask_schedule_probe":
        raise ValueError("project id drifted")
    if len(DATASETS) != 6 or set(BIOLOGICAL_DATASETS) | set(NONBIOLOGICAL_DATASETS) != set(DATASETS):
        raise ValueError("dataset panel must contain exactly six fixed sentinels")
    if len(SEEDS) != 3 or len(set(SEEDS)) != 3:
        raise ValueError("seeds must be three unique paired seeds")
    if ARMS != ("CLEAN_AE", "ALL_FIXED", "ACTIVE_FIXED", "ALL_VARIABLE", "ACTIVE_VARIABLE"):
        raise ValueError("main arm matrix drifted")
    if set(LEGAL_GPU_POOL) & set(FORBIDDEN_GPU_IDS):
        raise ValueError("GPU allow/deny lists overlap")
    if not (0.0 < VARIABLE_MASK_LOW < FIXED_MASK_RATIO < VARIABLE_MASK_HIGH < 1.0):
        raise ValueError("mask schedule bounds drifted")
    if SVD_COMPONENTS != 32:
        raise ValueError("SVD baseline dimension drifted")
    if PER_RUN_TIMEOUT_SECONDS <= 0 or HARD_WALL_SECONDS <= NEW_LAUNCH_CUTOFF_SECONDS:
        raise ValueError("timeout/hard-wall contract invalid")
    if any(not isinstance(x, str) for x in LOCKED_ROUTES):
        raise ValueError("locked route contract invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--write-config", type=Path)
    args = parser.parse_args()
    validate_contract()
    if args.write_config:
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_config.write_text(json.dumps(resolved_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "valid", "protocol_id": PROTOCOL_ID}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
