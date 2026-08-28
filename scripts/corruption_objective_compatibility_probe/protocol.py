"""Frozen protocol for the overnight corruption--objective probe.

This project is independent of the closed C2, M1, and D1 studies.  Existing
results are read-only controls; no result-driven choice is made by the runner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "corruption_objective_compatibility_probe"
PROTOCOL_ID = "corruption_objective_compatibility_probe_e0_e4_v1"
BASE_PROJECTS = {
    "c2": "sparse_corruption_principle_probe_c2_v1",
    "support_d1": "support_crossing_common_dose_probe_d0_d1_v1",
    "input": "representation_consumer_probe_s0_v1",
}

DEVELOPMENT_PANEL = (
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
    "Mouse_retina": "biological_scRNA_development",
    "Baron Human": "biological_scRNA_development",
    "Campbell": "biological_scRNA_boundary_control",
    "cnae9": "nonbiological_sparse_textlike",
    "hate_speech": "nonbiological_sparse_textlike",
    "sms_spam_collection": "nonbiological_sparse_textlike",
}
PRIMARY_SEEDS = (42, 123, 7)
GPU_POOL = (1, 2, 3, 4, 5, 6)
FORBIDDEN_GPU_IDS = (0, 7)

E1_ARMS = ("Clean", "P0_Random", "P2_SupportTarget")
E2_CORRUPTIONS = ("P0_Random", "P2_SupportTarget")
E2_OBJECTIVES = ("O0_GlobalMSE", "O1_ChangedOnlyMSE", "O2_BalancedMSE")
CHECKPOINT_EPOCHS = (1, 5, 10, 20, 30)

INPUT_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S0_freeze/datasets"
LABEL_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S1_oracle_v2"
C2_ROOT = PROJECT_ROOT / "result/sparse_corruption_principle_probe/C2_static_matrix"
SUPPORT_D1_ROOT = PROJECT_ROOT / "result/support_crossing_common_dose_probe/D1_feasibility"
RAW_ROOT = PROJECT_ROOT / "datasets"
RESULT_ROOT = PROJECT_ROOT / "result/corruption_objective_compatibility_probe"

MATERIAL_DELTA_ARI = 0.03
SUPPORT_THRESHOLD_RATIO = 0.05
CORRUPTION_RATE = 0.25
PAIR_COST_EPS = 1e-7
DOSE_EPS = 1e-8
E1_MIN_DATASET_COUNT = 2
E1_MIN_SEED_POSITIVE_COUNT = 2
E2_MIN_DATASET_COUNT = 4
E2_MIN_BIOLOGICAL_COUNT = 1
E2_MIN_NONBIOLOGICAL_COUNT = 1
E2_OPPOSING_SIGN_MAX_COUNT = 1

BACKBONE = {
    "encoder_dims": ["d_eff", 64, 32],
    "decoder_dims": [32, 64, "d_eff"],
    "activation": "ReLU",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "epochs": 30,
    "batch_size": 512,
    "standardization": "clean_H0_column_mean_std_fit_once_per_dataset",
    "readout": "clean_embedding_known_K_KMeans",
}

PER_RUN_TIMEOUT_SECONDS = 1800
HARD_WALL_SECONDS = int(11.5 * 60 * 60)
RETRYABLE_ERROR_TOKENS = (
    "cuda",
    "input/output error",
    "i/o error",
    "resource temporarily unavailable",
    "connection reset",
)

LABEL_FIREWALL = {
    "labels_used_during_fit": False,
    "labels_loaded_after_fit_for": ["benchmark_known_K", "ARI", "NMI", "ACC"],
    "forbidden_fit_inputs": ["y", "K", "ARI", "NMI", "ACC"],
}

LOCKED_ROUTES = (
    "support_d2",
    "support_matcher_optimization",
    "optimal_or_hungarian_matcher",
    "adaptive_policy",
    "GAN",
    "learned_generator",
    "transformer",
    "new_gate",
    "corruption_rate_sweep",
    "mask_predictor",
)


def resolved_config() -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "protocol_id": PROTOCOL_ID,
        "base_projects_read_only": dict(BASE_PROJECTS),
        "development_panel": list(DEVELOPMENT_PANEL),
        "biological_datasets": list(BIOLOGICAL_DATASETS),
        "nonbiological_datasets": list(NONBIOLOGICAL_DATASETS),
        "role_by_dataset": dict(ROLE_BY_DATASET),
        "primary_seeds": list(PRIMARY_SEEDS),
        "legal_gpu_pool": list(GPU_POOL),
        "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
        "e1": {
            "logical_arms": list(E1_ARMS),
            "new_gpu_runs": 36,
            "reused_bio_p0_p2_entries": 18,
            "raw_nofit_entries": len(DEVELOPMENT_PANEL) * len(E1_ARMS) * len(PRIMARY_SEEDS),
            "gate": {
                "minimum_nonbio_datasets": E1_MIN_DATASET_COUNT,
                "minimum_seed_positive_count": E1_MIN_SEED_POSITIVE_COUNT,
                "delta_random": "ARI(P2)-ARI(P0)",
                "delta_clean": "ARI(P2)-ARI(Clean)",
                "training_amplification": "(AE_P2-AE_P0)-(raw_P2-raw_P0)",
                "material_delta_ari": MATERIAL_DELTA_ARI,
            },
        },
        "e2": {
            "corruptions": list(E2_CORRUPTIONS),
            "objectives": list(E2_OBJECTIVES),
            "logical_runs": len(DEVELOPMENT_PANEL) * len(E2_CORRUPTIONS) * len(E2_OBJECTIVES) * len(PRIMARY_SEEDS),
            "new_gpu_runs": 72,
            "o0_reused_entries": 36,
            "estimands": {
                "delta_objective": "ARI(P2,O)-ARI(P0,O)",
                "objective_interaction": "Delta_O-Delta_O0",
            },
            "gate": {
                "minimum_datasets": E2_MIN_DATASET_COUNT,
                "minimum_biological": E2_MIN_BIOLOGICAL_COUNT,
                "minimum_nonbiological": E2_MIN_NONBIOLOGICAL_COUNT,
                "maximum_opposing_sign_datasets": E2_OPPOSING_SIGN_MAX_COUNT,
                "material_delta_ari": MATERIAL_DELTA_ARI,
            },
        },
        "e3": {
            "purpose": "raw-input descriptive audit only",
            "does_not_change_fit_or_gates": True,
            "support_semantics": "raw_X_zero_nonzero_descriptive_only; H0_threshold_support remains separate",
        },
        "backbone": dict(BACKBONE),
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "per_run_timeout_seconds": PER_RUN_TIMEOUT_SECONDS,
        "hard_wall_seconds": HARD_WALL_SECONDS,
        "retry_policy": {
            "max_same_config_retries": 1,
            "retryable_error_tokens": list(RETRYABLE_ERROR_TOKENS),
            "non_retryable": ["NaN", "shape mismatch", "budget mismatch", "label leakage", "protocol mismatch", "assertion"],
        },
        "label_firewall": dict(LABEL_FIREWALL),
        "locked_routes": list(LOCKED_ROUTES),
        "publication_exclusions": ["raw_data", "labels", "arrays", "embeddings", "predictions", "weights", "checkpoints", "logs", "caches"],
        "decision_options": [
            "STOP_GENERAL_CORRUPTION",
            "STATIC_CORRUPTION_REPLICATION",
            "CORRUPTION_AWARE_OBJECTIVE_OPPORTUNITY",
            "REPRESENTATION_NOT_OBJECTIVE",
        ],
    }


def validate_contract() -> None:
    if set(GPU_POOL) & set(FORBIDDEN_GPU_IDS):
        raise ValueError("GPU allow/deny lists overlap")
    if set(DEVELOPMENT_PANEL) != set(BIOLOGICAL_DATASETS) | set(NONBIOLOGICAL_DATASETS):
        raise ValueError("dataset roles do not partition the development panel")
    if len(PRIMARY_SEEDS) != 3 or len(set(PRIMARY_SEEDS)) != 3:
        raise ValueError("primary seeds must be exactly three unique paired seeds")
    if E1_ARMS != ("Clean", "P0_Random", "P2_SupportTarget"):
        raise ValueError("E1 arm contract drifted")
    if E2_OBJECTIVES != ("O0_GlobalMSE", "O1_ChangedOnlyMSE", "O2_BalancedMSE"):
        raise ValueError("E2 objective contract drifted")
    if CHECKPOINT_EPOCHS[-1] != BACKBONE["epochs"]:
        raise ValueError("checkpoint schedule must end at the frozen epoch budget")
    if PER_RUN_TIMEOUT_SECONDS <= 0 or HARD_WALL_SECONDS <= PER_RUN_TIMEOUT_SECONDS:
        raise ValueError("invalid timeout contract")
    if not all(stage in LOCKED_ROUTES for stage in ("support_d2", "adaptive_policy", "GAN", "transformer")):
        raise ValueError("locked routes drifted")
