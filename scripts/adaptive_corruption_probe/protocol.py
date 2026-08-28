"""Dependency-light B0 contract for the adaptive-corruption probe.

No corruption, model, loss, label or clustering implementation belongs here.
The module freezes only the input roles and promotion boundaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "adaptive_corruption_probe"
PROTOCOL_ID = "adaptive_corruption_probe_b0_v1"
BASE_COMMIT = "c80877cf904e41950315d37b95374825c33a7362"

DEVELOPMENT_PANEL = (
    "sms_spam_collection",
    "hate_speech",
    "Mouse_retina",
    "Baron Human",
    "cnae9",
    "Campbell",
)
ROLE_BY_DATASET = {
    "sms_spam_collection": "sparse_text_1",
    "hate_speech": "sparse_text_2",
    "Mouse_retina": "registered_scrna_count_1",
    "Baron Human": "registered_scrna_count_2",
    "Campbell": "registered_scrna_count_3_boundary_control",
    "cnae9": "generic_non_expression_sparse_high_dimensional_control",
}
CORRUPTION_ARMS = (
    "C_clean_no_corruption",
    "C0_MatchedRandom",
    "C1_ValueOnly",
    "C2_SupportOnly",
    "C3_MixedMatched",
    "C4_StaticHard",
)
PRIMARY_SEEDS = (42, 123, 7)
HOLDOUT_SEEDS = (42, 123, 7, 3032, 3033)
LEGAL_GPU_POOL = (1, 2, 3, 4, 5, 6)
FORBIDDEN_GPU_IDS = (0, 7)
MATERIAL_DELTA_ARI = 0.03
SIMPLE_MIN_DATASET_COUNT = 2
H0_SUPPORT_THRESHOLD_RATIO = 0.05
CORRUPTION_RATE = 0.25
BACKBONE_CONFIG = {
    "input": "audited_S0_H0",
    "standardization": "clean_H0_column_mean_std_fit_once_per_dataset",
    "encoder_dims": ["d_eff", 64, 32],
    "decoder_dims": [32, 64, "d_eff"],
    "activation": "ReLU",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "epochs": 30,
    "batch_size": 512,
    "corruption_rate": CORRUPTION_RATE,
    "pair_budget_rule": "m_i=min(ceil(rate*active_i), floor(active_i/2), inactive_i); all arms change 2*m_i coordinates",
    "support_definition": "abs(H0_ij)>=max(1e-6,0.05*row_max_abs)",
    "readout": "clean_embedding_known_K_KMeans",
}
STRUCTURED_ARMS = (
    "C1_ValueOnly",
    "C2_SupportOnly",
    "C3_MixedMatched",
    "C4_StaticHard",
)
TERMINAL_LABELS = (
    "completed_valid",
    "protocol_insensitive",
    "corruption_not_current_bottleneck",
    "random_corruption_sufficient",
    "simple_corruption_principle_sufficient",
    "adaptive_corruption_opportunity_present",
    "adaptive_model_not_necessary",
    "generator_not_necessary",
    "incomplete_compute",
    "protocol_mismatch",
)


def resolved_config() -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "protocol_id": PROTOCOL_ID,
        "base_commit": BASE_COMMIT,
        "old_projects_read_only": [
            "representation_consumer_probe",
            "relation_selection_probe",
        ],
        "development_panel": list(DEVELOPMENT_PANEL),
        "role_by_dataset": dict(ROLE_BY_DATASET),
        "corruption_arms": list(CORRUPTION_ARMS),
        "primary_seeds": list(PRIMARY_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "legal_gpu_pool": list(LEGAL_GPU_POOL),
        "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
        "material_delta_ari": MATERIAL_DELTA_ARI,
        "h0_support_threshold_ratio": H0_SUPPORT_THRESHOLD_RATIO,
        "corruption_rate": CORRUPTION_RATE,
        "backbone": dict(BACKBONE_CONFIG),
        "structured_arms": list(STRUCTURED_ARMS),
        "decision_hierarchy": {
            "level_1_corruption_matters": {
                "primary_contrast": "ARI(C0_MatchedRandom)-ARI(C_clean_no_corruption)",
                "library_contrast": "max_C Delta_clean(C)",
            },
            "level_2_structured_beats_random": {
                "contrast": "Delta_random(C)=ARI(C)-ARI(C0_MatchedRandom)",
                "arms": list(STRUCTURED_ARMS),
            },
            "level_3_adaptation_necessary": {
                "minimum_material_delta_random": MATERIAL_DELTA_ARI,
                "minimum_distinct_role_winners": 2,
                "terminal_if_not_met": "random_corruption_sufficient",
            },
            "simple_principle": {
                "minimum_material_datasets_for_same_arm": SIMPLE_MIN_DATASET_COUNT,
            },
        },
        "labels_used_during_fit": False,
        "k_source": "benchmark_oracle_from_y_outer_readout_only",
        "initial_stage": "B1",
        "locked_stages": ["B2", "B3", "B4", "B5"],
        "holdout_status": "not_yet_selected_outcome_independently_before_B5",
        "cross_track_holdout_disjointness_required": True,
        "development_overlap_allowed_but_audited": True,
        "final_holdout_overlap_forbidden": True,
        "positive_control": {
            "kind": "synthetic_support_value_sensitivity",
            "labels_used": False,
            "must_pass_before_real_null_decision": True,
            "is_clustering_evidence": False,
        },
        "matching_fields": [
            "requested_change_ratio",
            "effective_changed_coordinates",
            "support_change_ratio",
            "value_change_ratio",
            "total_absolute_change",
        ],
        "publication_exclusions": [
            "raw_data",
            "weights",
            "embeddings",
            "predictions",
            "graphs",
            "logs",
            "caches",
        ],
    }


def validate_contract() -> None:
    if set(LEGAL_GPU_POOL) & set(FORBIDDEN_GPU_IDS):
        raise ValueError("GPU allow/deny lists overlap")
    if len(DEVELOPMENT_PANEL) != 6 or set(DEVELOPMENT_PANEL) != set(ROLE_BY_DATASET):
        raise ValueError("B0 development roles must cover six fixed datasets")
    if len(CORRUPTION_ARMS) != 6 or CORRUPTION_ARMS[0] != "C_clean_no_corruption":
        raise ValueError("B0 corruption arm library drifted")
    if tuple(CORRUPTION_ARMS[2:]) != STRUCTURED_ARMS:
        raise ValueError("B0 structured corruption arm library drifted")
    if not 0.0 < H0_SUPPORT_THRESHOLD_RATIO < 1.0 or not 0.0 < CORRUPTION_RATE < 1.0:
        raise ValueError("B0 support threshold and corruption rate must be in (0,1)")
    if BASE_COMMIT != "c80877cf904e41950315d37b95374825c33a7362":
        raise ValueError("B0 base commit drifted")
