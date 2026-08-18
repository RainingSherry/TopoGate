"""Dependency-light A0 contract for the learned-relation-rule probe.

This module intentionally contains no feature extraction, labels, model,
selector or clustering code.  It only exposes the immutable protocol values
needed by the S0 audit and by future launchers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "learned_relation_rule_probe"
PROTOCOL_ID = "learned_relation_rule_probe_a0_v1"
BASE_COMMIT = "c80877cf904e41950315d37b95374825c33a7362"

DEVELOPMENT_DATASETS = ("cnae9", "Campbell", "sms_spam_collection")
SENTINEL_DATASETS = ("Mouse_retina", "Baron Human", "hate_speech")
ALL_PANEL_DATASETS = DEVELOPMENT_DATASETS + SENTINEL_DATASETS
PRIMARY_SEEDS = (42, 123, 7)
HOLDOUT_SEEDS = (42, 123, 7, 3032, 3033)
LEGAL_GPU_POOL = (1, 2, 3, 4, 5, 6)
FORBIDDEN_GPU_IDS = (0, 7)
MATERIAL_DELTA_ARI = 0.03
CAPTURE_THRESHOLD = 0.25
HOLDOUT_SOURCE = PROJECT_ROOT / "reports/representation_consumer_probe/STAGE5_HOLDOUT_MANIFEST.json"


def resolved_config() -> dict[str, Any]:
    """Return the frozen, JSON-safe configuration without reading labels."""
    return {
        "project_id": PROJECT_ID,
        "protocol_id": PROTOCOL_ID,
        "base_commit": BASE_COMMIT,
        "old_project": "relation_selection_probe",
        "old_project_status": "terminal_read_only",
        "development_datasets": list(DEVELOPMENT_DATASETS),
        "sentinel_datasets": list(SENTINEL_DATASETS),
        "all_panel_datasets": list(ALL_PANEL_DATASETS),
        "primary_seeds": list(PRIMARY_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "legal_gpu_pool": list(LEGAL_GPU_POOL),
        "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
        "material_delta_ari": MATERIAL_DELTA_ARI,
        "capture_threshold": CAPTURE_THRESHOLD,
        "candidate_pool": "inherited_frozen_pool_read_only",
        "reference_arm": "R_inherited_matched_random_read_only",
        "row_budget": "inherited_b_i=min(8,positive_count_i)",
        "consumer": "normalized_spectral_known_k_kmeans",
        "labels_used_during_fit": False,
        "diagnostic_supervision": {
            "allowed_stage": "A1_target_builder",
            "target": "pool_reference_membership",
            "deployable_method": False,
        },
        "authorized_initial_stage": "A1",
        "locked_stages": ["A2", "A3", "A4", "A5"],
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
    if BASE_COMMIT != "c80877cf904e41950315d37b95374825c33a7362":
        raise ValueError("A0 base commit drifted")
    if len(DEVELOPMENT_DATASETS) != 3 or len(SENTINEL_DATASETS) != 3:
        raise ValueError("A0 panel roles must remain 3+3")
    if len(set(PRIMARY_SEEDS)) != 3 or len(set(HOLDOUT_SEEDS)) != 5:
        raise ValueError("A0 seeds are not the frozen sets")
    if not 0 < MATERIAL_DELTA_ARI < 1 or not 0 < CAPTURE_THRESHOLD < 1:
        raise ValueError("A0 thresholds must be probabilities/effect sizes")
