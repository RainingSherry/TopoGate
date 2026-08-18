"""Frozen M0/M1 contract for ``support_target_validation_probe``.

The project does not extend the C2 static library.  M1 adds one control to the
already completed C2 P2 evidence and asks whether threshold-support crossing
survives a deterministic magnitude-matched, support-preserving intervention.
Labels are never accepted by the corruption or fit code; they are read only
after fitting for the benchmark-known-K outer readout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "support_target_validation_probe"
PROTOCOL_ID = "support_target_validation_probe_m0_m1_v1"
M0_PROTOCOL_ID = "support_target_validation_probe_m0_v1"
M1_PROTOCOL_ID = "support_target_validation_probe_m1_v1"

OLD_PROJECT_ID = "sparse_corruption_principle_probe"
OLD_C2_PROTOCOL_ID = "sparse_corruption_principle_probe_c2_v1"
OLD_C2_ROOT = PROJECT_ROOT / "result/sparse_corruption_principle_probe/C2_static_matrix"
OLD_HOLDOUT_ROOT = PROJECT_ROOT / "result/sparse_corruption_principle_probe/C0_holdout_inventory"
H0_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S0_freeze/datasets"
LABEL_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S1_oracle_v2"
RESULT_ROOT = PROJECT_ROOT / "result/support_target_validation_probe"

DEVELOPMENT_PANEL = ("Mouse_retina", "Baron Human", "Campbell")
PRIMARY_SEEDS = (42, 123, 7)
LEGAL_GPU_POOL = (1, 2, 3, 4, 5, 6)
FORBIDDEN_GPU_IDS = (0, 7)

M1_CONTROL = "P2_MM_SupportPreserve"
P2_PRINCIPLE = "P2_SupportTarget"
EPOCHS = 30
CORRUPTION_RATE = 0.25
H0_SUPPORT_THRESHOLD_RATIO = 0.05

# These are frozen before the new control is fitted.  The first tolerance is
# evaluated on the mean per-epoch dataset total L1; the second is the median
# row/epoch relative mismatch.  Rows with zero P2 movement contribute zero.
TOTAL_L1_REL_TOLERANCE = 0.05
MEDIAN_ROW_REL_TOLERANCE = 0.10
ROW_REL_EPS = 1e-7
MATERIAL_DELTA_ARI = 0.03

M2_M3_M4_LOCKED = True
ADAPTIVE_LOCKED = True
GAN_LOCKED = True

PUBLICATION_EXCLUSIONS = (
    "raw_data",
    "labels_true",
    "arrays",
    "embeddings",
    "predictions",
    "weights",
    "checkpoints",
    "logs",
    "caches",
)

LABEL_FIREWALL = {
    "fit_inputs": ["clean_H0", "seed", "replayed_P2_actions", "matched_partner_actions"],
    "forbidden_fit_inputs": ["y", "ARI", "NMI", "ACC", "cluster_purity"],
    "labels_allowed": "post_fit_outer_metrics_only",
    "K_source": "benchmark_oracle_from_y_outer_readout_only",
}


def resolved_config() -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "protocol_id": PROTOCOL_ID,
        "m0_protocol_id": M0_PROTOCOL_ID,
        "m1_protocol_id": M1_PROTOCOL_ID,
        "old_project_read_only": OLD_PROJECT_ID,
        "old_c2_protocol_id": OLD_C2_PROTOCOL_ID,
        "development_panel": list(DEVELOPMENT_PANEL),
        "primary_seeds": list(PRIMARY_SEEDS),
        "legal_gpu_pool": list(LEGAL_GPU_POOL),
        "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
        "p2_principle": P2_PRINCIPLE,
        "m1_control": M1_CONTROL,
        "epochs": EPOCHS,
        "corruption_rate": CORRUPTION_RATE,
        "support_definition": {
            "kind": "fixed_clean_row_threshold_on_dense_H0",
            "ratio": H0_SUPPORT_THRESHOLD_RATIO,
            "threshold": "max(1e-6,0.05*max(abs(clean_H0_row)))",
        },
        "matching": {
            "assignment": "deterministic_scipy_linear_sum_assignment",
            "target_pair_l1": "2*abs(H0[source]-H0[P2_inactive_destination])",
            "candidate_pair_l1": "2*abs(H0[source]-H0[active_partner])",
            "objective": "absolute_pair_l1_difference",
            "source_positions": "exactly_replayed_P2_active_sources",
            "partner_positions": "active_only_excluding_all_P2_sources",
            "one_to_one": True,
            "uses_labels": False,
        },
        "tolerances": {
            "dataset_total_l1_relative": TOTAL_L1_REL_TOLERANCE,
            "median_row_relative": MEDIAN_ROW_REL_TOLERANCE,
            "row_relative_eps": ROW_REL_EPS,
        },
        "primary_endpoint": "Delta_cross=ARI(P2_reused)-ARI(P2_MM_SupportPreserve)",
        "estimand_scope": "descriptive_matched_support_role_contrast; not strict causal isolation",
        "directional_caveat": (
            "MM swaps active values with active values and may be easier for reconstruction; "
            "therefore Delta_cross can be conservative/downward-biased for a support-crossing interpretation."
        ),
        "gate": {
            "minimum_passing_datasets": 2,
            "material_delta_ari": MATERIAL_DELTA_ARI,
            "minimum_positive_seed_signs_per_passing_dataset": 2,
            "maximum_strong_negative_dataset_count": 0,
            "strong_negative_threshold": -MATERIAL_DELTA_ARI,
        },
        "labels": dict(LABEL_FIREWALL),
        "locked_stages": ["M2_raw_x_bridge", "M3_holdout", "M4_full_backbone", "adaptive_policy", "GAN"],
        "support_interpretation_firewall": (
            "Support in C2/M1 denotes threshold-defined support of dense H0, "
            "not raw-X zero/nonzero support; raw sparse-support claims require a separate validation."
        ),
        "publication_exclusions": list(PUBLICATION_EXCLUSIONS),
    }


def validate_contract() -> None:
    if len(DEVELOPMENT_PANEL) != 3 or len(PRIMARY_SEEDS) != 3:
        raise ValueError("M1 is frozen to three datasets and three paired seeds")
    if set(LEGAL_GPU_POOL) & set(FORBIDDEN_GPU_IDS):
        raise ValueError("legal and forbidden GPU pools overlap")
    if M1_CONTROL != "P2_MM_SupportPreserve":
        raise ValueError("M1 control name drifted")
    if not (0.0 < CORRUPTION_RATE < 1.0 and 0.0 < H0_SUPPORT_THRESHOLD_RATIO < 1.0):
        raise ValueError("invalid frozen corruption/support ratio")
    if TOTAL_L1_REL_TOLERANCE <= 0.0 or MEDIAN_ROW_REL_TOLERANCE <= 0.0:
        raise ValueError("magnitude tolerances must be positive")
    if not (M2_M3_M4_LOCKED and ADAPTIVE_LOCKED and GAN_LOCKED):
        raise ValueError("later stages must remain locked during M1")
