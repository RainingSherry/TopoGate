"""Frozen D0/D1 contract for ``support_crossing_common_dose_probe``.

This project is deliberately smaller than a new model study.  It asks whether
the active/inactive swap and an active/active value swap have a constructive
common dose under the same row-specific changed-coordinate budget.  D0 and D1
are CPU-only; the D2 GPU matrix is locked until the feasibility gate passes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "support_crossing_common_dose_probe"
PROTOCOL_ID = "support_crossing_common_dose_probe_d0_d1_v1"
D0_PROTOCOL_ID = "support_crossing_common_dose_probe_d0_v1"
D1_PROTOCOL_ID = "support_crossing_common_dose_probe_d1_v1"

READ_ONLY_PROJECTS = (
    "sparse_corruption_principle_probe",
    "support_target_validation_probe",
    "representation_consumer_probe",
)

DEVELOPMENT_PANEL = ("Mouse_retina", "Baron Human", "Campbell")
PRIMARY_SEEDS = (42, 123, 7)

H0_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S0_freeze/datasets"
M1_ROOT = PROJECT_ROOT / "result/support_target_validation_probe/M1_preflight"
RESULT_ROOT = PROJECT_ROOT / "result/support_crossing_common_dose_probe"

CORRUPTION_RATE = 0.25
SUPPORT_THRESHOLD_RATIO = 0.05
PAIR_COST_EPS = 1e-7
DOSE_EPS = 1e-8

# These are frozen before looking at D1 outputs.  The range is explicitly a
# constructive witness range, not a claim that every attainable matching was
# enumerated.  This keeps the negative result interpretable and auditable.
MIN_COMMON_BUDGET_ROW_FRACTION = 0.95
TOTAL_DOSE_REL_TOLERANCE = 0.05
MEDIAN_ROW_DOSE_REL_TOLERANCE = 0.10

LEGAL_GPU_POOL = (1, 2, 3, 4, 5, 6)
FORBIDDEN_GPU_IDS = (0, 7)
D2_GPU_RUNS_STARTED = 0
D2_LOCKED = True
ADAPTIVE_LOCKED = True
GAN_LOCKED = True
RAW_X_BRIDGE_LOCKED = True
HOLDOUT_LOCKED = True

LABEL_FIREWALL = {
    "fit_inputs": ["clean_H0", "row_budget", "seed", "dose_target"],
    "forbidden_inputs": ["y", "ARI", "NMI", "ACC", "cluster_purity"],
    "labels_loaded": False,
    "purpose": "CPU feasibility only; no clustering consumer",
}

PUBLICATION_EXCLUSIONS = (
    "raw_data",
    "labels_true",
    "arrays",
    "pair_indices",
    "embeddings",
    "predictions",
    "weights",
    "checkpoints",
    "logs",
    "caches",
)


def resolved_config() -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "protocol_id": PROTOCOL_ID,
        "d0_protocol_id": D0_PROTOCOL_ID,
        "d1_protocol_id": D1_PROTOCOL_ID,
        "read_only_projects": list(READ_ONLY_PROJECTS),
        "development_panel": list(DEVELOPMENT_PANEL),
        "primary_seeds": list(PRIMARY_SEEDS),
        "support_definition": {
            "kind": "fixed_clean_row_threshold_on_dense_H0",
            "ratio": SUPPORT_THRESHOLD_RATIO,
            "threshold": "max(1e-6,0.05*max(abs(clean_H0_row)))",
        },
        "budget": {
            "source": "C2 row_budgets",
            "corruption_rate": CORRUPTION_RATE,
            "changed_coordinates": "2 * pair_count",
        },
        "arms": {
            "Cross": {
                "operation": "active_to_inactive value swap",
                "support_change": "strictly positive on every positive-budget row",
                "value_multiset": "preserved by swap",
            },
            "Preserve": {
                "operation": "active_to_active unequal-value swap",
                "support_change": "zero",
                "value_multiset": "preserved by swap",
            },
        },
        "dose": {
            "definition": "row_total_L1 = sum(abs(corrupted-clean))",
            "range": "constructive_min_max_witnesses; not exhaustive attainable set",
            "target": "midpoint of Cross/Preserve interval intersection",
            "target_matching": "deterministic nearest-per-pair greedy matching",
            "seed_semantics": "deterministic tie-break reproductions, not independent statistical samples",
        },
        "d1_gate": {
            "minimum_common_positive_budget_row_fraction": MIN_COMMON_BUDGET_ROW_FRACTION,
            "dataset_total_relative_mismatch": TOTAL_DOSE_REL_TOLERANCE,
            "median_row_relative_mismatch": MEDIAN_ROW_DOSE_REL_TOLERANCE,
            "exact_changed_count": True,
            "cross_support_change_positive": True,
            "preserve_support_change_zero": True,
            "row_value_multiset_preserved": True,
        },
        "label_firewall": dict(LABEL_FIREWALL),
        "legal_gpu_pool": list(LEGAL_GPU_POOL),
        "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
        "d2_gpu_runs_started": D2_GPU_RUNS_STARTED,
        "locked_stages": ["D2_gpu_matrix", "raw_x_bridge", "holdout", "adaptive_policy", "GAN"],
        "publication_exclusions": list(PUBLICATION_EXCLUSIONS),
        "estimand": "D2 Delta_cross = ARI(Cross)-ARI(Preserve) at matched common dose",
    }


def validate_contract() -> None:
    if len(DEVELOPMENT_PANEL) != 3 or len(PRIMARY_SEEDS) != 3:
        raise ValueError("D0/D1 require three datasets and three paired seeds")
    if set(LEGAL_GPU_POOL) & set(FORBIDDEN_GPU_IDS):
        raise ValueError("legal and forbidden GPU pools overlap")
    if not (0.0 < CORRUPTION_RATE < 1.0 and 0.0 < SUPPORT_THRESHOLD_RATIO < 1.0):
        raise ValueError("invalid frozen rate or threshold ratio")
    if not (0.0 < MIN_COMMON_BUDGET_ROW_FRACTION <= 1.0):
        raise ValueError("invalid common-row gate")
    if TOTAL_DOSE_REL_TOLERANCE <= 0.0 or MEDIAN_ROW_DOSE_REL_TOLERANCE <= 0.0:
        raise ValueError("dose tolerances must be positive")
    if not (D2_LOCKED and RAW_X_BRIDGE_LOCKED and HOLDOUT_LOCKED and ADAPTIVE_LOCKED and GAN_LOCKED):
        raise ValueError("D2 and later routes must remain locked")
