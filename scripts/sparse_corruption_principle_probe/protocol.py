"""Frozen C0 contract for ``sparse_corruption_principle_probe``.

The project asks a deliberately small question: which static corruption
principle, if any, changes clustering-relevant representations on naturally
sparse high-dimensional data?  This module freezes the panel, estimands,
principle names, resource boundary, and promotion locks before any new
performance matrix is launched.

Labels are not part of the corruption or fit contract.  They may only be read
by an outer benchmark evaluator after a fit has completed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ID = "sparse_corruption_principle_probe"
PROTOCOL_ID = "sparse_corruption_principle_probe_c0_v1"
C2_PROTOCOL_ID = "sparse_corruption_principle_probe_c2_v1"

# The repository currently has no usable Git object database.  These are
# evidence paths, not a fabricated commit claim.  The C0 artifact records
# their SHA256 values at execution time.
BASE_EVIDENCE = {
    "old_b1_protocol": "reports/adaptive_corruption_probe/PRE_REGISTRATION.md",
    "old_b1_compact_result": "result/adaptive_corruption_probe/B1_corruption_library",
    "s0_h0_root": "result/representation_consumer_probe/S0_freeze/datasets",
}
OLD_PROJECTS_READ_ONLY = (
    "adaptive_corruption_probe",
    "learned_relation_rule_probe",
    "representation_consumer_probe",
    "relation_selection_probe",
)

DEVELOPMENT_PANEL = ("Mouse_retina", "Baron Human", "Campbell")
ROLE_BY_DATASET = {
    "Mouse_retina": "corruption_presence_case",
    "Baron Human": "support_sensitive_case",
    "Campbell": "difficulty_sensitive_case",
}

# These are burned development datasets.  They are not an independent
# generalization denominator.
DEVELOPMENT_SOURCE_FILES = {
    "Mouse_retina": "datasets/Mouse_retina.npz",
    "Baron Human": "datasets/Baron Human.npz",
    "Campbell": "datasets/Campbell.npz",
}

PRIMARY_SEEDS = (42, 123, 7)
HOLDOUT_SEEDS = (42, 123, 7, 3032, 3033)
LEGAL_GPU_POOL = (1, 2, 3, 4, 5, 6)
FORBIDDEN_GPU_IDS = (0, 7)

MATERIAL_DELTA_ARI = 0.03
H0_SUPPORT_THRESHOLD_RATIO = 0.05
CORRUPTION_RATE = 0.25
GEOMETRY_K = 15
PAIRWISE_GEOMETRY_SAMPLE = 2048

# The formal primary C2 matrix is intentionally finite and fixed.  P5 is the
# high-geometry arm.  GeometrySafe is a paired negative-control fixture only;
# it is not an unregistered seventh arm or a post-hoc sweep.
PRINCIPLES = (
    "P0_Random",
    "P1_SupportPreserve",
    "P2_SupportTarget",
    "P3_FrequencyAware",
    "P4_ResidualHard",
    "P5_GeometryHard",
)
PRIMARY_PRINCIPLES = PRINCIPLES
NEGATIVE_CONTROL_FIXTURES = ("P5_GeometrySafe",)

H0_ROOT = PROJECT_ROOT / "result/representation_consumer_probe/S0_freeze/datasets"
B1_ROOT = PROJECT_ROOT / "result/adaptive_corruption_probe/B1_corruption_library"
RESULT_ROOT = PROJECT_ROOT / "result/sparse_corruption_principle_probe"

# The inventory script inspects these pre-existing, outcome-independent source
# families.  Selection from this universe is frozen before C2 results are
# visible and uses only label-free shape/sparsity/source characteristics.
HOLDOUT_UNIVERSE_ROOTS = (
    "/data/luolie/biopipeline/scCluBench/data/processed",
    "/data/luolie/biopipeline/scCluBench/data/processed_scmae",
    "/data/luolie/biopipeline/scCluBench/data/其他",
    "/data/luolie/biopipeline/test-datasets/test-datasets-modules/data/genomics/homo_sapiens/scrnaseq/h5ad",
    "/data/luolie/biopipeline/ScienceAgentBench/benchmark/datasets/pbmc68k",
    "/data/luolie/biopipeline/ScienceAgentBench/benchmark/datasets/pbmc_umap",
    "/data/luolie/biopipeline/dimension-reduction/scSSL-Bench/data/PBMC",
)

# Candidate names are fixed by source path, not by any metric.  The list is
# intentionally broader than the desired 8–12 holdout slots so the inventory
# can report a transparent shortfall or a deterministic maximin selection.
HOLDOUT_CANDIDATE_RELATIVE_PATHS = (
    "processed/Pollen.h5ad",
    "processed/Quake_Smart-seq2_Lung.h5ad",
    "processed/Wang.h5ad",
    "processed_scmae/Bach.h5ad",
    "processed_scmae/Guo.h5ad",
    "processed_scmae/Limb_Muscle.h5ad",
    "processed_scmae/Macosko.h5ad",
    "processed_scmae/Melanoma_5K.h5ad",
    "processed_scmae/Quake_10x_Spleen.h5ad",
    "processed_scmae/Shekhar.h5ad",
    "processed_scmae/Tosches.h5ad",
    "processed_scmae/Young.h5ad",
    "processed_scmae/worm_neuron_cell.h5ad",
    "其他/Mouse_Pancreas_1.h5ad",
)

HOLDOUT_MIN_DATASETS = 8
HOLDOUT_TARGET_DATASETS = 12

BACKBONE_CONTRACT = {
    "input": "audited_S0_H0",
    "standardization": "clean_H0_column_mean_std_fit_once_per_dataset",
    "encoder": "small_matched_reconstruction_probe",
    "encoder_dims": ["d_eff", 64, 32],
    "decoder_dims": [32, 64, "d_eff"],
    "activation": "ReLU",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "epochs": 30,
    "batch_size": 512,
    "warmup_epochs": 5,
    "readout": "clean_embedding_known_K_KMeans_outer_benchmark_only",
    "labels_used_during_fit": False,
}

LABEL_FIREWALL = {
    "fit_inputs": ["X_or_H0", "principle", "seed", "optional_frozen_residual"],
    "forbidden_fit_inputs": ["y", "ARI", "NMI", "ACC", "cluster_purity"],
    "labels_allowed": "post_fit_outer_metrics_and_synthetic_fixture_audit_only",
    "K_source": "benchmark_oracle_from_y_outer_readout_only",
}

PUBLICATION_EXCLUSIONS = (
    "raw_data",
    "labels_true",
    "arrays",
    "embeddings",
    "predictions",
    "weights",
    "checkpoints",
    "graphs",
    "logs",
    "caches",
)

DECISION_RULES = {
    "C1": {
        "purpose": "localize support/value/geometry changes without new model training",
        "labels_used": False,
    },
    "C2_library": {
        "primary": "Delta_P(d)=ARI(P,d)-ARI(P0_Random,d)",
        "material_delta_ari": MATERIAL_DELTA_ARI,
        "tested_library_envelope_name": "tested_static_library_opportunity",
        "not_oracle_upper_bound": True,
    },
    "adaptive_unlock": {
        "requires": [
            "C2_static_library_contract_pass",
            "heterogeneous_material_winners_on_frozen_development_panel",
            "outcome_independent_holdout_membership_frozen",
        ],
        "locked_until_then": ["adaptive_policy", "MLP_selector", "GAN", "learned_generator"],
    },
}


def resolved_config() -> dict[str, Any]:
    """Return a JSON-serializable snapshot used by every new run."""

    return {
        "project_id": PROJECT_ID,
        "protocol_id": PROTOCOL_ID,
        "c2_protocol_id": C2_PROTOCOL_ID,
        "old_projects_read_only": list(OLD_PROJECTS_READ_ONLY),
        "base_evidence": dict(BASE_EVIDENCE),
        "development_panel": list(DEVELOPMENT_PANEL),
        "role_by_dataset": dict(ROLE_BY_DATASET),
        "primary_seeds": list(PRIMARY_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "legal_gpu_pool": list(LEGAL_GPU_POOL),
        "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
        "material_delta_ari": MATERIAL_DELTA_ARI,
        "support_definition": {
            "kind": "fixed_clean_row_threshold",
            "threshold": "max(1e-6,0.05*max(abs(clean_row)))",
            "ratio": H0_SUPPORT_THRESHOLD_RATIO,
        },
        "corruption_rate": CORRUPTION_RATE,
        "geometry_k": GEOMETRY_K,
        "principles": list(PRINCIPLES),
        "negative_control_fixtures": list(NEGATIVE_CONTROL_FIXTURES),
        "formal_matrix": {
            "datasets": len(DEVELOPMENT_PANEL),
            "principles": len(PRIMARY_PRINCIPLES),
            "seeds": len(PRIMARY_SEEDS),
            "runs": len(DEVELOPMENT_PANEL) * len(PRIMARY_PRINCIPLES) * len(PRIMARY_SEEDS),
        },
        "backbone": dict(BACKBONE_CONTRACT),
        "label_firewall": dict(LABEL_FIREWALL),
        "decision_rules": dict(DECISION_RULES),
        "holdout": {
            "status": "membership_must_be_frozen_before_C2_results",
            "minimum": HOLDOUT_MIN_DATASETS,
            "target": HOLDOUT_TARGET_DATASETS,
            "selection_features": [
                "log_sample_count",
                "log_feature_count",
                "estimated_sparsity",
                "estimated_intrinsic_dimension_proxy",
                "source_family",
            ],
            "outcome_features_forbidden": ["ARI", "B1_effect", "historical_gain", "labels"],
        },
        "authorized_now": ["C0_freeze", "C1_structural_audit", "C2_static_library_implementation", "C2_54_run_matrix"],
        "locked_now": ["C3_holdout_runs", "adaptive_policy", "GAN", "learned_generator"],
        "c2_matrix_authorized": True,
        "support_interpretation_firewall": (
            "Support in C2 denotes the frozen threshold-defined support of dense H0, "
            "not raw-X zero/nonzero support; raw sparse-support claims require a separate validation."
        ),
        "publication_exclusions": list(PUBLICATION_EXCLUSIONS),
    }


def validate_contract() -> None:
    if len(DEVELOPMENT_PANEL) != 3 or set(DEVELOPMENT_PANEL) != set(ROLE_BY_DATASET):
        raise ValueError("C0 development roles must cover exactly three burned scRNA datasets")
    if set(LEGAL_GPU_POOL) & set(FORBIDDEN_GPU_IDS):
        raise ValueError("legal and forbidden GPU lists overlap")
    if any(gpu in LEGAL_GPU_POOL for gpu in FORBIDDEN_GPU_IDS):
        raise ValueError("forbidden GPU is present in legal pool")
    if len(PRINCIPLES) != 6 or PRINCIPLES != PRIMARY_PRINCIPLES:
        raise ValueError("C2 primary static library drifted")
    if "P5_GeometrySafe" in PRINCIPLES:
        raise ValueError("GeometrySafe must remain a fixture, not a seventh primary arm")
    if not (0.0 < CORRUPTION_RATE < 1.0):
        raise ValueError("corruption rate must be in (0,1)")
    if not (0.0 < H0_SUPPORT_THRESHOLD_RATIO < 1.0):
        raise ValueError("support threshold ratio must be in (0,1)")
    if len(PRIMARY_SEEDS) != 3 or len(set(PRIMARY_SEEDS)) != 3:
        raise ValueError("primary seeds must be three unique paired seeds")
    if not all(name in DEVELOPMENT_SOURCE_FILES for name in DEVELOPMENT_PANEL):
        raise ValueError("development source manifest is incomplete")


def validate_c2_authorization() -> None:
    """Validate the explicit user-authorized C2 matrix boundary."""

    validate_contract()
    if not C2_PROTOCOL_ID.endswith("_c2_v1"):
        raise ValueError("C2 protocol identifier drifted")
    if not resolved_config().get("c2_matrix_authorized"):
        raise ValueError("C2 matrix is not authorized in the frozen protocol")
    if not all(stage in resolved_config()["locked_now"] for stage in ("adaptive_policy", "GAN", "learned_generator")):
        raise ValueError("C2 must keep adaptive policy, GAN and learned generator locked")
