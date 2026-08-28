"""Frozen, independent protocol for the V26 Support Oracle Study v1."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ID = "V26_support_oracle"
PROTOCOL_ID = "support_oracle_study_v1"
IMPLEMENTATION_REVISION = "r1_prelaunch_review"
RESULT_ROOT = PROJECT_ROOT / "result" / PROJECT_ID
REPORT_ROOT = PROJECT_ROOT / "reports" / PROJECT_ID
FREEZE_ROOT = RESULT_ROOT / "FREEZE"

SEEDS = (42, 123, 7)
ARMS = ("CLEAN", "P0_RANDOM", "P1_SUPPORT_PRESERVE", "P2_SUPPORT_TARGET", "O_LABEL_ORACLE")
LEGAL_GPU_POOL = (1, 2, 3, 4, 5, 6)
FORBIDDEN_GPU_IDS = (0, 7)
CORRUPTION_RATE = 0.25
EPOCHS = 30
BATCH_CANDIDATES = (512, 256, 128, 64)
HIDDEN_DIM = 128
LATENT_DIM = 32
LEARNING_RATE = 1e-3
MATERIAL_DELTA_ARI = 0.03
SVD_COMPONENTS = 128
VALUE_PROFILE_QUANTILES = 128
SUPPORT_DIAGNOSTIC_MIN_ARI = 0.10


@dataclass(frozen=True)
class DatasetSpec:
    identifier: str
    display_name: str
    domain: str
    source_path: str
    source_type: str
    matrix_field: str | None
    label_field: str


DATASETS = (
    DatasetSpec("mouse", "Mouse_retina", "scRNA", "datasets/Mouse_retina.npz", "npz", "x", "y"),
    DatasetSpec("baron", "Baron Human", "scRNA", "datasets/Baron Human.npz", "npz", "x", "y"),
    DatasetSpec("campbell", "Campbell", "scRNA", "datasets/Campbell.npz", "npz", "x", "y"),
    DatasetSpec("macosko", "Macosko", "scRNA", "/data/luolie/biopipeline/scCluBench/data/processed_scmae/Macosko.h5ad", "h5ad", None, "resolved_label"),
    DatasetSpec("melanoma", "Melanoma_5K", "scRNA", "/data/luolie/biopipeline/scCluBench/data/processed_scmae/Melanoma_5K.h5ad", "h5ad", None, "resolved_label"),
    DatasetSpec("quake", "Quake_Smart-seq2_Lung", "scRNA", "/data/luolie/biopipeline/scCluBench/data/processed/Quake_Smart-seq2_Lung.h5ad", "h5ad", None, "resolved_label"),
    DatasetSpec("wang", "Wang", "scRNA", "/data/luolie/biopipeline/scCluBench/data/processed/Wang.h5ad", "h5ad", None, "resolved_label"),
    DatasetSpec("news20", "news20", "nonbiological", "datasets/external/v22_dataset_extension_round2_20260812/processed/news20.npz", "npz_csr", None, "y"),
    DatasetSpec("rcv1", "rcv1_train", "nonbiological", "datasets/external/v22_dataset_extension_round2_20260812/processed/rcv1_train.npz", "npz_csr", None, "y"),
    DatasetSpec("arcene", "arcene", "nonbiological", "datasets/external/v19_extended_sparse_20260811/processed/arcene.npz", "npz", "x", "y"),
    DatasetSpec("sms_spam", "sms_spam_collection", "nonbiological", "datasets/sms_spam_collection.npz", "npz", "x", "y"),
)
DATASET_BY_ID = {item.identifier: item for item in DATASETS}


def resolve_source(spec: DatasetSpec) -> Path:
    source = Path(spec.source_path)
    return source if source.is_absolute() else PROJECT_ROOT / source


def resolved_config() -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "protocol_id": PROTOCOL_ID,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "datasets": [asdict(item) for item in DATASETS],
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "corruption": {
            "rate": CORRUPTION_RATE,
            "P0": "uniform coordinate-pair value swap",
            "P1": "active-active value swap; support preserving",
            "P2": "active-inactive value swap; label free support target",
            "oracle": "class-conditional support-profile swap; non-own profile is a class-size-weighted mean; labels only select corruption coordinates",
        },
        "backbone": {
            "encoder": ["d", HIDDEN_DIM, LATENT_DIM],
            "decoder": [LATENT_DIM, HIDDEN_DIM, "d"],
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_candidates": list(BATCH_CANDIDATES),
            "readout": "clean_embedding_known_K_KMeans_outer_benchmark",
        },
        "diagnostics": {
            "support_only": "binary coordinate support -> TruncatedSVD -> known-K KMeans",
            "value_only": "per-row nonzero-value quantile profile; no feature locations or active-count padding -> known-K KMeans",
            "value_profile_quantiles": VALUE_PROFILE_QUANTILES,
        },
        "decision_rule": {
            "primary_metric": "ARI",
            "support_strong": f"support-only ARI >= {SUPPORT_DIAGNOSTIC_MIN_ARI} and support-only minus value-only ARI >= {MATERIAL_DELTA_ARI}",
            "oracle_gap": "mean paired ARI(O_LABEL_ORACLE - P2_SUPPORT_TARGET)",
            "material_gap_ari": MATERIAL_DELTA_ARI,
            "case_a": "support strong and absolute oracle gap < material threshold",
            "case_b": "support strong and positive oracle gap >= material threshold",
            "case_c": "support weak and oracle gap < material threshold",
            "otherwise": "inconclusive; do not force a three-case conclusion",
        },
        "label_protocol": {
            "P0_P1_P2_training": "labels unavailable to corruption and model fit",
            "O_LABEL_ORACLE": "labels used only to precompute per-row corruption scores; model fit receives no label vector",
            "evaluation": "labels used after fit for known-K and metrics",
        },
        "legal_gpu_pool": list(LEGAL_GPU_POOL),
        "forbidden_gpu_ids": list(FORBIDDEN_GPU_IDS),
    }


def implementation_sha256() -> str:
    """Digest every V26 source file that can alter a recorded result."""
    digest = hashlib.sha256()
    tracked = sorted(Path(__file__).resolve().parent.glob("*.py"))
    tracked.append(PROJECT_ROOT / "scripts" / "V26" / "run_matrix.py")
    for path in tracked:
        if not path.exists():
            continue
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_protocol() -> None:
    if set(LEGAL_GPU_POOL) & set(FORBIDDEN_GPU_IDS):
        raise ValueError("legal and forbidden GPU pools overlap")
    if len(DATASETS) != 11 or len(DATASET_BY_ID) != 11:
        raise ValueError("V26 must contain exactly the user-selected eleven datasets")
    if ARMS != ("CLEAN", "P0_RANDOM", "P1_SUPPORT_PRESERVE", "P2_SUPPORT_TARGET", "O_LABEL_ORACLE"):
        raise ValueError("V26 arm contract drifted")
    if len(SEEDS) != 3 or len(set(SEEDS)) != 3:
        raise ValueError("V26 requires three paired seeds")
    if not 0.0 < CORRUPTION_RATE < 1.0:
        raise ValueError("invalid support corruption rate")
