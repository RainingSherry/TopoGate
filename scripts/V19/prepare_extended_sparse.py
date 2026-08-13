#!/usr/bin/env python
"""Prepare a preregistered sparse/high-dimensional V19 extension panel.

The panel is selected from input semantics before reading any RG/scMAE
results.  New UCI archives are converted from the official train+validation
files; labels are retained only for the outer benchmark K/metric protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "datasets" / "external" / "v19_extended_sparse_20260811"
RAW_ROOT = DATA_ROOT / "raw" / "uci"
PROCESSED_ROOT = DATA_ROOT / "processed"
MANIFEST_ROOT = ROOT / "result" / "V19"
MANIFEST_PATH = MANIFEST_ROOT / "v19_rg_extended_sparse_manifest_20260811.json"
DERIVED_FEATURE_CAP = 2_000

UCI_SOURCES: dict[str, dict[str, Any]] = {
    "arcene": {
        "dataset_id": "arcene__uci_highdim",
        "uci_id": 167,
        "archive_name": "arcene.zip",
        "url": "https://archive.ics.uci.edu/dataset/167/arcene",
        "download_url": "https://archive.ics.uci.edu/static/public/167/arcene.zip",
        "data_type": "dense",
        "train_data": "ARCENE/arcene_train.data",
        "train_labels": "ARCENE/arcene_train.labels",
        "valid_data": "ARCENE/arcene_valid.data",
        "valid_labels": "arcene_valid.labels",
        "n_features": 10_000,
        "citation": "UCI Machine Learning Repository, Arcene",
    },
    "dexter": {
        "dataset_id": "dexter__uci_sparse_highdim",
        "uci_id": 168,
        "archive_name": "dexter.zip",
        "url": "https://archive.ics.uci.edu/dataset/168/dexter",
        "download_url": "https://archive.ics.uci.edu/static/public/168/dexter.zip",
        "data_type": "sparse",
        "train_data": "DEXTER/dexter_train.data",
        "train_labels": "DEXTER/dexter_train.labels",
        "valid_data": "DEXTER/dexter_valid.data",
        "valid_labels": "dexter_valid.labels",
        "n_features": 20_000,
        "citation": "UCI Machine Learning Repository, Dexter",
    },
    "dorothea": {
        "dataset_id": "dorothea__uci_sparse_highdim",
        "uci_id": 169,
        "archive_name": "dorothea.zip",
        "url": "https://archive.ics.uci.edu/dataset/169/dorothea",
        "download_url": "https://archive.ics.uci.edu/static/public/169/dorothea.zip",
        "data_type": "sparse",
        "train_data": "DOROTHEA/dorothea_train.data",
        "train_labels": "DOROTHEA/dorothea_train.labels",
        "valid_data": "DOROTHEA/dorothea_valid.data",
        "valid_labels": "dorothea_valid.labels",
        "n_features": 100_000,
        "citation": "UCI Machine Learning Repository, Dorothea",
    },
    "gisette": {
        "dataset_id": "gisette__uci_highdim_dense",
        "uci_id": 170,
        "archive_name": "gisette.zip",
        "url": "https://archive.ics.uci.edu/dataset/170/gisette",
        "download_url": "https://archive.ics.uci.edu/static/public/170/gisette.zip",
        "data_type": "dense",
        "train_data": "GISETTE/gisette_train.data",
        "train_labels": "GISETTE/gisette_train.labels",
        "valid_data": "GISETTE/gisette_valid.data",
        "valid_labels": "gisette_valid.labels",
        "n_features": 5_000,
        "citation": "UCI Machine Learning Repository, Gisette",
    },
    "madelon": {
        "dataset_id": "madelon__uci_highdim_control",
        "uci_id": 171,
        "archive_name": "madelon.zip",
        "url": "https://archive.ics.uci.edu/dataset/171/madelon",
        "download_url": "https://archive.ics.uci.edu/static/public/171/madelon.zip",
        "data_type": "dense",
        "train_data": "MADELON/madelon_train.data",
        "train_labels": "MADELON/madelon_train.labels",
        "valid_data": "MADELON/madelon_valid.data",
        "valid_labels": "madelon_valid.labels",
        "n_features": 500,
        "citation": "UCI Machine Learning Repository, Madelon",
    },
}

# This list is fixed before any extension-panel performance is inspected.
LOCAL_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "dataset_id": "fbis_wc__local_sparse_text",
        "name": "fbis.wc",
        "source_path": ROOT / "datasets" / "fbis.wc.npz",
        "input_protocol": "shared_text",
        "family": "sparse_text",
        "selection_basis": "existing local high-dimensional sparse text matrix",
        "source_provenance_status": "local_snapshot_unresolved_source_metadata",
    },
    {
        "dataset_id": "tr45_wc__local_sparse_text",
        "name": "tr45.wc",
        "source_path": ROOT / "datasets" / "tr45.wc.npz",
        "input_protocol": "shared_text",
        "family": "sparse_text",
        "selection_basis": "existing local high-dimensional sparse text matrix",
        "source_provenance_status": "local_snapshot_unresolved_source_metadata",
    },
    {
        "dataset_id": "fabert__local_sparse_text",
        "name": "fabert",
        "source_path": ROOT / "datasets" / "fabert.npz",
        "input_protocol": "shared_text",
        "family": "sparse_text",
        "selection_basis": "existing local high-dimensional sparse text matrix",
        "source_provenance_status": "local_snapshot_unresolved_source_metadata",
    },
    {
        "dataset_id": "micro_mass__local_sparse_highdim",
        "name": "micro-mass",
        "source_path": ROOT / "datasets" / "micro-mass.npz",
        "input_protocol": "clubench_bridge",
        "family": "sparse_highdim",
        "selection_basis": "existing local sparse high-dimensional mass-spectrum matrix",
        "source_provenance_status": "local_snapshot_unresolved_source_metadata",
    },
    {
        "dataset_id": "gina_prior2__local_sparse_highdim",
        "name": "gina_prior2",
        "source_path": ROOT / "datasets" / "gina_prior2.npz",
        "input_protocol": "clubench_bridge",
        "family": "sparse_highdim_control",
        "selection_basis": "existing local high-dimensional sparse benchmark matrix",
        "source_provenance_status": "local_snapshot_unresolved_source_metadata",
    },
    {
        "dataset_id": "internet_advertisements__uci_sparse",
        "name": "internet_advertisements",
        "source_path": ROOT / "datasets" / "external" / "v9_related_20260806" / "processed" / "internet_advertisements.npz",
        "input_protocol": "clubench_bridge",
        "family": "web_sparse",
        "selection_basis": "existing provenance-audited UCI sparse web matrix",
        "source_provenance_status": "reused_v9_external_manifest",
        "source_manifest": "datasets/external/v9_related_20260806/processed/v9_external_manifest.json",
    },
    {
        "dataset_id": "sms_spam_full__uci_sparse_text",
        "name": "sms_spam_collection_full_tfidf500",
        "source_path": ROOT / "datasets" / "external" / "v9_related_20260806" / "processed" / "sms_spam_collection_full_tfidf500.npz",
        "input_protocol": "shared_text",
        "family": "sparse_text",
        "selection_basis": "existing provenance-audited UCI sparse text matrix",
        "source_provenance_status": "reused_v9_external_manifest",
        "source_manifest": "datasets/external/v9_related_20260806/processed/v9_external_manifest.json",
    },
    {
        "dataset_id": "quake_smartseq2_lung__local_sparse_expression",
        "name": "Quake_Smart-seq2_Lung",
        "source_path": ROOT / "datasets" / "Quake_Smart-seq2_Lung.npz",
        "input_protocol": "clubench_bridge",
        "family": "scRNA_sparse_highdim",
        "selection_basis": "existing high-dimensional zero-inflated single-cell expression matrix",
        "source_provenance_status": "local_snapshot_existing_dataset_registry",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_lines(zf: zipfile.ZipFile, member: str) -> list[str]:
    with zf.open(member) as handle:
        return handle.read().decode("utf-8", errors="replace").splitlines()


def _read_labels(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    values: list[str] = []
    for line in _read_lines(zf, member):
        value = line.strip()
        if value:
            values.append(value)
    if not values:
        raise ValueError(f"empty label file: {member}")
    # Preserve signed labels as integers; the V19 outer runner encodes them
    # only after preprocessing and uses their cardinality for benchmark K.
    return np.asarray([int(float(value)) for value in values], dtype=np.int64)


def _read_dense(zf: zipfile.ZipFile, member: str, n_features: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for line in _read_lines(zf, member):
        if not line.strip():
            continue
        row = np.fromstring(line, sep=" ", dtype=np.float32)
        if row.size != int(n_features):
            raise ValueError(f"{member}: expected {n_features} values, got {row.size}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty dense data file: {member}")
    return np.vstack(rows).astype(np.float32, copy=False)


def _read_sparse(zf: zipfile.ZipFile, member: str, n_features: int) -> sp.csr_matrix:
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    row = 0
    for line in _read_lines(zf, member):
        if not line.strip():
            continue
        for token in line.strip().split():
            if ":" in token:
                index_text, value_text = token.split(":", 1)
            else:
                # Dorothea uses a sparse-binary format with bare one-based
                # feature indices; Dexter uses index:value pairs.
                index_text, value_text = token, "1.0"
            index = int(index_text) - 1
            if not 0 <= index < int(n_features):
                raise ValueError(f"{member}: feature index out of range: {index + 1}")
            value = float(value_text)
            if value != 0.0:
                row_indices.append(row)
                col_indices.append(index)
                values.append(value)
        row += 1
    if row == 0:
        raise ValueError(f"empty sparse data file: {member}")
    matrix = sp.csr_matrix(
        (np.asarray(values, dtype=np.float32), (row_indices, col_indices)),
        shape=(row, int(n_features)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix


def _write_npz(path: Path, X: np.ndarray | sp.csr_matrix, y: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sp.issparse(X):
        matrix = sp.csr_matrix(X, dtype=np.float32)
        np.savez_compressed(
            path,
            data=matrix.data.astype(np.float32, copy=False),
            indices=matrix.indices.astype(np.int64, copy=False),
            indptr=matrix.indptr.astype(np.int64, copy=False),
            shape=np.asarray(matrix.shape, dtype=np.int64),
            y=np.asarray(y, dtype=np.int64),
        )
        nnz = int(matrix.nnz)
        n_rows, n_features = matrix.shape
        storage = "csr_npz"
    else:
        array = np.asarray(X, dtype=np.float32)
        np.savez_compressed(path, x=array, y=np.asarray(y, dtype=np.int64))
        nnz = int(np.count_nonzero(array))
        n_rows, n_features = array.shape
        storage = "dense_npz"
    return {
        "n_samples": int(n_rows),
        "n_features": int(n_features),
        "nnz": nnz,
        "zero_fraction": float(1.0 - nnz / float(n_rows * n_features)),
        "storage": storage,
        "labels_unique": int(np.unique(y).size),
    }


def _variance_subset(
    X: np.ndarray | sp.csr_matrix,
    feature_cap: int,
) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray]:
    """Select a fixed number of highest-variance features without labels."""
    if int(X.shape[1]) <= int(feature_cap):
        return X, np.arange(int(X.shape[1]), dtype=np.int64)
    if sp.issparse(X):
        matrix = sp.csr_matrix(X, dtype=np.float32)
        mean = np.asarray(matrix.mean(axis=0)).reshape(-1).astype(np.float64)
        mean_square = np.asarray(matrix.multiply(matrix).mean(axis=0)).reshape(-1).astype(np.float64)
        variance = np.nan_to_num(mean_square - mean * mean, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    else:
        array = np.asarray(X, dtype=np.float32)
        variance = np.nan_to_num(np.var(array, axis=0, dtype=np.float64), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    order = np.lexsort((np.arange(variance.size, dtype=np.int64), -variance))
    selected = np.sort(order[: int(feature_cap)]).astype(np.int64, copy=False)
    return X[:, selected], selected


def prepare_uci_sources() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, source in UCI_SOURCES.items():
        archive = RAW_ROOT / source["archive_name"]
        if not archive.is_file():
            raise FileNotFoundError(f"missing downloaded UCI archive: {archive}")
        with zipfile.ZipFile(archive) as zf:
            train_labels = _read_labels(zf, source["train_labels"])
            valid_labels = _read_labels(zf, source["valid_labels"])
            if source["data_type"] == "sparse":
                train = _read_sparse(zf, source["train_data"], int(source["n_features"]))
                valid = _read_sparse(zf, source["valid_data"], int(source["n_features"]))
                X: np.ndarray | sp.csr_matrix = sp.vstack([train, valid], format="csr")
            else:
                train = _read_dense(zf, source["train_data"], int(source["n_features"]))
                valid = _read_dense(zf, source["valid_data"], int(source["n_features"]))
                X = np.vstack([train, valid]).astype(np.float32, copy=False)
            y = np.concatenate([train_labels, valid_labels]).astype(np.int64, copy=False)
        if int(X.shape[0]) != int(y.size):
            raise ValueError(f"{key}: X/y row mismatch")
        source_shape = tuple(int(value) for value in X.shape)
        selected_indices = None
        feature_selection = None
        if key == "dorothea":
            X, selected_indices = _variance_subset(X, DERIVED_FEATURE_CAP)
            feature_selection = {
                "strategy": "variance_top_k_label_free",
                "requested": DERIVED_FEATURE_CAP,
                "source_features": source_shape[1],
                "selected_features": int(X.shape[1]),
                "fit_rows": "all_train_plus_validation_rows",
                "labels_used": False,
            }
        output = PROCESSED_ROOT / f"{key}.npz"
        profile = _write_npz(output, X, y)
        profile["source_n_features"] = source_shape[1]
        if feature_selection is not None:
            profile["feature_selection"] = feature_selection
            np.save(PROCESSED_ROOT / f"{key}.selected_feature_indices.npy", selected_indices)
        rows.append(
            {
                "dataset_id": source["dataset_id"],
                "name": key,
                "source_path": str(output.resolve()),
                "source_kind": "UCI",
                "source_identity": f"uci:dataset_id={source['uci_id']}",
                "source_url": source["url"],
                "download_url": source["download_url"],
                "raw_archive": str(archive.resolve()),
                "raw_archive_sha256": _sha256(archive),
                "processed_sha256": _sha256(output),
                "citation": source["citation"],
                "input_protocol": "clubench_bridge",
                "input_kind": "sparse_highdim_features",
                "family": "uci_sparse_highdim" if source["data_type"] == "sparse" else "uci_highdim_control",
                "selection_uses_labels_or_outcomes": False,
                "selection_basis": "pre-registered UCI high-dimensional benchmark input semantics",
                "comparison_scope": "external_highdim_bridge_only",
                "train_valid_combined": True,
                "labels_used_during_fit": False,
                "profile": profile,
                "source_n_features": source_shape[1],
                "feature_selection": feature_selection,
                "status": "eligible",
            }
        )
    return rows


def inspect_local_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    source = Path(candidate["source_path"])
    if not source.is_file():
        return {**candidate, "source_path": str(source.resolve()), "status": "ineligible", "ineligible_reason": "source_missing"}
    with np.load(source, allow_pickle=False) as payload:
        x_key = next((key for key in ("X", "x", "features", "data") if key in payload.files), None)
        y_key = next((key for key in ("y", "labels", "label") if key in payload.files), None)
        if x_key is None:
            raise ValueError(f"{source}: no feature matrix key")
        X = payload[x_key]
        y = None if y_key is None else np.asarray(payload[y_key]).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"{source}: feature matrix must be 2-D")
        nnz = int(np.count_nonzero(X))
        profile = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "nnz": nnz,
            "zero_fraction": float(1.0 - nnz / float(X.size)),
            "storage": "dense_npz",
            "labels_unique": None if y is None else int(np.unique(y).size),
        }
    return {
        **candidate,
        "source_path": str(source.resolve()),
        "source_hash": "unavailable",
        "profile": profile,
        "status": "eligible",
        "selection_uses_labels_or_outcomes": False,
        "comparison_scope": "external_highdim_bridge_only",
    }


def build_manifest(uci_rows: list[dict[str, Any]]) -> dict[str, Any]:
    local_rows = [inspect_local_candidate(candidate) for candidate in LOCAL_CANDIDATES]
    rows = local_rows + uci_rows
    payload = {
        "protocol_id": "v19_rg_extended_sparse_v1",
        "manifest_id": "v19_rg_extended_sparse_manifest_20260811",
        "description": "Pre-registered sparse/high-dimensional extension panel for matched RG/scMAE",
        "selection_policy": {
            "selection_uses_labels_or_outcomes": False,
            "selection_basis": "dataset input semantics and source availability fixed before extension results",
            "target_claim": "at least five RG-positive datasets is a post-hoc success criterion, not a selection rule",
            "train_valid_combination": "UCI train and validation rows are combined before fitting; labels stay outside fit",
        },
        "variants": ["scmae_only", "rg_full"],
        "formal_seeds_in_order": [42, 123, 7],
        "expected_dataset_count": len(rows),
        "expected_runs_total": len(rows) * 2 * 3,
        "input_protocols": sorted({str(row["input_protocol"]) for row in rows}),
        "comparison_scope": "external_highdim_bridge_only; no archived SOTA row is implied",
        "datasets": rows,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    (DATA_ROOT / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "manifest.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-uci", action="store_true", help="convert downloaded UCI archives")
    args = parser.parse_args()
    UCI_ROWS = prepare_uci_sources() if args.prepare_uci else []
    payload = build_manifest(UCI_ROWS)
    print(json.dumps({
        "manifest": str(MANIFEST_PATH),
        "datasets": len(payload["datasets"]),
        "eligible": sum(row.get("status") == "eligible" for row in payload["datasets"]),
        "expected_runs": payload["expected_runs_total"],
        "uci_converted": len(UCI_ROWS),
    }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
