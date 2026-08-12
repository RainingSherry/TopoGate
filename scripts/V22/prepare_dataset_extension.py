#!/usr/bin/env python3
"""Download and register the fixed V22 dataset extension panel.

The candidate list is declared before any V22 result is inspected.  Labels in
the LibSVM files are retained only as outer benchmark metadata; conversion and
all model fitting use X alone.  PBMC3k is intentionally registered as an
unlabelled deployment-style dataset because the 10x archive has no cell-type
ground truth.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import requests
import urllib3

urllib3.disable_warnings()


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "datasets" / "external" / "v22_dataset_extension_20260812"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

# Fixed before reading any new model result.
LIBSVM_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "dataset_id": "sector__libsvm_sparse_highdim",
        "name": "sector",
        "family": "sparse_highdim_text",
        "input_protocol": "shared_text",
        "source_kind": "LibSVMTools",
        "source_identity": "LIBSVM multiclass/sector",
        "citation": "LIBSVM Data: sector multiclass benchmark",
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/sector/sector.scale.bz2",
        "archive_name": "sector.scale.bz2",
    },
    {
        "dataset_id": "real_sim__libsvm_sparse_highdim",
        "name": "real-sim",
        "family": "sparse_highdim_text",
        "input_protocol": "shared_text",
        "source_kind": "LibSVMTools",
        "source_identity": "LIBSVM binary/real-sim",
        "citation": "LIBSVM Data: real-sim binary benchmark",
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "archive_name": "real-sim.bz2",
    },
    {
        "dataset_id": "covtype__libsvm_dense_control",
        "name": "covtype",
        "family": "dense_highsample_control",
        "input_protocol": "clubench_bridge",
        "source_kind": "LibSVMTools",
        "source_identity": "LIBSVM multiclass/covtype",
        "citation": "LIBSVM Data: covtype multiclass benchmark",
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/covtype.scale.bz2",
        "archive_name": "covtype.scale.bz2",
    },
)

PBMC3K = {
    "dataset_id": "pbmc3k__10x_unlabelled_count",
    "name": "PBMC3k",
    "family": "scRNA_count_unlabelled",
    "input_protocol": "scRNA_count",
    "source_kind": "10x Genomics",
    "source_identity": "10x PBMC3k filtered gene-barcode matrix",
    "citation": "10x Genomics PBMC3k public count matrix",
    "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz",
    "archive_name": "pbmc3k_filtered_gene_bc_matrices.tar.gz",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    headers = {"User-Agent": "TopoGate-V22-dataset-preparer/1.0"}
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=(30, 180), verify=False)
            response.raise_for_status()
            with path.open("wb") as target:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        target.write(chunk)
            response.close()
            if path.stat().st_size <= 0:
                raise RuntimeError(f"empty download: {url}")
            return
        except Exception as exc:  # retry transient proxy disconnects
            last_error = exc
            if path.exists():
                path.unlink()
    raise RuntimeError(f"download failed after retries: {url}: {last_error}")


def _parse_libsvm(path: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    labels: list[float] = []
    max_col = -1
    row = 0
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            labels.append(float(tokens[0]))
            for token in tokens[1:]:
                index_text, value_text = token.split(":", 1)
                col = int(index_text) - 1
                if col < 0:
                    raise ValueError(f"invalid LIBSVM index in {path}: {token}")
                value = float(value_text)
                if value != 0.0:
                    rows.append(row)
                    cols.append(col)
                    values.append(value)
                max_col = max(max_col, col)
            row += 1
    if row == 0 or max_col < 0:
        raise ValueError(f"empty LIBSVM file: {path}")
    matrix = sp.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=(row, max_col + 1),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix, np.asarray(labels)


def _write_npz(path: Path, matrix: sp.csr_matrix, labels: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices.astype(np.int64, copy=False),
        indptr=matrix.indptr.astype(np.int64, copy=False),
        shape=np.asarray(matrix.shape, dtype=np.int64),
        y=np.asarray(labels),
    )
    return {
        "n_samples": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "zero_fraction": float(1.0 - matrix.nnz / float(matrix.shape[0] * matrix.shape[1])),
        "storage": "csr_npz",
        "labels_unique": int(np.unique(labels).size),
    }


def _prepare_libsvm(candidate: dict[str, Any]) -> dict[str, Any]:
    raw = RAW_ROOT / candidate["archive_name"]
    processed = PROCESSED_ROOT / f"{candidate['name'].replace('-', '_')}.npz"
    _download(candidate["url"], raw)
    matrix, labels = _parse_libsvm(raw)
    profile = _write_npz(processed, matrix, labels)
    return {
        **candidate,
        "status": "eligible",
        "source_path": str(processed.resolve()),
        "raw_path": str(raw.resolve()),
        "raw_sha256": _sha256(raw),
        "processed_sha256": _sha256(processed),
        "labels_used_during_fit": False,
        "selection_uses_labels_or_outcomes": False,
        "selection_basis": "fixed public LibSVM sparse/high-dimensional or control candidate",
        "profile": profile,
    }


def _prepare_pbmc3k() -> dict[str, Any]:
    raw = RAW_ROOT / PBMC3K["archive_name"]
    _download(PBMC3K["url"], raw)
    extract_root = DATA_ROOT / "pbmc3k_raw"
    extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".extracted"
    if not marker.exists():
        with tarfile.open(raw, "r:*") as archive:
            archive.extractall(extract_root)
        marker.write_text("ok\n", encoding="utf-8")
    matrix_file = next(extract_root.rglob("matrix.mtx"))
    matrix_dir = matrix_file.parent
    genes_file = matrix_dir / "genes.tsv"
    barcodes_file = matrix_dir / "barcodes.tsv"
    from scipy.io import mmread

    matrix = sp.csr_matrix(mmread(matrix_file), dtype=np.float32).T.tocsr()
    genes = np.loadtxt(genes_file, dtype=str, delimiter="\t", ndmin=2)
    barcodes = np.loadtxt(barcodes_file, dtype=str, ndmin=1)
    processed = PROCESSED_ROOT / "PBMC3k.npz"
    processed.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed,
        data=matrix.data,
        indices=matrix.indices.astype(np.int64, copy=False),
        indptr=matrix.indptr.astype(np.int64, copy=False),
        shape=np.asarray(matrix.shape, dtype=np.int64),
        gene_ids=np.asarray(genes[:, 0], dtype=str),
        barcodes=np.asarray(barcodes, dtype=str),
    )
    return {
        **PBMC3K,
        "status": "eligible_unlabelled",
        "source_path": str(processed.resolve()),
        "raw_path": str(raw.resolve()),
        "raw_sha256": _sha256(raw),
        "processed_sha256": _sha256(processed),
        "labels_used_during_fit": False,
        "selection_uses_labels_or_outcomes": False,
        "selection_basis": "fixed 10x count-matrix deployment control; no independent cell-type labels in source archive",
        "profile": {
            "n_samples": int(matrix.shape[0]),
            "n_features": int(matrix.shape[1]),
            "nnz": int(matrix.nnz),
            "zero_fraction": float(1.0 - matrix.nnz / float(matrix.shape[0] * matrix.shape[1])),
            "storage": "csr_npz",
            "labels_unique": None,
        },
        "evaluation_status": "no_ari_without_external_labels",
    }


def build_manifest(download: bool) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for candidate in LIBSVM_CANDIDATES:
        if download:
            records.append(_prepare_libsvm(candidate))
        else:
            records.append({**candidate, "status": "download_pending"})
    if download:
        records.append(_prepare_pbmc3k())
    else:
        records.append({**PBMC3K, "status": "download_pending"})
    return {
        "manifest_id": "v22_dataset_extension_manifest_20260812",
        "protocol_id": "v22_topology_discriminator_hard_mask_v1",
        "description": "New sparse/high-dimensional, dense control, and unlabelled scRNA count datasets for V22",
        "download_tls_verify": False,
        "download_tls_note": "The local proxy certificate chain is not trusted; URLs and raw SHA256 are retained for later independent verification.",
        "selection_policy": {
            "selection_uses_labels_or_outcomes": False,
            "primary_strata": ["sparse_highdim_text", "scRNA_count_unlabelled"],
            "control_strata": ["dense_highsample_control"],
            "labels_used_during_fit": False,
            "unlabelled_records_excluded_from_ari_aggregate": True,
        },
        "datasets": records,
        "formal_seeds": [42, 123, 7],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.download and args.dry_run:
        raise SystemExit("choose --download or --dry-run")
    payload = build_manifest(download=args.download)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if args.download:
        MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
