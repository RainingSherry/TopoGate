from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp


def _encode_labels(values: Any) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.dtype.kind in {"S", "U", "O"}:
        decoded = np.asarray(
            [value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value) for value in array]
        )
        _, encoded = np.unique(decoded, return_inverse=True)
        return encoded.astype(np.int64)
    _, encoded = np.unique(array, return_inverse=True)
    return encoded.astype(np.int64)


def _read_h5(path: Path, matrix_key: str, label_key: str, chunk_rows: int) -> tuple[sp.csr_matrix, np.ndarray]:
    with h5py.File(path, "r") as handle:
        matrix = handle[matrix_key]
        labels = _encode_labels(handle[label_key][...])
        blocks: list[sp.csr_matrix] = []
        for start in range(0, int(matrix.shape[0]), int(chunk_rows)):
            block = np.asarray(matrix[start : start + int(chunk_rows)])
            blocks.append(sp.csr_matrix(block))
    return sp.vstack(blocks, format="csr"), labels


def _read_h5ad(
    path: Path,
    label_key: str,
    matrix_key: str,
    chunk_rows: int,
) -> tuple[sp.csr_matrix, np.ndarray]:
    import anndata as ad

    data = ad.read_h5ad(path, backed="r")
    try:
        labels = _encode_labels(data.obs[label_key].to_numpy())
        if matrix_key == "X":
            matrix = data.X
        elif matrix_key == "raw.X":
            if data.raw is None:
                raise ValueError("H5AD does not contain raw.X")
            matrix = data.raw.X
        elif matrix_key.startswith("layers/"):
            matrix = data.layers[matrix_key.split("/", 1)[1]]
        else:
            raise ValueError(f"unsupported H5AD matrix key: {matrix_key}")
        blocks: list[sp.csr_matrix] = []
        for start in range(0, int(data.n_obs), int(chunk_rows)):
            blocks.append(sp.csr_matrix(matrix[start : start + int(chunk_rows)]))
    finally:
        data.file.close()
    return sp.vstack(blocks, format="csr"), labels


def convert(
    source: str | Path,
    output: str | Path,
    *,
    label_key: str = "Y",
    matrix_key: str = "X",
    count_semantics: str = "raw_count",
    chunk_rows: int = 256,
) -> dict[str, Any]:
    source_path = Path(source)
    output_path = Path(output)
    if source_path.suffix.lower() == ".h5ad":
        matrix, labels = _read_h5ad(source_path, label_key, matrix_key, chunk_rows)
    else:
        matrix, labels = _read_h5(source_path, matrix_key, label_key, chunk_rows)
    values = np.asarray(matrix.data, dtype=np.float64)
    if values.size and np.min(values) < 0.0:
        raise ValueError("source matrix contains negative values")
    if count_semantics == "log1p_count":
        recovered = np.rint(np.expm1(values))
        if values.size and not np.allclose(values, np.log1p(recovered), atol=3e-5, rtol=3e-5):
            raise ValueError("source matrix is not exactly recoverable log1p(count)")
        values = recovered
    elif values.size and not np.allclose(values, np.rint(values), atol=1e-6, rtol=0.0):
        raise ValueError("source matrix is not a non-negative integer count matrix")
    matrix = sp.csr_matrix((np.rint(values).astype(np.int64), matrix.indices, matrix.indptr), shape=matrix.shape)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.shape[0] != labels.size:
        raise ValueError("matrix rows and labels have different lengths")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        data=matrix.data,
        indices=matrix.indices.astype(np.int64),
        indptr=matrix.indptr.astype(np.int64),
        shape=np.asarray(matrix.shape, dtype=np.int64),
        y=labels,
    )
    metadata = {
        "source_path": str(source_path.resolve()),
        "output_path": str(output_path.resolve()),
        "n": int(matrix.shape[0]),
        "d": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "count_semantics": count_semantics,
        "label_key": label_key,
        "matrix_key": matrix_key,
        "labels_used_during_fit": False,
        "storage": "sparse_npz_csr",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw count H5/H5AD to V16.1 CSR bundle")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-key", default="Y")
    parser.add_argument("--matrix-key", default="X")
    parser.add_argument("--count-semantics", default="raw_count")
    parser.add_argument("--chunk-rows", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(convert(**vars(args)), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
