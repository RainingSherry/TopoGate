from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


REGISTERED_SEMANTICS = {
    "Campbell": "scRNA_count_registered_source",
    "Mouse_retina": "scRNA_count_registered_source",
    "hrvatin_geo_maintype_counts": "scRNA_count_registered_source",
    "fbis.wc": "word_count_registered_source",
    "tr45.wc": "word_count_registered_source",
}


def _npy_header(archive: zipfile.ZipFile, member: str) -> dict[str, Any] | None:
    try:
        raw_info = archive.getinfo(member)
    except KeyError:
        return None
    with archive.open(raw_info) as raw:
        version = np.lib.format.read_magic(raw)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(raw)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(raw)
        elif version == (3, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(raw)
        else:
            return None
    return {
        "shape": [int(value) for value in shape],
        "dtype": np.dtype(dtype).str,
        "fortran_order": bool(fortran_order),
        "compressed": raw_info.compress_type != zipfile.ZIP_STORED,
        "member_bytes": int(raw_info.file_size),
    }


def _read_small_shape(path: Path, member: str) -> list[int] | None:
    with np.load(path, allow_pickle=False) as payload:
        if member[:-4] not in payload.files:
            return None
        return [int(value) for value in np.asarray(payload[member[:-4]]).reshape(-1)]


def _matrix_member(names: set[str]) -> str | None:
    for name in ("x.npy", "X.npy", "data_matrix.npy"):
        if name in names:
            return name
    return None


def inspect_npz(path: Path) -> dict[str, Any]:
    dataset = path.name[:-4] if path.name.endswith(".npz") else path.name
    record: dict[str, Any] = {
        "dataset": dataset,
        "path": str(path.resolve()),
        "status": "missing",
        "source_semantics": REGISTERED_SEMANTICS.get(dataset),
        "source_semantics_verified": dataset in REGISTERED_SEMANTICS,
        "labels_used_during_fit": False,
    }
    if not path.exists():
        return record
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        csr_names = {"data.npy", "indices.npy", "indptr.npy", "shape.npy"}
        if csr_names.issubset(names):
            shape = _read_small_shape(path, "shape.npy")
            data_header = _npy_header(archive, "data.npy")
            indptr_header = _npy_header(archive, "indptr.npy")
            record.update(
                {
                    "storage": "csr_bundle",
                    "matrix_shape": shape,
                    "matrix_dtype": None if data_header is None else data_header["dtype"],
                    "nnz": None if data_header is None else int(np.prod(data_header["shape"])),
                    "indptr_length": None if indptr_header is None else int(np.prod(indptr_header["shape"])),
                    "matrix_member_compressed": None if data_header is None else data_header["compressed"],
                    "dense_input_warning": False,
                }
            )
        else:
            matrix_member = _matrix_member(names)
            header = None if matrix_member is None else _npy_header(archive, matrix_member)
            shape = None if header is None else header["shape"]
            record.update(
                {
                    "storage": "dense_member" if header is not None else "unknown",
                    "matrix_member": matrix_member,
                    "matrix_shape": shape,
                    "matrix_dtype": None if header is None else header["dtype"],
                    "matrix_member_bytes": None if header is None else header["member_bytes"],
                    "matrix_member_compressed": None if header is None else header["compressed"],
                    "dense_input_warning": header is not None,
                }
            )
        y_header = _npy_header(archive, "y.npy")
        record["label_shape"] = None if y_header is None else y_header["shape"]
    shape = record.get("matrix_shape")
    label_shape = record.get("label_shape")
    if shape is None:
        record["status"] = "matrix_member_not_found"
    elif record["dense_input_warning"]:
        record["status"] = "dense_storage_warning"
    elif not record["source_semantics_verified"]:
        record["status"] = "source_semantics_unverified"
    else:
        record["status"] = "sparse_input_candidate"
    if shape is not None and len(shape) == 2:
        record["n_samples"] = int(shape[0])
        record["n_features"] = int(shape[1])
        record["estimated_dense_bytes"] = int(np.prod(shape) * np.dtype(record["matrix_dtype"]).itemsize)
    if label_shape is not None and len(label_shape) > 0:
        record["n_labels"] = int(label_shape[0])
        record["shape_labels_match"] = shape is not None and int(label_shape[0]) == int(shape[0])
    else:
        record["n_labels"] = None
        record["shape_labels_match"] = None
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="V17 read-only NPZ input audit")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Campbell", "Mouse_retina", "hrvatin", "enron", "fbis.wc", "tr45.wc", "20newsgroups"],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    records = [inspect_npz(root / f"{name}.npz") for name in args.datasets]
    result = {
        "stage": "V17_input_audit",
        "status": "read_only",
        "labels_used_during_fit": False,
        "hashes_computed": False,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    for record in records:
        print(
            record["dataset"],
            record["status"],
            record.get("matrix_shape"),
            record.get("storage"),
            record.get("source_semantics"),
        )


if __name__ == "__main__":
    main()
