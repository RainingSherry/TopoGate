from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V23_cycle_response.data import load_matrix_only


FIXED_STAGE1_SOURCES: tuple[dict[str, str], ...] = (
    {
        "dataset_id": "quake_smartseq2_lung__scrna",
        "name": "Quake_Smart-seq2_Lung",
        "family": "scRNA",
        "input_protocol": "scRNA_count",
        "source_path": "datasets/Quake_Smart-seq2_Lung.npz",
    },
    {
        "dataset_id": "sector__sparse_text",
        "name": "sector",
        "family": "sparse_text",
        "input_protocol": "shared_text",
        "source_path": "datasets/external/v22_dataset_extension_20260812/processed/sector.npz",
    },
    {
        "dataset_id": "micro_mass__mass_spectrum",
        "name": "micro-mass",
        "family": "mass_spectrum",
        "input_protocol": "clubench_bridge",
        "source_path": "datasets/micro-mass.npz",
    },
    {
        "dataset_id": "internet_advertisements__web_sparse",
        "name": "internet_advertisements",
        "family": "web_sparse",
        "input_protocol": "clubench_bridge",
        "source_path": "datasets/external/v9_related_20260806/processed/internet_advertisements.npz",
    },
    {
        "dataset_id": "madelon__dense_redundant_control",
        "name": "madelon",
        "family": "dense_redundant_control",
        "input_protocol": "clubench_bridge",
        "source_path": "datasets/external/v19_extended_sparse_20260811/processed/madelon.npz",
    },
    {
        "dataset_id": "gisette__image_features",
        "name": "gisette",
        "family": "image_features",
        "input_protocol": "clubench_bridge",
        "source_path": "datasets/external/v19_extended_sparse_20260811/processed/gisette.npz",
    },
)


def _load_source(path: Path) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray | None]:
    matrix = load_matrix_only(path)
    labels = None
    with np.load(path, allow_pickle=False) as payload:
        for key in ("y", "labels", "label"):
            if key in payload.files:
                labels = np.asarray(payload[key]).reshape(-1)
                break
    return matrix, labels


def prepare(output_root: Path, selected: set[str] | None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source_record in FIXED_STAGE1_SOURCES:
        if selected is not None and source_record["dataset_id"] not in selected:
            continue
        source_path = (ROOT / source_record["source_path"]).resolve()
        matrix, labels = _load_source(source_path)
        dataset_dir = output_root / source_record["dataset_id"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = dataset_dir / "matrix_only.npz"
        labels_path = dataset_dir / "labels_true.npy"
        if sp.issparse(matrix):
            sp.save_npz(matrix_path, matrix.tocsr().astype(np.float32))
        else:
            np.savez_compressed(matrix_path, X=np.asarray(matrix, dtype=np.float32))
        if labels is None:
            labels_path_value = None
        else:
            np.save(labels_path, labels)
            labels_path_value = str(labels_path.resolve())
        records.append(
            source_record
            | {
                "source_path": str(source_path),
                "matrix_path": str(matrix_path.resolve()),
                "labels_path": labels_path_value,
                "n_samples": int(matrix.shape[0]),
                "n_features_original": int(matrix.shape[1]),
                "labels_available_outer_only": labels is not None,
                "labels_accessible_during_fit": False,
                "labels_accessible_during_profile": False,
                "selection_uses_labels_or_results": False,
            }
        )
    found = {record["dataset_id"] for record in records}
    if selected is not None and found != selected:
        raise ValueError(f"unknown or missing dataset ids: {sorted(selected - found)}")
    manifest = {
        "manifest_id": "v23_cycle_response_stage1_cross_domain_v1",
        "protocol_id": "v23_cycle_response_protocol_a_v1",
        "selection_uses_labels_or_results": False,
        "stage1_panel_frozen_before_results": True,
        "seeds": [42, 123, 7],
        "records": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare physically label-isolated V23 cross-domain inputs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "datasets" / "external" / "v23_cycle_response_stage1",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    manifest = prepare(args.output_root, None if args.datasets is None else set(args.datasets))
    print(json.dumps({"manifest_id": manifest["manifest_id"], "records": len(manifest["records"])}, indent=2))


if __name__ == "__main__":
    main()
