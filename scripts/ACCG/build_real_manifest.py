#!/usr/bin/env python3
"""Freeze an outcome-independent ACCG real panel and its main/ablation jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 7)
MAIN_CONFIG = ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_joint.yaml"
ABLATION_CONFIGS = {
    "coordinate": ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_coordinate.yaml",
    "shuffled_graph": ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_shuffled_graph.yaml",
    "marginal_only": ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_marginal_only.yaml",
    "abstention_sensitivity": ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_joint_abstain.yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix_profile(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        labels = None
        for key in ("y", "labels", "label"):
            if key in payload.files:
                labels = np.asarray(payload[key]).reshape(-1)
                break
        if {"data", "indices", "indptr", "shape"}.issubset(payload.files):
            shape = tuple(int(value) for value in np.asarray(payload["shape"]).reshape(-1))
            matrix = sp.csr_matrix(
                (payload["data"], payload["indices"], payload["indptr"]), shape=shape
            )
            zero_fraction = 1.0 - float(matrix.nnz / max(1, shape[0] * shape[1]))
            n_samples, n_features, sparse_storage = shape[0], shape[1], True
        else:
            matrix = None
            for key in ("X", "x", "features", "data"):
                if key in payload.files:
                    matrix = np.asarray(payload[key])
                    break
            if matrix is None:
                raise ValueError(f"cannot locate a matrix in {path}")
            if matrix.ndim != 2:
                raise ValueError(f"matrix must be 2D: {path}")
            n_samples, n_features, sparse_storage = matrix.shape[0], matrix.shape[1], False
            zero_fraction = float(np.mean(matrix == 0.0))
    if labels is not None and labels.size != n_samples:
        raise ValueError(f"label length does not match matrix rows: {path}")
    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "sparse_storage": bool(sparse_storage),
        "zero_fraction": float(zero_fraction),
        "labels_present": labels is not None,
        "labels_unique": None if labels is None else int(np.unique(labels.astype(str)).size),
    }


def build_manifest(
    spec_path: Path,
    output_root: Path,
    *,
    manifest_id: str = "accg_locked_real_panel_v1",
    protocol_id: str = "accg_action_conditional_joint_v1",
) -> dict[str, Any]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    rows = spec.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset spec must contain a non-empty datasets list")
    forbidden = {"ari", "nmi", "acc", "result", "outcome", "selected_because"}
    records = []
    for source_row in rows:
        if not isinstance(source_row, dict) or forbidden.intersection(source_row):
            raise ValueError("dataset rows cannot contain outcomes or result-dependent selection fields")
        for key in ("dataset_id", "name", "source_path", "domain", "source_family", "input_protocol", "license"):
            if not source_row.get(key):
                raise ValueError(f"dataset row is missing {key}")
        source = Path(str(source_row["source_path"]))
        if not source.is_absolute():
            source = spec_path.parent / source
        if not source.is_file():
            raise FileNotFoundError(source)
        matrix_profile = _matrix_profile(source)
        if matrix_profile["n_features"] > 2000:
            raise ValueError(f"{source_row['dataset_id']} exceeds the frozen 2000-feature interface")
        explicit_k = source_row.get("n_clusters")
        if explicit_k is not None and int(explicit_k) <= 1:
            raise ValueError(f"{source_row['dataset_id']} has invalid n_clusters")
        if explicit_k is None and not matrix_profile["labels_present"]:
            raise ValueError(f"{source_row['dataset_id']} needs labels or an explicit n_clusters")
        if (
            explicit_k is not None
            and matrix_profile["labels_unique"] is not None
            and int(explicit_k) != int(matrix_profile["labels_unique"])
        ):
            raise ValueError(f"{source_row['dataset_id']} explicit n_clusters disagrees with benchmark labels")
        records.append(
            {
                **source_row,
                "source_path": str(source.resolve()),
                "source_sha256": _sha256(source),
                **matrix_profile,
                "K_source": "explicit_n_clusters" if explicit_k is not None else "benchmark_oracle_from_y",
                "labels_used_for_selection": False,
                "outcomes_inspected_before_freeze": False,
                "status": "eligible",
            }
        )
    domains = {str(row["domain"]) for row in records}
    if len(records) < 8 or len(records) > 12:
        raise ValueError("the first ACCG panel must contain 8-12 datasets")
    if len(domains) < 2:
        raise ValueError("the ACCG panel must cover at least two domains")
    development = set(str(value) for value in spec.get("development_subset", []))
    unknown = development - {str(row["dataset_id"]) for row in records}
    if unknown:
        raise ValueError(f"unknown development dataset ids: {sorted(unknown)}")
    jobs = []
    for record in records:
        dataset_id = str(record["dataset_id"])
        for seed in SEEDS:
            main_output = output_root / "main" / dataset_id / f"seed{seed}"
            jobs.append(
                {
                    "run_key": f"{manifest_id}::{dataset_id}::main::seed{seed}",
                    "dataset_id": dataset_id,
                    "seed": int(seed),
                    "role": "main",
                    "config": str(MAIN_CONFIG.resolve()),
                    "config_sha256": _sha256(MAIN_CONFIG),
                    "output_dir": str(main_output.resolve()),
                    "record": record,
                    "reused_from": None,
                    "status": "queued_manifest_only",
                }
            )
            if dataset_id in development:
                for ablation, config in ABLATION_CONFIGS.items():
                    jobs.append(
                        {
                            "run_key": f"{manifest_id}::{dataset_id}::{ablation}::seed{seed}",
                            "dataset_id": dataset_id,
                            "seed": int(seed),
                            "role": "ablation",
                            "ablation": ablation,
                            "config": str(config.resolve()),
                            "config_sha256": _sha256(config),
                            "output_dir": str((output_root / "ablations" / ablation / dataset_id / f"seed{seed}").resolve()),
                            "record": record,
                            "reused_from": str(main_output.resolve()),
                            "reused_controls": ["N", "R", "T_s"],
                            "status": "queued_manifest_only",
                        }
                    )
    return {
        "manifest_id": manifest_id,
        "protocol_id": protocol_id,
        "dataset_spec_path": str(spec_path.resolve()),
        "dataset_spec_sha256": _sha256(spec_path),
        "selection_uses_labels_or_outcomes": False,
        "labels_used_during_fit": False,
        "feature_cap": 2000,
        "seeds": list(SEEDS),
        "domains": sorted(domains),
        "development_subset": sorted(development),
        "datasets": records,
        "jobs": jobs,
        "expected_main_panels": len(records) * len(SEEDS),
        "expected_ablation_arms": len(development) * len(SEEDS) * len(ABLATION_CONFIGS),
        "formal_training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--manifest-id", default="accg_locked_real_panel_v1")
    parser.add_argument("--protocol-id", default="accg_action_conditional_joint_v1")
    args = parser.parse_args()
    payload = build_manifest(
        args.dataset_spec,
        args.output_root,
        manifest_id=args.manifest_id,
        protocol_id=args.protocol_id,
    )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "frozen_not_run", "main_panels": payload["expected_main_panels"], "ablation_arms": payload["expected_ablation_arms"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
