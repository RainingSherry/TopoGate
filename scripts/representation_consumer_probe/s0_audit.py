"""Run the CPU-only S0 contract audit for representation_consumer_probe.

S0 intentionally produces no clustering metrics and no trained model.  It verifies
the frozen input/graph/loss contract, saves one common H0 per stress dataset, and
records whether the existing V21/V25 code contains a semantically faithful
sample-edge adapter.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.representation_consumer_probe.protocol import (  # noqa: E402
    CONFIG,
    STRESS_DATASETS,
    audit_adapter_semantics,
    build_candidate_pool,
    build_h0,
    budget_profile,
    contract_manifest,
    jsonable,
    numerical_loss_contract,
    sha256_file,
    sha256_array,
    synthetic_apparatus_sanity,
)

E1_MANIFEST = ROOT / "result/V25_systematic_mechanism_study/E1/e1_manifest.json"
E1_MANIFEST_SHA256 = "edf2d57bba15cc1a56b18d12dd72efd320e0cbc4a730875a513b79814c577339"


EXPECTED: dict[str, dict[str, Any]] = {
    "cnae9": {
        "source_path": "/data/luolie/ToPoGate/datasets/cnae9.npz",
        "source_sha256": "e3a22bcfa761a837b8881a3a1966f6008d911b77948b76a4e4a8ca2aae65a71e",
        "input_protocol": "shared_text",
        "shape": [1080, 856],
    },
    "Mouse_retina": {
        "source_path": "/data/luolie/ToPoGate/datasets/Mouse_retina.npz",
        "source_sha256": "d3bc2eb08d95acd12d324f668c537ae4208de57c355862f8e5800d4ba1e727c1",
        "input_protocol": "clubench_bridge",
        "shape": [8352, 6198],
    },
    "sms_spam_collection": {
        "source_path": "/data/luolie/ToPoGate/datasets/sms_spam_collection.npz",
        "source_sha256": "1d9c068c43dc4f58e759169906dae9dff0ac6a8def7315ce7e4989a3b28d7070",
        "input_protocol": "shared_text",
        "shape": [835, 500],
    },
    "Baron Human": {
        "source_path": "/data/luolie/ToPoGate/datasets/Baron Human.npz",
        "source_sha256": "99c5bb7a272f5b22d64949179c6d49b4b52220670f3a7cef38e9e405902249d8",
        "input_protocol": "clubench_bridge",
        "shape": [8451, 20125],
    },
    "Campbell": {
        "source_path": "/data/luolie/ToPoGate/datasets/Campbell.npz",
        "source_sha256": "f26c9568baf8e8fade7a61e6fed80f16ceb3980afe38b7c9ccbe9aaad2b96e04",
        "input_protocol": "clubench_bridge",
        "shape": [9993, 26774],
    },
    "hate_speech": {
        "source_path": "/data/luolie/ToPoGate/datasets/hate_speech.npz",
        "source_sha256": "c39fc08366c0423b8a2f4e3543df6d5b06f2e0b065813e120263b99e7d1e44bb",
        "input_protocol": "shared_text",
        "shape": [3221, 100],
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_hash_manifest(root: Path) -> dict[str, Any]:
    """Hash the completed S0 artifact tree without including the manifest itself."""
    records: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"):
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {
        "manifest_id": "representation_consumer_probe_s0_artifacts_v1",
        "root": str(root.resolve()),
        "file_count": len(records),
        "files": records,
    }


def _load_matrix_without_labels(path: Path) -> tuple[sp.csr_matrix, int | None, list[str]]:
    """Load X and only count y for the outer K audit; never return labels."""
    with np.load(path, allow_pickle=True) as archive:
        files = list(archive.files)
        if "x" in files:
            matrix = sp.csr_matrix(np.asarray(archive["x"], dtype=np.float32))
        elif {"data", "indices", "indptr", "shape"}.issubset(files):
            matrix = sp.csr_matrix(
                (
                    np.asarray(archive["data"], dtype=np.float32),
                    np.asarray(archive["indices"], dtype=np.int64),
                    np.asarray(archive["indptr"], dtype=np.int64),
                ),
                shape=tuple(int(v) for v in archive["shape"]),
            )
        else:
            raise ValueError(f"unsupported dataset archive keys for {path}: {files}")
        k = None
        if "y" in files:
            y = np.asarray(archive["y"])
            k = int(np.unique(y).size)
    return matrix, k, files


def _audit_one_dataset(dataset: str, target: Path) -> dict[str, Any]:
    entry = dict(EXPECTED[dataset])
    source = Path(entry["source_path"])
    if not source.exists():
        return {"dataset": dataset, **entry, "status": "protocol_mismatch", "reason": "missing_source"}
    actual_hash = sha256_file(source)
    entry["actual_source_sha256"] = actual_hash
    if actual_hash != entry["source_sha256"]:
        return {"dataset": dataset, **entry, "status": "protocol_mismatch", "reason": "source_sha256_mismatch"}
    matrix, k, archive_keys = _load_matrix_without_labels(source)
    entry["actual_shape"] = list(matrix.shape)
    entry["archive_keys"] = archive_keys
    entry["K_source"] = CONFIG.k_source
    entry["labels_unique_outer_only"] = k
    entry["labels_vector_used_in_fit"] = False
    if list(matrix.shape) != entry["shape"]:
        return {"dataset": dataset, **entry, "status": "protocol_mismatch", "reason": "shape_mismatch"}

    dataset_dir = target / dataset.replace("/", "_")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    h0, h0_profile = build_h0(matrix)
    np.save(dataset_dir / "H0.npy", h0)
    pool = build_candidate_pool(h0)
    np.savez_compressed(
        dataset_dir / "candidate_pool.npz",
        indices=pool.indices,
        cosine=pool.cosine,
        positive_counts=pool.positive_counts,
        effective_budget=pool.effective_budget,
    )
    _write_json(dataset_dir / "budget_manifest.json", budget_profile(pool))
    loss = numerical_loss_contract(h0, sp.csr_matrix((matrix.shape[0], matrix.shape[0]), dtype=np.float32))
    # The empty graph is a numerical zero-degree test; no graph arm is being run here.
    result = {
        "dataset": dataset,
        **entry,
        "status": "completed_valid",
        "H0_path": str((dataset_dir / "H0.npy").resolve()),
        "H0_sha256": sha256_array(h0),
        "H0_profile": h0_profile,
        "candidate_pool_path": str((dataset_dir / "candidate_pool.npz").resolve()),
        "candidate_pool_profile": pool.profile,
        "budget_cap": int(CONFIG.budget_cap),
        "effective_budget_profile": budget_profile(pool),
        "positive_budget_status": "row_specific_feasibility_cap",
        "loss_zero_degree_sanity": loss,
    }
    del matrix, h0, pool
    gc.collect()
    return result


def run(output_dir: Path, *, datasets: tuple[str, ...] = STRESS_DATASETS) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    from scripts.representation_consumer_probe.protocol import resolved_config

    _write_json(
        output_dir / "resolved_config.json",
        {"config": resolved_config(), "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}},
    )
    adapter = audit_adapter_semantics(ROOT)
    _write_json(output_dir / "selection_to_relation_adapter.json", adapter)
    e1_manifest_exists = E1_MANIFEST.exists()
    e1_manifest_actual = sha256_file(E1_MANIFEST) if e1_manifest_exists else None
    provenance = {
        "e1_manifest_path": str(E1_MANIFEST),
        "e1_manifest_expected_sha256": E1_MANIFEST_SHA256,
        "e1_manifest_actual_sha256": e1_manifest_actual,
        "e1_manifest_match": bool(e1_manifest_actual == E1_MANIFEST_SHA256),
    }
    _write_json(output_dir / "input_provenance.json", provenance)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        rows.append(_audit_one_dataset(dataset, output_dir / "datasets"))
        _write_json(output_dir / "dataset_manifest.json", rows)
    synthetic = synthetic_apparatus_sanity()
    _write_json(output_dir / "synthetic_apparatus.json", synthetic)
    audit = {
        "source_preflight_complete": all(row.get("status") == "completed_valid" for row in rows),
        "e1_manifest_match": provenance["e1_manifest_match"],
        "H0_snapshot_count": sum("H0_path" in row for row in rows),
        "budget_cap": int(CONFIG.budget_cap),
        "effective_budget_profiles": {
            row["dataset"]: row.get("effective_budget_profile", {}) for row in rows
        },
        "row_specific_budget_datasets": [row["dataset"] for row in rows],
        "adapter_status": adapter.get("status"),
        "graph_numerical_sanity_passes": bool(
            all(synthetic.get("graph_numerical_sanity", {}).values())
        ),
        "spectral_recovery_sanity_passes": bool(
            all(
                synthetic.get("direction_checks", {}).get(key, False)
                for key in (
                    "spectral_clean_recovery",
                    "spectral_clean_beats_contaminated",
                    "spectral_embeddings_finite",
                    "spectral_isolate_policy",
                )
            )
        ),
        "synthetic_sanity_passes": bool(all(synthetic["direction_checks"].values())),
        "labels_used_during_fit": False,
        "oracle_non_tuning": True,
    }
    _write_json(output_dir / "graph_loss_contract.json", audit)
    s0_status = "adapter_valid" if adapter.get("status") == "adapter_valid" else adapter.get("status", "protocol_mismatch")
    if (
        not audit["source_preflight_complete"]
        or not audit["e1_manifest_match"]
        or not audit["graph_numerical_sanity_passes"]
        or not audit["spectral_recovery_sanity_passes"]
    ):
        s0_status = "protocol_mismatch"
    decision = {
        "project_id": CONFIG.project_id,
        "protocol_id": CONFIG.protocol_id,
        "status": s0_status,
        "s1_opportunity_only_allowed": s0_status == "adapter_not_estimable",
        "s2_opportunity_confirmation_allowed": s0_status == "adapter_not_estimable",
        "s3_unlocked": False,
        "s4_unlocked": False,
        "s5_unlocked": False,
        "reason": "S0 contract audit only; current project stops at opportunity-only S1/S2/Decision",
        "audit": audit,
    }
    _write_json(output_dir / "s0_decision.json", decision)
    manifest = contract_manifest(adapter, rows)
    manifest["status"] = s0_status
    _write_json(output_dir / "s0_manifest.json", manifest)
    _write_json(output_dir / "artifact_hashes.json", _artifact_hash_manifest(output_dir))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "result/representation_consumer_probe/S0_freeze",
    )
    parser.add_argument("--dataset", action="append", choices=STRESS_DATASETS)
    args = parser.parse_args()
    selected = tuple(args.dataset) if args.dataset else STRESS_DATASETS
    decision = run(args.output_dir, datasets=selected)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["status"] in {"adapter_valid", "adapter_not_estimable"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
