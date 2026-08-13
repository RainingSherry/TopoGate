#!/usr/bin/env python
"""Audit V18 run-key coverage and artifact contracts without retuning or hashing."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VARIANTS = [
    "scmae_only", "latent_candidate_spectral", "latent_C_exactzero", "latent_GW_frozen", "v18_full",
    "v18_shuffled_E0", "v18_no_recurrence", "v18_no_stability", "v18_mask04", "v18_leiden",
]
SEEDS = (42, 123, 7)
EXPECTED_PROTOCOL_ID = "v18_scmae_mainline_v2_2"
COMMON = {"summary.json", "resolved_config.json", "status.json", "predictions.npy", "latent_final.npy",
          "latent_mae.npy", "embedding_final.npy", "abstained_mask.npy", "metrics.json"}
GRAPH = {"candidate_graph.npz", "coefficient_matrix.npz", "affinity_matrix.npz", "gate_relation_slots.npz"}


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def audit(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = _json(manifest_path)
    if manifest is None:
        raise ValueError(f"invalid manifest: {manifest_path}")
    manifest_id = str(manifest.get("manifest_id"))
    if manifest.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError(
            f"expected manifest protocol_id={EXPECTED_PROTOCOL_ID!r}; "
            f"got {manifest.get('protocol_id')!r}"
        )
    eligible = [row for row in manifest.get("datasets", []) if row.get("status") == "eligible"]
    expected: dict[str, tuple[str, str, int]] = {}
    for row in eligible:
        for variant in VARIANTS:
            for seed in SEEDS:
                key = f"{row['dataset_id']}::{variant}::seed{seed}"
                expected[key] = (str(row["dataset_id"]), variant, seed)
    observed: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    malformed: list[str] = []
    bad_contract: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for key, (dataset_id, variant, seed) in expected.items():
        run_dir = output_root / dataset_id / variant / f"seed{seed}"
        record = _json(run_dir / "run_record.json")
        summary = _json(run_dir / "summary.json")
        status = _json(run_dir / "status.json")
        if record is None:
            missing.append(key)
            status_counts["missing"] += 1
            continue
        state = str(record.get("status", "unknown"))
        status_counts[state] += 1
        observed[key] = {"status": state, "path": str(run_dir)}
        if state == "completed":
            if summary is None or status is None:
                malformed.append(key)
                continue
            required = set(COMMON) | (set() if variant == "scmae_only" else GRAPH)
            missing_files = sorted(required - {p.name for p in run_dir.iterdir()})
            contract = {
                "key": key,
                "missing_files": missing_files,
                "protocol_id": summary.get("protocol_id", record.get("protocol_id")),
                "dataset_id": summary.get("dataset_id", record.get("dataset_id")),
                "summary_status": summary.get("status"),
                "status_json": status.get("status"),
                "status_protocol_id": status.get("protocol_id"),
                "status_dataset_id": status.get("dataset_id"),
                "manifest_id": summary.get("manifest_id", record.get("manifest_id")),
                "seed": summary.get("seed", record.get("seed")),
                "labels_used_during_fit": summary.get("labels_used_during_fit"),
                "K_source": summary.get("K_source"),
                "benchmark_oracle_from_y": summary.get("benchmark_oracle_from_y"),
                "K_used_only_in_readout": summary.get("K_used_only_in_readout"),
                "n_clusters": summary.get("n_clusters"),
            }
            seed_ok = False
            try:
                seed_ok = int(contract["seed"]) == int(seed)
            except (TypeError, ValueError):
                seed_ok = False
            if variant == "v18_leiden":
                k_contract_ok = (
                    contract["K_source"] == "not_applicable_leiden"
                    and contract["benchmark_oracle_from_y"] is False
                    and contract["K_used_only_in_readout"] is False
                    and contract["n_clusters"] is None
                )
            else:
                k_contract_ok = (
                    contract["K_source"] == "benchmark_oracle_from_y"
                    and contract["benchmark_oracle_from_y"] is True
                    and contract["K_used_only_in_readout"] is True
                    and contract["n_clusters"] is not None
                )
            if missing_files or contract["protocol_id"] != EXPECTED_PROTOCOL_ID \
                    or contract["dataset_id"] != dataset_id \
                    or contract["status_protocol_id"] != EXPECTED_PROTOCOL_ID \
                    or contract["status_dataset_id"] != dataset_id \
                    or contract["summary_status"] != "completed" or contract["status_json"] != "completed" \
                    or contract["manifest_id"] != manifest_id or not seed_ok \
                    or contract["labels_used_during_fit"] is not False or not k_contract_ok:
                bad_contract.append(contract)
    result = {
        "manifest_id": manifest_id,
        "eligible_datasets": len(eligible),
        "expected_run_keys": len(expected),
        "observed_run_keys": len(observed),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_run_keys": missing,
        "malformed_run_keys": malformed,
        "bad_contracts": bad_contract,
        "complete": not missing and not malformed and not bad_contract and
                    status_counts.get("completed", 0) + status_counts.get("incomplete_compute", 0) +
                    status_counts.get("domain_not_supported", 0) + status_counts.get("code_error", 0) == len(expected),
        "hashes_recomputed": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit V18 matrix coverage")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.output_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"expected": result["expected_run_keys"], "observed": result["observed_run_keys"],
                      "status_counts": result["status_counts"], "complete": result["complete"],
                      "bad_contracts": len(result["bad_contracts"]), "missing": len(result["missing_run_keys"])}, ensure_ascii=True))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
