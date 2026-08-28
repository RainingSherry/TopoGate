#!/usr/bin/env python3
"""Freeze the V22 cooperative Full panel across the original and extension data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE_MANIFEST = ROOT / "datasets" / "external" / "v22_full_single_seed_20260812" / "manifest.json"
ROUND2_MANIFEST = ROOT / "datasets" / "external" / "v22_dataset_extension_round2_20260812" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "datasets" / "external" / "v22_full_cooperative_single_seed_20260812" / "manifest.json"
VARIANT = "v22_topology_discriminator_cooperative_keep_gate"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be an object: {path}")
    return payload


def _round2_record(record: dict[str, Any], source_manifest: Path) -> dict[str, Any]:
    profile = record.get("profile")
    if not isinstance(profile, dict):
        raise ValueError(f"round-2 record lacks profile: {record.get('dataset_id')}")
    unlabelled = str(record.get("status", "")).endswith("unlabelled")
    labels_unique = profile.get("labels_unique")
    source_path = Path(str(record["source_path"]))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    normalized = dict(record)
    normalized.update(
        {
            "source_sha256": record["processed_sha256"],
            "n_samples": int(profile["n_samples"]),
            "n_features_original": int(profile["n_features"]),
            "labels_available_outer_only": not unlabelled,
            "expected_labels_unique": None if labels_unique is None else int(labels_unique),
            "k_source": "explicit_n_clusters" if unlabelled else "benchmark_oracle_from_y",
            "source_manifest": str(source_manifest.relative_to(ROOT)),
            "selection_uses_labels_or_outcomes": False,
        }
    )
    if unlabelled:
        # K is an outer readout protocol parameter, never a fitting input.
        normalized["n_clusters"] = 8
    return normalized


def prepare(base: Path, round2: Path, output: Path) -> dict[str, Any]:
    base_payload = _read(base)
    round2_payload = _read(round2)
    if base_payload.get("seeds") != [42] or len(base_payload.get("records", [])) != 12:
        raise ValueError("the base V22 Full panel must contain its frozen 12 records and seed 42")
    base_records = [dict(row) for row in base_payload["records"]]
    extra_records = [
        _round2_record(dict(row), round2)
        for row in round2_payload.get("datasets", [])
        if str(row.get("status", "")).startswith("eligible")
    ]
    records = base_records + extra_records
    seen: set[str] = set()
    for row in records:
        dataset_id = str(row["dataset_id"])
        if dataset_id in seen:
            raise ValueError(f"duplicate dataset_id: {dataset_id}")
        seen.add(dataset_id)
        source = Path(str(row["source_path"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        if not row.get("source_sha256"):
            raise ValueError(f"missing source_sha256: {dataset_id}")
        if not row.get("labels_available_outer_only", True) and row.get("n_clusters") is None:
            raise ValueError(f"unlabelled record requires explicit K: {dataset_id}")
    if len(records) != 16:
        raise ValueError(f"expected 16 records (12 base + 4 round2), got {len(records)}")
    payload: dict[str, Any] = {
        "manifest_id": "v22_full_cooperative_keep_single_seed_20260812_v1",
        "protocol_id": "v22_topology_discriminator_cooperative_keep_gate_v1",
        "phase": "full_components_single_seed_cooperative",
        "description": "Frozen V22 cooperative Keep-Gate Full single-seed panel: original eight, round-one four, and round-two four datasets.",
        "variants": [VARIANT],
        "seeds": [42],
        "epochs": 80,
        "batch_size": 4096,
        "selection_policy": {
            "selection_uses_labels_or_outcomes": False,
            "labels_used_during_fit": False,
            "K_used_during_fit": False,
            "unlabelled_records_excluded_from_ari_aggregate": True,
            "panel_frozen_before_cooperative_full_results": True,
        },
        "source_manifests": [str(base.relative_to(ROOT)), str(round2.relative_to(ROOT))],
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE_MANIFEST)
    parser.add_argument("--round2", type=Path, default=ROUND2_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = prepare(args.base, args.round2, args.output)
    print(json.dumps({key: payload[key] for key in ("manifest_id", "protocol_id", "phase", "variants", "seeds")}, indent=2))
    print(json.dumps({"records": len(payload["records"]), "unlabelled": [r["dataset_id"] for r in payload["records"] if not r.get("labels_available_outer_only", True)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
