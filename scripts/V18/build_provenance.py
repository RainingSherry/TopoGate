#!/usr/bin/env python
"""Build a V18 provenance sidecar from already recorded source metadata.

This command deliberately does not open dataset payloads and never computes a
hash.  It only reuses hashes already present in repository manifests; sources
without an existing hash are recorded as ``unavailable``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL_ID = "v18_scmae_mainline_v2_2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _ahdpc_hashes(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"invalid AHDPC manifest datasets field: {path}")
    result: dict[str, dict[str, Any]] = {}
    for name, record in datasets.items():
        if not isinstance(record, dict):
            continue
        processed = record.get("processed")
        if isinstance(processed, dict) and processed.get("sha256"):
            result[str(name)] = {
                "sha256": str(processed["sha256"]),
                "source": str(path),
                "recorded_path": processed.get("path"),
            }
    return result


def build_provenance(manifest_path: Path, output_path: Path, *, ahdpc_manifest: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    protocol_id = manifest.get("protocol_id")
    if protocol_id != EXPECTED_PROTOCOL_ID:
        raise ValueError(f"expected protocol_id={EXPECTED_PROTOCOL_ID!r}, got {protocol_id!r}")
    ahdpc = _ahdpc_hashes(ahdpc_manifest) if ahdpc_manifest.exists() else {}
    records: list[dict[str, Any]] = []
    for item in manifest.get("datasets", []):
        if not isinstance(item, dict):
            continue
        dataset_id = str(item.get("dataset_id"))
        source_path = str(item.get("source_path"))
        name = Path(source_path).stem
        known = ahdpc.get(name) if item.get("source_kind") == "ahdpc_prepared" else None
        if known is None:
            hash_status = "unavailable"
            source_hash = None
            hash_source = None
            recorded_path = None
        else:
            hash_status = "reused_existing_manifest_value"
            source_hash = known["sha256"]
            hash_source = known["source"]
            recorded_path = known.get("recorded_path")
        records.append({
            "dataset_id": dataset_id,
            "source_path": source_path,
            "source_kind": item.get("source_kind"),
            "source_version": item.get("source_version"),
            "source_hash": source_hash,
            "source_hash_status": hash_status,
            "source_hash_manifest": hash_source,
            "recorded_path": recorded_path,
        })
    result = {
        "protocol_id": protocol_id,
        "manifest_id": manifest.get("manifest_id"),
        "source_hash_policy": "reuse_existing_metadata_only_no_hash_computation",
        "hashes_recomputed": False,
        "source_manifest": str(manifest_path),
        "datasets": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V18 provenance sidecar without hashing dataset files")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ahdpc-manifest", type=Path,
                        default=Path("datasets/AHDPC/MANIFEST.json"))
    args = parser.parse_args()
    result = build_provenance(args.manifest, args.output, ahdpc_manifest=args.ahdpc_manifest)
    from collections import Counter
    print(json.dumps({"output": str(args.output), "datasets": len(result["datasets"]),
                      "hash_status": dict(Counter(row["source_hash_status"] for row in result["datasets"])),
                      "hashes_recomputed": False}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
