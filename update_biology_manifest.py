"""Merge the completed structural H5 audit into a biology manifest safely."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.root / "manifest.json"
    audit_path = args.root / "inventory" / "biology_h5_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_by_id = {
        row["dataset_id"]: row
        for row in json.loads(audit_path.read_text(encoding="utf-8"))
        if row["dataset_id"] != "hrvatin_geo"
    }

    updated = 0
    for row in manifest["datasets"]:
        audit = audit_by_id.get(row["dataset_id"])
        if row.get("panel") != "biology" or audit is None:
            continue
        for key in (
            "canonical_path",
            "matrix_key",
            "label_key",
            "n_samples",
            "n_features",
            "sparsity",
            "n_clusters",
        ):
            row[key] = audit[key]
        row.update(
            {
                "input_kind": "pending_verification",
                "representation": "unverified scMAE H5 expression",
                "K_source": "embedded Y audited",
                "normalized_status": "pending_verification",
                "log1p_status": "pending_verification",
                "data_sha256": "not_computed_raw_data_policy",
                "labels_sha256": "embedded",
                "eligible": "pending_verification",
                "ineligible_reason": (
                    "expression representation semantics require source confirmation "
                    "before preprocessing"
                ),
                "notes": (
                    "X/Y shape, finite values, non-empty labels and K passed structural "
                    "audit; formal tuning not started."
                ),
            }
        )
        updated += 1

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps({"updated": updated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
