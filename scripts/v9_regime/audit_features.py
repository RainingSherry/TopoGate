#!/usr/bin/env python3
from __future__ import annotations

"""Compute the frozen, label-free V9 graph/input feature table."""

import argparse
import json
from pathlib import Path

from scripts.v9_regime.protocol import (
    SPLIT_SEED,
    build_x_only_features,
    get_record,
    load_x,
    read_manifest,
    write_csv,
    write_json,
)


def audit(manifest_path: Path, output_csv: Path, output_json: Path | None, seed: int, max_samples: int, max_features: int, datasets: set[str] | None) -> list[dict]:
    manifest = read_manifest(manifest_path)
    rows: list[dict] = []
    for record in manifest["datasets"]:
        dataset_id = str(record["dataset_id"])
        if datasets and dataset_id not in datasets:
            continue
        row = {
            "dataset_id": dataset_id,
            "name": record.get("name"),
            "family": record.get("family"),
            "status": record.get("status"),
            "source_kind": record.get("source_kind"),
            "source_identity": record.get("source_identity"),
            "source_path": record.get("source_path"),
            "manifest_labels_used_during_fit": False,
            "feature_seed": int(seed),
        }
        if record.get("status") != "eligible":
            row["feature_error"] = f"manifest_status:{record.get('status')}:{record.get('status_reason')}"
            rows.append(row)
            continue
        try:
            x = load_x(Path(record["source_path"]))
            features = build_x_only_features(
                x,
                seed=seed,
                max_analysis_samples=max_samples,
                max_analysis_features=max_features,
            )
            row.update(features)
            row["feature_error"] = None
        except Exception as exc:
            row["feature_error"] = f"{type(exc).__name__}:{exc}"
        rows.append(row)
        print(f"{dataset_id}\t{row.get('feature_error') or 'ok'}", flush=True)
    write_csv(output_csv, rows)
    if output_json is not None:
        write_json(output_json, {"protocol_id": manifest.get("protocol_id"), "rows": rows})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    rows = audit(
        args.manifest,
        args.output_csv,
        args.output_json,
        args.seed,
        args.max_samples,
        args.max_features,
        set(args.datasets) if args.datasets else None,
    )
    completed = sum(1 for row in rows if not row.get("feature_error"))
    print(json.dumps({"rows": len(rows), "completed": completed, "failed_or_skipped": len(rows) - completed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

