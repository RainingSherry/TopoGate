#!/usr/bin/env python3
"""Freeze a same-protocol recovery panel for unfinished cooperative Full jobs.

The recovery manifest deliberately contains only jobs that ended as
``incomplete_compute`` in the original cooperative Full queue.  It keeps the
same dataset records, variant, seed, epochs, and batch size; the launcher may
reuse only the exact topology-statistics cache written by that interrupted run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "datasets" / "external" / "v22_full_cooperative_single_seed_20260812" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "datasets" / "external" / "v22_full_cooperative_recovery_20260812" / "manifest.json"
DEFAULT_DATASETS = ("real_sim__libsvm_sparse_highdim", "covtype__libsvm_dense_control")
VARIANT = "v22_topology_discriminator_cooperative_keep_gate"


def prepare(source: Path, output: Path, dataset_ids: tuple[str, ...]) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("variants") != [VARIANT] or payload.get("seeds") != [42]:
        raise ValueError("cooperative recovery requires the frozen cooperative seed-42 manifest")
    if int(payload.get("epochs", -1)) != 80 or int(payload.get("batch_size", -1)) != 4096:
        raise ValueError("recovery must preserve the original cooperative Full budget")
    requested = tuple(dict.fromkeys(str(value) for value in dataset_ids))
    if not requested or len(requested) != len(dataset_ids):
        raise ValueError("recovery dataset ids must be non-empty and unique")
    records = [dict(row) for row in payload.get("records", []) if str(row.get("dataset_id")) in requested]
    found = {str(row.get("dataset_id")) for row in records}
    missing = sorted(set(requested) - found)
    if missing:
        raise ValueError(f"unknown cooperative Full dataset ids: {missing}")
    if len(records) != len(requested):
        raise ValueError("recovery records are not one-to-one with requested dataset ids")
    for row in records:
        source_path = Path(str(row["source_path"]))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if not row.get("source_sha256"):
            raise ValueError(f"missing source_sha256: {row.get('dataset_id')}")

    recovery = dict(payload)
    recovery.update(
        {
            "manifest_id": "v22_full_cooperative_recovery_20260812_v1",
            "phase": "full_components_single_seed_cooperative_recovery",
            "description": (
                "Same-protocol recovery of cooperative Full jobs that reached the resource "
                "boundary before writing a complete summary."
            ),
            "recovery_from_manifest": str(source.resolve()),
            "recovery_dataset_ids": list(requested),
            "recovery_attempt_boundary": "prior_control_window_interrupted_before_worker_summary; restart_allowed_from_same_manifest",
            "topology_cache_reuse": "explicit_same_dataset_run_cache_only",
            "selection_policy": dict(payload.get("selection_policy", {}))
            | {
                "recovery_selection_uses_labels_or_outcomes": False,
                "recovery_preserves_variant_seed_epochs_batch_size": True,
                "recovery_reuses_only_exact_same_dataset_topology_cache": True,
            },
            "records": records,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(recovery, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return recovery


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    args = parser.parse_args()
    payload = prepare(args.source, args.output, tuple(args.datasets))
    print(
        json.dumps(
            {
                "manifest_id": payload["manifest_id"],
                "protocol_id": payload["protocol_id"],
                "phase": payload["phase"],
                "records": [row["dataset_id"] for row in payload["records"]],
                "epochs": payload["epochs"],
                "batch_size": payload["batch_size"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
