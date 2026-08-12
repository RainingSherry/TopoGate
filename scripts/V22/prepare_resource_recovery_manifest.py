#!/usr/bin/env python3
"""Freeze a resource-recovery manifest without changing the dataset panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "datasets" / "external" / "v22_full_single_seed_20260812" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "datasets" / "external" / "v22_full_resource_recovery_20260812" / "manifest.json"


def prepare(source: Path, output: Path, batch_size: int) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("variants") != ["v22_topology_discriminator_hard_gate"] or payload.get("seeds") != [42]:
        raise ValueError("resource recovery requires the frozen V22 Full single-seed panel")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    recovery = dict(payload)
    recovery.update(
        {
            "manifest_id": "v22_full_resource_recovery_20260812_v1",
            "protocol_id": "v22_topology_discriminator_hard_mask_resource_recovery_v1",
            "phase": "full_components_single_seed_resource_recovery",
            "description": "The same twelve-dataset V22 Full panel with explicit topology-cache reuse and a larger batch size for resource recovery.",
            "recovery_from_manifest": str(source.resolve()),
            "recovery_batch_size": int(batch_size),
            "topology_cache_reuse": "explicit_same_dataset_run_cache_only",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(recovery, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return recovery


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    payload = prepare(args.source, args.output, args.batch_size)
    print(json.dumps({key: payload[key] for key in ("manifest_id", "protocol_id", "phase", "recovery_batch_size")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
