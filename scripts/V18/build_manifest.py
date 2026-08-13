#!/usr/bin/env python
"""Freeze the V18 dataset universe without repeated source hashing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v9_regime.build_manifest import build_manifest
from scripts.v9_regime.protocol import MAX_ELEMENTS, json_default


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a one-time V18 dataset manifest without recomputing hashes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-json", type=Path, default=None)
    parser.add_argument("--base-manifest", type=Path, default=None,
                        help="reuse an existing frozen manifest without rescanning or hashing sources")
    parser.add_argument("--max-elements", type=int, default=MAX_ELEMENTS)
    parser.add_argument("--manifest-id", default="v18_scmae_mainline_v2_2_20260808")
    parser.add_argument("--protocol-id", default="v18_scmae_mainline_v2_2")
    args = parser.parse_args()
    if args.base_manifest is not None:
        payload = json.loads(args.base_manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), list):
            raise ValueError(f"invalid base manifest: {args.base_manifest}")
        payload = dict(payload)
    else:
        payload = build_manifest(args.max_elements, args.external_json)
    payload["protocol_id"] = str(args.protocol_id)
    payload["manifest_id"] = str(args.manifest_id)
    payload["selection_policy"] = {
        "source": f"frozen base manifest: {args.base_manifest}" if args.base_manifest else
                   "existing V9 registry plus registered external records",
        "selection_uses_labels_or_outcomes": False,
        "hashes_recomputed_during_experiments": False,
        "eligible_records_run": True,
        "ineligible_or_unresolved_records_preserved": True,
        "source_manifest_reused_without_rescan": args.base_manifest is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=json_default), encoding="utf-8")
    counts: dict[str, int] = {}
    for record in payload["datasets"]:
        counts[str(record.get("status"))] = counts.get(str(record.get("status")), 0) + 1
    print(json.dumps({"output": str(args.output), "manifest_id": args.manifest_id,
                      "datasets": len(payload["datasets"]), "status": counts}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
