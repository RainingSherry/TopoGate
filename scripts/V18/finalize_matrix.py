#!/usr/bin/env python
"""Wait for the frozen V18 matrix, then run its terminal audits once.

The watcher never launches model jobs, changes configs, selects results, or
computes hashes.  It only waits for all manifest run keys to reach a terminal
run_record status and then invokes the existing audit/summarizer utilities.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.V18.audit_matrix import audit
from scripts.V18.build_provenance import build_provenance
from scripts.V18.repair_leiden_metadata import repair as repair_leiden_metadata
from scripts.V18.summarize import summarize


TERMINAL = {"completed", "incomplete_compute", "domain_not_supported", "code_error"}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _expected_keys(manifest: dict[str, Any]) -> int:
    eligible = [row for row in manifest.get("datasets", []) if row.get("status") == "eligible"]
    variants = ("scmae_only", "latent_candidate_spectral", "latent_C_exactzero", "latent_GW_frozen",
                "v18_full", "v18_shuffled_E0", "v18_no_recurrence", "v18_no_stability",
                "v18_mask04", "v18_leiden")
    return len(eligible) * len(variants) * 3


def _counts(root: Path) -> Counter[str]:
    values: Counter[str] = Counter()
    for path in root.glob("*/*/seed*/run_record.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            values["malformed"] += 1
            continue
        values[str(value.get("status", "unknown"))] += 1
    return values


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = _expected_keys(manifest)
    status_path = args.status_output
    while True:
        counts = _counts(args.output_root)
        observed = sum(counts.values())
        state = {
            "status": "waiting",
            "manifest_id": manifest.get("manifest_id"),
            "expected_run_keys": expected,
            "observed_run_keys": observed,
            "status_counts": dict(counts),
            "hashes_recomputed": False,
        }
        if observed == expected and not (set(counts) - TERMINAL):
            break
        _write(status_path, state)
        time.sleep(max(5, int(args.poll_seconds)))

    metadata_repair = repair_leiden_metadata(args.output_root)
    audit_payload = audit(args.manifest, args.output_root)
    _write(args.audit_output, audit_payload)
    summary_payload = summarize(args.output_root, str(manifest.get("manifest_id")))
    _write(args.summary_output, summary_payload)
    provenance_payload = build_provenance(args.manifest, args.provenance_output,
                                           ahdpc_manifest=args.ahdpc_manifest)
    final = {
        "status": "completed",
        "manifest_id": manifest.get("manifest_id"),
        "audit_complete": bool(audit_payload.get("complete")),
        "audit_status_counts": audit_payload.get("status_counts", {}),
        "summary_runs": summary_payload.get("run_summaries", 0),
        "provenance_datasets": len(provenance_payload.get("datasets", [])),
        "leiden_metadata_repair": metadata_repair,
        "hashes_recomputed": False,
        "outputs": {
            "audit": str(args.audit_output),
            "summary": str(args.summary_output),
            "provenance": str(args.provenance_output),
        },
    }
    _write(status_path, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize V18 after every run key reaches terminal state")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--ahdpc-manifest", type=Path, default=Path("datasets/AHDPC/MANIFEST.json"))
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    result = finalize(args)
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result["audit_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
