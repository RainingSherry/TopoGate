#!/usr/bin/env python3
from __future__ import annotations

"""Discover external numeric OpenML candidates without downloading data.

The output is a candidate registry.  Fetching and conversion is a separate
explicit step, so an API failure is recorded as unresolved rather than being
silently replaced by another dataset.
"""

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import requests

from scripts.v9_regime.protocol import write_json


def _quality_map(entry: dict) -> dict[str, str]:
    """Normalize OpenML's current quality-array and legacy flat fields."""
    values = {
        str(item.get("name")): str(item.get("value"))
        for item in entry.get("quality", [])
        if isinstance(item, dict) and item.get("name") is not None
    }
    for key in ("NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "NumberOfSymbolicFeatures"):
        if key in entry and entry[key] is not None:
            values.setdefault(key, str(entry[key]))
    return values


def _entry_metadata(entry: dict) -> tuple[int, int, int, int] | None:
    quality = _quality_map(entry)
    try:
        n = int(float(quality.get("NumberOfInstances", "0")))
        d = int(float(quality.get("NumberOfFeatures", "0")))
        k = int(float(quality.get("NumberOfClasses", "0")))
        symbolic = int(float(quality.get("NumberOfSymbolicFeatures", "0")))
    except (TypeError, ValueError):
        return None
    return n, d, k, symbolic


def discover(output: Path, limit: int, timeout: int) -> dict:
    url = f"https://www.openml.org/api/v1/json/data/list/limit/{int(limit)}"
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "ToPoGate-v9-regime/1.0"})
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        result = {
            "source": "OpenML",
            "status": "unresolved",
            "error": f"{type(exc).__name__}:{exc}",
            "datasets": [],
            "network_attempted": True,
        }
        write_json(output, result)
        return result

    entries = payload.get("data", {}).get("dataset", [])
    candidates = []
    for entry in entries:
        metadata = _entry_metadata(entry)
        if metadata is None:
            continue
        n, d, k, symbolic = metadata
        # OpenML reports NumberOfClasses=0 for many numeric datasets whose
        # target type is only known after fetching. Keep those records as
        # explicit candidates; fetch_openml.py performs the final K audit and
        # marks regression/continuous-target records unresolved.
        if n < 100 or k > 50 or symbolic != 0:
            continue
        did = str(entry.get("did") or entry.get("id"))
        name = str(entry.get("name", did))
        candidates.append(
            {
                "dataset_id": f"openml__{did}",
                "name": name,
                "source_kind": "openml",
                "source_identity": f"openml:did={did}",
                "source_version": str(entry.get("version", "unknown")),
                "source_url": f"https://www.openml.org/d/{quote(did)}",
                "download_url": f"https://www.openml.org/data/v1/download/{quote(did)}",
                "license": "OpenML dataset metadata; verify per record",
                "citation": "OpenML record; verify dataset paper",
                "family": "external_numeric",
                "metadata_n": n,
                "metadata_d": d,
                "metadata_k": k,
                "status": "candidate_unfetched",
                "status_reason": (
                    "requires_target_audit_after_fetch" if k < 2
                    else "requires_explicit_fetch_and_source_audit"
                ),
            }
        )
    result = {
        "source": "OpenML",
        "status": "completed",
        "api_url": url,
        "network_attempted": True,
        "candidate_count": len(candidates),
        "datasets": candidates,
    }
    write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    result = discover(args.output, args.limit, args.timeout)
    print(json.dumps({key: result.get(key) for key in ("source", "status", "candidate_count", "error")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
