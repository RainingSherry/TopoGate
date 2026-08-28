#!/usr/bin/env python3
from __future__ import annotations

"""Explicitly fetch audited OpenML numeric candidates into NPZ files."""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from scripts.v9_regime.protocol import MAX_CLUSTERS, MIN_CLUSTERS, MIN_SAMPLES, write_json


def fetch_candidates(candidates_json: Path, output_dir: Path, registry_output: Path, limit: int = 0) -> dict:
    payload = json.loads(candidates_json.read_text(encoding="utf-8"))
    candidates = payload.get("datasets", [])[: limit if limit > 0 else None]
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = []
    for index, candidate in enumerate(candidates, start=1):
        row = dict(candidate)
        did = str(candidate.get("dataset_id", "")).split("__", 1)[-1]
        path = output_dir / f"{candidate.get('name', did)}.npz"
        row["source_path"] = str(path.resolve())
        print(f"[{index}/{len(candidates)}] {candidate.get('dataset_id')}", flush=True)
        try:
            bunch = fetch_openml(data_id=int(did), as_frame=False, parser="auto", cache=True)
            x = np.asarray(bunch.data)
            if x.ndim != 2:
                raise ValueError(f"X is not 2-D: {x.shape}")
            x = x.astype(np.float32)
            y_raw = np.asarray(bunch.target).reshape(-1)
            if len(y_raw) != x.shape[0]:
                raise ValueError(f"X/y mismatch: {x.shape} vs {y_raw.shape}")
            y = LabelEncoder().fit_transform(y_raw).astype(np.int64)
            n, d = x.shape
            k = int(np.unique(y).size)
            if n < MIN_SAMPLES or k < MIN_CLUSTERS or k > MAX_CLUSTERS:
                raise ValueError(f"post-fetch eligibility failed: n={n}, K={k}")
            np.savez_compressed(path, x=x, y=y)
            row.update({"status": "fetched", "status_reason": None, "n": n, "d": d, "n_clusters": k, "labels_encoded": True})
        except Exception as exc:
            row.update({"status": "unresolved", "status_reason": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc()})
        registry.append(row)
    result = {"source": "OpenML", "candidate_registry": str(candidates_json), "datasets": registry}
    write_json(registry_output, result)
    print(json.dumps({"fetched": sum(row.get('status') == 'fetched' for row in registry), "unresolved": sum(row.get('status') == 'unresolved' for row in registry)}))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    fetch_candidates(args.candidates, args.output_dir, args.registry_output, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

