#!/usr/bin/env python3
"""Write a conservative, reproducible progress audit for a transductive run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.root / "manifest.json")
    rows = [row for row in manifest["datasets"] if row.get("panel") == "clubench"]
    datasets = args.root / "datasets"
    completed_search = []
    completed_final = []
    hash_failures = []
    artifact_coverage = {
        "predictions": 0,
        "embeddings": 0,
        "model_weights": 0,
        "preprocessors": 0,
        "cluster_centers": 0,
    }
    unfinished_overwrite_evidence = []

    for row in rows:
        path = datasets / row["dataset_id"]
        if (path / "search_summary.json").exists():
            completed_search.append(row["dataset_id"])
        else:
            candidate_dirs = sorted(path.glob("candidate_*"))
            completed_indices = sorted(
                int(candidate.name.rsplit("_", 1)[1])
                for candidate in candidate_dirs
                if len(list(candidate.glob("record_seed_*.json"))) == 3
            )
            empty_indices = sorted(
                int(candidate.name.rsplit("_", 1)[1])
                for candidate in candidate_dirs
                if not list(candidate.glob("record_seed_*.json"))
            )
            if completed_indices and empty_indices and max(empty_indices) > max(completed_indices):
                unfinished_overwrite_evidence.append({
                    "dataset_id": row["dataset_id"],
                    "completed_candidate_max": max(completed_indices),
                    "empty_candidate_indices": empty_indices,
                    "interpretation": (
                        "high-index empty candidate directories predate the current contiguous "
                        "record sequence; retry provenance is incomplete and must not be called audited"
                    ),
                })
        final_path = path / "final_summary.json"
        if not final_path.exists():
            continue
        completed_final.append(row["dataset_id"])
        final = read_json(final_path)
        selected_path = path / "selected.json"
        if not selected_path.exists():
            hash_failures.append({"dataset_id": row["dataset_id"], "reason": "selected.json missing"})
        else:
            selected = read_json(selected_path)
            if selected.get("selection_hash") != final.get("selection_hash"):
                hash_failures.append({"dataset_id": row["dataset_id"], "reason": "selection_hash mismatch"})
        final_dirs = list(path.glob("final_seed_*")) + list((path / "final").glob("seed_*"))
        files = [item for directory in final_dirs if directory.is_dir() for item in directory.iterdir()]
        artifact_coverage["predictions"] += any("prediction" in item.name.lower() for item in files)
        artifact_coverage["embeddings"] += any("embedding" in item.name.lower() for item in files)
        artifact_coverage["model_weights"] += any(item.suffix in {".pt", ".pth", ".ckpt"} for item in files)
        artifact_coverage["preprocessors"] += any("preprocessor" in item.name.lower() for item in files)
        artifact_coverage["cluster_centers"] += any("cluster_centers" in item.name.lower() for item in files)

    audit = {
        "protocol_id": "topogate_transductive_perdataset_64x3_v1",
        "panel": "clubench",
        "registered_datasets": len(rows),
        "search_summary_count": len(completed_search),
        "final_summary_count": len(completed_final),
        "pending_search": sorted(set(row["dataset_id"] for row in rows) - set(completed_search)),
        "pending_final": sorted(set(completed_search) - set(completed_final)),
        "selection_hash_failures": hash_failures,
        "unfinished_overwrite_evidence": unfinished_overwrite_evidence,
        "final_artifact_coverage": artifact_coverage,
        "status": "interim_not_formal_completion",
    }
    reports = args.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    temporary = reports / "transductive_progress_audit.json.tmp"
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(reports / "transductive_progress_audit.json")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
