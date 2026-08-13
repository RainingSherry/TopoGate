#!/usr/bin/env python3
"""Audit current NPZ names against the local CLUBench/CLM table.

This parser is provenance-aware: the README table is useful for mapping and
exploration, but it is not declared an externally verified CLM source unless a
fixed external commit is supplied separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _normalise(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\^*+` ]", "", value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_clubench_table(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\|\s*\[?(\d+)\]?\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*"
        r"(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*"
        r"([-+]?\d*\.\d+)\s*\|\s*([-+]?\d*\.\d+)\s*\|\s*([-+]?\d*\.\d+)\s*\|"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        dataset = match.group(2).strip()
        rows.append(
            {
                "id": int(match.group(1)),
                "dataset": dataset,
                "type": match.group(3).strip(),
                "n": int(match.group(4)),
                "d": int(match.group(5)),
                "K": int(match.group(6)),
                "r_mm": float(match.group(7)),
                "r_ma": float(match.group(8)),
                "IR": float(match.group(9)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--clubench-readme", type=Path, default=ROOT / "baseline" / "CLUBench" / "README.md")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    external_rows = parse_clubench_table(args.clubench_readme)
    by_name = {_normalise(row["dataset"]): row for row in external_rows}
    npz_paths = sorted(args.dataset_root.rglob("*.npz"))
    mappings: list[dict[str, Any]] = []
    for path in npz_paths:
        name = path.stem
        row = by_name.get(_normalise(name))
        mappings.append(
            {
                "dataset": name,
                "path": str(path.resolve()),
                "clubench_match": row is not None,
                "clubench": row,
                "clm_source": str(args.clubench_readme.resolve()) if row is not None else None,
                "clm_metric": "r_mm" if row is not None else None,
                "clm_value": None if row is None else row["r_mm"],
                "clm_commit_verified": False,
            }
        )
    requested = {
        "Mouse_retina",
        "cnae9",
        "imdb",
        "sms_spam_collection",
        "secom",
        "enron",
        "reuters",
        "20newsgroups",
        "cifar10",
        "CIFAR10_CLIP",
        "labeled_faces_in_the_wild",
        "flickr_material_database",
        "ISOLET",
        "olivetti_faces",
        "mnist64",
        "seeds",
        "HIVA",
    }
    current_names = {_normalise(path.stem) for path in npz_paths}
    payload = {
        "schema_version": "V15-mapping-1",
        "clubench_readme": str(args.clubench_readme.resolve()),
        "clubench_readme_sha256": _sha256(args.clubench_readme),
        "clubench_commit_verified": False,
        "external_row_count": len(external_rows),
        "current_npz_count": len(npz_paths),
        "requested_missing_from_npz": sorted(name for name in requested if _normalise(name) not in current_names),
        "mappings": mappings,
        "note": "README r_mm mapping is exploratory until hj-n/labeled-datasets and hj-n/clm commits are fixed and hashed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"external_rows": len(external_rows), "npz": len(npz_paths), "output": str(args.output)}))


if __name__ == "__main__":
    main()
