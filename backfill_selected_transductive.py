#!/usr/bin/env python3
"""Create auditable selected.json files from immutable completed search records.

This script never re-ranks candidates or re-runs a model.  It copies the
already recorded ``search_summary.json.winner`` only after checking that the
completed final (when present) used the same config hash.  Backfilled files
are explicitly labelled so they cannot be confused with a file written at the
instant the search completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def final_config_hash(summary: dict) -> str | None:
    nested_winner = summary.get("winner")
    nested_hash = nested_winner.get("config_hash") if isinstance(nested_winner, dict) else None
    return summary.get("winner_config_hash") or summary.get("config_hash") or nested_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Experiment root containing datasets/")
    parser.add_argument(
        "--include-without-final",
        action="store_true",
        help="Also backfill a completed search that has not started final evaluation.",
    )
    parser.add_argument(
        "--annotate-final",
        action="store_true",
        help="Add the verified selected.json hash to an existing final summary without changing metrics.",
    )
    args = parser.parse_args()

    created: list[str] = []
    existing: list[str] = []
    annotated_final: list[str] = []
    skipped: dict[str, list[str]] = {
        "missing_search": [], "missing_final": [], "hash_mismatch": [], "selection_hash_mismatch": []
    }
    for dataset in sorted((args.root / "datasets").iterdir()):
        if not dataset.is_dir():
            continue
        selected_path = dataset / "selected.json"
        if selected_path.exists():
            existing.append(dataset.name)
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
        else:
            search_path = dataset / "search_summary.json"
            if not search_path.exists():
                skipped["missing_search"].append(dataset.name)
                continue
            search = json.loads(search_path.read_text(encoding="utf-8"))
            winner = search.get("winner")
            if not isinstance(winner, dict) or not winner.get("config_hash"):
                skipped["missing_search"].append(dataset.name)
                continue
        final_path = dataset / "final_summary.json"
        if not final_path.exists() and not args.include_without_final:
            skipped["missing_final"].append(dataset.name)
            continue
        if final_path.exists():
            summary = json.loads(final_path.read_text(encoding="utf-8"))
            observed = final_config_hash(summary)
            expected = selected["winner"]["config_hash"] if selected_path.exists() else winner["config_hash"]
            matching_selection_hash = (
                selected_path.exists()
                and summary.get("selection_hash") == selected.get("selection_hash")
            )
            if observed != expected and not matching_selection_hash:
                skipped["hash_mismatch"].append(dataset.name)
                continue
        if not selected_path.exists():
            selected = {
                "dataset_id": dataset.name,
                "protocol_id": search.get("protocol_id"),
                "screen_candidates": search.get("screen_candidates"),
                "selection_seeds": search.get("selection_seeds"),
                "selection_metric": "validation_ari_mean",
                "selection_tiebreakers": ["validation_ari_std", "config_hash"],
                "winner": winner,
                "selection_hash": canonical_hash(winner),
                "provenance": {
                    "source": "backfilled_from_existing_search_summary",
                    "search_summary_sha256": hashlib.sha256(search_path.read_bytes()).hexdigest(),
                    "final_config_hash_verified": final_path.exists(),
                    "note": "Derived audit artifact; no search, ranking, or final metric was recomputed.",
                },
            }
            atomic_json(selected_path, selected)
            created.append(dataset.name)
        if args.annotate_final and final_path.exists():
            summary = json.loads(final_path.read_text(encoding="utf-8"))
            frozen_hash = selected["selection_hash"]
            present_hash = summary.get("selection_hash")
            if present_hash is not None and present_hash != frozen_hash:
                skipped["selection_hash_mismatch"].append(dataset.name)
                continue
            if present_hash is None:
                summary["selection_hash"] = frozen_hash
                summary["audit_metadata"] = {
                    "selection_hash_source": "selected.json",
                    "final_config_hash_verified": True,
                    "note": "Audit metadata backfilled without recomputing or changing final metrics.",
                }
                atomic_json(final_path, summary)
                annotated_final.append(dataset.name)
    print(json.dumps({"created": created, "existing": len(existing), "annotated_final": annotated_final, "skipped": skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()
