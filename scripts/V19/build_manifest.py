#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "v19_rg_selected_advantage_v1"
MANIFEST_ID = "v19_rg_advantage_inputs_20260808_v1"
BIOLOGICAL = {
    "mouse_retina": ("Mouse_retina", "Mouse_retina.npz", "log1p_expression"),
    "campbell": ("Campbell", "Campbell.npz", "log1p_expression"),
    "baron_human": ("Baron Human", "Baron Human.npz", "raw_count"),
}
TEXT = {
    "sms_spam_collection": "sms_spam_collection.npz",
    "cnae9": "cnae9.npz",
    "imdb": "imdb.npz",
    "hate_speech": "hate_speech.npz",
    "sentiment_labeld_sentences": "sentiment_labeld_sentences.npz",
}


def _record(
    dataset_id: str,
    name: str,
    filename: str,
    input_protocol: str,
    input_kind: str,
) -> dict[str, Any]:
    source = ROOT / "datasets" / filename
    return {
        "dataset_id": dataset_id,
        "name": name,
        "source_path": str(source.resolve()),
        "source_hash": "unavailable",
        "source_hash_policy": "reuse_existing_or_unavailable_no_recomputation",
        "status": "eligible" if source.is_file() else "ineligible",
        "ineligible_reason": None if source.is_file() else "source_missing",
        "input_protocol": input_protocol,
        "input_kind": input_kind,
        "selection_uses_labels_or_outcomes": False,
        "comparison_scope": (
            "archived_sota_bridge_eligible"
            if input_protocol in {"clubench_bridge", "shared_text"}
            else "internal_rg_native_only"
        ),
    }


def build_manifest(output: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for slug, (name, filename, input_kind) in BIOLOGICAL.items():
        rows.append(_record(f"{slug}__rg_native", name, filename, "rg_native", input_kind))
        rows.append(
            _record(
                f"{slug}__clubench_bridge",
                name,
                filename,
                "clubench_bridge",
                input_kind,
            )
        )
    for slug, filename in TEXT.items():
        rows.append(_record(f"{slug}__shared_text", slug, filename, "shared_text", "sparse_text_features"))
    payload = {
        "protocol_id": PROTOCOL_ID,
        "manifest_id": MANIFEST_ID,
        "description": "Fixed RG-advantage sparse/count CLUBench subset for V19",
        "selection_policy": {
            "selection_uses_labels_or_outcomes": False,
            "selection_basis": "input semantics fixed before V19 runs: sparse text or biological count/expression",
            "text_native_bridge_deduplicated": True,
            "biological_protocols_kept_separate": True,
        },
        "variants": ["scmae_only", "rg_full"],
        "formal_seeds_in_order": [42, 123, 7],
        "expected_input_strata": 11,
        "expected_runs_total": 66,
        "hash_policy": "do not recompute SHA/hash; use existing provenance or unavailable",
        "datasets": rows,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the fixed V19 selected-dataset manifest")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "result" / "V19" / "v19_rg_dataset_manifest_20260808.json",
    )
    args = parser.parse_args()
    payload = build_manifest(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(payload["datasets"]),
                "eligible": sum(row["status"] == "eligible" for row in payload["datasets"]),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
