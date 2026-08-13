#!/usr/bin/env python3
"""Build the frozen V21 readout-fix extension manifest from archived sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "result/V19/v19_rg_extended_sparse_manifest_20260811.json"
DEFAULT_OUTPUT = ROOT / "result/V21/v21_extended13_readoutfix_manifest_20260811.json"
MANIFEST_ID = "v21_extended13_readoutfix_manifest_20260811"
PROTOCOL_ID = "v21_assignment_adversarial_extended13_readoutfix_v1"
MODEL_PROTOCOL_ID = "v21_assignment_adversarial_v3_readoutfix_v1"
DATASET_IDS = (
    "fbis_wc__local_sparse_text",
    "tr45_wc__local_sparse_text",
    "fabert__local_sparse_text",
    "micro_mass__local_sparse_highdim",
    "gina_prior2__local_sparse_highdim",
    "internet_advertisements__uci_sparse",
    "sms_spam_full__uci_sparse_text",
    "quake_smartseq2_lung__local_sparse_expression",
    "arcene__uci_highdim",
    "dexter__uci_sparse_highdim",
    "dorothea__uci_sparse_highdim",
    "gisette__uci_highdim_dense",
    "madelon__uci_highdim_control",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_manifest(source_path: Path = SOURCE_MANIFEST) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("protocol_id") != "v19_rg_extended_sparse_v1":
        raise ValueError("unexpected archived source manifest protocol")
    if source.get("selection_policy", {}).get("selection_uses_labels_or_outcomes") is not False:
        raise ValueError("source panel was not selected independently of outcomes")
    records = {str(row.get("dataset_id")): row for row in source.get("datasets", [])}
    missing = [dataset_id for dataset_id in DATASET_IDS if dataset_id not in records]
    if missing:
        raise ValueError(f"source manifest is missing V21 extension datasets: {missing}")
    selected = []
    for dataset_id in DATASET_IDS:
        record = dict(records[dataset_id])
        if record.get("status") != "eligible":
            raise ValueError(f"V21 extension dataset is not eligible: {dataset_id}")
        if not Path(str(record["source_path"])).is_file():
            raise FileNotFoundError(f"V21 extension source is missing: {record['source_path']}")
        record["v21_selection_uses_labels_or_outcomes"] = False
        selected.append(record)
    return {
        "manifest_id": MANIFEST_ID,
        "protocol_id": PROTOCOL_ID,
        "model_protocol_id": MODEL_PROTOCOL_ID,
        "description": "Outcome-independent 13-dataset transfer panel for the V21 readout correction",
        "source_manifest": str(source_path.resolve()),
        "source_manifest_id": source.get("manifest_id"),
        "selection_policy": {
            "selection_uses_labels_or_outcomes": False,
            "selection_basis": "all 13 datasets from the archived primary sparse/high-dimensional panel",
            "prior_v19_outcomes_used_for_selection": False,
            "v21_six_dataset_outcomes_used_for_selection": False,
            "overlap_with_v21_development_six": [],
            "sms_note": "sms_spam_collection_full_tfidf500 is a distinct input from the six-dataset sms_spam_collection matrix",
        },
        "development_provenance": {
            "hyperparameters": "transferred from the completed V21 six-dataset ARI development/confirmation layer",
            "selection_uses_labels": True,
            "extension_labels_used_for_selection": False,
            "primary_readout": "kmeans_embedding_known_k",
            "student_t_head_role": "differentiable training surrogate and diagnostic only",
        },
        "variants": ["topology_assignment_adversarial", "scmae_only"],
        "seeds": [42, 123, 7],
        "expected_dataset_count": len(selected),
        "expected_runs_total": len(selected) * 2 * 3,
        "comparison_scope": "held-out extension transfer; Full versus matched scMAE-only",
        "labels_used_during_fit": False,
        "datasets": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest(args.source_manifest)
    _write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "datasets": len(payload["datasets"]), "runs": payload["expected_runs_total"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
