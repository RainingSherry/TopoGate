#!/usr/bin/env python3
"""Audit generated ACCG worlds for support and marginal shortcut leakage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.ACCG_action_constrained_gate.synthetic import SyntheticConfig, SyntheticWorld
from methods.TopoGate.ACCG_action_constrained_gate.synthetic_audit import audit_shortcuts


def _load_world(record: dict[str, object]) -> SyntheticWorld:
    matrix = np.load(str(record["matrix_path"]), allow_pickle=False)["X"]
    labels = np.load(str(record["labels_path"]), allow_pickle=False)
    oracle = np.load(str(record["oracle_path"]), allow_pickle=False)
    alternative_path = record.get("alternative_labels_path")
    alternative = None if not alternative_path else np.load(str(alternative_path), allow_pickle=False)
    metadata_path = Path(str(record["matrix_path"])).with_name("metadata.json")
    return SyntheticWorld(
        name=str(record["world"]),
        family=str(record["family"]),
        X=matrix,
        labels=labels,
        alternative_labels=alternative,
        clean_reference=oracle["clean_reference"],
        repair_mask=oracle["repair_mask"],
        protect_mask=oracle["protect_mask"],
        nuisance_mask=oracle["nuisance_mask"],
        module_ids=oracle["module_ids"],
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = SyntheticConfig(**manifest["config"])
    grouped: dict[tuple[str, int], dict[str, SyntheticWorld]] = {}
    for record in manifest["records"]:
        key = (str(record["family"]), int(record["seed"]))
        grouped.setdefault(key, {})[str(record["world"])] = _load_world(record)
    rows = []
    for (family, seed), worlds in sorted(grouped.items()):
        rows.append({"family": family, "seed": seed, **audit_shortcuts(worlds, config=config, seed=seed)})
    payload = {
        "protocol_id": config.protocol_id,
        "status": "audit_only_no_training",
        "valid": bool(rows and all(row["valid"] for row in rows)),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": payload["valid"], "audits": len(rows)}, indent=2))
    return 0 if payload["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
