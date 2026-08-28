#!/usr/bin/env python3
"""Evaluate completed ACCG synthetic runs using outer-only oracle metadata."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.ACCG_action_constrained_gate.synthetic import SyntheticWorld
from methods.TopoGate.ACCG_action_constrained_gate.synthetic_audit import oracle_action_metrics
from scripts.ACCG.run_synthetic_matrix import ABLATIONS, CORE_WORLDS


def _world(record: dict[str, object]) -> SyntheticWorld:
    matrix = np.load(str(record["matrix_path"]), allow_pickle=False)["X"]
    labels = np.load(str(record["labels_path"]), allow_pickle=False)
    oracle = np.load(str(record["oracle_path"]), allow_pickle=False)
    alternative = record.get("alternative_labels_path")
    return SyntheticWorld(
        name=str(record["world"]),
        family=str(record["family"]),
        X=matrix,
        labels=labels,
        alternative_labels=None if not alternative else np.load(str(alternative), allow_pickle=False),
        clean_reference=oracle["clean_reference"],
        repair_mask=oracle["repair_mask"],
        protect_mask=oracle["protect_mask"],
        nuisance_mask=oracle["nuisance_mask"],
        module_ids=oracle["module_ids"],
        metadata={},
    )


def _evaluate_output(output: Path, world: SyntheticWorld, arms: tuple[str, ...]) -> dict[str, object] | None:
    if not (output / "summary.json").is_file():
        return None
    result: dict[str, object] = {}
    for arm in arms:
        prediction_path = output / arm / "predictions.npy"
        if not prediction_path.is_file():
            continue
        predictions = np.load(prediction_path, allow_pickle=False)
        arm_result: dict[str, object] = {
            "ari": float(adjusted_rand_score(world.labels, predictions)),
            "nmi": float(normalized_mutual_info_score(world.labels, predictions)),
        }
        if world.alternative_labels is not None:
            arm_result["ari_alternative_partition"] = float(
                adjusted_rand_score(world.alternative_labels, predictions)
            )
        trace_path = output / arm / "selection_trace.npz"
        if trace_path.is_file():
            trace = np.load(trace_path, allow_pickle=False)
            row_ids = trace["row_ids"].astype(np.int64)
            if row_ids.size:
                sampled_world = SyntheticWorld(
                    name=world.name,
                    family=world.family,
                    X=world.X[row_ids],
                    labels=world.labels[row_ids],
                    alternative_labels=None,
                    clean_reference=world.clean_reference[row_ids],
                    repair_mask=world.repair_mask[row_ids],
                    protect_mask=world.protect_mask[row_ids],
                    nuisance_mask=world.nuisance_mask[row_ids],
                    module_ids=world.module_ids,
                    metadata={},
                )
                arm_result["oracle_action_metrics"] = oracle_action_metrics(trace["hard_masks"], sampled_world)
        result[arm] = arm_result
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    incomplete = []
    for record in manifest["records"]:
        if record["world"] not in CORE_WORLDS:
            continue
        world = _world(record)
        base = args.output_root / record["family"] / record["world"] / f"seed{record['seed']}"
        main = _evaluate_output(base / "main", world, ("N", "R", "T_s", "T_c"))
        if main is None:
            incomplete.append(record["dataset_id"])
            continue
        row = {"dataset_id": record["dataset_id"], "family": record["family"], "world": record["world"], "seed": record["seed"], "main": main}
        if record["world"] == "W5_joint_interaction":
            ablations = {name: _evaluate_output(base / name, world, ("T_c",)) for name in ABLATIONS}
            missing_ablations = [name for name, value in ablations.items() if value is None]
            if missing_ablations:
                incomplete.extend(f"{record['dataset_id']}::{name}" for name in missing_ablations)
            row["ablations"] = ablations
        rows.append(row)
    payload = {
        "status": "complete" if not incomplete else "incomplete_compute",
        "labels_used_during_fit": False,
        "oracle_metadata_used_after_fit_only": True,
        "rows": rows,
        "incomplete": incomplete,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows)}, indent=2))
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
