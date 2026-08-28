#!/usr/bin/env python3
"""Build W1-W5 oracle action probes and incremental-information audits without training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.ACCG_action_constrained_gate.config import load_config
from methods.TopoGate.ACCG_action_constrained_gate.synthetic import SyntheticConfig, SyntheticWorld
from methods.TopoGate.ACCG_action_constrained_gate.synthetic_probe import (
    ActionProbeConfig,
    build_action_probe,
    evaluate_incremental_information,
    leave_family_out_information,
)


PROBE_WORLDS = frozenset(
    {"W1_isolated_corruption", "W2_rare_coherent_signal", "W3_coherent_nuisance", "W5_joint_interaction"}
)
IDENTIFIABILITY_WORLDS = frozenset(
    {"W1_isolated_corruption", "W2_rare_coherent_signal", "W5_joint_interaction"}
)
BOUNDARY_WORLDS = frozenset({"W3_coherent_nuisance"})


def _decision(information: dict[str, object], *, auc_floor: float, required: bool) -> dict[str, object]:
    valid = bool(information.get("valid"))
    auc_joint = float(information.get("auc_joint", float("nan")))
    ci_low = float(information.get("delta_auc_ci_low", float("nan")))
    delta_pr = float(information.get("delta_pr", float("nan")))
    passes = bool(valid and np.isfinite(auc_joint) and auc_joint >= auc_floor and ci_low > 0.0 and delta_pr > 0.0)
    return {
        "required_for_identifiability_gate": bool(required),
        "passes": passes if required else None,
        "criteria": {
            "auc_joint_at_least": float(auc_floor),
            "delta_auc_ci_low_positive": True,
            "delta_pr_positive": True,
        },
    }


def _load_world(record: dict[str, object]) -> SyntheticWorld:
    matrix = np.load(str(record["matrix_path"]), allow_pickle=False)["X"]
    labels = np.load(str(record["labels_path"]), allow_pickle=False)
    oracle = np.load(str(record["oracle_path"]), allow_pickle=False)
    metadata = json.loads(Path(str(record["matrix_path"])).with_name("metadata.json").read_text(encoding="utf-8"))
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
        metadata=metadata,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--accg-config",
        type=Path,
        default=ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs/accg_joint.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--actions-per-row", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    synthetic = SyntheticConfig(**manifest["config"])
    accg = load_config(args.accg_config)
    probe_config = ActionProbeConfig(max_rows=args.max_rows, actions_per_row=args.actions_per_row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    pooled: dict[str, list[np.ndarray]] = {
        "baseline": [],
        "joint": [],
        "target": [],
        "family": [],
    }
    for record in manifest["records"]:
        if record["world"] not in PROBE_WORLDS:
            continue
        world = _load_world(record)
        probe = build_action_probe(
            world,
            synthetic_config=synthetic,
            constraint_config=accg.constraint,
            probe_config=probe_config,
            seed=int(record["seed"]),
        )
        target = np.asarray(probe["target"], dtype=np.int64)
        baseline = np.column_stack(
            (
                probe["sample_hardness"],
                probe["donor_magnitude"],
                probe["marginal_delta"],
            )
        )
        information = evaluate_incremental_information(
            baseline,
            -np.asarray(probe["joint_delta"]),
            target,
            seed=int(record["seed"]),
            bootstrap_replicates=probe_config.bootstrap_replicates,
            groups=np.asarray(probe["row"], dtype=np.int64),
        )
        out = args.output_dir / record["family"] / record["world"] / f"seed{record['seed']}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "action_probe.npz",
            **{key: value for key, value in probe.items() if key != "profile"},
        )
        required = str(record["world"]) in IDENTIFIABILITY_WORLDS
        if not required and str(record["world"]) not in BOUNDARY_WORLDS:
            raise ValueError(f"unclassified action-probe world: {record['world']}")
        payload = {
            "profile": probe["profile"],
            "contract_role": "identifiability" if required else "coherent_nuisance_boundary",
            "incremental_information": information,
            "decision": _decision(information, auc_floor=probe_config.auc_floor, required=required),
        }
        (out / "action_probe_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        summaries.append({"record": record["dataset_id"], **payload})
        if information.get("valid") and required:
            pooled["baseline"].append(baseline)
            pooled["joint"].append(-np.asarray(probe["joint_delta"]))
            pooled["target"].append(target)
            pooled["family"].append(np.full(target.size, record["family"], dtype=object))
    family_holdout = {"valid": False}
    if pooled["target"]:
        family_holdout = leave_family_out_information(
            np.vstack(pooled["baseline"]),
            np.concatenate(pooled["joint"]),
            np.concatenate(pooled["target"]),
            np.concatenate(pooled["family"]),
        )
    required_decisions = [
        bool(row["decision"]["passes"])
        for row in summaries
        if row["contract_role"] == "identifiability"
    ]
    holdout_passes = bool(
        family_holdout.get("valid")
        and float(family_holdout.get("auc_joint", float("nan"))) >= probe_config.auc_floor
        and float(family_holdout.get("delta_auc", float("nan"))) > 0.0
        and float(family_holdout.get("delta_pr", float("nan"))) > 0.0
    )
    payload = {
        "status": "probe_only_no_training",
        "summaries": summaries,
        "generator_family_holdout": family_holdout,
        "contract_decision": {
            "valid": bool(required_decisions),
            "per_record_required_passes": int(sum(required_decisions)),
            "per_record_required_total": len(required_decisions),
            "all_required_records_pass": bool(required_decisions and all(required_decisions)),
            "generator_family_holdout_pass": holdout_passes,
            "passes": bool(required_decisions and all(required_decisions) and holdout_passes),
            "W3_is_boundary_not_a_required_positive": True,
        },
        "pass_threshold": {"joint_auc_lower_target": probe_config.auc_floor},
        "labels_used_by_method": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"probes": len(summaries), "training_started": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
