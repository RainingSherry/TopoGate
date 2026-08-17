#!/usr/bin/env python3
"""Evaluate the v2 ACCG action contract with world-stratified estimands."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.ACCG_action_constrained_gate.synthetic_probe import (
    evaluate_incremental_information,
    leave_family_out_information,
)


PRIMARY_WORLD = "W5_joint_interaction"
SECONDARY_WORLD = "W2_rare_coherent_signal"
NEGATIVE_CONTROL_WORLD = "W1_isolated_corruption"
PRIMARY_AUC_FLOOR = 0.65
SECONDARY_AUC_FLOOR = 0.60
FAMILY_AUC_FLOOR = PRIMARY_AUC_FLOOR


def _finite(value: object) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _load_probe(path: Path, *, group_offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        target = np.asarray(payload["target"], dtype=np.int64)
        baseline = np.column_stack(
            (
                payload["sample_hardness"],
                payload["donor_magnitude"],
                payload["marginal_delta"],
            )
        ).astype(np.float64)
        joint = -np.asarray(payload["joint_delta"], dtype=np.float64)
        rows = np.asarray(payload["row"], dtype=np.int64) + int(group_offset)
    if not (baseline.shape[0] == joint.size == target.size == rows.size):
        raise ValueError(f"probe arrays have inconsistent lengths: {path}")
    if np.unique(target).size != 2:
        raise ValueError(f"probe target is not binary: {path}")
    return baseline, joint, target, rows, np.full(target.size, group_offset, dtype=np.int64)


def _records(root: Path, world: str) -> list[tuple[str, int, Path]]:
    records: list[tuple[str, int, Path]] = []
    for family_dir in sorted(root.iterdir()):
        if not family_dir.is_dir():
            continue
        for path in sorted((family_dir / world).glob("seed*/action_probe.npz")):
            seed = int(path.parent.name.removeprefix("seed"))
            records.append((family_dir.name, seed, path))
    if not records:
        raise FileNotFoundError(f"no action probes found for {world} under {root}")
    return records


def _concat(root: Path, world: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    baselines: list[np.ndarray] = []
    joints: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    families: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    offset = 0
    for family, seed, path in _records(root, world):
        baseline, joint, target, row_groups, _record_groups = _load_probe(path, group_offset=offset)
        baselines.append(baseline)
        joints.append(joint)
        targets.append(target)
        groups.append(row_groups)
        families.append(np.full(target.size, family, dtype=object))
        metadata.append({"family": family, "seed": seed, "path": str(path.resolve()), "records": int(target.size)})
        offset += int(np.max(row_groups) + 1) if row_groups.size else 1
    return (
        np.vstack(baselines),
        np.concatenate(joints),
        np.concatenate(targets),
        np.concatenate(groups),
        np.concatenate(families),
        metadata,
    )


def _positive_delta(result: dict[str, object]) -> bool:
    return bool(
        result.get("valid")
        and _finite(result.get("auc_joint"))
        and _finite(result.get("delta_auc"))
        and _finite(result.get("delta_pr"))
        and float(result["delta_auc"]) > 0.0
        and float(result["delta_pr"]) > 0.0
    )


def _leave_one_family_out(
    baseline: np.ndarray,
    joint: np.ndarray,
    target: np.ndarray,
    families: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Return held-out metrics separately for every generator family."""
    values = np.asarray(families).astype(str)
    full = np.column_stack((baseline, joint))
    result: dict[str, dict[str, object]] = {}
    for family in sorted(np.unique(values)):
        train = values != family
        test = ~train
        if np.unique(target[train]).size != 2 or np.unique(target[test]).size != 2:
            result[family] = {"valid": False, "reason": "held-out split lacks both oracle classes"}
            continue
        baseline_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
            baseline[train], target[train]
        )
        full_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
            full[train], target[train]
        )
        baseline_score = baseline_model.predict_proba(baseline[test])[:, 1]
        full_score = full_model.predict_proba(full[test])[:, 1]
        auc_baseline = float(roc_auc_score(target[test], baseline_score))
        auc_joint = float(roc_auc_score(target[test], full_score))
        pr_baseline = float(average_precision_score(target[test], baseline_score))
        pr_joint = float(average_precision_score(target[test], full_score))
        result[family] = {
            "valid": True,
            "records": int(test.sum()),
            "auc_baseline": auc_baseline,
            "auc_joint": auc_joint,
            "delta_auc": auc_joint - auc_baseline,
            "pr_baseline": pr_baseline,
            "pr_joint": pr_joint,
            "delta_pr": pr_joint - pr_baseline,
        }
    return result


def _world_summary(root: Path, world: str, *, seed: int) -> dict[str, object]:
    baseline, joint, target, groups, families, metadata = _concat(root, world)
    family_holdout = leave_family_out_information(baseline, joint, target, families)
    held_out_families = _leave_one_family_out(baseline, joint, target, families)
    grouped = evaluate_incremental_information(
        baseline,
        joint,
        target,
        seed=seed,
        bootstrap_replicates=1000,
        groups=groups,
    )
    per_family: dict[str, dict[str, object]] = {}
    for family in sorted(set(families.tolist())):
        selected = families == family
        per_family[family] = evaluate_incremental_information(
            baseline[selected],
            joint[selected],
            target[selected],
            seed=seed,
            bootstrap_replicates=1000,
            groups=groups[selected],
        )
    return {
        "world": world,
        "records": int(target.size),
        "positive_rate": float(np.mean(target)),
        "families": sorted(set(families.tolist())),
        "source_records": metadata,
        "family_holdout": family_holdout,
        "held_out_family_metrics": held_out_families,
        "grouped_all_families": grouped,
        "per_family": per_family,
        "labels_used_by_method": False,
    }


def _primary_pass(summary: dict[str, object]) -> bool:
    holdout = summary["family_holdout"]
    held_out_families = summary["held_out_family_metrics"]
    holdout_pass = bool(
        holdout.get("valid")
        and float(holdout.get("auc_joint", float("nan"))) >= PRIMARY_AUC_FLOOR
        and float(holdout.get("delta_auc", float("nan"))) > 0.0
        and float(holdout.get("delta_pr", float("nan"))) > 0.0
    )
    family_pass = bool(
        held_out_families
        and all(
            value.get("valid")
            and float(value.get("auc_joint", float("nan"))) >= FAMILY_AUC_FLOOR
            and float(value.get("delta_auc", float("nan"))) > 0.0
            and float(value.get("delta_pr", float("nan"))) > 0.0
            for value in held_out_families.values()
        )
    )
    summary["decision"] = {
        "role": "primary_joint_action_estimand",
        "passes": bool(holdout_pass and family_pass),
        "family_holdout_pass": holdout_pass,
        "per_family_pass": family_pass,
        "criteria": {
            "family_holdout_auc_at_least": PRIMARY_AUC_FLOOR,
            "each_held_out_family_auc_at_least": FAMILY_AUC_FLOOR,
            "delta_auc_positive": True,
            "delta_pr_positive": True,
        },
    }
    return bool(summary["decision"]["passes"])


def _secondary_pass(summary: dict[str, object]) -> bool:
    holdout = summary["family_holdout"]
    passed = bool(
        holdout.get("valid")
        and float(holdout.get("auc_joint", float("nan"))) >= SECONDARY_AUC_FLOOR
        and float(holdout.get("delta_auc", float("nan"))) > 0.0
        and float(holdout.get("delta_pr", float("nan"))) > 0.0
    )
    summary["decision"] = {
        "role": "secondary_rare_signal_protection",
        "passes": passed,
        "criteria": {
            "family_holdout_auc_at_least": SECONDARY_AUC_FLOOR,
            "delta_auc_positive": True,
            "delta_pr_positive": True,
        },
    }
    return passed


def _negative_control(summary: dict[str, object]) -> bool:
    holdout = summary["family_holdout"]
    baseline_auc = float(holdout.get("auc_baseline", float("nan")))
    passed = bool(holdout.get("valid") and _finite(baseline_auc) and baseline_auc >= PRIMARY_AUC_FLOOR)
    summary["decision"] = {
        "role": "negative_control_marginal_repair",
        "passes": passed,
        "joint_increment_required": False,
        "criteria": {"baseline_family_holdout_auc_at_least": PRIMARY_AUC_FLOOR},
    }
    return passed


def evaluate(
    *,
    manifest: Path,
    shortcut_audit: Path,
    selector_audit: Path,
    probe_root: Path,
    output: Path,
) -> dict[str, Any]:
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    shortcut = json.loads(shortcut_audit.read_text(encoding="utf-8"))
    selector = json.loads(selector_audit.read_text(encoding="utf-8"))
    seed_values = sorted({int(record["seed"]) for record in manifest_payload["records"]})
    audit_seed = int(seed_values[0])
    worlds = {
        PRIMARY_WORLD: _world_summary(probe_root, PRIMARY_WORLD, seed=audit_seed),
        SECONDARY_WORLD: _world_summary(probe_root, SECONDARY_WORLD, seed=audit_seed),
        NEGATIVE_CONTROL_WORLD: _world_summary(probe_root, NEGATIVE_CONTROL_WORLD, seed=audit_seed),
    }
    w5_pass = _primary_pass(worlds[PRIMARY_WORLD])
    w2_pass = _secondary_pass(worlds[SECONDARY_WORLD])
    w1_pass = _negative_control(worlds[NEGATIVE_CONTROL_WORLD])
    selector_pass = bool(
        int(selector.get("exact_feasible_rows", 0)) > 0
        and float(selector.get("greedy_feasible_given_exact_rate", 0.0)) >= 0.95
        and _finite(selector.get("mean_hardness_gap"))
    )
    shortcut_pass = bool(shortcut.get("valid"))
    payload: dict[str, Any] = {
        "protocol_id": "accg_synthetic_contract_v2",
        "status": "contract_audit_only_no_training",
        "source_manifest": str(manifest.resolve()),
        "source_shortcut_audit": str(shortcut_audit.resolve()),
        "source_selector_audit": str(selector_audit.resolve()),
        "source_probe_root": str(probe_root.resolve()),
        "seed_values": seed_values,
        "world_estimands": worlds,
        "dependencies": {
            "shortcut_pass": shortcut_pass,
            "selector_pass": selector_pass,
            "w5_primary_pass": w5_pass,
            "w2_secondary_pass": w2_pass,
            "w1_negative_control_pass": w1_pass,
        },
        "promotion_decision": {
            "passes": bool(shortcut_pass and selector_pass and w5_pass),
            "primary_requires": ["shortcut_pass", "selector_pass", "w5_primary_pass"],
            "w2_is_secondary_not_required": True,
            "w1_is_negative_control_not_joint_positive": True,
            "labels_used_by_method": False,
            "formal_training_started": False,
        },
        "protocol_rationale": (
            "W5 is the only primary joint-action estimand. W1 tests a marginal repair control and "
            "therefore does not require incremental joint information. W2 is stratified secondary "
            "evidence; W3 remains a nuisance boundary and W4 an observational alias boundary."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shortcut-audit", type=Path, required=True)
    parser.add_argument("--selector-audit", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate(
        manifest=args.manifest,
        shortcut_audit=args.shortcut_audit,
        selector_audit=args.selector_audit,
        probe_root=args.probe_root,
        output=args.output,
    )
    print(json.dumps(payload["promotion_decision"], indent=2, ensure_ascii=True))
    return 0 if payload["promotion_decision"]["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
