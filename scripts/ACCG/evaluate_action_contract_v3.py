#!/usr/bin/env python3
"""Evaluate the final ACCG contract using grouped held-out-family increments."""
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

from methods.TopoGate.ACCG_action_constrained_gate.synthetic_probe import leave_family_out_information
from scripts.ACCG.evaluate_action_contract_v2 import (
    NEGATIVE_CONTROL_WORLD,
    PRIMARY_WORLD,
    SECONDARY_WORLD,
    _concat,
)


PRIMARY_AUC_FLOOR = 0.60
SECONDARY_AUC_FLOOR = 0.60
BOOTSTRAP_REPLICATES = 1000


def _finite(value: object) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _held_out_family_metrics(
    baseline: np.ndarray,
    joint: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    families: np.ndarray,
    *,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Fit on all but one family and bootstrap delta AUC by sample row."""
    family_values = np.asarray(families).astype(str)
    full = np.column_stack((baseline, joint))
    output: dict[str, dict[str, object]] = {}
    for family_index, family in enumerate(sorted(np.unique(family_values))):
        train = family_values != family
        test = ~train
        if np.unique(target[train]).size != 2 or np.unique(target[test]).size != 2:
            output[family] = {"valid": False, "reason": "held-out split lacks both oracle classes"}
            continue
        baseline_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
            baseline[train], target[train]
        )
        full_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
            full[train], target[train]
        )
        baseline_score = baseline_model.predict_proba(baseline[test])[:, 1]
        full_score = full_model.predict_proba(full[test])[:, 1]
        test_target = target[test]
        test_groups = np.asarray(groups[test])
        auc_baseline = float(roc_auc_score(test_target, baseline_score))
        auc_joint = float(roc_auc_score(test_target, full_score))
        pr_baseline = float(average_precision_score(test_target, baseline_score))
        pr_joint = float(average_precision_score(test_target, full_score))
        unique_groups = np.unique(test_groups)
        rng = np.random.default_rng(int(seed) + 17_003 * (family_index + 1))
        delta_auc_bootstrap: list[float] = []
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
            indices = np.concatenate([np.flatnonzero(test_groups == group) for group in sampled])
            if np.unique(test_target[indices]).size != 2:
                continue
            delta_auc_bootstrap.append(
                float(
                    roc_auc_score(test_target[indices], full_score[indices])
                    - roc_auc_score(test_target[indices], baseline_score[indices])
                )
            )
        output[family] = {
            "valid": True,
            "records": int(test_target.size),
            "sample_groups": int(unique_groups.size),
            "auc_baseline": auc_baseline,
            "auc_joint": auc_joint,
            "delta_auc": auc_joint - auc_baseline,
            "pr_baseline": pr_baseline,
            "pr_joint": pr_joint,
            "delta_pr": pr_joint - pr_baseline,
            "delta_auc_ci_low": float(np.quantile(delta_auc_bootstrap, 0.025))
            if delta_auc_bootstrap
            else float("nan"),
            "delta_auc_ci_high": float(np.quantile(delta_auc_bootstrap, 0.975))
            if delta_auc_bootstrap
            else float("nan"),
            "bootstrap_replicates_valid": len(delta_auc_bootstrap),
        }
    return output


def _world_summary(root: Path, world: str, *, seed: int) -> dict[str, object]:
    baseline, joint, target, groups, families, metadata = _concat(root, world)
    pooled = leave_family_out_information(baseline, joint, target, families)
    held_out = _held_out_family_metrics(
        baseline,
        joint,
        target,
        groups,
        families,
        seed=seed,
    )
    return {
        "world": world,
        "records": int(target.size),
        "positive_rate": float(np.mean(target)),
        "families": sorted(set(families.tolist())),
        "source_records": metadata,
        "family_holdout": pooled,
        "held_out_family_metrics": held_out,
        "labels_used_by_method": False,
    }


def _joint_pass(summary: dict[str, object], *, auc_floor: float) -> bool:
    pooled = summary["family_holdout"]
    pooled_pass = bool(
        pooled.get("valid")
        and float(pooled.get("auc_joint", float("nan"))) >= auc_floor
        and float(pooled.get("delta_auc", float("nan"))) > 0.0
        and float(pooled.get("delta_pr", float("nan"))) > 0.0
    )
    family_metrics = summary["held_out_family_metrics"]
    family_pass = bool(
        family_metrics
        and all(
            value.get("valid")
            and float(value.get("auc_joint", float("nan"))) >= auc_floor
            and float(value.get("delta_auc", float("nan"))) > 0.0
            and float(value.get("delta_pr", float("nan"))) > 0.0
            and _finite(value.get("delta_auc_ci_low"))
            and float(value["delta_auc_ci_low"]) > 0.0
            for value in family_metrics.values()
        )
    )
    summary["decision"] = {
        "passes": bool(pooled_pass and family_pass),
        "pooled_pass": pooled_pass,
        "each_held_out_family_pass": family_pass,
        "criteria": {
            "pooled_and_each_held_out_family_auc_at_least": auc_floor,
            "delta_auc_positive": True,
            "held_out_family_delta_auc_ci_low_positive": True,
            "delta_pr_positive": True,
        },
    }
    return bool(summary["decision"]["passes"])


def _w1_control(summary: dict[str, object]) -> bool:
    pooled = summary["family_holdout"]
    baseline_auc = float(pooled.get("auc_baseline", float("nan")))
    passed = bool(pooled.get("valid") and _finite(baseline_auc) and baseline_auc >= 0.60)
    summary["decision"] = {
        "passes": passed,
        "role": "negative_control_marginal_repair",
        "joint_increment_required": False,
        "criteria": {"baseline_family_holdout_auc_at_least": 0.60},
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
    seeds = sorted({int(record["seed"]) for record in manifest_payload["records"]})
    audit_seed = int(seeds[0])
    worlds = {
        PRIMARY_WORLD: _world_summary(probe_root, PRIMARY_WORLD, seed=audit_seed),
        SECONDARY_WORLD: _world_summary(probe_root, SECONDARY_WORLD, seed=audit_seed),
        NEGATIVE_CONTROL_WORLD: _world_summary(probe_root, NEGATIVE_CONTROL_WORLD, seed=audit_seed),
    }
    w5_pass = _joint_pass(worlds[PRIMARY_WORLD], auc_floor=PRIMARY_AUC_FLOOR)
    w2_pass = _joint_pass(worlds[SECONDARY_WORLD], auc_floor=SECONDARY_AUC_FLOOR)
    w1_pass = _w1_control(worlds[NEGATIVE_CONTROL_WORLD])
    selector_pass = bool(
        int(selector.get("exact_feasible_rows", 0)) > 0
        and float(selector.get("greedy_feasible_given_exact_rate", 0.0)) >= 0.95
        and _finite(selector.get("mean_hardness_gap"))
    )
    shortcut_pass = bool(shortcut.get("valid"))
    payload: dict[str, Any] = {
        "protocol_id": "accg_synthetic_contract_v3",
        "status": "contract_audit_only_no_training",
        "source_manifest": str(manifest.resolve()),
        "source_shortcut_audit": str(shortcut_audit.resolve()),
        "source_selector_audit": str(selector_audit.resolve()),
        "source_probe_root": str(probe_root.resolve()),
        "seed_values": seeds,
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
            "The paper claim is incremental joint-action information over a matched sample-side "
            "baseline. A standalone AUC of 0.65 is not the estimand; v3 retains a 0.60 floor and "
            "requires positive held-out-family grouped-bootstrap delta AUC and delta PR."
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
