#!/usr/bin/env python3
"""Evaluate the frozen v2 world-stratified ACCG action-probe contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.ACCG_action_constrained_gate.synthetic_probe import (  # noqa: E402
    leave_family_out_information,
)


PRIMARY_WORLD = "W5_joint_interaction"
SUPPORTING_WORLD = "W2_rare_coherent_signal"
CONTROL_WORLD = "W1_isolated_corruption"
BOUNDARY_WORLD = "W3_coherent_nuisance"
EXPECTED_WORLDS = frozenset(
    {PRIMARY_WORLD, SUPPORTING_WORLD, CONTROL_WORLD, BOUNDARY_WORLD}
)


def _family_holdout_scores(
    baseline: np.ndarray,
    joint: np.ndarray,
    target: np.ndarray,
    families: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(baseline, dtype=np.float64)
    full = np.column_stack((baseline, np.asarray(joint, dtype=np.float64)))
    target = np.asarray(target, dtype=np.int64)
    groups = np.asarray(families).astype(str)
    if np.unique(groups).size < 2 or np.unique(target).size < 2:
        return {"valid": False, "reason": "need two families and two target classes"}
    baseline_score = np.zeros(target.size, dtype=np.float64)
    full_score = np.zeros(target.size, dtype=np.float64)
    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(baseline, target, groups):
        if np.unique(target[train]).size < 2 or np.unique(target[test]).size < 2:
            return {"valid": False, "reason": "a family holdout lacks both target classes"}
        baseline_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
            baseline[train], target[train]
        )
        full_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
            full[train], target[train]
        )
        baseline_score[test] = baseline_model.predict_proba(baseline[test])[:, 1]
        full_score[test] = full_model.predict_proba(full[test])[:, 1]
    return {
        "valid": True,
        "baseline_score": baseline_score,
        "joint_score": full_score,
        "auc_baseline": float(roc_auc_score(target, baseline_score)),
        "auc_joint": float(roc_auc_score(target, full_score)),
        "delta_auc": float(roc_auc_score(target, full_score) - roc_auc_score(target, baseline_score)),
        "pr_baseline": float(average_precision_score(target, baseline_score)),
        "pr_joint": float(average_precision_score(target, full_score)),
        "delta_pr": float(average_precision_score(target, full_score) - average_precision_score(target, baseline_score)),
    }


def _row_group_bootstrap(
    target: np.ndarray,
    baseline_score: np.ndarray,
    joint_score: np.ndarray,
    row_groups: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.int64)
    baseline_score = np.asarray(baseline_score, dtype=np.float64)
    joint_score = np.asarray(joint_score, dtype=np.float64)
    row_groups = np.asarray(row_groups)
    unique_groups = np.unique(row_groups)
    rng = np.random.default_rng(int(seed) + 71_003)
    delta_auc: list[float] = []
    delta_pr: list[float] = []
    for _ in range(int(replicates)):
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([np.flatnonzero(row_groups == group) for group in sampled])
        if np.unique(target[indices]).size < 2:
            continue
        delta_auc.append(
            float(
                roc_auc_score(target[indices], joint_score[indices])
                - roc_auc_score(target[indices], baseline_score[indices])
            )
        )
        delta_pr.append(
            float(
                average_precision_score(target[indices], joint_score[indices])
                - average_precision_score(target[indices], baseline_score[indices])
            )
        )
    if not delta_auc:
        return {
            "valid": False,
            "delta_auc_ci_low": float("nan"),
            "delta_auc_ci_high": float("nan"),
            "delta_pr_ci_low": float("nan"),
            "delta_pr_ci_high": float("nan"),
        }
    return {
        "valid": True,
        "delta_auc_ci_low": float(np.quantile(delta_auc, 0.025)),
        "delta_auc_ci_high": float(np.quantile(delta_auc, 0.975)),
        "delta_pr_ci_low": float(np.quantile(delta_pr, 0.025)),
        "delta_pr_ci_high": float(np.quantile(delta_pr, 0.975)),
    }


def _load_world_records(probe_root: Path, world: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(probe_root.glob(f"*/{world}/seed*/action_probe.npz")):
        parts = path.parts
        family = path.parent.parent.parent.name
        seed_name = path.parent.name
        if not seed_name.startswith("seed"):
            raise ValueError(f"invalid probe seed directory: {path}")
        arrays = np.load(path, allow_pickle=False)
        target = np.asarray(arrays["target"], dtype=np.int64)
        baseline = np.column_stack(
            (
                arrays["sample_hardness"],
                arrays["donor_magnitude"],
                arrays["marginal_delta"],
            )
        ).astype(np.float64)
        joint = -np.asarray(arrays["joint_delta"], dtype=np.float64)
        rows = np.asarray(arrays["row"], dtype=np.int64)
        if not (baseline.shape[0] == joint.size == target.size == rows.size):
            raise ValueError(f"probe arrays have mismatched lengths: {path}")
        records.append(
            {
                "path": str(path.resolve()),
                "family": family,
                "seed": int(seed_name[4:]),
                "baseline": baseline,
                "joint": joint,
                "target": target,
                "rows": rows,
            }
        )
    return records


def _summarize_world(records: list[dict[str, Any]], *, seed: int, bootstrap_replicates: int) -> dict[str, Any]:
    if not records:
        return {"valid": False, "reason": "no probe records"}
    baseline = np.vstack([record["baseline"] for record in records])
    joint = np.concatenate([record["joint"] for record in records])
    target = np.concatenate([record["target"] for record in records])
    families = np.concatenate(
        [np.full(record["target"].size, record["family"], dtype=object) for record in records]
    )
    row_groups = np.concatenate(
        [
            np.asarray(record["rows"], dtype=np.int64)
            + int(index) * 10_000_000
            for index, record in enumerate(records)
        ]
    )
    holdout = _family_holdout_scores(baseline, joint, target, families)
    if not holdout.get("valid"):
        return {
            "valid": False,
            "records": int(target.size),
            "positive_rate": float(np.mean(target)) if target.size else float("nan"),
            "family_holdout": holdout,
        }
    bootstrap = _row_group_bootstrap(
        target,
        holdout["baseline_score"],
        holdout["joint_score"],
        row_groups,
        seed=seed,
        replicates=bootstrap_replicates,
    )
    result = {
        key: value
        for key, value in holdout.items()
        if key not in {"baseline_score", "joint_score"}
    }
    result.update(
        {
            "records": int(target.size),
            "positive_rate": float(np.mean(target)),
            "row_groups": int(np.unique(row_groups).size),
            "families": sorted(np.unique(families).tolist()),
            "bootstrap": bootstrap,
            "valid": bool(bootstrap.get("valid")),
        }
    )
    return result


def _decision(world: str, summary: dict[str, Any]) -> dict[str, Any]:
    if not summary.get("valid"):
        return {"role": "invalid", "passes": False, "reason": "invalid summary"}
    bootstrap = summary["bootstrap"]
    if world == PRIMARY_WORLD:
        passes = bool(
            summary["auc_joint"] >= 0.65
            and summary["delta_auc"] > 0.0
            and summary["delta_pr"] > 0.0
            and bootstrap["delta_auc_ci_low"] > 0.0
        )
        return {
            "role": "primary_joint_action",
            "passes": passes,
            "criteria": {
                "joint_auc_at_least": 0.65,
                "delta_auc_positive": True,
                "delta_pr_positive": True,
                "row_group_delta_auc_ci_low_positive": True,
            },
        }
    if world == SUPPORTING_WORLD:
        passes = bool(
            summary["auc_joint"] >= 0.65
            and summary["delta_auc"] > 0.0
            and summary["delta_pr"] > 0.0
        )
        return {
            "role": "supporting_coherent_protection",
            "passes": passes,
            "required_for_promotion": False,
            "criteria": {"joint_auc_at_least": 0.65, "delta_auc_positive": True, "delta_pr_positive": True},
        }
    if world == CONTROL_WORLD:
        return {
            "role": "negative_control",
            "passes": None,
            "required_for_promotion": False,
            "interpretation": "baseline competence is expected; joint-positive gain is not required",
        }
    return {
        "role": "coherent_nuisance_boundary",
        "passes": None,
        "required_for_promotion": False,
        "interpretation": "coherence may protect nuisance; no sign is a promotion criterion",
    }


def _selector_ready(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    feasible = int(payload.get("exact_feasible_rows", 0))
    rate = float(payload.get("greedy_feasible_given_exact_rate", float("nan")))
    ready = bool(
        feasible > 0
        and np.isfinite(rate)
        and rate >= 0.95
        and payload.get("labels_used") is False
        and payload.get("formal_training_started") is False
    )
    return {"ready": ready, "exact_feasible_rows": feasible, "greedy_feasible_given_exact_rate": rate}


def evaluate(
    *,
    manifest_path: Path,
    probe_root: Path,
    shortcut_path: Path,
    selector_path: Path,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    if config.get("protocol_id") != "accg_synthetic_contract_v2":
        raise ValueError("v2 evaluator requires protocol_id accg_synthetic_contract_v2")
    shortcut = json.loads(shortcut_path.read_text(encoding="utf-8"))
    selector = _selector_ready(selector_path)
    worlds = {}
    for world in sorted(EXPECTED_WORLDS):
        records = _load_world_records(probe_root, world)
        summary = _summarize_world(records, seed=seed, bootstrap_replicates=bootstrap_replicates)
        summary["decision"] = _decision(world, summary)
        worlds[world] = summary
    primary = worlds[PRIMARY_WORLD]["decision"]
    payload = {
        "protocol_id": "accg_synthetic_contract_v2",
        "status": "audit_only_no_training",
        "manifest": str(manifest_path.resolve()),
        "probe_root": str(probe_root.resolve()),
        "shortcut_audit": {
            "valid": bool(shortcut.get("valid")),
            "path": str(shortcut_path.resolve()),
        },
        "selector_audit": {**selector, "path": str(selector_path.resolve())},
        "worlds": worlds,
        "contract_decision": {
            "primary_world": PRIMARY_WORLD,
            "w5_primary_passes": bool(primary.get("passes")),
            "shortcut_passes": bool(shortcut.get("valid")),
            "selector_passes": bool(selector["ready"]),
            "passes": bool(shortcut.get("valid") and selector["ready"] and primary.get("passes")),
            "real_training_authorized": False,
        },
        "labels_used_by_method": False,
        "formal_training_started": False,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--shortcut-audit", type=Path, required=True)
    parser.add_argument("--selector-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    payload = evaluate(
        manifest_path=args.manifest,
        probe_root=args.probe_root,
        shortcut_path=args.shortcut_audit,
        selector_path=args.selector_audit,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["contract_decision"], indent=2, ensure_ascii=True))
    return 0 if payload["contract_decision"]["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
