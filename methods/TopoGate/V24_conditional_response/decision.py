from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .config import V24Q1Config


NULL_WORLDS = ("W0_global_null", "W1_mean_only", "W2_support_only", "W3_marginal_only")
REQUIRED_WORLDS = (*NULL_WORLDS, "W4_dependency_only", "W5_mixed_realistic")


def _world_summary(records: list[dict[str, Any]], config: V24Q1Config) -> dict[str, Any]:
    deltas = np.asarray([float(record["delta_auc"]) for record in records], dtype=np.float64)
    lower = np.asarray([float(record["ci95_low"]) for record in records], dtype=np.float64)
    upper = np.asarray([float(record["ci95_high"]) for record in records], dtype=np.float64)
    return {
        "seeds": [int(record["seed"]) for record in records],
        "delta_auc_mean": float(np.mean(deltas)),
        "delta_auc_std": float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
        "positive_seed_count": int(np.count_nonzero(deltas > 0.0)),
        "inside_equivalence_seed_count": int(np.count_nonzero(np.abs(deltas) <= config.equivalence_margin)),
        "ci95_low_min": float(np.min(lower)),
        "ci95_high_max": float(np.max(upper)),
        "contracts_valid": bool(all(bool(record.get("contract_valid", True)) for record in records)),
    }


def _records_have_finite_ci(records: list[dict[str, Any]]) -> bool:
    try:
        values = np.asarray(
            [(float(record["ci95_low"]), float(record["ci95_high"])) for record in records],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(values.size and np.isfinite(values).all())


def _equivalence_pass(row: dict[str, Any], config: V24Q1Config) -> bool:
    return bool(
        row["contracts_valid"]
        and abs(row["delta_auc_mean"]) <= config.null_point_margin
        and row["ci95_low_min"] >= -config.equivalence_margin
        and row["ci95_high_max"] <= config.equivalence_margin
        and row["inside_equivalence_seed_count"] >= 4
    )


def decide_q1(
    records: list[dict[str, Any]],
    config: V24Q1Config,
    *,
    calibration: dict[str, Any] | None = None,
    postmortem: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply frozen Q1 rules without treating observed controls as independence tests."""

    config.validate()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["world"])].append(record)
    missing = [world for world in REQUIRED_WORLDS if len(grouped[world]) != len(config.primary_seeds)]
    if missing:
        return {
            "decision": "incomplete_compute",
            "promotion_to_q2": False,
            "missing_or_incomplete_worlds": missing,
            "automatic_go_decision": False,
        }
    if not _records_have_finite_ci(records):
        return {
            "decision": "incomplete_uncertainty",
            "promotion_to_q2": False,
            "reason": "formal Q1 requires finite per-seed Poisson-bootstrap confidence intervals",
            "automatic_go_decision": False,
        }

    summaries = {world: _world_summary(grouped[world], config) for world in REQUIRED_WORLDS}
    null_passes = {world: _equivalence_pass(summaries[world], config) for world in NULL_WORLDS}
    w4 = summaries["W4_dependency_only"]
    w5 = summaries["W5_mixed_realistic"]
    w4_pass = bool(
        w4["contracts_valid"]
        and w4["delta_auc_mean"] >= config.w4_delta_min
        and w4["ci95_low_min"] > 0.0
        and w4["positive_seed_count"] >= 4
    )
    w5_pass = bool(
        w5["contracts_valid"]
        and w5["delta_auc_mean"] >= config.w5_delta_min
        and w5["ci95_low_min"] > 0.0
        and w5["positive_seed_count"] >= 4
    )
    p1_pass = bool(all(null_passes.values()) and w4_pass and w5_pass)
    calibration_pass = bool(calibration and calibration.get("calibration_passes") is True)
    postmortem_completed = bool(postmortem and postmortem.get("status") == "completed")

    if not p1_pass:
        decision = "no_go_stop_response_direction"
        promotion = False
        interpretation = "corrected_P1_criteria_not_met"
    elif not calibration_pass:
        decision = "calibration_incomplete_or_failed"
        promotion = False
        interpretation = "P1_passed_but_estimator_calibration_is_not_a_promotion_gate_pass"
    elif not postmortem_completed:
        decision = "p0_postmortem_incomplete"
        promotion = False
        interpretation = "P1_passed_but_V23_read_only_postmortem_must_be_recorded_before_Q2"
    else:
        decision = "go_q2_preregistration_required"
        promotion = True
        interpretation = "conditional_incremental_utility_after_observed_controls_only"
    return {
        "decision": decision,
        "promotion_to_q2": promotion,
        "automatic_go_decision": False,
        "interpretation": interpretation,
        "p1_synthetic_pass": p1_pass,
        "null_world_equivalence": null_passes,
        "w4_dependency_pass": w4_pass,
        "w5_mixed_pass": w5_pass,
        "estimator_calibration_pass": calibration_pass,
        "p0_postmortem_completed": postmortem_completed,
        "worlds": summaries,
    }
