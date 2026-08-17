#!/usr/bin/env python3
"""Compare ACCG's greedy joint selector with brute force on small W5 instances."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.ACCG_action_constrained_gate.calibration import calibrate_epsilon
from methods.TopoGate.ACCG_action_constrained_gate.config import FeatureConstraintConfig
from methods.TopoGate.ACCG_action_constrained_gate.feature_model import fit_cross_fitted_feature_model
from methods.TopoGate.ACCG_action_constrained_gate.selector import exact_constrained_action, select_action
from methods.TopoGate.ACCG_action_constrained_gate.synthetic import SyntheticConfig, generate_worlds


def run_audit(seed: int, rows: int) -> dict[str, object]:
    synthetic = SyntheticConfig(
        n_samples=160,
        n_features=20,
        n_clusters=4,
        module_size=2,
        zero_fraction=0.60,
        families=("lognormal_sparse",),
    )
    world = generate_worlds(synthetic, family="lognormal_sparse", seed=seed)["W5_joint_interaction"]
    constraint = FeatureConstraintConfig(
        max_features=24,
        graph_k=2,
        graph_crossfit_folds=4,
        epsilon_rounds=8,
        selector_pair_lookahead=20,
        exact_solver_max_features=24,
    )
    model = fit_cross_fitted_feature_model(world.X, config=constraint, seed=seed)
    calibration = calibrate_epsilon(world.X, model, mask_ratio=0.15, config=constraint, seed=seed)
    z = model.transform_matrix(world.X).astype(np.float64)
    donor_indices = np.roll(np.arange(world.X.shape[0]), 1)
    donor = z[donor_indices]
    rng = np.random.default_rng(int(seed) + 77)
    records = []
    for row in range(min(int(rows), world.X.shape[0])):
        eligible = donor[row] != z[row]
        budget = min(int(np.ceil(eligible.sum() * 0.15)), 3)
        if budget <= 0:
            continue
        scores = rng.normal(size=world.X.shape[1])
        greedy = select_action(
            scores[None, :],
            eligible[None, :],
            z[row : row + 1],
            donor[row : row + 1],
            row_ids=np.asarray([row]),
            epsilon=np.asarray([calibration.epsilon[row]]),
            model=model,
            mask_ratio=budget / max(1, int(eligible.sum())),
            selector_mode="joint",
            greedy_passes=2,
            pair_lookahead=20,
            fallback="least_violation",
        )
        exact_mask, exact = exact_constrained_action(
            scores,
            eligible,
            budget,
            z[row],
            donor[row],
            epsilon=float(calibration.epsilon[row]),
            fold=model.fold_for_row(row),
        )
        greedy_hardness = float(scores[greedy.hard_mask[0].astype(bool)].sum())
        exact_hardness = float(scores[exact_mask].sum())
        records.append(
            {
                "row": row,
                "budget": budget,
                "exact_feasible": bool(exact["feasible"]),
                "greedy_infeasible": bool(greedy.constraint_infeasible[0]),
                "greedy_joint_delta": float(greedy.joint_delta[0]),
                "exact_joint_delta": float(exact["joint_delta"]),
                "hardness_gap_exact_minus_greedy": exact_hardness - greedy_hardness,
            }
        )
    feasible = [row for row in records if row["exact_feasible"]]
    return {
        "protocol": "accg_small_w5_exact_gap_audit_v1",
        "seed": int(seed),
        "rows": records,
        "exact_feasible_rows": len(feasible),
        "greedy_feasible_given_exact_rate": float(
            np.mean([not row["greedy_infeasible"] for row in feasible]) if feasible else np.nan
        ),
        "mean_hardness_gap": float(
            np.mean([row["hardness_gap_exact_minus_greedy"] for row in feasible]) if feasible else np.nan
        ),
        "labels_used": False,
        "formal_training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_audit(args.seed, args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(payload["rows"]), "formal_training_started": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
