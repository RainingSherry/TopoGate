from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FeatureConstraintConfig
from .feature_model import CrossFittedFeatureModel


@dataclass(frozen=True)
class EpsilonCalibration:
    epsilon: np.ndarray
    sampled_deltas: np.ndarray
    profile: dict[str, Any]


def calibrate_epsilon(
    X_model: np.ndarray,
    model: CrossFittedFeatureModel,
    *,
    mask_ratio: float,
    config: FeatureConstraintConfig,
    seed: int,
) -> EpsilonCalibration:
    """Calibrate structural-damage tolerance from label-free random joint actions."""

    X = np.asarray(X_model, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] != model.fold_ids.size or X.shape[1] != model.n_features:
        raise ValueError("X_model does not match the fitted feature model")
    if not 0.0 < float(mask_ratio) <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1]")
    z = model.transform_matrix(X).astype(np.float64)
    rng = np.random.default_rng(int(seed) + int(config.epsilon_seed_offset))
    deltas = np.zeros((X.shape[0], int(config.epsilon_rounds)), dtype=np.float64)
    budget_fill = np.zeros_like(deltas)
    for round_index in range(int(config.epsilon_rounds)):
        offset = int(rng.integers(1, X.shape[0])) if X.shape[0] > 1 else 0
        donor = np.roll(z, shift=offset, axis=0)
        eligible = np.abs(donor - z) > 0.0
        random_scores = rng.random(z.shape)
        for row in range(X.shape[0]):
            candidates = np.flatnonzero(eligible[row])
            budget = min(candidates.size, int(np.ceil(candidates.size * float(mask_ratio))))
            mask = np.zeros(X.shape[1], dtype=np.bool_)
            if budget:
                order = candidates[np.argsort(random_scores[row, candidates])[::-1]]
                mask[order[:budget]] = True
            delta, _clean, _action = model.fold_for_row(row).action_delta(z[row], donor[row], mask)
            deltas[row, round_index] = delta
            budget_fill[row, round_index] = float(mask.sum() / max(1, budget))
    global_epsilon = float(np.quantile(deltas, float(config.epsilon_quantile)))
    if config.epsilon_scope == "global":
        epsilon = np.full(X.shape[0], global_epsilon, dtype=np.float64)
    else:
        epsilon = np.quantile(deltas, float(config.epsilon_quantile), axis=1).astype(np.float64)
    profile = {
        "protocol": "label_free_random_joint_action_null_v1",
        "scope": config.epsilon_scope,
        "quantile": float(config.epsilon_quantile),
        "rounds": int(config.epsilon_rounds),
        "seed": int(seed) + int(config.epsilon_seed_offset),
        "global_epsilon": global_epsilon,
        "epsilon_min": float(np.min(epsilon)),
        "epsilon_median": float(np.median(epsilon)),
        "epsilon_max": float(np.max(epsilon)),
        "null_delta_mean": float(np.mean(deltas)),
        "null_delta_std": float(np.std(deltas)),
        "null_delta_min": float(np.min(deltas)),
        "null_delta_max": float(np.max(deltas)),
        "budget_fill_mean": float(np.mean(budget_fill)),
        "labels_used": False,
        "outcomes_used": False,
    }
    return EpsilonCalibration(epsilon=epsilon, sampled_deltas=deltas, profile=profile)
