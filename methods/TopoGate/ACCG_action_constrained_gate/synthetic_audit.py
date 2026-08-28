from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .synthetic import SyntheticConfig, SyntheticWorld


def _macro_auc(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    classes, counts = np.unique(labels, return_counts=True)
    folds = min(5, int(counts.min()))
    if classes.size < 2 or folds < 2:
        return float("nan")
    probabilities = np.zeros((labels.size, classes.size), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    for train, test in splitter.split(features, labels):
        model = LogisticRegression(max_iter=500, solver="lbfgs")
        model.fit(features[train], labels[train])
        probabilities[test] = model.predict_proba(features[test])
    return float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))


def _support_features(X: np.ndarray) -> np.ndarray:
    active = X != 0.0
    return np.column_stack(
        (
            active.mean(axis=1),
            active[:, ::2].mean(axis=1),
            active[:, 1::2].mean(axis=1),
        )
    ).astype(np.float32)


def _marginal_features(X: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            X.mean(axis=1),
            X.std(axis=1),
            np.quantile(X, 0.25, axis=1),
            np.quantile(X, 0.50, axis=1),
            np.quantile(X, 0.75, axis=1),
            np.linalg.norm(X, axis=1),
        )
    ).astype(np.float32)


def audit_shortcuts(
    worlds: dict[str, SyntheticWorld],
    *,
    config: SyntheticConfig,
    seed: int,
) -> dict[str, Any]:
    """Test whether declared matched summaries identify the world."""

    if not worlds:
        raise ValueError("worlds cannot be empty")
    names = sorted(worlds)
    matrices = [worlds[name].X for name in names]
    if len({matrix.shape for matrix in matrices}) != 1:
        raise ValueError("shortcut audit requires shape-matched worlds")
    labels = np.concatenate([np.full(matrix.shape[0], index, dtype=np.int64) for index, matrix in enumerate(matrices)])
    support_auc = _macro_auc(np.vstack([_support_features(matrix) for matrix in matrices]), labels, seed)
    marginal_auc = _macro_auc(np.vstack([_marginal_features(matrix) for matrix in matrices]), labels, seed + 1)
    support_hashes = [np.packbits(matrix != 0.0, axis=None).tobytes() for matrix in matrices]
    support_exactly_matched = len(set(support_hashes)) == 1
    column_summaries = [
        np.column_stack((matrix.mean(axis=0), matrix.std(axis=0), (matrix == 0.0).mean(axis=0)))
        for matrix in matrices
    ]
    max_column_summary_gap = max(
        float(np.max(np.abs(column_summaries[left] - column_summaries[right])))
        for left in range(len(names))
        for right in range(left + 1, len(names))
    )
    valid = bool(
        support_exactly_matched
        and np.isfinite(support_auc)
        and np.isfinite(marginal_auc)
        and support_auc <= config.shortcut_auc_ceiling
        and marginal_auc <= config.shortcut_auc_ceiling
    )
    return {
        "valid": valid,
        "worlds": names,
        "support_exactly_matched": support_exactly_matched,
        "support_macro_auc": support_auc,
        "marginal_macro_auc": marginal_auc,
        "auc_ceiling": float(config.shortcut_auc_ceiling),
        "max_column_summary_gap": max_column_summary_gap,
        "labels_used_for_generator_audit_only": True,
        "formal_method_result": False,
    }


def oracle_action_metrics(mask: np.ndarray, world: SyntheticWorld) -> dict[str, float]:
    selected = np.asarray(mask, dtype=np.bool_)
    if selected.shape != world.X.shape:
        raise ValueError("mask shape must match the synthetic matrix")
    selected_count = max(1, int(selected.sum()))
    repair_count = max(1, int(world.repair_mask.sum()))
    protect_count = max(1, int(world.protect_mask.sum()))
    nuisance_count = max(1, int(world.nuisance_mask.sum()))
    return {
        "noise_intervention_precision": float(np.logical_and(selected, world.repair_mask).sum() / selected_count),
        "noise_intervention_recall": float(np.logical_and(selected, world.repair_mask).sum() / repair_count),
        "rare_signal_preservation": float(np.logical_and(~selected, world.protect_mask).sum() / protect_count),
        "coherent_nuisance_false_protection": float(np.logical_and(~selected, world.nuisance_mask).sum() / nuisance_count),
    }
