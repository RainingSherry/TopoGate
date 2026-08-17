from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from methods.TopoGate.V21_assignment_adversarial_gate.graph import (
    build_svd_knn_graph,
    compute_topology_statistics,
)
from methods.TopoGate.V21_assignment_adversarial_gate.input_adapter import prepare_dual_input

from .calibration import calibrate_epsilon
from .config import FeatureConstraintConfig
from .feature_model import fit_cross_fitted_feature_model
from .synthetic import SyntheticConfig, SyntheticWorld


@dataclass(frozen=True)
class ActionProbeConfig:
    mask_ratio: float = 0.10
    max_rows: int = 256
    actions_per_row: int = 4
    auc_floor: float = 0.65
    bootstrap_replicates: int = 1000

    def validate(self) -> None:
        if not 0.0 < self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in (0, 1]")
        if self.max_rows <= 0 or self.actions_per_row < 2:
            raise ValueError("invalid action probe sample counts")
        if not 0.5 < self.auc_floor < 1.0 or self.bootstrap_replicates <= 0:
            raise ValueError("invalid action probe decision parameters")


def _force_exact_budget(
    eligible: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    *,
    include: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.zeros(eligible.size, dtype=np.bool_)
    include_values = np.empty(0, dtype=np.int64) if include is None else np.asarray(include, dtype=np.int64)
    include_values = include_values[eligible[include_values]]
    if include_values.size > budget:
        include_values = include_values[:budget]
    mask[include_values] = True
    candidates = np.flatnonzero(eligible & ~mask)
    if exclude is not None:
        candidates = candidates[~np.isin(candidates, np.asarray(exclude, dtype=np.int64))]
    needed = budget - int(mask.sum())
    if candidates.size < needed:
        fallback = np.flatnonzero(eligible & ~mask)
        candidates = np.unique(np.concatenate((candidates, fallback)))
    if needed > 0:
        mask[rng.choice(candidates, size=needed, replace=False)] = True
    return mask


def _candidate_mask(
    world: SyntheticWorld,
    row: int,
    eligible: np.ndarray,
    budget: int,
    action_index: int,
    rng: np.random.Generator,
    synthetic: SyntheticConfig,
) -> np.ndarray:
    if world.name == "W1_isolated_corruption":
        repair = np.flatnonzero(world.repair_mask[row])
        return _force_exact_budget(
            eligible,
            budget,
            rng,
            include=repair[:1] if action_index % 2 == 0 else None,
            exclude=repair if action_index % 2 else None,
        )
    if world.name == "W2_rare_coherent_signal":
        protected = np.flatnonzero(world.protect_mask[row])
        return _force_exact_budget(
            eligible,
            budget,
            rng,
            include=protected[:1] if action_index % 2 == 0 else None,
            exclude=protected if action_index % 2 else None,
        )
    if world.name == "W3_coherent_nuisance":
        nuisance = np.flatnonzero(world.nuisance_mask[row])
        signal = np.arange(synthetic.n_clusters * synthetic.module_size, dtype=np.int64)
        return _force_exact_budget(
            eligible,
            budget,
            rng,
            include=nuisance[:1] if action_index % 2 == 0 else signal[:1],
            exclude=signal if action_index % 2 == 0 else nuisance,
        )
    if world.name == "W5_joint_interaction":
        pair = np.asarray(world.metadata["interaction_pair"], dtype=np.int64)
        eligible_pair = pair[eligible[pair]]
        if action_index % 2 == 0 and eligible_pair.size == pair.size and pair.size <= budget:
            return _force_exact_budget(eligible, budget, rng, include=pair)
        include = eligible_pair[:1]
        exclude = pair[~np.isin(pair, include)]
        return _force_exact_budget(eligible, budget, rng, include=include, exclude=exclude)
    return _force_exact_budget(eligible, budget, rng)


def _oracle_target(
    world: SyntheticWorld,
    row: int,
    donor_row: int,
    mask: np.ndarray,
    synthetic: SyntheticConfig,
) -> bool:
    if world.name == "W1_isolated_corruption":
        selected_repair = mask & world.repair_mask[row]
        if not selected_repair.any():
            return False
        anchor_error = np.square(world.X[row] - world.clean_reference[row])
        donor_error = np.square(world.X[donor_row] - world.clean_reference[row])
        return bool(np.mean(donor_error[selected_repair]) < np.mean(anchor_error[selected_repair]))
    if world.name == "W2_rare_coherent_signal":
        return bool(not np.any(mask & world.protect_mask[row]))
    if world.name == "W3_coherent_nuisance":
        signal = np.zeros(mask.size, dtype=np.bool_)
        signal[: synthetic.n_clusters * synthetic.module_size] = True
        return bool(np.any(mask & world.nuisance_mask[row]) and not np.any(mask & signal))
    if world.name == "W5_joint_interaction":
        pair = np.asarray(world.metadata["interaction_pair"], dtype=np.int64)
        touched = mask[pair]
        return bool(touched.any() and touched.all())
    return False


def build_action_probe(
    world: SyntheticWorld,
    *,
    synthetic_config: SyntheticConfig,
    constraint_config: FeatureConstraintConfig,
    probe_config: ActionProbeConfig,
    seed: int,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Build outer-only oracle action records; no model or selector sees targets."""

    probe_config.validate()
    prepared = prepare_dual_input(world.X, dataset_name=world.name, input_protocol="clubench_bridge")
    feature_model = fit_cross_fitted_feature_model(prepared.X_model, config=constraint_config, seed=seed)
    calibration = calibrate_epsilon(
        prepared.X_model,
        feature_model,
        mask_ratio=probe_config.mask_ratio,
        config=constraint_config,
        seed=seed,
    )
    sample_graph = build_svd_knn_graph(
        prepared.X_graph,
        neighbor_k=min(20, max(1, prepared.X_model.shape[0] - 1)),
        svd_target=0.95,
        svd_min_dim=min(10, max(1, prepared.X_model.shape[0] - 1)),
        svd_max_dim=min(50, max(1, prepared.X_model.shape[0] - 1)),
        seed=seed,
    )
    topology, _profile = compute_topology_statistics(
        prepared.X_model,
        sample_graph,
        block_size=256,
        cache_dir=None,
        clip=5.0,
    )
    z = feature_model.transform_matrix(prepared.X_model).astype(np.float64)
    rng = np.random.default_rng(int(seed) + 90_001)
    candidate_rows = np.arange(world.X.shape[0])
    if world.name == "W1_isolated_corruption":
        candidate_rows = np.flatnonzero(world.repair_mask.any(axis=1))
    elif world.name == "W2_rare_coherent_signal":
        candidate_rows = np.flatnonzero(world.protect_mask.any(axis=1))
    candidate_rows = candidate_rows[: min(candidate_rows.size, probe_config.max_rows)]
    records: dict[str, list[float | int | bool]] = {
        "row": [],
        "donor_row": [],
        "sample_hardness": [],
        "donor_magnitude": [],
        "marginal_delta": [],
        "joint_delta": [],
        "epsilon": [],
        "target": [],
    }
    for row in candidate_rows:
        for action_index in range(probe_config.actions_per_row):
            donor_row = int((row + 1 + action_index * 17) % world.X.shape[0])
            eligible = z[donor_row] != z[row]
            budget = min(int(eligible.sum()), int(np.ceil(eligible.sum() * probe_config.mask_ratio)))
            if budget <= 0:
                continue
            mask = _candidate_mask(world, int(row), eligible, budget, action_index, rng, synthetic_config)
            fold = feature_model.fold_for_row(int(row))
            joint_delta, _clean, _action = fold.action_delta(z[row], z[donor_row], mask)
            singleton_values = []
            for feature in np.flatnonzero(mask):
                singleton = np.zeros(mask.size, dtype=np.bool_)
                singleton[feature] = True
                singleton_values.append(fold.action_delta(z[row], z[donor_row], singleton)[0])
            records["row"].append(int(row))
            records["donor_row"].append(donor_row)
            records["sample_hardness"].append(float(np.mean(topology[row, mask, 0])))
            records["donor_magnitude"].append(float(np.mean(np.abs(z[donor_row, mask] - z[row, mask]))))
            records["marginal_delta"].append(float(np.mean(singleton_values)))
            records["joint_delta"].append(float(joint_delta))
            records["epsilon"].append(float(calibration.epsilon[row]))
            records["target"].append(_oracle_target(world, int(row), donor_row, mask, synthetic_config))
    arrays = {name: np.asarray(values) for name, values in records.items()}
    arrays["target"] = arrays["target"].astype(np.int64)
    return {
        **arrays,
        "profile": {
            "world": world.name,
            "family": world.family,
            "seed": int(seed),
            "records": int(arrays["target"].size),
            "positive_rate": float(np.mean(arrays["target"])) if arrays["target"].size else float("nan"),
            "labels_used_by_method": False,
            "oracle_used_after_action_features_only": True,
            "epsilon_profile": calibration.profile,
        },
    }


def _cv_splits(
    features: np.ndarray,
    target: np.ndarray,
    seed: int,
    groups: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    classes, counts = np.unique(target, return_counts=True)
    if classes.size != 2:
        return []
    if groups is None:
        folds = min(5, int(counts.min()))
        if folds < 2:
            return []
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        return [(train, test) for train, test in splitter.split(features, target)]
    group_values = np.asarray(groups)
    if group_values.shape != target.shape:
        raise ValueError("groups must have one value per action record")
    class_group_counts = [np.unique(group_values[target == label]).size for label in classes]
    folds = min(5, int(np.unique(group_values).size), min(class_group_counts))
    if folds < 2:
        return []
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    splits = [(train, test) for train, test in splitter.split(features, target, group_values)]
    if any(np.unique(target[train]).size < 2 or np.unique(target[test]).size < 2 for train, test in splits):
        return []
    return splits


def _fit_scores(features: np.ndarray, target: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if not splits:
        return np.full(target.size, np.nan)
    output = np.zeros(target.size, dtype=np.float64)
    for train, test in splits:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, solver="lbfgs"),
        )
        model.fit(features[train], target[train])
        output[test] = model.predict_proba(features[test])[:, 1]
    return output


def _bootstrap_indices(
    target_size: int,
    rng: np.random.Generator,
    groups: np.ndarray | None,
) -> np.ndarray:
    if groups is None:
        return rng.integers(0, target_size, size=target_size)
    group_values = np.asarray(groups)
    unique_groups = np.unique(group_values)
    sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
    return np.concatenate([np.flatnonzero(group_values == group) for group in sampled_groups])


def evaluate_incremental_information(
    baseline: np.ndarray,
    joint_feature: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    bootstrap_replicates: int,
    groups: np.ndarray | None = None,
) -> dict[str, float | bool]:
    target = np.asarray(target, dtype=np.int64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    if baseline_values.ndim == 1:
        baseline_values = baseline_values[:, None]
    if baseline_values.ndim != 2 or baseline_values.shape[0] != target.size:
        raise ValueError("baseline must be [records, features]")
    joint_values = np.column_stack((baseline_values, np.asarray(joint_feature, dtype=np.float64)))
    splits = _cv_splits(baseline_values, target, seed, groups)
    baseline_score = _fit_scores(baseline_values, target, splits)
    joint_score = _fit_scores(joint_values, target, splits)
    if np.isnan(baseline_score).any() or np.isnan(joint_score).any():
        return {"valid": False, "auc_baseline": float("nan"), "auc_joint": float("nan"), "delta_auc": float("nan")}
    auc_baseline = float(roc_auc_score(target, baseline_score))
    auc_joint = float(roc_auc_score(target, joint_score))
    pr_baseline = float(average_precision_score(target, baseline_score))
    pr_joint = float(average_precision_score(target, joint_score))
    rng = np.random.default_rng(int(seed) + 3)
    delta_auc = []
    for _ in range(int(bootstrap_replicates)):
        sample = _bootstrap_indices(target.size, rng, groups)
        if np.unique(target[sample]).size < 2:
            continue
        delta_auc.append(
            float(roc_auc_score(target[sample], joint_score[sample]) - roc_auc_score(target[sample], baseline_score[sample]))
        )
    return {
        "valid": True,
        "auc_baseline": auc_baseline,
        "auc_joint": auc_joint,
        "delta_auc": auc_joint - auc_baseline,
        "pr_baseline": pr_baseline,
        "pr_joint": pr_joint,
        "delta_pr": pr_joint - pr_baseline,
        "delta_auc_ci_low": float(np.quantile(delta_auc, 0.025)) if delta_auc else float("nan"),
        "delta_auc_ci_high": float(np.quantile(delta_auc, 0.975)) if delta_auc else float("nan"),
    }


def leave_family_out_information(
    features: np.ndarray,
    joint_feature: np.ndarray,
    target: np.ndarray,
    families: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(features, dtype=np.float64)
    full = np.column_stack((baseline, np.asarray(joint_feature, dtype=np.float64)))
    target = np.asarray(target, dtype=np.int64)
    groups = np.asarray(families).astype(str)
    if np.unique(groups).size < 2 or np.unique(target).size < 2:
        return {"valid": False}
    baseline_score = np.zeros(target.size, dtype=np.float64)
    full_score = np.zeros(target.size, dtype=np.float64)
    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(baseline, target, groups):
        if np.unique(target[train]).size < 2 or np.unique(target[test]).size < 2:
            return {"valid": False, "reason": "a family holdout lacks both oracle classes"}
        baseline_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
            baseline[train], target[train]
        )
        full_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
            full[train], target[train]
        )
        baseline_score[test] = baseline_model.predict_proba(baseline[test])[:, 1]
        full_score[test] = full_model.predict_proba(full[test])[:, 1]
    auc_baseline = float(roc_auc_score(target, baseline_score))
    auc_joint = float(roc_auc_score(target, full_score))
    pr_baseline = float(average_precision_score(target, baseline_score))
    pr_joint = float(average_precision_score(target, full_score))
    return {
        "valid": True,
        "auc_baseline": auc_baseline,
        "auc_joint": auc_joint,
        "delta_auc": auc_joint - auc_baseline,
        "pr_baseline": pr_baseline,
        "pr_joint": pr_joint,
        "delta_pr": pr_joint - pr_baseline,
    }
