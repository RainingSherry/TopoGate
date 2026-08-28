from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .config import V24Q1Config


GROUPED_SUPPORT_WORLDS = frozenset({"W1_mean_only", "W3_marginal_only", "W4_dependency_only"})


@dataclass(frozen=True)
class ContractAudit:
    world: str
    valid: bool
    metrics: dict[str, float | bool | str]


def _macro_ovr_auc(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    folds: int = 5,
    groups: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes = np.unique(labels)
    if classes.size < 2 or min(np.count_nonzero(labels == value) for value in classes) < folds:
        return float("nan"), np.empty((0, classes.size), dtype=np.float32)
    probabilities = np.zeros((labels.size, classes.size), dtype=np.float64)
    if groups is None:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        splits = splitter.split(features, labels)
    else:
        groups = np.asarray(groups).reshape(-1)
        if groups.shape != labels.shape:
            raise ValueError("groups must have one entry per sample")
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        splits = splitter.split(features, labels, groups)
    for train, test in splits:
        model = LogisticRegression(
            C=1.0,
            max_iter=500,
            solver="lbfgs",
        )
        model.fit(features[train], labels[train])
        probabilities[test] = model.predict_proba(features[test])
    return float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")), probabilities


def _support_template_groups(matrix: np.ndarray) -> np.ndarray:
    """Group exact support templates so their class copies cannot cross CV sides."""

    packed = np.packbits(matrix > 0.0, axis=1)
    _, groups = np.unique(packed, axis=0, return_inverse=True)
    return np.asarray(groups, dtype=np.int64)


def _featurewise_marginal_auc(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    groups: np.ndarray | None,
    max_features: int = 16,
) -> tuple[float, float, float, int]:
    """Probe one scalar feature at a time, excluding multifeature dependence."""

    feature_indices = np.unique(
        np.linspace(0, matrix.shape[1] - 1, num=min(max_features, matrix.shape[1]), dtype=np.int64)
    )
    values: list[float] = []
    for offset, feature in enumerate(feature_indices):
        auc, _ = _macro_ovr_auc(
            matrix[:, feature : feature + 1],
            labels,
            seed=seed + offset,
            groups=groups,
        )
        values.append(auc)
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(array)),
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
        int(array.size),
    )


def _bootstrap_auc_ci(labels: np.ndarray, probabilities: np.ndarray, *, seed: int, replicates: int = 128) -> tuple[float, float]:
    if probabilities.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    values: list[float] = []
    for _ in range(int(replicates)):
        indices = rng.integers(0, labels.size, size=labels.size)
        sampled = labels[indices]
        if np.unique(sampled).size != np.unique(labels).size:
            continue
        values.append(float(roc_auc_score(sampled, probabilities[indices], multi_class="ovr", average="macro")))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _chance_classifier_pass(metrics: dict[str, float | bool | str], prefix: str, config: V24Q1Config) -> bool:
    """Bound detectable classification, without misusing a conditional CI as a null test.

    A per-seed bootstrap interval estimates the held-out AUC conditional on one
    fitted probe. It need not contain 0.5 for every independently generated
    null seed. The panel-level mean check supplies the corresponding fixed-seed
    null-centering guard.
    """

    auc = float(metrics.get(f"{prefix}_macro_ovr_auc", float("nan")))
    return bool(np.isfinite(auc) and auc <= config.classifier_chance_ceiling)


def _ci_contains_chance(metrics: dict[str, float | bool | str], prefix: str) -> bool:
    low = float(metrics.get(f"{prefix}_auc_ci_low", float("nan")))
    high = float(metrics.get(f"{prefix}_auc_ci_high", float("nan")))
    return bool(np.isfinite(low) and np.isfinite(high) and low <= 0.5 <= high)


def audit_global_null_panel(
    seed_audits: dict[int, ContractAudit],
    config: V24Q1Config,
) -> dict[str, object]:
    """Apply the predeclared fixed-five-seed global-null centering guard."""

    required = tuple(int(seed) for seed in config.primary_seeds)
    observed = tuple(sorted(int(seed) for seed in seed_audits))
    complete = observed == tuple(sorted(required))
    support_values: list[float] = []
    marginal_values: list[float] = []
    individual_valid = complete
    for seed in required:
        audit = seed_audits.get(seed)
        if audit is None:
            individual_valid = False
            continue
        individual_valid = bool(individual_valid and audit.valid)
        try:
            support_values.append(float(audit.metrics["support_macro_ovr_auc"]))
            marginal_values.append(float(audit.metrics["marginal_macro_ovr_auc"]))
        except (KeyError, TypeError, ValueError):
            individual_valid = False
    support_mean = float(np.mean(support_values)) if len(support_values) == len(required) else float("nan")
    marginal_mean = float(np.mean(marginal_values)) if len(marginal_values) == len(required) else float("nan")
    support_centered = bool(np.isfinite(support_mean) and abs(support_mean - 0.5) <= config.null_panel_mean_auc_margin)
    marginal_centered = bool(np.isfinite(marginal_mean) and abs(marginal_mean - 0.5) <= config.null_panel_mean_auc_margin)
    return {
        "scope": "fixed_primary_seed_global_null_panel",
        "required_seeds": list(required),
        "observed_seeds": list(observed),
        "individual_contracts_valid": individual_valid,
        "support_macro_ovr_auc_mean": support_mean,
        "marginal_macro_ovr_auc_mean": marginal_mean,
        "mean_auc_target": 0.5,
        "mean_auc_margin": config.null_panel_mean_auc_margin,
        "support_mean_centered": support_centered,
        "marginal_mean_centered": marginal_centered,
        "valid": bool(individual_valid and support_centered and marginal_centered),
    }


def _support_cooccurrence_distance(matrix: np.ndarray, labels: np.ndarray, n_clusters: int) -> float:
    supports = matrix > 0.0
    cooccurrences: list[np.ndarray] = []
    for cluster in range(n_clusters):
        values = supports[labels == cluster].astype(np.float32)
        cooccurrences.append((values.T @ values) / max(1, values.shape[0]))
    distances = [
        np.linalg.norm(cooccurrences[left] - cooccurrences[right], ord="fro") / matrix.shape[1]
        for left in range(n_clusters)
        for right in range(left + 1, n_clusters)
    ]
    return float(max(distances, default=0.0))


def _marginal_differences(matrix: np.ndarray, labels: np.ndarray, n_clusters: int) -> dict[str, float]:
    zero_rate = []
    mean = []
    variance = []
    quantile_differences = []
    for feature in range(matrix.shape[1]):
        pooled_nonzero = matrix[:, feature][matrix[:, feature] > 0.0]
        if pooled_nonzero.size == 0:
            continue
        pooled_mean = float(np.mean(pooled_nonzero))
        pooled_std = max(float(np.std(pooled_nonzero)), 1e-8)
        pooled_var = max(float(np.var(pooled_nonzero)), 1e-8)
        class_quantiles: list[np.ndarray] = []
        for cluster in range(n_clusters):
            values = matrix[labels == cluster, feature]
            zero_rate.append(abs(float(np.mean(values == 0.0)) - float(np.mean(matrix[:, feature] == 0.0))))
            nonzero = values[values > 0.0]
            if nonzero.size:
                mean.append(abs(float(np.mean(nonzero)) - pooled_mean) / pooled_std)
                variance.append(abs(float(np.var(nonzero)) - pooled_var) / pooled_var)
                class_quantiles.append(np.quantile(nonzero, (0.25, 0.50, 0.75)))
        for left in range(len(class_quantiles)):
            for right in range(left + 1, len(class_quantiles)):
                quantile_differences.append(
                    float(np.max(np.abs(class_quantiles[left] - class_quantiles[right]) / pooled_std))
                )
    return {
        "zero_rate_max_abs_difference": float(max(zero_rate, default=0.0)),
        "nonzero_mean_max_standardized_difference": float(max(mean, default=0.0)),
        "nonzero_variance_max_relative_difference": float(max(variance, default=0.0)),
        "nonzero_quantile_max_standardized_difference": float(max(quantile_differences, default=0.0)),
    }


def _block_dependency_separation(matrix: np.ndarray, labels: np.ndarray, config: V24Q1Config) -> float:
    cluster_blocks: list[list[np.ndarray]] = []
    for cluster in range(config.n_clusters):
        class_matrix = matrix[labels == cluster]
        blocks: list[np.ndarray] = []
        for block in range(config.n_blocks):
            start = block * config.block_size
            block_values = class_matrix[:, start : start + config.block_size]
            active = np.any(block_values > 0.0, axis=1)
            values = block_values[active]
            if values.shape[0] < 3:
                blocks.append(np.zeros((config.block_size, config.block_size), dtype=np.float32))
            else:
                blocks.append(np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0).astype(np.float32))
        cluster_blocks.append(blocks)
    distances: list[float] = []
    for left in range(config.n_clusters):
        for right in range(left + 1, config.n_clusters):
            per_block = [
                np.linalg.norm(cluster_blocks[left][block] - cluster_blocks[right][block], ord="fro") / config.block_size
                for block in range(config.n_blocks)
            ]
            distances.append(float(np.mean(per_block)))
    return float(np.mean(distances)) if distances else 0.0


def audit_world(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    world: str,
    config: V24Q1Config,
    seed: int,
    run_classifiers: bool = True,
) -> ContractAudit:
    """Evaluate generator contracts before fit; this is outer-only label use."""

    config.validate()
    matrix = np.asarray(matrix, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if matrix.shape != (config.n_samples, config.n_features):
        raise ValueError("matrix shape differs from V24 configuration")
    if labels.size != matrix.shape[0]:
        raise ValueError("label count differs from matrix rows")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("synthetic matrix must be finite and nonnegative")
    metrics: dict[str, float | bool | str] = {
        "observed_zero_fraction": float(np.mean(matrix == 0.0)),
        "expected_zero_fraction": float(config.zero_fraction),
        "finite": bool(np.isfinite(matrix).all()),
    }
    marginal = _marginal_differences(matrix, labels, config.n_clusters)
    metrics.update(marginal)
    if run_classifiers:
        groups = _support_template_groups(matrix) if world in GROUPED_SUPPORT_WORLDS else None
        support_auc, support_probabilities = _macro_ovr_auc(
            (matrix > 0.0).astype(np.float32),
            labels,
            seed=seed,
            groups=groups,
        )
        # Probe one feature at a time. A raw-vector linear classifier can
        # combine non-Gaussian correlated coordinates and become a surrogate
        # dependency probe, which would invalidate the W4 marginal contract.
        marginal_auc, marginal_low, marginal_high, marginal_features = _featurewise_marginal_auc(
            matrix,
            labels,
            seed=seed + 1,
            groups=groups,
        )
        support_low, support_high = _bootstrap_auc_ci(labels, support_probabilities, seed=seed + 3)
        metrics.update(
            {
                "support_macro_ovr_auc": support_auc,
                "support_auc_ci_low": support_low,
                "support_auc_ci_high": support_high,
                "marginal_macro_ovr_auc": marginal_auc,
                "marginal_auc_ci_low": marginal_low,
                "marginal_auc_ci_high": marginal_high,
                "marginal_probe": "featurewise_scalar_linear_cv",
                "marginal_probe_feature_count": marginal_features,
                "classifier_cv": "stratified_group_by_support_template" if groups is not None else "stratified",
            }
        )
        metrics["support_classifier_chance_pass"] = _chance_classifier_pass(metrics, "support", config)
        metrics["marginal_classifier_chance_pass"] = _chance_classifier_pass(metrics, "marginal", config)
        metrics["support_auc_ci_contains_chance"] = _ci_contains_chance(metrics, "support")
        metrics["marginal_auc_ci_contains_chance"] = _ci_contains_chance(metrics, "marginal")
        metrics["classifier_chance_contract"] = "per_seed_auc_ceiling_with_panel_mean_centering"
    cooccurrence = _support_cooccurrence_distance(matrix, labels, config.n_clusters)
    metrics["support_cooccurrence_max_frobenius_distance"] = cooccurrence
    if world == "W4_dependency_only":
        dependency = _block_dependency_separation(matrix, labels, config)
        metrics["block_dependency_mean_frobenius_separation"] = dependency
        valid = bool(
            abs(float(metrics["observed_zero_fraction"]) - config.zero_fraction) <= 1e-7
            and marginal["zero_rate_max_abs_difference"] <= config.marginal_equality_tolerance
            and marginal["nonzero_mean_max_standardized_difference"] <= config.marginal_equality_tolerance
            and marginal["nonzero_variance_max_relative_difference"] <= config.marginal_equality_tolerance
            and marginal["nonzero_quantile_max_standardized_difference"] <= config.marginal_equality_tolerance
            and cooccurrence <= config.support_cooccurrence_max
            and dependency >= config.dependency_separation_min
            and (not run_classifiers or bool(metrics["support_classifier_chance_pass"]))
            and (not run_classifiers or bool(metrics["marginal_classifier_chance_pass"]))
        )
    elif world == "W0_global_null":
        valid = bool(
            abs(float(metrics["observed_zero_fraction"]) - config.zero_fraction) <= 1e-7
            and (not run_classifiers or bool(metrics["support_classifier_chance_pass"]))
            and (not run_classifiers or bool(metrics["marginal_classifier_chance_pass"]))
        )
    elif world == "W1_mean_only":
        valid = bool(
            abs(float(metrics["observed_zero_fraction"]) - config.zero_fraction) <= 1e-7
            and marginal["nonzero_mean_max_standardized_difference"] >= config.mean_shift_min
            and (not run_classifiers or bool(metrics["support_classifier_chance_pass"]))
        )
    elif world == "W2_support_only":
        valid = bool(
            abs(float(metrics["observed_zero_fraction"]) - config.zero_fraction) <= 1e-7
            and (not run_classifiers or float(metrics["support_macro_ovr_auc"]) >= config.support_signal_auc_floor)
        )
    elif world == "W3_marginal_only":
        dispersion_signal = max(
            marginal["nonzero_variance_max_relative_difference"],
            marginal["nonzero_quantile_max_standardized_difference"],
        )
        metrics["marginal_dispersion_signal"] = dispersion_signal
        valid = bool(
            abs(float(metrics["observed_zero_fraction"]) - config.zero_fraction) <= 1e-7
            and dispersion_signal >= config.marginal_dispersion_min
            and (not run_classifiers or bool(metrics["support_classifier_chance_pass"]))
        )
    else:
        valid = bool(abs(float(metrics["observed_zero_fraction"]) - config.zero_fraction) <= 1e-7)
    metrics["contract_scope"] = "generator_pre_fit_outer_only"
    return ContractAudit(world=world, valid=valid, metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a V24 corrected synthetic world before fitting")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with np.load(args.matrix, allow_pickle=False) as loaded:
        matrix = np.asarray(loaded["X"], dtype=np.float32)
    labels = np.load(args.labels, allow_pickle=False)
    audit = audit_world(matrix, labels, world=args.world, config=V24Q1Config(), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"world": audit.world, "valid": audit.valid, "metrics": audit.metrics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"world": audit.world, "valid": audit.valid}, ensure_ascii=False))


if __name__ == "__main__":
    main()
