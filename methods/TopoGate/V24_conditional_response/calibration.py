from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import multiprocessing

import numpy as np

from .config import V24Q1Config
from .evaluation import conditional_pair_utility


@dataclass(frozen=True)
class CalibrationResult:
    null_deltas: np.ndarray
    alternative_deltas: np.ndarray
    summary: dict[str, float | bool | int]


def _simulate_representations(config: V24Q1Config, *, seed: int, alternative: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed estimator calibration with matching N, K, T and nuisance dimensions.

    The raw feature dimension/sparsity enter through the support distribution;
    this is an estimator calibration, not a substitute for a corrected world.
    """

    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(config.n_clusters, dtype=np.int64), config.n_samples // config.n_clusters)
    rng.shuffle(labels)
    state = rng.normal(size=(config.n_samples, 128)).astype(np.float32)
    support = rng.binomial(
        n=max(1, int(round(config.n_features * config.fingerprint_mask_ratio))),
        p=0.187,
        size=(config.n_samples, config.fingerprint_masks),
    ).astype(np.float32)
    marginal = rng.normal(size=(config.n_samples, config.fingerprint_masks, 9)).astype(np.float32)
    response = np.empty((config.n_samples, config.fingerprint_masks), dtype=np.float32)
    coefficient = rng.normal(scale=0.05, size=(128, config.fingerprint_masks)).astype(np.float32)
    for intervention in range(config.fingerprint_masks):
        response[:, intervention] = (
            state @ coefficient[:, intervention]
            + 0.02 * support[:, intervention]
            + 0.03 * marginal[:, intervention, 0]
            + rng.normal(scale=1.0, size=config.n_samples)
        )
    if alternative:
        # Fixed weak residual class signal, selected before any data are observed.
        class_code = (labels - labels.mean()) / max(1.0, labels.std())
        response += (0.10 * class_code[:, None] * rng.normal(size=(1, config.fingerprint_masks))).astype(np.float32)
    return state, support, marginal, response, labels


def _calibrate_replicate(task: tuple[V24Q1Config, int]) -> tuple[float, float]:
    config, replicate = task
    values: list[float] = []
    for alternative in (False, True):
        state, support, marginal, response, labels = _simulate_representations(
            config,
            seed=17_003 + replicate * 23 + int(alternative),
            alternative=alternative,
        )
        result = conditional_pair_utility(
            state,
            support,
            marginal,
            response,
            labels=labels,
            outer_folds=config.outer_folds,
            inner_folds=config.inner_folds,
            seed=31_007 + replicate,
            alpha=config.ridge_alpha,
            pair_count_per_fold=min(400, config.pair_count_per_fold),
        )
        values.append(result.delta_auc)
    return float(values[0]), float(values[1])


def calibrate_estimator(
    config: V24Q1Config,
    *,
    replicates: int | None = None,
    workers: int = 1,
) -> CalibrationResult:
    """Run the preregistered null/weak-alternative estimator calibration."""

    config.validate()
    count = int(config.calibration_replicates if replicates is None else replicates)
    worker_count = int(workers)
    if count <= 0 or worker_count <= 0:
        raise ValueError("replicates and workers must be positive")
    tasks = [(config, replicate) for replicate in range(count)]
    if worker_count == 1:
        paired_values = [_calibrate_replicate(task) for task in tasks]
    else:
        # Avoid forking a process that may already host numerical-library
        # threads; every task is deterministic and picklable under spawn.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            paired_values = list(executor.map(_calibrate_replicate, tasks))
    null_values = [pair[0] for pair in paired_values]
    alternative_values = [pair[1] for pair in paired_values]
    null_array = np.asarray(null_values, dtype=np.float32)
    alternative_array = np.asarray(alternative_values, dtype=np.float32)
    false_positive = float(np.mean(null_array > config.equivalence_margin))
    power = float(np.mean(alternative_array >= config.w4_delta_min))
    null_mean = float(np.mean(null_array))
    null_centered = bool(abs(null_mean) <= config.null_point_margin)
    return CalibrationResult(
        null_deltas=null_array,
        alternative_deltas=alternative_array,
        summary={
            "replicates": count,
            "workers": worker_count,
            "matching_n_samples": config.n_samples,
            "matching_n_features": config.n_features,
            "matching_n_clusters": config.n_clusters,
            "matching_zero_fraction": config.zero_fraction,
            "matching_fingerprint_masks": config.fingerprint_masks,
            "matching_marginal_dimension": 9,
            "null_false_positive_rate": false_positive,
            "alternative_power_at_w4_threshold": power,
            "null_mean_delta_auc": null_mean,
            "null_std_delta_auc": float(np.std(null_array, ddof=1)) if null_array.size > 1 else 0.0,
            "null_q025_delta_auc": float(np.quantile(null_array, 0.025)),
            "null_q975_delta_auc": float(np.quantile(null_array, 0.975)),
            "null_centered_for_null_world_gate": null_centered,
            "alternative_mean_delta_auc": float(np.mean(alternative_array)),
            "calibration_passes": bool(
                false_positive <= 0.05
                and power >= 0.80
                and null_centered
            ),
        },
    )
