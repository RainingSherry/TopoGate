from __future__ import annotations

from dataclasses import dataclass
import concurrent.futures
import multiprocessing

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ResidualResult:
    residual: np.ndarray
    prediction: np.ndarray
    r2_by_intervention: np.ndarray
    fold_indices: tuple[np.ndarray, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ConditionalUtilityResult:
    base_auc: float
    plus_auc: float
    delta_auc: float
    fold_deltas: np.ndarray
    fold_indices: tuple[np.ndarray, ...]
    residual_diagnostics: dict[str, object]
    records: dict[str, np.ndarray]


_BOOTSTRAP_CONTEXT: tuple[object, ...] | None = None


def _evaluate_bootstrap_process(task: tuple[int, np.ndarray]) -> float | None:
    """Evaluate one bootstrap replicate in a forked worker with shared inputs."""

    if _BOOTSTRAP_CONTEXT is None:
        raise RuntimeError("bootstrap process context is not initialized")
    (
        state,
        support,
        marginal,
        response,
        labels,
        outer_folds,
        inner_folds,
        seed,
        alpha,
        pair_count_per_fold,
    ) = _BOOTSTRAP_CONTEXT
    replicate, weights = task
    labels = np.asarray(labels, dtype=np.int64)
    positive_by_class = [np.count_nonzero(weights[labels == value] > 0.0) for value in np.unique(labels)]
    if min(positive_by_class, default=0) < int(outer_folds):
        return None
    try:
        result = conditional_pair_utility(
            state,
            support,
            marginal,
            response,
            labels=labels,
            outer_folds=int(outer_folds),
            inner_folds=int(inner_folds),
            seed=int(seed) + int(replicate) + 1,
            alpha=float(alpha),
            pair_count_per_fold=int(pair_count_per_fold),
            sample_weights=weights,
        )
    except ValueError:
        return None
    return float(result.delta_auc)


def make_sample_folds(n_samples: int, *, n_splits: int, seed: int) -> tuple[np.ndarray, ...]:
    if n_samples < n_splits:
        raise ValueError("n_samples must be at least n_splits")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return tuple(np.asarray(test, dtype=np.int64) for _, test in splitter.split(np.arange(n_samples)))


def _validate_inputs(state: np.ndarray, support: np.ndarray, marginal: np.ndarray, response: np.ndarray) -> None:
    state = np.asarray(state)
    support = np.asarray(support)
    marginal = np.asarray(marginal)
    response = np.asarray(response)
    if state.ndim != 2 or support.ndim != 2 or marginal.ndim != 3 or response.ndim != 2:
        raise ValueError("state/support/marginal/response have invalid dimensions")
    if state.shape[0] != support.shape[0] or state.shape[0] != marginal.shape[0] or state.shape[0] != response.shape[0]:
        raise ValueError("representation sample dimensions differ")
    if support.shape[1] != response.shape[1] or marginal.shape[:2] != response.shape:
        raise ValueError("intervention dimensions differ")
    if not all(np.isfinite(value).all() for value in (state, support, marginal, response)):
        raise ValueError("representations must be finite")


def _design_for_intervention(state: np.ndarray, support: np.ndarray, marginal: np.ndarray, intervention: int) -> np.ndarray:
    return np.concatenate((state, support[:, intervention : intervention + 1], marginal[:, intervention, :]), axis=1).astype(
        np.float32,
        copy=False,
    )


def _fit_predict_intervention(
    design: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    alpha: float,
    train_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    train_variance = np.var(design[train], axis=0)
    valid = train_variance > 1e-12
    if not np.any(valid):
        if train_weights is None:
            baseline = float(np.mean(target[train]))
        else:
            weights = np.asarray(train_weights, dtype=np.float64).reshape(-1)
            if weights.shape != (train.size,) or np.any(weights < 0.0) or not np.any(weights > 0.0):
                raise ValueError("train_weights must be nonnegative with one positive entry")
            baseline = float(np.average(target[train], weights=weights))
        return np.full(test.size, baseline, dtype=np.float32), int(design.shape[1])
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    fit_kwargs: dict[str, np.ndarray] = {}
    if train_weights is not None:
        weights = np.asarray(train_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (train.size,) or np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("train_weights must be nonnegative with one positive entry")
        # Keep feature scaling on unique rows while weighting the Ridge
        # objective. This realizes a sample bootstrap without materializing
        # duplicated rows that could cross an outer train/test boundary.
        fit_kwargs["ridge__sample_weight"] = weights
    model.fit(design[train][:, valid], target[train], **fit_kwargs)
    return np.asarray(model.predict(design[test][:, valid]), dtype=np.float32), int(np.count_nonzero(~valid))


def crossfit_residual_response(
    state: np.ndarray,
    support: np.ndarray,
    marginal: np.ndarray,
    response: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    alpha: float,
    sample_weights: np.ndarray | None = None,
) -> ResidualResult:
    """Cross-fit residual C after observed State, Support, and Marginal controls.

    This function intentionally has no labels or K input. Folds are generated
    from sample indices only.
    """

    _validate_inputs(state, support, marginal, response)
    n_samples, n_interventions = response.shape
    weights = np.ones(n_samples, dtype=np.float64) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if weights.shape != (n_samples,) or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("sample_weights must be nonnegative with one positive sample")
    folds = make_sample_folds(n_samples, n_splits=n_splits, seed=seed)
    prediction = np.zeros_like(response, dtype=np.float32)
    dropped_columns: list[int] = []
    all_rows = np.arange(n_samples, dtype=np.int64)
    for held_out in folds:
        train_mask = np.ones(n_samples, dtype=np.bool_)
        train_mask[held_out] = False
        train = all_rows[train_mask]
        for intervention in range(n_interventions):
            current, dropped = _fit_predict_intervention(
                _design_for_intervention(state, support, marginal, intervention),
                response[:, intervention],
                train,
                held_out,
                alpha=alpha,
                train_weights=weights[train],
            )
            prediction[held_out, intervention] = current
            dropped_columns.append(dropped)
    residual = np.asarray(response - prediction, dtype=np.float32)
    r2 = np.asarray(
        [r2_score(response[:, intervention], prediction[:, intervention]) for intervention in range(n_interventions)],
        dtype=np.float32,
    )
    return ResidualResult(
        residual=residual,
        prediction=prediction,
        r2_by_intervention=r2,
        fold_indices=folds,
        diagnostics={
            "residualizer": "cross_fitted_ridge",
            "ridge_alpha": float(alpha),
            "state_dimension": int(state.shape[1]),
            "support_dimension_per_intervention": 1,
            "marginal_dimension_per_intervention": int(marginal.shape[2]),
            "design_dimension_per_intervention": int(state.shape[1] + 1 + marginal.shape[2]),
            "mean_cross_fitted_r2": float(np.mean(r2)),
            "min_cross_fitted_r2": float(np.min(r2)),
            "max_cross_fitted_r2": float(np.max(r2)),
            "mean_dropped_zero_variance_columns": float(np.mean(dropped_columns)) if dropped_columns else 0.0,
            "labels_accessible": False,
            "K_accessible": False,
            "sample_weights_used": bool(sample_weights is not None),
        },
    )


def _outer_test_residual(
    state: np.ndarray,
    support: np.ndarray,
    marginal: np.ndarray,
    response: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    alpha: float,
    train_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    output = np.zeros((test.size, response.shape[1]), dtype=np.float32)
    dropped: list[int] = []
    for intervention in range(response.shape[1]):
        current, removed = _fit_predict_intervention(
            _design_for_intervention(state, support, marginal, intervention),
            response[:, intervention],
            train,
                test,
                alpha=alpha,
                train_weights=train_weights,
        )
        output[:, intervention] = response[test, intervention] - current
        dropped.append(removed)
    return output, {"mean_dropped_zero_variance_columns": float(np.mean(dropped)) if dropped else 0.0}


def _balanced_pairs(
    labels: np.ndarray,
    rows: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rows = np.asarray(rows, dtype=np.int64)
    weights = np.ones(labels.size, dtype=np.float64) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if weights.shape != labels.shape or np.any(weights < 0.0):
        raise ValueError("sample_weights must align with labels and be nonnegative")
    classes = np.unique(labels[rows])
    by_class = {int(value): rows[labels[rows] == value] for value in classes}
    positive_classes = [value for value, values in by_class.items() if np.count_nonzero(weights[values] > 0.0) >= 2]
    if len(positive_classes) < 2:
        raise ValueError("outer fold cannot form balanced same/different pairs")

    def choose(values: np.ndarray, *, size: int) -> np.ndarray:
        positive = values[weights[values] > 0.0]
        if positive.size < size:
            raise ValueError("bootstrap weights removed too many class members for a pair")
        probabilities = weights[positive]
        probabilities = probabilities / probabilities.sum()
        return np.asarray(rng.choice(positive, size=size, replace=False, p=probabilities), dtype=np.int64)

    first: list[int] = []
    second: list[int] = []
    targets: list[int] = []
    for _ in range(int(count)):
        label = int(rng.choice(positive_classes))
        pair = choose(by_class[label], size=2)
        first.append(int(pair[0]))
        second.append(int(pair[1]))
        targets.append(1)
    for _ in range(int(count)):
        left, right = rng.choice(positive_classes, size=2, replace=False)
        first.append(int(choose(by_class[int(left)], size=1)[0]))
        second.append(int(choose(by_class[int(right)], size=1)[0]))
        targets.append(0)
    return np.column_stack((first, second)).astype(np.int64), np.asarray(targets, dtype=np.int64)


def _pair_features(representation: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    left = representation[pairs[:, 0]]
    right = representation[pairs[:, 1]]
    difference = np.abs(left - right)
    dot = np.sum(left * right, axis=1, keepdims=True)
    norm = np.linalg.norm(left, axis=1, keepdims=True) * np.linalg.norm(right, axis=1, keepdims=True)
    cosine = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 1e-12)
    return np.concatenate((difference, cosine), axis=1).astype(np.float32, copy=False)


def _fit_pair_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=0),
    )
    fit_kwargs: dict[str, np.ndarray] = {}
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (train_targets.size,) or np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("pair sample weights must be nonnegative and align with pair targets")
        fit_kwargs["logisticregression__sample_weight"] = weights
    model.fit(train_features, train_targets, **fit_kwargs)
    return np.asarray(model.predict_proba(test_features)[:, 1], dtype=np.float32)


def conditional_pair_utility(
    state: np.ndarray,
    support: np.ndarray,
    marginal: np.ndarray,
    response: np.ndarray,
    *,
    labels: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    alpha: float,
    pair_count_per_fold: int,
    sample_weights: np.ndarray | None = None,
) -> ConditionalUtilityResult:
    """Outer-only conditional same-cluster utility with sample-disjoint pairs."""

    _validate_inputs(state, support, marginal, response)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size != state.shape[0]:
        raise ValueError("outer labels must match representation rows")
    weights = np.ones(labels.size, dtype=np.float64) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if weights.shape != labels.shape or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("sample_weights must be nonnegative with one positive sample")
    folds = make_sample_folds(labels.size, n_splits=outer_folds, seed=seed)
    all_rows = np.arange(labels.size, dtype=np.int64)
    base_representation = np.concatenate((state, support, marginal.reshape(labels.size, -1)), axis=1).astype(np.float32)
    all_targets: list[np.ndarray] = []
    base_scores: list[np.ndarray] = []
    plus_scores: list[np.ndarray] = []
    fold_deltas: list[float] = []
    record_pairs: list[np.ndarray] = []
    record_folds: list[np.ndarray] = []
    residual_records: list[dict[str, object]] = []

    for fold_number, test in enumerate(folds):
        train_mask = np.ones(labels.size, dtype=np.bool_)
        train_mask[test] = False
        train = all_rows[train_mask]
        # Inner OOF residuals prevent the pair learner from receiving in-sample Ridge residuals.
        train_residual_result = crossfit_residual_response(
            state[train],
            support[train],
            marginal[train],
            response[train],
            n_splits=inner_folds,
            seed=seed + 1_003 + fold_number,
            alpha=alpha,
            sample_weights=weights[train],
        )
        train_residual = train_residual_result.residual
        test_residual, test_residual_diagnostics = _outer_test_residual(
            state,
            support,
            marginal,
            response,
            train,
            test,
            alpha=alpha,
            train_weights=weights[train],
        )
        residual_records.append(
            {
                "inner_mean_cross_fitted_r2": float(
                    train_residual_result.diagnostics["mean_cross_fitted_r2"]
                ),
                **test_residual_diagnostics,
            }
        )
        train_residual_full = np.zeros_like(response, dtype=np.float32)
        test_residual_full = np.zeros_like(response, dtype=np.float32)
        train_residual_full[train] = train_residual
        test_residual_full[test] = test_residual

        rng = np.random.default_rng(seed + 10_000 + fold_number)
        train_pairs, train_targets = _balanced_pairs(
            labels,
            train,
            count=pair_count_per_fold,
            rng=rng,
            sample_weights=weights,
        )
        test_pairs, test_targets = _balanced_pairs(
            labels,
            test,
            count=pair_count_per_fold,
            rng=rng,
            sample_weights=weights,
        )
        train_base = _pair_features(base_representation, train_pairs)
        test_base = _pair_features(base_representation, test_pairs)
        train_plus = np.concatenate((train_base, _pair_features(train_residual_full, train_pairs)), axis=1)
        test_plus = np.concatenate((test_base, _pair_features(test_residual_full, test_pairs)), axis=1)
        # Pair sampling already draws endpoints in proportion to their Poisson
        # masses. Applying the same weights again inside LogisticRegression
        # would square the bootstrap mass of a duplicated sample.
        current_base = _fit_pair_model(train_base, train_targets, test_base)
        current_plus = _fit_pair_model(train_plus, train_targets, test_plus)
        base_auc = float(roc_auc_score(test_targets, current_base))
        plus_auc = float(roc_auc_score(test_targets, current_plus))
        fold_deltas.append(plus_auc - base_auc)
        all_targets.append(test_targets)
        base_scores.append(current_base)
        plus_scores.append(current_plus)
        record_pairs.append(test_pairs)
        record_folds.append(np.full(test_targets.shape, fold_number, dtype=np.int64))

    joined_targets = np.concatenate(all_targets)
    joined_base = np.concatenate(base_scores)
    joined_plus = np.concatenate(plus_scores)
    return ConditionalUtilityResult(
        base_auc=float(roc_auc_score(joined_targets, joined_base)),
        plus_auc=float(roc_auc_score(joined_targets, joined_plus)),
        delta_auc=float(roc_auc_score(joined_targets, joined_plus) - roc_auc_score(joined_targets, joined_base)),
        fold_deltas=np.asarray(fold_deltas, dtype=np.float32),
        fold_indices=folds,
        residual_diagnostics={
            "outer_folds": int(outer_folds),
            "inner_folds": int(inner_folds),
            "sample_disjoint_pairs": True,
            "pair_observations_treated_as_iid": False,
            "outer_fold_residuals": residual_records,
            "sample_weights_used": bool(sample_weights is not None),
            "outer_train_test_original_sample_disjoint": True,
        },
        records={
            "targets": joined_targets,
            "base_scores": joined_base,
            "plus_scores": joined_plus,
            "pairs": np.concatenate(record_pairs),
            "pair_fold": np.concatenate(record_folds),
        },
    )


def bootstrap_conditional_delta(
    state: np.ndarray,
    support: np.ndarray,
    marginal: np.ndarray,
    response: np.ndarray,
    *,
    labels: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    alpha: float,
    pair_count_per_fold: int,
    replicates: int,
    workers: int = 1,
) -> np.ndarray:
    """Poisson-weighted full-pipeline bootstrap with fixed disjoint outer folds.

    Rows are never physically duplicated. Each replicate changes only the
    weight of an original sample, while outer KFold membership remains fixed.
    Consequently an original sample cannot appear in both the train and test
    side of an outer pair model merely because it was resampled.
    """

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("bootstrap workers must be positive")
    rng = np.random.default_rng(seed + 71_003)
    weights_by_replicate = [
        rng.poisson(1.0, size=labels.size).astype(np.float64)
        for _ in range(int(replicates))
    ]

    tasks = list(enumerate(weights_by_replicate))
    if worker_count == 1:
        global _BOOTSTRAP_CONTEXT
        _BOOTSTRAP_CONTEXT = (
            state,
            support,
            marginal,
            response,
            labels,
            outer_folds,
            inner_folds,
            seed,
            alpha,
            pair_count_per_fold,
        )
        values = [value for task in tasks if (value := _evaluate_bootstrap_process(task)) is not None]
    else:
        _BOOTSTRAP_CONTEXT = (
            state,
            support,
            marginal,
            response,
            labels,
            outer_folds,
            inner_folds,
            seed,
            alpha,
            pair_count_per_fold,
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            results = executor.map(_evaluate_bootstrap_process, tasks)
            values = [value for value in results if value is not None]
        _BOOTSTRAP_CONTEXT = None
    return np.asarray(values, dtype=np.float32)
