from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from methods.TopoGate.ACCG_action_constrained_gate.calibration import calibrate_epsilon
from methods.TopoGate.ACCG_action_constrained_gate.config import FeatureConstraintConfig, load_config
from methods.TopoGate.ACCG_action_constrained_gate.feature_model import fit_cross_fitted_feature_model
from methods.TopoGate.ACCG_action_constrained_gate.protocol import run_matched_panel


ROOT = Path(__file__).resolve().parents[4]


def _config(**overrides: object) -> FeatureConstraintConfig:
    values = {
        "max_features": 20,
        "graph_k": 2,
        "graph_crossfit_folds": 3,
        "epsilon_rounds": 4,
    }
    values.update(overrides)
    return FeatureConstraintConfig(**values)


def test_all_frozen_configs_validate() -> None:
    root = ROOT / "methods/TopoGate/ACCG_action_constrained_gate/configs"
    for name in (
        "accg_joint.yaml",
        "accg_coordinate.yaml",
        "accg_shuffled_graph.yaml",
        "accg_joint_abstain.yaml",
        "accg_marginal_only.yaml",
    ):
        load_config(root / name).validate()


def test_cross_fitted_graph_is_nonnegative_normalized_and_self_free() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(18, 8)).astype(np.float32)
    model = fit_cross_fitted_feature_model(X, config=_config(), seed=7)
    assert model.fold_ids.shape == (18,)
    assert len(model.folds) == 3
    for fold in model.folds:
        assert np.count_nonzero(fold.prediction.diagonal()) == 0
        assert np.all(fold.prediction.data >= 0.0)
        assert np.allclose(np.asarray(fold.prediction.sum(axis=1)).reshape(-1), 1.0)
        heldout = set(np.flatnonzero(model.fold_ids == fold.fold).tolist())
        train = set(np.flatnonzero(model.fold_ids != fold.fold).tolist())
        assert heldout
        assert heldout.isdisjoint(train)


def test_null_epsilon_calibration_is_deterministic_and_label_free() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 6)).astype(np.float32)
    config = _config(graph_crossfit_folds=2)
    model = fit_cross_fitted_feature_model(X, config=config, seed=42)
    first = calibrate_epsilon(X, model, mask_ratio=0.4, config=config, seed=42)
    second = calibrate_epsilon(X, model, mask_ratio=0.4, config=config, seed=42)
    assert np.array_equal(first.epsilon, second.epsilon)
    assert np.array_equal(first.sampled_deltas, second.sampled_deltas)
    assert first.profile["labels_used"] is False
    assert first.profile["outcomes_used"] is False


def test_fit_entrypoint_cannot_receive_labels() -> None:
    assert "y" not in inspect.signature(run_matched_panel).parameters
    assert "labels" not in inspect.signature(fit_cross_fitted_feature_model).parameters
