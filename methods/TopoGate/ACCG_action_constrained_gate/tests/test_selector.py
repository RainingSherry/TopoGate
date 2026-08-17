from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from methods.TopoGate.ACCG_action_constrained_gate.config import FeatureConstraintConfig
from methods.TopoGate.ACCG_action_constrained_gate.feature_model import (
    CrossFittedFeatureModel,
    FeatureFoldModel,
    RobustTransform,
    fit_cross_fitted_feature_model,
)
from methods.TopoGate.ACCG_action_constrained_gate.selector import (
    exact_constrained_action,
    select_action,
    straight_through_mask,
)


def _paired_model() -> CrossFittedFeatureModel:
    prediction = sp.csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    operator = (sp.eye(2, format="csr") - prediction).tocsc()
    footprints = (np.asarray([0, 1], dtype=np.int64), np.asarray([0, 1], dtype=np.int64))
    fold = FeatureFoldModel(
        fold=0,
        neighbors=np.asarray([[1], [0]], dtype=np.int64),
        weights=np.ones((2, 1), dtype=np.float32),
        prediction=prediction,
        residual_operator_csc=operator,
        residual_scale=np.ones(2, dtype=np.float64),
        footprints=footprints,
        footprint_indptr=np.asarray([0, 2, 4], dtype=np.int64),
        footprint_indices=np.asarray([0, 1, 0, 1], dtype=np.int64),
        profile={},
    )
    transform = RobustTransform(
        center=np.zeros(2, dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
        clip=8.0,
        profile={},
    )
    return CrossFittedFeatureModel(
        transform=transform,
        fold_ids=np.asarray([0], dtype=np.int64),
        folds=(fold,),
        profile={},
    )


def test_w5_pair_lookahead_recovers_jointly_admissible_action() -> None:
    model = _paired_model()
    z = np.asarray([[-1.0, -1.0]])
    donor = np.asarray([[1.0, 1.0]])
    common = dict(
        hardness_scores=np.asarray([[2.0, 1.0]]),
        eligible=np.ones((1, 2), dtype=np.bool_),
        z=z,
        donor_z=donor,
        row_ids=np.asarray([0]),
        epsilon=0.1,
        model=model,
        mask_ratio=1.0,
        greedy_passes=1,
        pair_lookahead=4,
        fallback="least_violation",
    )
    joint = select_action(selector_mode="joint", **common)
    coordinate = select_action(selector_mode="coordinate", **common)
    assert joint.selected_counts.tolist() == [2]
    assert joint.constraint_infeasible.tolist() == [False]
    assert abs(float(joint.joint_delta[0])) < 1e-10
    assert coordinate.constraint_infeasible.tolist() == [True]
    assert coordinate.fallback_counts.tolist() == [2]


def test_exact_solver_matches_joint_pair_and_reports_feasibility() -> None:
    model = _paired_model()
    mask, profile = exact_constrained_action(
        np.asarray([2.0, 1.0]),
        np.asarray([True, True]),
        2,
        np.asarray([-1.0, -1.0]),
        np.asarray([1.0, 1.0]),
        epsilon=0.1,
        fold=model.folds[0],
    )
    assert mask.tolist() == [True, True]
    assert profile["feasible"] is True
    assert profile["combinations"] == 1


def test_straight_through_mask_preserves_hard_forward_and_gate_gradient() -> None:
    logits = torch.randn(2, 4, requires_grad=True)
    eligible = torch.ones_like(logits, dtype=torch.bool)
    gumbel = torch.zeros_like(logits)
    hard = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    budgets = torch.tensor([2, 2])
    mask = straight_through_mask(
        logits,
        eligible,
        gumbel,
        hard,
        budgets,
        gumbel_scale=1.0,
        tau=0.5,
    )
    assert torch.allclose(mask.detach(), hard, atol=1e-7, rtol=0.0)
    mask.square().mean().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_incremental_joint_selector_matches_full_post_action_recomputation() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(15, 7)).astype(np.float32)
    config = FeatureConstraintConfig(
        max_features=10,
        graph_k=3,
        graph_crossfit_folds=3,
        epsilon_rounds=4,
    )
    model = fit_cross_fitted_feature_model(X, config=config, seed=7)
    z = model.transform_matrix(X).astype(np.float64)
    donor = np.roll(z, 1, axis=0)
    eligible = donor != z
    scores = rng.normal(size=z.shape)
    selection = select_action(
        scores,
        eligible,
        z,
        donor,
        row_ids=np.arange(X.shape[0]),
        epsilon=np.full(X.shape[0], 1e9),
        model=model,
        mask_ratio=0.4,
        selector_mode="joint",
        greedy_passes=2,
        pair_lookahead=16,
        fallback="least_violation",
    )
    assert np.array_equal(selection.selected_counts, selection.budgets)
    for row in range(X.shape[0]):
        delta, clean, action = model.fold_for_row(row).action_delta(
            z[row], donor[row], selection.hard_mask[row].astype(np.bool_)
        )
        assert np.isclose(selection.joint_delta[row], delta, atol=1e-9)
        assert np.isclose(selection.clean_energy[row], clean, atol=1e-9)
        assert np.isclose(selection.action_energy[row], action, atol=1e-9)
