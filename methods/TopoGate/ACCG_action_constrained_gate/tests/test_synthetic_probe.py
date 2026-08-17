from __future__ import annotations

import numpy as np

from methods.TopoGate.ACCG_action_constrained_gate.synthetic import SyntheticConfig, generate_worlds
from methods.TopoGate.ACCG_action_constrained_gate.synthetic_probe import (
    _candidate_mask,
    _cv_splits,
    _oracle_target,
    evaluate_incremental_information,
)


def _config() -> SyntheticConfig:
    return SyntheticConfig(n_samples=120, n_features=80, module_size=5)


def test_w3_and_w5_probe_actions_are_exact_budget_and_oracle_separated() -> None:
    config = _config()
    worlds = generate_worlds(config, family="lognormal_sparse", seed=42)
    eligible = np.ones(config.n_features, dtype=np.bool_)
    rng = np.random.default_rng(9)

    w3 = worlds["W3_coherent_nuisance"]
    nuisance_rows = np.flatnonzero(w3.nuisance_mask.any(axis=1))
    row = int(nuisance_rows[0])
    w3_positive = _candidate_mask(w3, row, eligible, 8, 0, rng, config)
    w3_negative = _candidate_mask(w3, row, eligible, 8, 1, rng, config)
    assert int(w3_positive.sum()) == int(w3_negative.sum()) == 8
    assert _oracle_target(w3, row, 1, w3_positive, config) is True
    assert _oracle_target(w3, row, 1, w3_negative, config) is False

    w5 = worlds["W5_joint_interaction"]
    w5_positive = _candidate_mask(w5, 0, eligible, 8, 0, rng, config)
    w5_negative = _candidate_mask(w5, 0, eligible, 8, 1, rng, config)
    assert int(w5_positive.sum()) == int(w5_negative.sum()) == 8
    assert _oracle_target(w5, 0, 1, w5_positive, config) is True
    assert _oracle_target(w5, 0, 1, w5_negative, config) is False


def test_incremental_information_uses_group_disjoint_matched_folds() -> None:
    rng = np.random.default_rng(17)
    groups = np.repeat(np.arange(60), 4)
    target = np.tile(np.asarray([0, 1, 0, 1], dtype=np.int64), 60)
    baseline = rng.normal(size=(target.size, 3))
    joint = target.astype(np.float64) + rng.normal(scale=0.05, size=target.size)
    splits = _cv_splits(baseline, target, seed=42, groups=groups)
    assert splits
    for train, test in splits:
        assert set(groups[train]).isdisjoint(set(groups[test]))
    result = evaluate_incremental_information(
        baseline,
        joint,
        target,
        seed=42,
        bootstrap_replicates=100,
        groups=groups,
    )
    assert result["valid"] is True
    assert float(result["auc_joint"]) > 0.95
    assert float(result["delta_auc_ci_low"]) > 0.0

