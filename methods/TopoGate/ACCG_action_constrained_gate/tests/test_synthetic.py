from __future__ import annotations

import numpy as np

from methods.TopoGate.ACCG_action_constrained_gate.synthetic import SyntheticConfig, generate_worlds
from methods.TopoGate.ACCG_action_constrained_gate.synthetic_audit import audit_shortcuts, oracle_action_metrics


def _config() -> SyntheticConfig:
    return SyntheticConfig(n_samples=120, n_features=80, module_size=5)


def test_worlds_share_support_and_feature_marginals() -> None:
    config = _config()
    worlds = generate_worlds(config, family="lognormal_sparse", seed=42)
    matrices = [world.X for world in worlds.values()]
    reference = matrices[0]
    for matrix in matrices[1:]:
        assert np.array_equal(matrix != 0.0, reference != 0.0)
        assert np.allclose(np.sort(matrix, axis=0), np.sort(reference, axis=0))


def test_w4_reuses_observations_with_two_task_partitions() -> None:
    worlds = generate_worlds(_config(), family="count_sparse", seed=7)
    w0 = worlds["W0_matched_null"]
    w4 = worlds["W4_observational_alias"]
    assert np.array_equal(w0.X, w4.X)
    assert w4.alternative_labels is not None
    assert not np.array_equal(w4.labels, w4.alternative_labels)


def test_shortcut_audit_and_oracle_metrics_have_explicit_boundaries() -> None:
    config = _config()
    worlds = generate_worlds(config, family="lognormal_sparse", seed=42)
    audit = audit_shortcuts(worlds, config=config, seed=42)
    assert audit["support_exactly_matched"] is True
    assert audit["max_column_summary_gap"] < 1e-5
    w1 = worlds["W1_isolated_corruption"]
    metrics = oracle_action_metrics(w1.repair_mask, w1)
    assert metrics["noise_intervention_precision"] == 1.0
    assert metrics["noise_intervention_recall"] == 1.0


def test_oracle_masks_only_mark_observed_coordinates() -> None:
    worlds = generate_worlds(_config(), family="lognormal_sparse", seed=42)
    support = worlds["W0_matched_null"].X != 0.0
    assert np.all(~worlds["W1_isolated_corruption"].repair_mask | support)
    assert np.all(~worlds["W2_rare_coherent_signal"].protect_mask | support)
    assert np.all(~worlds["W3_coherent_nuisance"].nuisance_mask | support)


def test_w5_declares_a_sparse_coherent_interaction_pair() -> None:
    world = generate_worlds(_config(), family="lognormal_sparse", seed=7)["W5_joint_interaction"]
    pair = np.asarray(world.metadata["interaction_pair"], dtype=np.int64)
    assert pair.shape == (2,)
    assert np.array_equal(world.X[:, pair[0]] != 0.0, world.X[:, pair[1]] != 0.0)
    active = np.flatnonzero(world.X[:, pair[0]] != 0.0)
    first_rank = np.argsort(np.argsort(world.X[active, pair[0]], kind="mergesort"), kind="mergesort")
    second_rank = np.argsort(np.argsort(world.X[active, pair[1]], kind="mergesort"), kind="mergesort")
    assert float(np.corrcoef(first_rank, second_rank)[0, 1]) > 0.9
