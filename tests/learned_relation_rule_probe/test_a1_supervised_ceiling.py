import numpy as np
import pytest
from pathlib import Path

from scripts.learned_relation_rule_probe.a1_supervised_ceiling import (
    FULL_VIEW,
    NO_GEOMETRY_VIEW,
    NO_RANK_VIEW,
    _grouped_oof_scores,
    _select_graph_from_scores,
    _view_columns,
    engineering_preflight,
)


def test_a1_views_preserve_expected_geometry_contract():
    names = (
        "cosine", "cosine_rank", "cosine_percentile", "distance",
        "mutual", "stability_recurrence",
    )
    assert _view_columns(names, FULL_VIEW).tolist() == list(range(len(names)))
    assert set(names[i] for i in _view_columns(names, NO_GEOMETRY_VIEW)) == {
        "mutual", "stability_recurrence"
    }
    assert "cosine_rank" not in set(names[i] for i in _view_columns(names, NO_RANK_VIEW))
    assert "cosine" in set(names[i] for i in _view_columns(names, NO_RANK_VIEW))


def test_a1_grouped_oof_is_complete_and_anchor_disjoint():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(12), 4)
    x = rng.normal(size=(groups.size, 3)).astype(np.float32)
    target = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(np.int64)
    scores, folds, audit = _grouped_oof_scores(x, target, groups, "Logistic", seed=42)
    assert scores.shape == target.shape
    assert np.isfinite(scores).all()
    assert len(folds) == 5
    assert audit["oof_coverage_100pct"] is True
    assert audit["anchor_disjoint"] is True


def test_a1_selection_preserves_row_budget():
    rows = np.repeat(np.arange(3), 3)
    cols = np.array([1, 2, 0, 0, 2, 1, 0, 1, 2])
    cosine = np.linspace(0.9, 0.1, rows.size).astype(np.float32)
    budget = np.array([2, 1, 0], dtype=np.int64)
    scores = np.arange(rows.size, dtype=np.float64)
    graph, selected = _select_graph_from_scores(rows, cols, cosine, budget, scores, 3)
    assert selected.sum() == 3
    assert graph.shape == (3, 3)


@pytest.mark.skipif(
    not (Path("result/relation_selection_probe/RS1_information/features/cnae9/edge_features.npz").exists()),
    reason="private audited RS1 artifacts are intentionally excluded from the public release",
)
def test_a1_engineering_preflight_passes_on_frozen_primary_panel():
    result = engineering_preflight()
    assert result["status"] == "completed_valid"
    assert all(row["groupkfold_anchor_disjoint"] for row in result["datasets"])
    assert all(row["budget_capacity_sufficient"] for row in result["datasets"])
