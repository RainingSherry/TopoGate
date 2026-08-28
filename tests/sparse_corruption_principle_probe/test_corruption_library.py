import numpy as np
import pytest

from scripts.sparse_corruption_principle_probe.corruption_library import (
    compact_audit,
    corrupt_matrix,
    geometry_importance,
    geometry_safe_fixture,
    residual_proxy,
    row_budgets,
    support_mask,
)


def _matrix():
    rng = np.random.default_rng(3)
    x = np.zeros((12, 20), dtype=np.float32)
    x[:, :8] = rng.uniform(0.8, 2.0, size=(12, 8)).astype(np.float32)
    return x


def test_all_primary_arms_use_exact_changed_coordinate_budget():
    x = _matrix()
    requested, _ = row_budgets(x)
    active = support_mask(x, reference=x)
    residual = residual_proxy(x)
    geometry = geometry_importance(x, k=5)
    for principle in (
        "P0_Random",
        "P1_SupportPreserve",
        "P2_SupportTarget",
        "P3_FrequencyAware",
        "P4_ResidualHard",
        "P5_GeometryHard",
    ):
        corrupted, audit = corrupt_matrix(
            x,
            principle,
            np.random.default_rng(42),
            residual_scores=residual,
            geometry_scores=geometry,
        )
        assert audit["exact_budget"] is True
        assert np.array_equal(audit["effective_changed_counts"], requested)
        assert np.isfinite(corrupted).all()
        assert compact_audit(audit)["labels_used"] is False
        if principle in {"P1_SupportPreserve", "P3_FrequencyAware", "P4_ResidualHard", "P5_GeometryHard"}:
            assert np.array_equal(support_mask(corrupted, reference=x), active)


def test_support_target_moves_values_without_changing_the_row_value_multiset():
    x = _matrix()
    corrupted, audit = corrupt_matrix(x, "P2_SupportTarget", np.random.default_rng(11))
    assert audit["exact_budget"] is True
    assert np.any(audit["support_changed_mask"])
    for before, after in zip(x, corrupted, strict=True):
        assert np.allclose(np.sort(before[before != 0]), np.sort(after[after != 0]))


def test_support_target_preserves_dense_proxy_row_values():
    """Threshold-inactive H0 entries may still be raw nonzero values."""

    x = np.array(
        [
            [1.0, 0.01, 0.02, 0.03, 0.04, 0.05],
            [1.0, 0.02, 0.03, 0.04, 0.05, 0.06],
        ],
        dtype=np.float32,
    )
    corrupted, audit = corrupt_matrix(x, "P2_SupportTarget", np.random.default_rng(4))
    assert audit["exact_budget"] is True
    for before, after in zip(x, corrupted, strict=True):
        assert np.allclose(np.sort(before[before != 0]), np.sort(after[after != 0]))


def test_geometry_safe_is_fixture_only_but_keeps_budget_and_support():
    x = _matrix()
    geometry = geometry_importance(x, k=5)
    corrupted, audit = geometry_safe_fixture(x, np.random.default_rng(19), geometry_scores=geometry)
    assert audit["principle"] == "P5_GeometrySafe"
    assert audit["exact_budget"] is True
    assert np.array_equal(support_mask(corrupted, reference=x), support_mask(x, reference=x))


def test_residual_hard_requires_explicit_scores():
    with pytest.raises(ValueError, match="requires frozen residual_scores"):
        corrupt_matrix(_matrix(), "P4_ResidualHard", np.random.default_rng(1))


def test_explicit_reference_support_is_validated_and_reused():
    x = _matrix()
    active = support_mask(x, reference=x)
    requested, _ = row_budgets(x, reference_support=active)
    assert requested.shape == (x.shape[0],)
    with pytest.raises(ValueError, match="reference_support"):
        row_budgets(x, reference_support=active[:, :-1])


@pytest.mark.parametrize("n_rows", [2, 3, 4, 5])
def test_geometry_importance_handles_small_sample_counts(n_rows):
    """The kNN contract must remain valid when k reaches n-1."""

    x = np.zeros((n_rows, 6), dtype=np.float32)
    x[:, :3] = np.arange(1, 4, dtype=np.float32)
    scores = geometry_importance(x, k=15)
    assert scores.shape == x.shape
    assert np.isfinite(scores).all()
