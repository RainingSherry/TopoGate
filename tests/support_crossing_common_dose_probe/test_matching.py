import numpy as np

from scripts.support_crossing_common_dose_probe.matching import (
    audit_swap,
    build_common_dose_row,
    row_constructive_ranges,
)
from scripts.sparse_corruption_principle_probe.corruption_library import support_mask


def test_constructive_common_dose_preserves_contract_on_toy_row():
    clean = np.asarray([1.0, 0.6, 0.051, 0.049, 0.02, -0.01], dtype=np.float32)
    support = support_mask(clean[None, :], reference=clean[None, :])[0]
    ranges = row_constructive_ranges(clean, row=0, seed=42, pair_count=1)
    assert ranges["common_exists"] is True
    result = build_common_dose_row(clean, row=0, seed=42, pair_count=1, ranges=ranges)
    assert result["match_ok"] is True
    assert result["cross_audit"]["exact_changed_count"] is True
    assert result["preserve_audit"]["exact_changed_count"] is True
    assert result["cross_audit"]["support_change_positive"] is True
    assert result["preserve_audit"]["support_change_zero"] is True
    assert result["cross_audit"]["row_value_multiset_ok"] is True
    assert result["preserve_audit"]["row_value_multiset_ok"] is True
    assert support.sum() == 3


def test_zero_budget_row_is_audited_without_fake_support_change():
    clean = np.asarray([1e-8, 2e-8, 3e-8], dtype=np.float32)
    support = support_mask(clean[None, :], reference=clean[None, :])[0]
    audit = audit_swap(
        clean,
        clean,
        reference_support=support,
        requested_changed_count=0,
        expect_support_change=False,
    )
    assert audit["exact_changed_count"] is True
    assert audit["support_change_zero"] is True
    assert audit["row_value_multiset_ok"] is True


def test_degenerate_active_values_are_audited_as_range_failure():
    clean = np.asarray([1.0, 1.0, 0.01, 0.005], dtype=np.float32)
    ranges = row_constructive_ranges(clean, row=0, seed=42, pair_count=1)
    assert ranges["nonzero_budget"] is True
    assert ranges["common_exists"] is False
    assert ranges["range_failure"] == "insufficient_positive_dose_matching"
