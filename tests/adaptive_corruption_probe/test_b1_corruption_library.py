import numpy as np

from scripts.adaptive_corruption_probe.b1_corruption_library import (
    corrupt_h0,
    positive_control,
    support_mask,
)


def test_positive_control_passes_without_labels():
    result = positive_control()
    assert result["status"] == "completed_valid"
    assert result["labels_used"] is False
    assert all(result["checks"].values())


def test_support_value_arm_semantics_are_separated():
    rng = np.random.default_rng(11)
    h0 = np.zeros((12, 16), dtype=np.float32)
    h0[:, :5] = rng.uniform(1.0, 2.0, size=(12, 5))
    active = support_mask(h0)
    value, _ = corrupt_h0(h0, "C1_ValueOnly", rng)
    moved, _ = corrupt_h0(h0, "C2_SupportOnly", rng)
    mixed, _ = corrupt_h0(h0, "C3_MixedMatched", rng)
    assert np.array_equal(support_mask(value), active)
    assert np.any(support_mask(moved) != active)
    assert np.any(support_mask(mixed) != active)
    assert np.any(np.abs(mixed - h0) > 1e-7)


def test_clean_arm_is_identity():
    rng = np.random.default_rng(7)
    h0 = rng.normal(size=(8, 12)).astype(np.float32)
    out, stats = corrupt_h0(h0, "C_clean_no_corruption", rng)
    assert np.array_equal(out, h0)
    assert stats["effective_changed_coordinate_rate"] == 0.0
