import numpy as np

from scripts.raw_sparse_mask_schedule_probe import raw_adapter


def test_zero_preserving_scale_keeps_exact_pattern():
    x = np.array([[0.0, 2.0, -1.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    x0, scale = raw_adapter.zero_preserving_scale(x)
    assert np.array_equal(x == 0, x0 == 0)
    assert np.all(np.isfinite(x0))
    assert np.all(scale > 0)


def test_zero_preserving_scale_rejects_non_2d():
    try:
        raw_adapter.zero_preserving_scale(np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("one-dimensional input must be rejected")

