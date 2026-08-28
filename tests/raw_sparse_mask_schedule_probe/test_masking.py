import numpy as np

from scripts.raw_sparse_mask_schedule_probe import masking


def _toy():
    x = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    return x, x != 0


def test_all_fixed_nominal_budget_and_determinism():
    x, active = _toy()
    first = masking.make_mask(x, active, target_space="ALL", schedule="FIXED", seed=42, epoch=0)
    second = masking.make_mask(x, active, target_space="ALL", schedule="FIXED", seed=42, epoch=0)
    assert np.array_equal(first.mask, second.mask)
    expected = np.ceil(0.25 * active.sum(axis=1)).astype(int)
    assert np.array_equal(first.mask.sum(axis=1), expected)
    assert first.audit["std_sampled_mask_ratio"] == 0.0


def test_active_fixed_selects_only_nonzero_and_zero_rows_unchanged():
    x, active = _toy()
    result = masking.make_mask(x, active, target_space="ACTIVE", schedule="FIXED", seed=7, epoch=1)
    assert not np.any(result.mask & ~active)
    assert np.array_equal(result.corrupted[1], x[1])
    assert result.audit["zero_budget_rows"] == 1


def test_variable_ratio_bounds_and_change():
    x = np.tile(np.arange(1, 21, dtype=np.float32), (8, 1))
    active = x != 0
    result = masking.make_mask(x, active, target_space="ACTIVE", schedule="VARIABLE", seed=123, epoch=0)
    assert 0.05 <= result.audit["sampled_ratio_min"] <= result.audit["sampled_ratio_max"] <= 0.45
    assert result.audit["sampled_ratio_max"] > result.audit["sampled_ratio_min"]


def test_masked_only_loss_ignores_unselected_coordinates():
    import torch

    pred = torch.tensor([[1.0, 100.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])
    loss = masking.masked_mse(pred, target, mask)
    assert float(loss) == 1.0
    loss.backward()
    assert pred.grad.tolist() == [[2.0, 0.0]]

