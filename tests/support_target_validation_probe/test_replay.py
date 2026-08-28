import numpy as np

from scripts.sparse_corruption_principle_probe.corruption_library import corrupt_matrix
from scripts.support_target_validation_probe import protocol
from scripts.support_target_validation_probe.replay import build_magnitude_matched_epoch, replay_p2_epoch


def _matrix():
    rng = np.random.default_rng(11)
    x = np.zeros((16, 24), dtype=np.float32)
    x[:, :16] = rng.uniform(0.2, 1.5, size=(16, 16)).astype(np.float32)
    x[:, 16:] = rng.uniform(0.001, 0.06, size=(16, 8)).astype(np.float32)
    return x


def test_replay_matches_frozen_c2_p2_action_and_values():
    x = _matrix()
    old_rng = np.random.default_rng(42)
    replay_rng = np.random.default_rng(42)
    old, old_audit = corrupt_matrix(x, protocol.P2_PRINCIPLE, old_rng, rate=protocol.CORRUPTION_RATE)
    replay, replay_audit = replay_p2_epoch(x, replay_rng)
    assert np.array_equal(old, replay)
    for key in ("changed_mask", "source_mask", "destination_mask", "support_changed_mask", "effective_changed_counts"):
        assert np.array_equal(old_audit[key], replay_audit[key])


def test_magnitude_matched_control_preserves_support_sources_and_row_multiset():
    x = _matrix()
    p2, p2_audit = replay_p2_epoch(x, np.random.default_rng(7))
    mm, mm_audit = build_magnitude_matched_epoch(x, p2, p2_audit)
    assert np.array_equal(p2_audit["source_mask"], mm_audit["source_mask"])
    assert mm_audit["exact_budget"] is True
    assert mm_audit["support_change_rate"] == 0.0
    assert mm_audit["row_value_multiset_mismatch_count"] == 0
    assert np.isfinite(mm).all()
    assert mm_audit["labels_used"] is False

