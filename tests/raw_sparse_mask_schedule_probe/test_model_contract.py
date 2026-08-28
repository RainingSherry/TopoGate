import numpy as np

from scripts.raw_sparse_mask_schedule_probe import model


def test_paired_initial_state_hash_and_batch_schedule():
    x = np.array([[1, 0, 2], [0, 3, 0], [4, 5, 0], [0, 0, 0]], dtype=np.float32)
    active = x != 0
    clean = model.fit_autoencoder(x, active, arm="CLEAN_AE", seed=42, device="cpu", epochs=1, batch_size=2)
    masked = model.fit_autoencoder(x, active, arm="ACTIVE_FIXED", seed=42, device="cpu", epochs=1, batch_size=2)
    assert clean.model_init_hash == masked.model_init_hash
    assert clean.batch_schedule_hash == masked.batch_schedule_hash
    assert clean.embedding.shape == (4, 32)


def test_zero_budget_training_is_finite():
    x = np.zeros((3, 4), dtype=np.float32)
    result = model.fit_autoencoder(x, x != 0, arm="ACTIVE_FIXED", seed=1, device="cpu", epochs=1, batch_size=2)
    assert np.isfinite(result.embedding).all()
    assert np.isfinite([row["loss"] for row in result.history]).all()


def test_sparse_dense_projection_equivalence():
    from scripts.raw_sparse_mask_schedule_probe.benchmark_sparse_compute import projection_equivalence

    x = np.array([[0, 1, 0], [2, 0, 3]], dtype=np.float32)
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    dense, sparse, error = projection_equivalence(x, w)
    assert np.allclose(dense, sparse)
    assert error <= 1e-5

