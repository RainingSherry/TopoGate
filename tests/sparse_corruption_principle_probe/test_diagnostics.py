import numpy as np

from scripts.sparse_corruption_principle_probe.corruption_library import corrupt_matrix, geometry_importance, residual_proxy
from scripts.sparse_corruption_principle_probe.mechanism_diagnostics import (
    combined_diagnostics,
    representation_diagnostics,
)


def test_structural_diagnostics_are_finite_and_label_free():
    rng = np.random.default_rng(5)
    x = np.zeros((30, 16), dtype=np.float32)
    x[:, :6] = rng.uniform(0.5, 2.0, size=(30, 6)).astype(np.float32)
    z, audit = corrupt_matrix(
        x,
        "P5_GeometryHard",
        np.random.default_rng(6),
        residual_scores=residual_proxy(x),
        geometry_scores=geometry_importance(x, k=7),
    )
    diagnostics = combined_diagnostics(x, z)
    assert audit["labels_used"] is False
    assert diagnostics
    assert all(np.isfinite(float(value)) for value in diagnostics.values())


def test_representation_diagnostics_detect_constant_embedding():
    z = np.ones((20, 4), dtype=np.float32)
    result = representation_diagnostics(z)
    assert result["effective_rank"] == 0.0
    assert result["low_variance_dimension_ratio"] == 1.0

