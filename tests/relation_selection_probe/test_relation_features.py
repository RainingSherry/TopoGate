import numpy as np

from scripts.relation_selection_probe.relation_features import (
    FEATURE_FAMILIES,
    extract_edge_features,
)
from scripts.representation_consumer_probe.protocol import build_candidate_pool


def test_feature_extraction_is_finite_and_label_free():
    rng = np.random.default_rng(4)
    h0 = rng.normal(size=(24, 12)).astype(np.float32)
    pool = build_candidate_pool(h0)
    table = extract_edge_features(h0, {
        "indices": pool.indices,
        "cosine": pool.cosine,
        "positive_counts": pool.positive_counts,
        "effective_budget": pool.effective_budget,
    })
    assert table.rows.size > 0
    assert table.features.shape == (table.rows.size, 17)
    assert np.isfinite(table.features).all()
    assert table.metadata["labels_used"] is False
    assert set(FEATURE_FAMILIES["G+T+S"]) == set(table.feature_names)
    assert np.all((table.feature("stability_recurrence") >= 0.0) & (table.feature("stability_recurrence") <= 1.0))
