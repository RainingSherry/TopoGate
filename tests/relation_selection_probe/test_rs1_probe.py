import numpy as np

from scripts.relation_selection_probe.relation_features import extract_edge_features
from scripts.relation_selection_probe.rs1_information_probe import grouped_probe
from scripts.representation_consumer_probe.protocol import build_candidate_pool


def test_grouped_probe_returns_primary_metrics_without_leakage():
    rng = np.random.default_rng(21)
    h0 = rng.normal(size=(40, 18)).astype(np.float32)
    pool = build_candidate_pool(h0)
    table = extract_edge_features(h0, {
        "indices": pool.indices,
        "cosine": pool.cosine,
        "positive_counts": pool.positive_counts,
        "effective_budget": pool.effective_budget,
    })
    target = (table.feature("cosine") >= np.median(table.feature("cosine"))).astype(np.int64)
    result, scores = grouped_probe(table, target, "G", n_splits=5)
    assert result["status"] == "completed_valid"
    assert np.isfinite(scores).all()
    assert "average_precision" in result
    assert "delta_ap" in result
    assert result["labels_used_in_feature_extraction"] is False
    assert result["labels_used_in_diagnostic_target"] is True
