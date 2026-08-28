import numpy as np

from scripts.relation_selection_probe.relation_features import extract_edge_features
from scripts.relation_selection_probe.selectors import SELECTORS, selected_graph, selector_mask
from scripts.representation_consumer_probe.protocol import build_candidate_pool


def _table():
    rng = np.random.default_rng(9)
    h0 = rng.normal(size=(30, 16)).astype(np.float32)
    pool = build_candidate_pool(h0)
    table = extract_edge_features(h0, {
        "indices": pool.indices,
        "cosine": pool.cosine,
        "positive_counts": pool.positive_counts,
        "effective_budget": pool.effective_budget,
    })
    return table


def test_all_fixed_selectors_preserve_directed_row_budget():
    table = _table()
    for selector in SELECTORS:
        mask = selector_mask(table, selector)
        counts = np.bincount(table.rows[mask], minlength=table.n_samples)
        assert np.array_equal(counts, table.budget)
        graph, returned_mask = selected_graph(table, selector)
        assert np.array_equal(mask, returned_mask)
        assert graph.shape == (table.n_samples, table.n_samples)
        assert np.isfinite(graph.data).all()
        assert graph.nnz > 0

