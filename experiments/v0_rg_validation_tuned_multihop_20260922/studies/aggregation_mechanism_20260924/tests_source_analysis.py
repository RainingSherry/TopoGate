"""Small unit checks for the frozen-source aggregation calculations."""
import unittest
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from source_analysis import (
    analyze_sources, embedding_scale, feature_pairs, fragment_ids,
    observed_fragment_distances, selected_illustrations,
    verify_frozen_candidates,
)


class SourceAnalysisChecks(unittest.TestCase):
    def test_fragments_deterministic_and_label_free(self):
        neighbors = np.asarray([[1], [0], [3], [2]])
        weights = np.ones((4, 1))
        a, graph = fragment_ids(neighbors, weights, 4)
        b, _ = fragment_ids(neighbors, weights, 4)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(len(np.unique(a)), 2)
        self.assertEqual(a[0], a[1])
        self.assertNotEqual(a[0], a[2])

    def test_single_directed_edge_has_same_weight_in_either_direction(self):
        forward = np.asarray([[1], [1]])
        backward = np.asarray([[0], [0]])
        weights = np.asarray([[.3], [0.]])
        _, graph_forward = fragment_ids(forward, weights, 2)
        _, graph_backward = fragment_ids(backward, np.asarray([[0.], [.3]]), 2)
        self.assertAlmostEqual(graph_forward[0][1]["weight"], .3)
        self.assertAlmostEqual(graph_backward[0][1]["weight"], .3)

    def test_raw_feature_similarity_is_pairwise_and_zero_safe(self):
        raw = sp.csr_matrix([[1, 2, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        l2 = np.sqrt(np.asarray(raw.multiply(raw).sum(axis=1)).ravel())
        cosine, jaccard = feature_pairs(raw, 0, np.asarray([1, 2]),
                                        np.diff(raw.indptr), l2)
        np.testing.assert_allclose(cosine, [1 / np.sqrt(5), 0])
        np.testing.assert_allclose(jaccard, [.5, 0])

    def test_mc_mass_and_cross_fragment_coverage(self):
        y = np.asarray([0, 0, 1, 0])
        fragments = np.asarray([0, 1, 2, 1])
        raw = np.asarray([[1, 0], [1, 1], [0, 1], [1, 2]], dtype=float)
        h1 = np.asarray([[0, 0], [1, 0], [0, 2], [1, 1]], dtype=float)
        h8 = np.asarray([[0, 0], [.5, 0], [0, 2], [.5, .5]], dtype=float)
        endpoints = [
            dict(anchor_index="0", source_index="1", hop="1", candidate_probability="0.5",
                 mc_actual_weight="0.75", mc_draw_hits="2"),
            dict(anchor_index="0", source_index="2", hop="5", candidate_probability="0.25",
                 mc_actual_weight="0.25", mc_draw_hits="1"),
            dict(anchor_index="0", source_index="3", hop="5", candidate_probability="0.25",
                 mc_actual_weight="0", mc_draw_hits="0"),
        ]
        pairs, anchors, summary = analyze_sources(endpoints, [0], y, fragments,
                                                  raw, h1, h8, "raw_count")
        self.assertEqual(len(pairs), 3)
        self.assertEqual(len(anchors), 3)
        self.assertEqual(len(summary), 12)
        h1row = next(r for r in anchors if r["hop_group"] == "H1")
        self.assertAlmostEqual(h1row["same_class_cross_fragment_mc_mass"], .75)
        self.assertAlmostEqual(h1row["same_class_cross_fragment_coverage_of_global"], 1)
        far = next(r for r in anchors if r["hop_group"] == "H5_8")
        self.assertEqual(far["same_class_cross_fragment_candidate_fragments"], 1)
        self.assertEqual(far["same_class_cross_fragment_sampled_fragments"], 0)
        self.assertEqual(pairs[0]["source_library_size"], 2)
        self.assertLess(pairs[0]["h8_distance_global_rms"], pairs[0]["h1_distance_global_rms"])
        centroids = observed_fragment_distances(pairs, y, fragments, h1, h8)
        self.assertEqual(len(centroids), 2)
        self.assertEqual(sum(r["candidate_pair_count"] for r in centroids), 3)
        self.assertTrue(any(r["same_class"] == 1 for r in centroids))
        self.assertTrue(any(r["same_class"] == 0 for r in centroids))

    def test_invalid_embedding_scale_and_feature_only_illustration_rule(self):
        with self.assertRaises(ValueError):
            embedding_scale(np.ones((3, 2)))
        rows = [dict(anchor_index=i, far_actual_sampling_mass=x)
                for i, x in enumerate([.9, .1, .5, .7, .3])]
        chosen = selected_illustrations(rows)
        self.assertEqual([r["anchor_index"] for r in chosen], [1, 2, 0])

    def test_frozen_candidate_reconstruction_checks_nodes_hops_and_probability(self):
        expected = {"indices": np.asarray([3, 4]), "hops": np.asarray([1, 5]),
                    "probs": np.asarray([.4, .6])}
        prior = SimpleNamespace(frozen_row=lambda *args: expected)
        cfg = SimpleNamespace(hop_variant="H8_B2", high_hop_probability_multiplier=1.0)
        core = {"_transition_matrix": lambda *args: None}
        saved = [dict(anchor_index="0", source_index=str(j), hop=str(h),
                      candidate_probability=str(p)) for j, h, p in ((3, 1, .4), (4, 5, .6))]
        self.assertEqual(verify_frozen_candidates(saved, [0], None, np.zeros(1), cfg, prior, core), 2)
        for field, bad in (("source_index", "5"), ("hop", "2"),
                           ("candidate_probability", ".2")):
            altered = [dict(row) for row in saved]
            altered[1][field] = bad
            with self.assertRaises(ValueError):
                verify_frozen_candidates(altered, [0], None, np.zeros(1), cfg, prior, core)


if __name__ == "__main__":
    unittest.main()
