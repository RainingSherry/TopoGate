"""Portable tests for the unified operator and leakage boundaries."""
import importlib.util
import unittest
from dataclasses import asdict, replace

import numpy as np

from methods.TopoGate.V0_RG.config import V0_RGConfig
from methods.TopoGate.V0_RG.graph import build_pca_knn_graph, compute_edge_reliability
from methods.TopoGate.V0_RG.mixing import compute_node_gate, estimate_neighbors
from methods.TopoGate.V0_RG.tuning import (SplitPreprocessor, initial_candidates,
    rank_candidates, split_rows)


class UnifiedTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(9).normal(size=(20, 6)).astype(np.float32)
        self.graph = build_pca_knn_graph(self.X, k=5, pca_dim=3, tau=.2, seed=42)

    def gate(self, config):
        lo, hi = config.gate_bounds()
        return compute_node_gate(self.graph, self.graph.probs, 'topology', lo, hi,
            config.beta_mutual, config.beta_snn, config.beta_perturb, 0)[0]

    def test_center_radius_matches_legacy_gate(self):
        c = V0_RGConfig(gate_center=.08, gate_adaptivity=.5)
        legacy = replace(c, gate_center=None, gate_adaptivity=None, gate_min=.04, gate_max=.12)
        np.testing.assert_array_equal(self.gate(c), self.gate(legacy))
        c = replace(c, gate_adaptivity=0.)
        np.testing.assert_array_equal(self.gate(c), np.full(20, .08, dtype=np.float32))

    def test_edge_zero_exactly_recovers_base_probabilities(self):
        _, w, _ = compute_edge_reliability(self.graph, 'none', 1, 1, 1, 1)
        np.testing.assert_array_equal(w, self.graph.probs)
        _, z, _ = compute_edge_reliability(self.graph, 'sim_mutual_snn_distance', 0, 0, 0, 0)
        np.testing.assert_allclose(z, self.graph.probs, atol=1e-7)

    def test_current_estimator_matches_original_loop_and_rng(self):
        rng = np.random.default_rng(43)
        ids = np.array([1, 3, 7])
        expected = []
        for i in ids:
            probs = self.graph.probs[i]
            choices = rng.choice(5, size=4, replace=True, p=probs / np.clip(probs.sum(), 1e-12, None))
            weights = probs[choices].astype(np.float32)
            weights /= max(float(weights.sum()), 1e-12)
            expected.append(np.sum(self.X[self.graph.indices[i, choices]] * weights[:, None], axis=0))
        actual = estimate_neighbors(self.X, ids, self.graph, self.graph.probs, 4,
                                    np.random.default_rng(43), 'current')
        np.testing.assert_array_equal(actual, np.array(expected))

    def test_full_estimator_and_monte_carlo_expectation(self):
        ids = np.array([2])
        exact = estimate_neighbors(self.X, ids, self.graph, self.graph.probs, 4,
                                   np.random.default_rng(0), 'full')[0]
        rng = np.random.default_rng(0)
        samples = [estimate_neighbors(self.X, ids, self.graph, self.graph.probs, 4, rng,
                                      'uniform_sample')[0] for _ in range(4000)]
        np.testing.assert_allclose(np.mean(samples, axis=0), exact, atol=.04)

    def test_preprocessing_uses_training_statistics(self):
        train = np.array([[0., -1, 2], [2, 1, 4], [4, 3, 6]])
        p = SplitPreprocessor('general', 3).fit(train)
        before = p.fingerprint
        result = p.transform(np.array([[100., 101., 102.]]))
        np.testing.assert_allclose(result, (np.array([[100, 101, 102]])-train.mean(0))/train.std(0))
        self.assertEqual(before, p.fingerprint)
        with self.assertRaises(ValueError):
            p.transform(np.array([[np.nan, 1, 2]]))
        with self.assertRaises(ValueError):
            SplitPreprocessor('raw_count').fit(train)

    def test_splits_and_constant_anchors(self):
        parts = split_rows(100)
        self.assertEqual([len(p) for p in parts], [60, 20, 20])
        self.assertEqual(len(set(np.concatenate(parts))), 100)
        a = initial_candidates(V0_RGConfig())
        self.assertEqual(a[0].gate_bounds(), (.1, .1))
        self.assertEqual(a[0].edge_reliability_mode, 'none')
        self.assertEqual(a[2].gate_bounds(), (.1, .1))
        self.assertEqual(a[3].edge_reliability_mode, 'none')

    def test_selection_rejects_test_duplicates_and_different_budgets(self):
        def record(config, seed, score):
            return dict(config=asdict(config), seed=seed, validation_ari=score,
                        stage='validation', identity='one', status='completed')
        a = record(V0_RGConfig(), 42, .4)
        b = record(V0_RGConfig(epochs=40), 42, .9)
        with self.assertRaises(ValueError): rank_candidates([a, b], (42,))
        with self.assertRaises(ValueError): rank_candidates([a, a], (42,))
        with self.assertRaises(ValueError): rank_candidates([dict(a, test_metrics={})], (42,))
        self.assertEqual(rank_candidates([a], (42, 123)), [])
        with self.assertRaises(ValueError):
            rank_candidates([a, dict(a, seed=123, identity='other')], (42, 123))

    def test_invalid_parameters_rejected(self):
        for kwargs in [dict(gate_center=.1), dict(gate_center=.8, gate_adaptivity=1),
                       dict(gate_center=.1, gate_adaptivity=0, gate_max=.2),
                       dict(lr=float('nan')), dict(beta_mutual=float('inf')),
                       dict(auxiliary_weighting='other'), dict(neighbor_estimator='other')]:
            with self.assertRaises(ValueError): V0_RGConfig(**kwargs)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
    def test_fit_readout_uses_only_training_rows(self):
        from methods.TopoGate.V0_RG.trainer import fit_predict
        c = replace(initial_candidates(V0_RGConfig())[0], hidden_size=4, epochs=1, batch_size=8)
        train = self.X[:12]
        score = self.X[12:]
        pred, emb, diag = fit_predict(score, fit_X=train, n_clusters=2, config=c, seed=42, device='cpu')
        self.assertEqual(pred.shape, (8,))
        self.assertEqual(diag['neighbor_indices'].shape[0], 12)
        self.assertEqual(diag['core_summary']['readout_fit_scope'], 'fit_X')
        self.assertEqual(diag['core_summary']['gate_mode'], 'constant')
        np.testing.assert_allclose(diag['node_gate'], .1)
        # Fewer held-out rows than registered centers is a valid prediction task.
        short, _, _ = fit_predict(score[:2], fit_X=train, n_clusters=3, config=c, seed=42, device='cpu')
        self.assertEqual(short.shape, (2,))
        # Changing only held-out rows must leave graph, gates, and training losses unchanged.
        _, _, second = fit_predict(score + 100., fit_X=train, n_clusters=2, config=c, seed=42, device='cpu')
        np.testing.assert_array_equal(diag['neighbor_indices'], second['neighbor_indices'])
        np.testing.assert_array_equal(diag['training_history']['loss'], second['training_history']['loss'])


if __name__ == '__main__': unittest.main()
