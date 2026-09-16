"""Portable tests for the unified operator and leakage boundaries."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from dataclasses import asdict, replace

import numpy as np

from methods.TopoGate.V0_RG.config import V0_RGConfig
from methods.TopoGate.V0_RG.graph import build_pca_knn_graph, compute_edge_reliability
from methods.TopoGate.V0_RG.mixing import compute_node_gate, estimate_neighbors
from methods.TopoGate.V0_RG.tuning import (SplitPreprocessor, build_experiment_identity,
    _required_run_artifacts, initial_candidates, rank_candidates, source_fingerprint, split_rows)


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

    def test_selection_uses_sample_standard_deviation(self):
        config = V0_RGConfig()
        records = [
            dict(config=asdict(config), seed=seed, validation_ari=score,
                 stage='validation', identity='one', status='completed')
            for seed, score in [(42, .2), (123, .4), (7, .6)]
        ]
        ranked = rank_candidates(records, (42, 123, 7))
        self.assertAlmostEqual(ranked[0]['validation_ari_std'], np.std([.2, .4, .6], ddof=1))

    def test_completion_marker_is_written_after_other_required_artifacts(self):
        run_dir = Path("/audit/run")
        completed = run_dir / "completed.json"
        self.assertIn(completed, _required_run_artifacts(run_dir, final=False))
        self.assertNotIn(
            completed,
            _required_run_artifacts(run_dir, final=False, include_completion=False),
        )

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'requires h5py')
    def test_h5ad_csr_categorical_and_legacy_h5_loading(self):
        import h5py
        from methods.TopoGate.V0_RG.input_adapter import load_matrix

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            h5ad = directory / 'tiny.h5ad'
            with h5py.File(h5ad, 'w') as handle:
                matrix = handle.create_group('X')
                matrix.create_dataset('data', data=np.array([1., 2., 3.], dtype=np.float32))
                matrix.create_dataset('indices', data=np.array([0, 2, 1], dtype=np.int64))
                matrix.create_dataset('indptr', data=np.array([0, 2, 3], dtype=np.int64))
                matrix.attrs['shape'] = np.array([2, 3], dtype=np.int64)
                labels = handle.create_group('obs').create_group('resolved_label')
                labels.create_dataset('categories', data=np.array([b'A', b'B']))
                labels.create_dataset('codes', data=np.array([1, 0], dtype=np.int64))
                # This intentionally malformed group is ignored by the direct reader.
                handle.create_group('uns').create_dataset('unsupported', data=np.array([1]))
            loaded = load_matrix(h5ad, label_key='obs/resolved_label')
            self.assertEqual(loaded.X.shape, (2, 3))
            np.testing.assert_array_equal(loaded.labels, np.array(['B', 'A']))
            self.assertEqual(loaded.profile['ignored_h5ad_groups'], ['uns'])

            legacy = directory / 'legacy.h5'
            with h5py.File(legacy, 'w') as handle:
                handle.create_dataset('X', data=np.eye(3, dtype=np.float32))
                handle.create_dataset('Y', data=np.array([b'x', b'y', b'x']))
            legacy_loaded = load_matrix(legacy, matrix_key='X', label_key='Y')
            np.testing.assert_array_equal(legacy_loaded.labels, np.array(['x', 'y', 'x']))

    @unittest.skipUnless(importlib.util.find_spec('pandas'), 'requires pandas')
    def test_hrvatin_geo_csv_alignment_and_metadata_fingerprint(self):
        import pandas as pd
        from methods.TopoGate.V0_RG.input_adapter import load_matrix

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            matrix_path = directory / 'matrix.csv.gz'
            labels_path = directory / 'labels.csv.gz'
            pd.DataFrame(
                {'gene': ['g1', 'g2'], 'cell_a': [1., 2.], 'cell_b': [3., 4.], 'cell_c': [5., 6.]}
            ).to_csv(matrix_path, index=False, compression='gzip')
            pd.DataFrame({'maintype': ['B', 'A', None]}, index=['cell_b', 'cell_a', 'cell_c']).to_csv(
                labels_path, compression='gzip'
            )
            loaded = load_matrix(
                matrix_path,
                labels_path,
                input_adapter='hrvatin_geo_csv',
                label_column='maintype',
            )
            self.assertEqual(loaded.X.shape, (2, 2))
            np.testing.assert_array_equal(loaded.labels, np.array(['B', 'A']))
            self.assertEqual(loaded.profile['n_samples_excluded_missing_label'], 1)
            self.assertEqual(loaded.profile['source_cell_column_indices'], [1, 0])
            fingerprint = source_fingerprint(matrix_path)
            self.assertEqual(fingerprint['content_sha256'], 'not_computed')
            self.assertIn('metadata_hash', fingerprint)

    def test_gpu_is_excluded_from_scientific_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'matrix.npy'
            np.save(source, self.X)
            row = {
                'dataset_id': 'tiny', 'path': str(source), 'input_kind': 'general',
                'n_clusters': 2, 'panel': 'biology', 'source_group': 'tiny', 'representation': 'fixture',
            }
            args = SimpleNamespace(epochs=80, trials=32, search_seed=9102026, feature_limit=2000)
            identity = build_experiment_identity(
                row=row,
                loaded_profile={'format': 'npy', 'n_samples_original': 20, 'n_features_original': 6},
                train=np.arange(12), validation=np.arange(12, 16), test=np.arange(16, 20), args=args,
            )
            self.assertNotIn('device', identity)
            self.assertNotIn('gpu', identity)

    def test_invalid_parameters_rejected(self):
        for kwargs in [dict(gate_center=.1), dict(gate_center=.8, gate_adaptivity=1),
                       dict(gate_center=.1, gate_adaptivity=0, gate_max=.2),
                       dict(lr=float('nan')), dict(beta_mutual=float('inf')),
                       dict(auxiliary_weighting='other'), dict(neighbor_estimator='other')]:
            with self.assertRaises(ValueError): V0_RGConfig(**kwargs)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
    def test_fit_readout_uses_only_training_rows(self):
        from methods.TopoGate.V0_RG.trainer import fit_predict, predict_from_saved_state
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
        reloaded_predictions, reloaded_embedding = predict_from_saved_state(
            score,
            config=c,
            device='cpu',
            model_state_dict=diag['model_state_dict'],
            cluster_centers=diag['cluster_centers'],
        )
        np.testing.assert_array_equal(pred, reloaded_predictions)
        np.testing.assert_allclose(emb, reloaded_embedding, rtol=0, atol=1e-7)


if __name__ == '__main__': unittest.main()
