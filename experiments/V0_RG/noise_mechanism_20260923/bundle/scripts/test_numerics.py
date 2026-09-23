"""Small scientific invariants, not model-training tests. Run with Python."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from metrics import (displacement, distribution_metrics, graph_comparison,
                     knn_self_excluded, representation_metrics,
                     applied_direction_family, subset_prediction_metrics)
from summarize import aggregate, probe_paths, ari_delta


class NumericalChecks(unittest.TestCase):
    def test_frobenius_ratio_not_mean_sample_norm(self):
        clean = np.asarray([[0., 0.], [2., 0.]], np.float32)
        noisy = clean + np.asarray([[3., 4.], [0., 0.]], np.float32)
        metric, per_sample = displacement(clean, noisy)
        self.assertAlmostEqual(metric['normalized_displacement'], 5/np.sqrt(2), places=6)
        rescaled, _ = displacement(clean*7, noisy*7)
        self.assertAlmostEqual(metric['normalized_displacement'], rescaled['normalized_displacement'], places=6)
        self.assertEqual(per_sample.shape, (2,))

    def test_constant_reference_not_false_zero_drift(self):
        metric, per_sample = displacement(np.ones((4,2)), np.ones((4,2))*2)
        self.assertIsNone(metric['normalized_displacement'])
        self.assertTrue(np.isnan(per_sample).all())

    def test_distribution_can_stay_equal_while_pairing_changes(self):
        x = np.random.default_rng(8).normal(size=(32, 4))
        shifted_rows = np.roll(x, 1, axis=0)
        effect = distribution_metrics(x, shifted_rows)
        self.assertAlmostEqual(effect['mmd']['mmd2'], 0., places=10)
        self.assertGreater(displacement(x, shifted_rows)[0]['normalized_displacement'], 0.5)
        same = distribution_metrics(x, x)
        self.assertAlmostEqual(same['mmd']['mmd2'], 0., places=10)
        self.assertAlmostEqual(same['correlation']['rms_correlation_change'], 0., places=10)
        contrast = same['mmd']['reference_contrast']
        self.assertAlmostEqual(contrast['mean_difference'], 0., places=10)

    def test_duplicate_points_do_not_leave_self_in_diagnostic_neighbors(self):
        x = np.asarray([[1.,0.],[1.,0.],[0.,1.],[-1.,0.]],np.float32)
        idx = knn_self_excluded(x,2)
        self.assertFalse(np.any(idx==np.arange(4)[:,None]))
        g,_ = graph_comparison(x,x,np.asarray([0,0,1,1]),np.arange(4),2)
        self.assertEqual(g['neighbor_retention'],1.)
        self.assertEqual(g['purity_micro_change'],0.)

    def test_boundary_crossing_changes_signed_margin_and_prediction(self):
        z = np.asarray([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]],np.float32)
        z1=z.copy(); z1[0]=[2.,0.]
        centers = np.asarray([[-1.5,0.],[1.5,0.]],np.float32)
        _, p, arr = representation_metrics(z,z1,centers,np.asarray([0,0,1,1]),np.arange(4))
        self.assertEqual(p['flip_rate'],.25)
        self.assertLess(arr['noisy_margin'][0],0.)
        self.assertGreater(arr['clean_margin'][0],0.)
        self.assertEqual(arr['clean_embedding'].shape,z.shape)
        self.assertEqual(arr['noisy_embedding'].shape,z.shape)

    def test_model_seeds_get_equal_weight_and_noise_sd_is_separate(self):
        rows=[]
        for model_seed,values in [(42,[0.,0.,0.]),(123,[1.])]:
            for noise_seed,value in enumerate(values):
                rows.append(dict(protocol='p',stage='observe',dataset='har',variant='full',
                                 corruption='gaussian',severity=.2,metric='prediction.noisy_ari',
                                 model_seed=model_seed,noise_seed=noise_seed,value=value))
        _,summary=aggregate(rows)
        self.assertEqual(summary[0]['mean'],.5)
        self.assertFalse(summary[0]['equal_noise_seed_coverage'])
        self.assertEqual(summary[0]['n_model_seeds'],2)
        self.assertAlmostEqual(ari_delta({'prediction':{'clean_ari':.8,'noisy_ari':.7}}),-.1)

    def test_formal_results_do_not_absorb_smoke_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rel=Path('har/observe/full/model_seed_42/replacement_0.2_seed3001/result.json')
            for p in (root/rel,root/'validation/smoke_v2'/rel):
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps({'status':'completed'}))
            self.assertEqual(len(list(probe_paths(root))),1)
            self.assertEqual(len(list(probe_paths(root/'validation/smoke_v2'))),1)

    def test_equal_nominal_norm_is_not_equal_applied_norm_in_float32(self):
        clean = np.asarray([[1e10, 1e10], [0., 0.]], np.float32)
        horizontal = np.asarray([[1., 0.], [1., 0.]], np.float32)
        vertical = np.asarray([[0., 1.], [0., 1.]], np.float32)
        valid, target = np.ones(2, bool), np.ones(2)
        families = {name: (delta, valid, target) for name, delta in
                    [('native', horizontal), ('random', vertical), ('shuffled', -horizontal)]}
        applied, common = applied_direction_family(clean, families, 1.)
        self.assertEqual(common.tolist(), [False, True])
        for family in applied.values():
            self.assertEqual(family['norm'][0], 0.)
            self.assertAlmostEqual(family['norm'][1], 1.)

    def test_direction_metrics_use_common_rows_and_empty_is_undefined(self):
        labels = np.asarray([0, 0, 1, 1])
        noisy = np.asarray([1, 0, 1, 1])
        result = subset_prediction_metrics(labels, labels, noisy, np.asarray([False, True, True, True]))
        self.assertEqual(result['n'], 3)
        self.assertEqual(result['flip_rate'], 0.)
        self.assertEqual(result['noisy_ari'], 1.)
        empty = subset_prediction_metrics(labels, labels, noisy, np.zeros(4, bool))
        self.assertIsNone(empty['flip_rate'])
        self.assertIsNone(empty['noisy_ari'])


if __name__=='__main__':
    unittest.main(verbosity=2)
