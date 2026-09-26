"""Evaluate fixed validation-selected multihop and H1 predictions on complete X.

Labels are loaded for reporting only, after all configurations were fixed by
validation. This is descriptive full-sample transductive scoring, not heldout
generalization or another parameter-selection stage.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from v0rg_tunable_multihop.input_adapter import encode_labels, load_matrix


ROOT = Path('<PROJECT_ROOT>/experiments/v0_rg_validation_tuned_multihop_20260922')
TOP = ROOT / 'top5_h1_h8_20260923'
OUT = TOP / 'heldout_eval'
SEEDS = (42, 123, 7)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(y, pred):
    classes, ci = np.unique(y, return_inverse=True)
    clusters, pi = np.unique(pred, return_inverse=True)
    table = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    np.add.at(table, (ci, pi), 1)
    rr, cc = linear_sum_assignment(-table)
    return {'ari': float(adjusted_rand_score(y, pred)),
            'nmi': float(normalized_mutual_info_score(y, pred)),
            'acc': float(table[rr, cc].sum() / len(y))}


def main():
    candidate = read(OUT / 'validation_best_multihop_vs_h1_test_60.json')
    locked = read(OUT / 'selection_lock.json')
    if (candidate['dataset_count'] != 60 or locked['dataset_count'] != 60
            or candidate['summary_sha256'] != locked['summary_sha256']
            or candidate['candidate_selection_scope'] != 'validation_only'):
        raise ValueError('candidate/selection lock mismatch')
    lock_rows = {r['dataset']: r for r in locked['rows']}
    old_tasks = read(TOP / 'inputs/old_top5_tasks.json')['tasks']
    sources = {}
    for task in old_tasks:
        sources.setdefault(task['dataset'], set()).add(task['data_path'])
    fixed = []
    for row in candidate['rows']:
        d = row['dataset']
        if d not in lock_rows or len(sources.get(d, ())) != 1:
            raise ValueError(f'no unique source for {d}')
        fixed.append((row, lock_rows[d], next(iter(sources[d]))))
    results = []
    for row, lock, source_path in fixed:
        d = row['dataset']
        original = load_matrix(source_path)
        y, _ = encode_labels(original.labels)
        y = np.asarray(y).reshape(-1)
        if len(y) != lock['n_samples'] or original.X.shape[0] != len(y):
            raise ValueError(f'original source row mismatch {d}')
        split = read(lock['tuning_manifest_path'])
        final = read(Path(lock['tuning_manifest_path']).with_name('final_scoring_manifest.json'))
        seen = np.zeros(len(y), dtype=np.int8)
        train = np.asarray(split['train_indices'], dtype=np.int64)
        seen[train] += 1
        for sidecar_path, index_hash in ((split['validation_labels_path'], 'validation_indices_sha256'),
                                         (final['test_labels_path'], 'test_indices_sha256')):
            with np.load(sidecar_path, allow_pickle=False) as z:
                idx = np.asarray(z['indices'], dtype=np.int64)
                labels = np.asarray(z['labels']).reshape(-1)
            if (len(idx) != len(labels) or np.any(idx < 0) or np.any(idx >= len(y))
                    or not np.array_equal(y[idx], labels)):
                raise ValueError(f'original label/sidecar mismatch {d}: {sidecar_path}')
            seen[idx] += 1
        if not np.all(seen == 1):
            raise ValueError(f'split does not partition complete X: {d}')
        per_seed = {}
        for seed in SEEDS:
            per_seed[str(seed)] = {}
            for arm, paths in (('multihop', row['multihop_record_paths']), ('h1', lock['h1_records'])):
                if paths is None:
                    continue
                p = Path(paths[str(seed)])
                rec = read(p)
                if (rec.get('test_labels_loaded') is not False
                        or rec.get('selection_label_scope') != 'validation_only'
                        or rec.get('feature_sha256') != lock['feature_sha256']):
                    raise ValueError(f'non-isolated prediction record: {p}')
                with np.load(p.with_name(f'predictions_seed_{seed}.npz'), allow_pickle=False) as z:
                    pred = np.asarray(z['predictions']).reshape(-1)
                if len(pred) != len(y):
                    raise ValueError(f'prediction length mismatch {d}/{arm}/{seed}')
                per_seed[str(seed)][arm] = metrics(y, pred)
        scores = {}
        for metric in ('ari', 'nmi', 'acc'):
            for arm in ('multihop', 'h1'):
                values = [per_seed[str(s)][arm][metric] for s in SEEDS] if arm == 'multihop' or lock['h1_records'] else None
                scores[f'{arm}_{metric}_mean'] = mean(values) if values else None
                scores[f'{arm}_{metric}_std'] = pstdev(values) if values else None
            scores[f'delta_{metric}_mean'] = (scores[f'multihop_{metric}_mean'] - scores[f'h1_{metric}_mean']
                                               if lock['h1_records'] else None)
        if lock['h1_records']:
            scores['positive_ari_seed_count'] = sum(per_seed[str(s)]['multihop']['ari'] > per_seed[str(s)]['h1']['ari'] for s in SEEDS)
        else:
            scores['positive_ari_seed_count'] = None
        results.append({'dataset': d, 'source': row['source'], 'source_path': source_path,
                        'n_samples': len(y), 'multihop_config_id': row['multihop_config_id'],
                        'h1_config_id': row['h1_config_id'], **scores, 'per_seed': per_seed})
        print(f'completed {len(results)}/60 {d}', flush=True)
    json_path = OUT / 'validation_best_multihop_vs_h1_full_x_60.json'
    csv_path = OUT / 'validation_best_multihop_vs_h1_full_x_60.csv'
    if json_path.exists() or csv_path.exists():
        raise FileExistsError('refusing to overwrite completed full-X report')
    payload = {'evaluation_scope': 'all_rows_of_original_X',
               'fitting_scope': 'all_samples_unlabeled_transductive',
               'candidate_selection_scope': 'validation_only',
               'test_scores_used_for_selection': False,
               'warning': 'Full-X labels include validation and test rows; these descriptive scores are not heldout estimates.',
               'dataset_count': len(results), 'rows': results}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    fields = [k for k in results[0] if k not in ('source_path', 'per_seed')]
    with csv_path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    print(json.dumps({'datasets': len(results), 'paired': sum(r['delta_ari_mean'] is not None for r in results),
                      'ari_positive': sum(r['delta_ari_mean'] is not None and r['delta_ari_mean'] > 1e-12 for r in results),
                      'ari_negative': sum(r['delta_ari_mean'] is not None and r['delta_ari_mean'] < -1e-12 for r in results)}))


if __name__ == '__main__':
    main()
