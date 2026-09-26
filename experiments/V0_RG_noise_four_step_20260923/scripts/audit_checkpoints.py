"""Read-only frozen-checkpoint audit; never deserializes or trains a model.

Summary statistics establish an intervention, not its efficacy. Equal prediction
hashes are reported as evidence and never used to redefine the intervention.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

REQUIRED = ('model.pt', 'preprocessor.npz', 'preprocessor.json',
            'cluster_centers.npy', 'embedding_all.npy', 'predictions_all.npy',
            'test_indices.npy', 'final_summary.json', 'graph_gate_summary.json')
ALLOWED = {
    'full': set(),
    'base_scmae': {'variant', 'pseudo_weight', 'edge_reliability_mode'},
    'no_edge_modulation': {'edge_modulation_enabled', 'edge_reliability_mode'},
    'fixed_mix_original_aux': {'mix_strength_mode', 'fixed_mix_strength'},
    'mixed_observation_reconstruction': {'mixed_observation_target'},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def array_info(a):
    numeric = a.dtype.kind in 'biufc'
    return {'shape': list(a.shape), 'dtype': str(a.dtype),
            'finite': bool(np.isfinite(a).all()) if numeric else None}


def checkpoint_path(entry, variant, seed):
    root = Path(entry['checkpoint_root'])
    # Resolve sibling variants before resolving symlinks: current can be a link.
    if root.name != 'current' or root.parent.name != 'full':
        raise ValueError('cohort checkpoint_root must end in full/current')
    root = root.parent.parent / variant / 'current'
    candidates = [root / f'seed_{seed}', root / f'seed{seed}']
    present = [p for p in candidates if p.is_dir()]
    if len(present) > 1:
        raise ValueError('ambiguous seed directory: ' + str(present))
    return present[0] if present else candidates[0]


def inspect(entry, variant, seed):
    row = dict(dataset=entry['dataset'], variant=variant, model_seed=seed,
               status='ready', checkpoint=None, files={}, fit_scope=None,
               eval_scope=None, effective_intervention=None, interpretation='',
               config_diff={}, unexpected_config_diff={}, comparison={}, issues=[])
    arrays, summary, graph, pre = {}, {}, {}, {}
    try:
        p = checkpoint_path(entry, variant, seed)
        row['checkpoint'] = str(p)
        for name in REQUIRED:
            f = p / name
            info = {'path': str(f), 'exists': f.is_file(), 'sha256': None}
            row['files'][name] = info
            if not f.is_file():
                row['issues'].append('missing: ' + name)
                continue
            info['sha256'] = sha256(f)
            info['bytes'] = f.stat().st_size
            if not info['bytes']:
                row['issues'].append('empty: ' + name)
            if name.endswith('.npy'):
                a = np.load(f, allow_pickle=False, mmap_mode='r')
                arrays[name] = a
                info.update(array_info(a))
                if info['finite'] is not True:
                    row['issues'].append('non-numeric/non-finite array: ' + name)
            elif name == 'preprocessor.npz':
                with np.load(f, allow_pickle=False) as z:
                    info['arrays'] = {k: array_info(z[k]) for k in z.files}
                    if any(v['finite'] is not True for v in info['arrays'].values()):
                        row['issues'].append('invalid preprocessor arrays')
                    if not {'features', 'mean', 'scale'} <= set(z.files):
                        row['issues'].append('preprocessor missing features/mean/scale')
                    else:
                        if any(z[k].ndim != 1 for k in ['features', 'mean', 'scale']) or not (len(z['features']) == len(z['mean']) == len(z['scale'])):
                            row['issues'].append('preprocessor shape mismatch')
                        if z['features'].dtype.kind not in 'iu' or np.any(z['features'] < 0) or np.unique(z['features']).size != z['features'].size or np.any(z['scale'] <= 0):
                            row['issues'].append('preprocessor invalid features/scales')
        for name, dest in [('final_summary.json', summary), ('graph_gate_summary.json', graph), ('preprocessor.json', pre)]:
            if (p / name).is_file():
                dest.update(read_json(p / name))
        row['fit_scope'] = summary.get('fit_scope')
        row['eval_scope'] = summary.get('evaluation_scope')
        row['config'] = summary.get('config', {})
        row['graph_gate_summary'] = graph
        row['preprocessor_fingerprint'] = pre.get('fingerprint')
        if summary:
            if (summary.get('dataset'), summary.get('variant'), summary.get('seed')) != (entry['dataset'], variant, seed):
                row['issues'].append('summary identity mismatch')
            if summary.get('status') != 'completed':
                row['issues'].append('source checkpoint not completed')
            if row['fit_scope'] != 'all_samples_unlabeled' or row['eval_scope'] != 'fixed_test_only' or pre.get('fit_scope') != row['fit_scope']:
                row['issues'].append('unsupported fit/evaluation scope')
            if graph.get('core', {}).get('labels_used_during_fit') is not False:
                row['issues'].append('label-free fit not certified')
        if len(arrays) == 4:
            e, y, t, c = [arrays[k] for k in ('embedding_all.npy', 'predictions_all.npy', 'test_indices.npy', 'cluster_centers.npy')]
            if e.ndim != 2 or c.ndim != 2 or y.ndim != 1 or t.ndim != 1:
                row['issues'].append('invalid array rank')
            elif e.shape[0] != y.size or e.shape[1] != c.shape[1] or not c.size:
                row['issues'].append('embedding/prediction/centers shape mismatch')
            elif t.dtype.kind not in 'iu' or not t.size or t.min() < 0 or t.max() >= y.size or np.unique(t).size != t.size:
                row['issues'].append('invalid test indices')
            elif y.dtype.kind not in 'iu' or y.min() < 0 or y.max() >= c.shape[0]:
                row['issues'].append('invalid predictions')
    except Exception as exc:
        row['issues'].append(type(exc).__name__ + ': ' + str(exc))
    if row['issues']:
        row['status'] = 'missing_artifacts' if any(x.startswith('missing:') for x in row['issues']) else 'invalid_artifacts'
    return row


def effective(full, row):
    c, v = full['config'], row['config']
    fg, vg = full['graph_gate_summary'], row['graph_gate_summary']
    variant = row['variant']
    if variant == 'full':
        return None, 'Reference full model; no ablation.'
    if variant == 'base_scmae':
        if v.get('variant') == 'scmae_only' and v.get('pseudo_weight') == 0 and vg.get('core', {}).get('graph_enabled') is False and vg.get('core', {}).get('pseudo_enabled') is False:
            return True, 'Compound scMAE baseline disables graph pathway and pseudo auxiliary; not an edge-only intervention.'
    elif variant == 'no_edge_modulation':
        if v.get('edge_modulation_enabled') is not False:
            return None, 'Variant does not disable edge modulation.'
        if c.get('edge_reliability_mode') == 'none' or c.get('edge_modulation_enabled') is False:
            return False, 'Full model already has no edge reliability modulation; similarity-weighted graph can remain active.'
        gammas = [c.get(k) for k in ('gamma_distance', 'gamma_mutual', 'gamma_sim', 'gamma_snn')]
        if all(x == 0 for x in gammas):
            return False, 'Full reliability modulation coefficients are all zero.'
        a, b = fg.get('edge', {}), vg.get('edge', {})
        keys = ['edge_weight_entropy', 'effective_neighbor_count', 'max_edge_weight_mean', 'max_edge_weight_p95']
        changed = [k for k in keys if k in a and k in b and not np.isclose(a[k], b[k], rtol=1e-7, atol=1e-9)]
        if changed:
            return True, 'Nonzero full modulation and observed edge summary changes: ' + ', '.join(changed)
        return None, 'Nonzero modulation configured but edge summary does not establish changed weights; raw edge replay required.'
    elif variant == 'fixed_mix_original_aux':
        if v.get('mix_strength_mode') != 'fixed_train_mean' or v.get('fixed_mix_strength') is None:
            return None, 'Variant is not a fixed training-mean mix.'
        g = fg.get('gate', {})
        lo, hi, mean = g.get('min_node_gate'), g.get('max_node_gate'), g.get('mean_node_gate')
        if mean is not None and not np.isclose(v['fixed_mix_strength'], mean, rtol=1e-6, atol=1e-8):
            return None, 'Fixed strength does not match full clean training gate mean.'
        constant = lo is not None and hi is not None and np.isclose(lo, hi, rtol=1e-7, atol=1e-9)
        if constant or c.get('gate_adaptivity') == 0 or (all(c.get(k) == 0 for k in ['beta_mutual', 'beta_perturb', 'beta_snn', 'beta_uncertainty'])):
            return False, 'Full node gate is constant; fixed mean is a numerical no-op control (float32 reduction rounding may differ).'
        if lo is not None and hi is not None and hi > lo:
            return True, 'Full observed node gate varies; replaced by its clean training mean.'
    elif variant == 'mixed_observation_reconstruction':
        if c.get('mixed_observation_target') is False and v.get('mixed_observation_target') is True and vg.get('core', {}).get('mixed_observation_target') is True:
            if fg.get('gate', {}).get('max_node_gate', 0) > 0 and vg.get('core', {}).get('graph_enabled') is True:
                return True, 'Reconstruction target changes to mixed observation with an active nonzero graph mix.'
            return False, 'Target flag changes but graph mix is inactive/zero.'
    return None, 'Requested intervention could not be established from checkpoint evidence.'


def pair(full, row):
    if full['status'] != 'ready' or row['status'] != 'ready':
        if row['status'] == 'ready':
            row['status'] = 'pairing_mismatch'
            row['issues'].append('full reference is invalid')
        return
    f, v = full['config'], row['config']
    diff = {k: {'full': f.get(k), 'variant': v.get(k)} for k in sorted(set(f) | set(v)) if f.get(k) != v.get(k) or (k in f) != (k in v)}
    row['config_diff'] = diff
    row['unexpected_config_diff'] = {k: val for k, val in diff.items() if k not in ALLOWED.get(row['variant'], set())}
    # These are exact archived runner resolutions, not unrestricted allowed
    # changes. In particular an alternate reliability mode is not an ablation.
    expected_targets = {
        'base_scmae': {'variant': 'scmae_only', 'pseudo_weight': 0.0, 'edge_reliability_mode': 'none'},
        'no_edge_modulation': {'edge_modulation_enabled': False, 'edge_reliability_mode': 'none'},
    }.get(row['variant'], {})
    for key, target in expected_targets.items():
        actual = v.get(key)
        if actual != target or (isinstance(target, bool) and actual is not target):
            row['unexpected_config_diff'][key] = {
                'full': f.get(key), 'variant': actual, 'required_target': target,
            }
    def eq_file(name):
        return full['files'][name]['sha256'] == row['files'][name]['sha256']
    row['comparison'] = {
        'preprocessor_hash_equal': eq_file('preprocessor.npz') and eq_file('preprocessor.json'),
        'test_indices_hash_equal': eq_file('test_indices.npy'),
        'centers_shape_equal': full['files']['cluster_centers.npy']['shape'] == row['files']['cluster_centers.npy']['shape'],
        'model_hash_equal': eq_file('model.pt'),
        'predictions_hash_equal': eq_file('predictions_all.npy'),
        'embedding_hash_equal': eq_file('embedding_all.npy'),
        'model_hash_note': 'Serialized checkpoint bytes only; unequal hashes do not prove unequal tensors.',
    }
    row['effective_intervention'], row['interpretation'] = effective(full, row)
    if row['unexpected_config_diff']:
        row['status'] = 'config_mismatch'
        row['issues'].append('unexpected configuration differences')
        row['effective_intervention'] = None
    elif not all(row['comparison'][k] for k in ['preprocessor_hash_equal', 'test_indices_hash_equal', 'centers_shape_equal']):
        row['status'] = 'pairing_mismatch'
        row['issues'].append('clean preprocessing/test mapping/centers dimensions mismatch')
        row['effective_intervention'] = None
    elif row['effective_intervention'] is False:
        row['status'] = 'no_op_control'
    elif row['effective_intervention'] is None and row['variant'] != 'full':
        row['status'] = 'unresolved_intervention'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config.json')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    entries = {e['dataset']: e for e in read_json(config['cohort'])['entries']}
    rows = []
    for dataset in config['datasets']:
        entry = entries.get(dataset, {'dataset': dataset, 'checkpoint_root': None})
        for seed in config['model_seeds']:
            full = inspect(entry, 'full', seed)
            for variant in config['variants']:
                row = full if variant == 'full' else inspect(entry, variant, seed)
                pair(full, row)
                rows.append(row)
    out = (args.output_root or Path(config['output_root'])) / 'inventories'
    counts = dict(Counter(row['status'] for row in rows))
    payload = {'schema': 'four_step_checkpoint_inventory_v1', 'protocol': config['protocol'],
               'config_sha256': sha256(args.config), 'cohort_sha256': sha256(config['cohort']),
               'script_sha256': sha256(__file__), 'status_counts': counts,
               'expected_rows': len(config['datasets']) * len(config['variants']) * len(config['model_seeds']),
               'rows': rows}
    atomic_json(out / 'checkpoints.json', payload)
    fields = ['dataset', 'variant', 'model_seed', 'status', 'checkpoint', 'fit_scope', 'eval_scope',
              'effective_intervention', 'interpretation', 'config_diff', 'unexpected_config_diff', 'comparison', 'files', 'issues']
    tmp = out / 'checkpoints.csv.tmp'
    with tmp.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in fields})
    tmp.replace(out / 'checkpoints.csv')
    print(json.dumps({'rows': len(rows), 'status_counts': counts, 'output': str(out)}, ensure_ascii=False))
    return 0 if all(r['status'] in {'ready', 'no_op_control'} for r in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
