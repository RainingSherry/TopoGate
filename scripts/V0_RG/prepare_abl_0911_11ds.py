from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path('/data/luolie/ToPoGate/result/v0_rg_biology_ablation_transductive_0911winners_20260916')
OLD = Path('/data/luolie/ToPoGate/result/topogate_transductive_clubench131_scmae16_64x3_20260911/datasets')
DATA = Path('/data/luolie/biopipeline/scCluBench/data/scMAE')
DATASETS = ['Bach','Baron','Guo','Limb_Muscle','Macosko','Melanoma_5K','Quake_10x_Spleen','Quake_Smart-seq2_Lung','Tosches','Young','worm_neuron_cell']
VARIANTS = ['full','base_scmae','anchor_equal_mask','neighbor_equal','no_edge_modulation','fixed_mix_original_aux','mixed_observation_reconstruction']
ESTIMATOR_DATASETS = {'Quake_Smart-seq2_Lung','Macosko'}
ESTIMATORS = ['current','uniform_sample','full']
SEEDS = [42, 123, 7]

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, sort_keys=True))

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'protocol_id': 'v0_rg_0911winners_transductive_ablation_20260916_v1',
        'parent_protocol': 'topogate_transductive_perdataset_64x3_v1',
        'winner_source': str(OLD),
        'evaluation_mode': 'transductive',
        'split': {'train': 0.6, 'validation': 0.2, 'test': 0.2, 'split_seed': 9102026},
        'fit_scope': 'all_samples_unlabeled',
        'evaluation_scope': 'fixed_test_only',
        'labels_used_during_fit': False,
        'selection_labels': 'validation_only',
        'test_used_for_selection': False,
        'seeds': SEEDS,
        'epochs': 80,
        'variants': VARIANTS,
        'estimator_comparison': {'datasets': sorted(ESTIMATOR_DATASETS), 'estimators': ESTIMATORS},
        'fixed_mix_strength_definition': 'mean per-sample node gate of the frozen full model over all unlabeled data (transductive training set), per seed',
        'gpu_pool': [1, 2, 3, 5],
        'forbidden_gpus': [0, 7],
        'raw_sha256': 'not_computed_per_user_policy',
        'history_policy': 'old winner configs are configuration sources only; all metrics and artifacts are rerun in this root; original formal results are never overwritten',
        'full_acceptance': 'ARI/NMI/ACC must reproduce old final per seed within 1e-10 before interpreting other variants',
    }
    write(ROOT / 'protocol.json', protocol)
    manifest = []
    for n in DATASETS:
        sel = json.loads((OLD / n / 'selected.json').read_text())
        manifest.append({'dataset': n, 'data_path': str(DATA / (n + '.h5')),
                         'selected_path': str(OLD / n / 'selected.json'),
                         'selection_hash': sel.get('selection_hash'),
                         'winner_candidate': sel['winner']['candidate'],
                         'winner_config_hash': sel['winner']['config_hash'],
                         'winner_config': sel['winner']['config']})
    write(ROOT / 'manifest.json', {'datasets': manifest})
    tasks = []
    for n in DATASETS:
        for v in VARIANTS:
            tasks += [{'dataset': n, 'variant': v, 'estimator': 'current', 'seed': s} for s in SEEDS]
    for n in sorted(ESTIMATOR_DATASETS):
        for e in ['uniform_sample', 'full']:
            tasks += [{'dataset': n, 'variant': 'full', 'estimator': e, 'seed': s} for s in SEEDS]
    (ROOT / 'scheduler').mkdir(parents=True, exist_ok=True)
    with (ROOT / 'scheduler' / 'tasks.tsv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'variant', 'estimator', 'seed'], delimiter='\t')
        w.writeheader(); w.writerows(tasks)
    write(ROOT / 'scheduler' / 'status.json', {'total': len(tasks), 'completed': 0, 'failed': 0, 'pending': len(tasks)})
    (ROOT / 'README.md').write_text('Paired transductive ablation fixed to 0911 per-dataset frozen winners. 11 datasets x 7 settings x 3 seeds + 2 datasets x 2 estimators x 3 seeds. Original results are read-only.\n')
    print(json.dumps({'root': str(ROOT), 'tasks': len(tasks)}))

if __name__ == '__main__':
    main()

