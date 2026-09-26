"""Restartable, validation-only H9+ search for the three H8-leading datasets.

There is no maximum hop depth. For each active dataset, search B and one-step
neighbors of q/a around its validation incumbent at the next depth. Stop only
after four consecutive completed depths fail to beat the best validation ARI.
Every attempt retains its checkpoint. This module never opens test labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from statistics import mean, pstdev

ROOT = Path('<PROJECT_ROOT>/experiments/v0_rg_validation_tuned_multihop_20260922')
EXP = ROOT / 'h8_adaptive_depth_20260924'
TOP = ROOT / 'top5_h1_h8_20260923'
NAMES = ('Tosches', 'fabert', 'Baron')
SEEDS = (42, 123, 7)
Q_GRID = (0.25, 0.5, 1.0, 2.0)
A_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)
GPU_SLOTS = ((1, 8), (2, 12), (4, 20))
IMPROVEMENT_EPS = 1e-8
PYTHON = '<REMOTE_PATH>'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def token(value):
    return f'{value:g}'.replace('.', 'p')


def decode(config_id):
    variant, tail = config_id.split('_q', 1) if '_q' in config_id else (config_id, '1_a1')
    q, a = tail.split('_a', 1)
    return int(variant.split('_B')[1]), float(q.replace('p', '.')), float(a.replace('p', '.'))


def adjacent(grid, current):
    idx = min(range(len(grid)), key=lambda i: abs(grid[i]-current))
    if abs(grid[idx]-current) > 1e-9:
        raise ValueError(f'incumbent parameter {current} is outside the frozen search grid')
    return [grid[j] for j in (idx-1, idx+1) if 0 <= j < len(grid)]


def initial_state():
    old = {r['dataset']: r for r in read(ROOT / 'tuning19/tuning19_summary.json')['rows']}
    new = {r['dataset']: r for r in read(TOP / 'tuning_summary.json')['rows']}
    locked = {r['dataset']: r for r in read(TOP / 'heldout_eval/selection_lock.json')['rows']}
    templates = read(TOP / 'depth_tasks.json')['tasks'] + read(ROOT / 'tuning19/stage_a_tasks.json')['tasks']
    template = {n: next(t for t in templates if t['dataset'] == n and int(t['seed']) == 42) for n in NAMES}
    state = {'protocol': 'h8_plus_adaptive_depth_validation_only_v1',
             'selection_metric': '3seed_mean_validation_ari',
             'improvement_epsilon': IMPROVEMENT_EPS,
             'stop_after_consecutive_nonimproving_depths': 4,
             'test_labels_loaded': False, 'data_copies_created': False,
             'gpu_slots': {str(g): n for g, n in GPU_SLOTS}, 'datasets': {}}
    for n in NAMES:
        original = new[n] if n in new else old[n]
        config_id = original['best_config_id']
        if not config_id.startswith('H8_B'):
            raise ValueError(f'expected H8 validation incumbent: {n}: {config_id}')
        lock = locked[n]
        if lock['selected_config_id'] != config_id or original.get('test_labels_loaded', False) is not False:
            raise ValueError(f'incumbent validation lock mismatch: {n}')
        split = read(lock['tuning_manifest_path'])
        if (split.get('test_labels_loaded') is not False or split.get('test_label_path_exposed') is not False
                or 'test_labels_path' in split):
            raise ValueError(f'test path leaked to tuning manifest: {n}')
        records = lock['selected_records']
        for seed in SEEDS:
            rec = read(records[str(seed)])
            if rec.get('test_labels_loaded') is not False or rec.get('selection_label_scope') != 'validation_only':
                raise ValueError(f'invalid H8 record: {n}/{seed}')
            if not Path(records[str(seed)]).with_name(f'model_seed_{seed}.pt').is_file():
                raise FileNotFoundError(f'missing H8 checkpoint: {n}/{seed}')
        state['datasets'][n] = {'status': 'active', 'next_hop': 9, 'no_improve_streak': 0,
                                'best_hop': 8, 'best_config_id': config_id,
                                'best_validation_ari_mean': float(original['validation_ari_mean']),
                                'best_record_paths': records,
                                'best_checkpoint_paths': {str(s): str(Path(records[str(s)]).with_name(f'model_seed_{s}.pt')) for s in SEEDS},
                                'incumbent_config': read(records['42'])['config'],
                                'input_kind': template[n]['input_kind'],
                                'feature_limit': int(template[n]['feature_limit']),
                                'tuning_manifest_path': lock['tuning_manifest_path'],
                                'feature_sha256': lock['feature_sha256'],
                                'validation_labels_sha256': lock['validation_labels_sha256'],
                                'history': []}
    return state


def combinations(hop, state):
    c = state['incumbent_config']
    width, q, a = decode(state['best_config_id'])
    options = {(b, q, a) for b in (1, 2, 4)}
    options.update((width, x, a) for x in adjacent(Q_GRID, q))
    options.update((width, q, x) for x in adjacent(A_GRID, a))
    return sorted(options)


def build_stage(state):
    active = {n: v for n, v in state['datasets'].items() if v['status'] == 'active'}
    stage_id = '_'.join(f'{n}-H{v["next_hop"]}' for n, v in active.items())
    stage_dir = EXP / 'stages' / stage_id
    tasks = []
    for n, v in active.items():
        hop = v['next_hop']
        for breadth, q, a in combinations(hop, v):
            variant = f'H{hop}_B{breadth}'
            config_id = f'{variant}_q{token(q)}_a{token(a)}'
            for seed in SEEDS:
                config = copy.deepcopy(v['incumbent_config'])
                config.update(hop_variant=variant, high_hop_probability_multiplier=q, mix_amplitude_multiplier=a)
                tasks.append({'dataset': n, 'hop_variant': variant, 'config_id': config_id,
                              'seed': seed, 'config': config,
                              'tuning_manifest_path': v['tuning_manifest_path'],
                              'input_kind': v['input_kind'], 'feature_limit': v['feature_limit'],
                              'test_labels_loaded': False})
    manifest = {'protocol': state['protocol'], 'test_labels_loaded': False,
                'selection_label_scope': 'validation_only', 'stage_id': stage_id,
                'task_count': len(tasks), 'tasks': tasks}
    path = stage_dir / 'tasks.json'
    if path.exists():
        if read(path) != manifest:
            raise ValueError(f'previous stage manifest changed: {path}')
    else:
        write(path, manifest)
    return stage_dir, path, tasks


def record_for(task, runs_dir):
    seed = int(task['seed'])
    folder = runs_dir / 'runs' / task['dataset'] / task['config_id'] / f'seed_{seed}'
    for p in sorted(folder.glob(f'attempt_*/record_seed_{seed}.json'), reverse=True):
        try:
            r = read(p)
            split = read(task['tuning_manifest_path'])
            if (r.get('status') == 'completed' and r.get('seed') == seed
                    and r.get('config') == task['config']
                    and r.get('test_labels_loaded') is False
                    and r.get('selection_label_scope') == 'validation_only'
                    and r.get('feature_sha256') == split['feature_sha256']
                    and r.get('validation_labels_sha256') == split['validation_labels_sha256']
                    and p.with_name(f'model_seed_{seed}.pt').is_file()
                    and p.with_name(f'predictions_seed_{seed}.npz').is_file()
                    and p.with_name(f'hop_sampling_seed_{seed}.json').is_file()):
                return p
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            pass
    return None


def missing_tasks(tasks, runs_dir):
    return [t for t in tasks if record_for(t, runs_dir) is None]


def run_pass(stage_dir, manifest_path, worker_counts, pass_name, max_attempts):
    log_dir = stage_dir / 'logs' / pass_name
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_count = sum(slots for _, slots in worker_counts)
    processes = []
    worker_index = 0
    environment = os.environ.copy()
    environment['PYTHONPATH'] = f'{EXP}:/home/luolie/ToPoGate'
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[var] = '1'
    for gpu, slots in worker_counts:
        for _ in range(slots):
            env = environment.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            log = (log_dir / f'gpu{gpu}_worker{worker_index:02d}.log').open('a', encoding='utf-8')
            command = [PYTHON, '-m', 'v0rg_tunable_multihop.run_pilot_queue',
                       '--manifest', str(manifest_path), '--output-root', str(stage_dir / 'output'),
                       '--device', 'cuda:0', '--worker-index', str(worker_index),
                       '--worker-count', str(worker_count), '--max-attempts', str(max_attempts), '--continue-on-error']
            proc = subprocess.Popen(command, cwd=EXP, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((proc, log))
            worker_index += 1
    for proc, log in processes:
        proc.wait()
        log.close()
    return [proc.returncode for proc, _ in processes]


def summarize_stage(state, stage_dir, tasks):
    groups = {}
    failures = {}
    for t in tasks:
        path = record_for(t, stage_dir / 'output')
        if path is None:
            failures.setdefault(t['dataset'], []).append(f"{t['config_id']}/seed_{t['seed']}")
            continue
        groups.setdefault(t['dataset'], {}).setdefault(t['config_id'], {})[int(t['seed'])] = path
    for n, data in state['datasets'].items():
        if data['status'] != 'active':
            continue
        hop = data['next_hop']
        if failures.get(n):
            data['status'] = 'failed'
            data['failure'] = {'hop': hop, 'missing_tasks': failures[n]}
            continue
        scores = {}
        for config_id, paths in groups[n].items():
            if set(paths) != set(SEEDS):
                raise ValueError(f'incomplete group: {n}/{config_id}')
            values = [float(read(paths[s])['validation_metrics']['ari']) for s in SEEDS]
            scores[config_id] = {'validation_ari_mean': mean(values), 'validation_ari_std': pstdev(values),
                                 'seed_ari': {str(s): v for s, v in zip(SEEDS, values)},
                                 'record_paths': {str(s): str(paths[s]) for s in SEEDS},
                                 'checkpoint_paths': {str(s): str(paths[s].with_name(f'model_seed_{s}.pt')) for s in SEEDS}}
        selected = max(scores, key=lambda k: (scores[k]['validation_ari_mean'], -decode(k)[0], -abs(decode(k)[1]-1), -abs(decode(k)[2]-1)))
        current = float(scores[selected]['validation_ari_mean'])
        improved = current > float(data['best_validation_ari_mean']) + IMPROVEMENT_EPS
        data['history'].append({'hop': hop, 'best_stage_config_id': selected,
                                'best_stage_validation_ari_mean': current,
                                'prior_global_best_ari_mean': data['best_validation_ari_mean'],
                                'improved_global_best': improved, 'complete_config_count': len(scores),
                                'scores': scores})
        if improved:
            data['best_hop'] = hop
            data['best_config_id'] = selected
            data['best_validation_ari_mean'] = current
            data['best_record_paths'] = scores[selected]['record_paths']
            data['best_checkpoint_paths'] = scores[selected]['checkpoint_paths']
            data['incumbent_config'] = read(scores[selected]['record_paths']['42'])['config']
            data['no_improve_streak'] = 0
        else:
            data['no_improve_streak'] += 1
        data['next_hop'] = hop + 1
        if data['no_improve_streak'] >= 4:
            data['status'] = 'stopped_after_four_nonimproving_depths'
    write(stage_dir / 'stage_summary.json', {'protocol': state['protocol'],
                                             'test_labels_loaded': False,
                                             'datasets': {n: {'status': v['status'],
                                                             'best_hop': v['best_hop'],
                                                             'best_validation_ari_mean': v['best_validation_ari_mean'],
                                                             'no_improve_streak': v['no_improve_streak']}
                                                          for n, v in state['datasets'].items()}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize-only', action='store_true')
    args = parser.parse_args()
    state_path = EXP / 'state.json'
    state = read(state_path) if state_path.exists() else initial_state()
    if not state_path.exists():
        write(state_path, state)
    if args.initialize_only:
        print(json.dumps({n: {'incumbent': v['best_config_id'], 'ari': v['best_validation_ari_mean'],
                              'next_config_count': len(combinations(9, v))}
                          for n, v in state['datasets'].items()}), flush=True)
        return
    while any(v['status'] == 'active' for v in state['datasets'].values()):
        stage_dir, manifest_path, tasks = build_stage(state)
        print(f'stage {stage_dir.name}: {len(tasks)} tasks, missing {len(missing_tasks(tasks, stage_dir / "output"))}', flush=True)
        if missing_tasks(tasks, stage_dir / 'output'):
            codes = run_pass(stage_dir, manifest_path, GPU_SLOTS, 'primary', 2)
            print(f'primary workers complete; nonzero exits={sum(x != 0 for x in codes)}', flush=True)
        if missing_tasks(tasks, stage_dir / 'output'):
            retry_slots = tuple((gpu, max(1, slots // 2)) for gpu, slots in GPU_SLOTS)
            codes = run_pass(stage_dir, manifest_path, retry_slots, 'retry', 4)
            print(f'retry workers complete; nonzero exits={sum(x != 0 for x in codes)}', flush=True)
        summarize_stage(state, stage_dir, tasks)
        write(state_path, state)
        print(json.dumps({n: {'status': v['status'], 'best_hop': v['best_hop'],
                              'best_val_ari': v['best_validation_ari_mean'],
                              'no_improve_streak': v['no_improve_streak']}
                          for n, v in state['datasets'].items()}), flush=True)
    completed = all(v['status'] == 'stopped_after_four_nonimproving_depths' for v in state['datasets'].values())
    write(EXP / 'best_checkpoints.json', {n: {'best_hop': v['best_hop'],
                                               'best_config_id': v['best_config_id'],
                                               'validation_ari_mean': v['best_validation_ari_mean'],
                                               'checkpoint_paths': v['best_checkpoint_paths']}
                                            for n, v in state['datasets'].items()})
    (EXP / ('PIPELINE_COMPLETED' if completed else 'PIPELINE_PARTIAL')).touch()
    print('pipeline_completed' if completed else 'pipeline_partial', flush=True)


if __name__ == '__main__':
    main()
