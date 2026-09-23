"""Explicit, resumable stage launcher. Full experiments require an explicit CLI stage."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import load_config, atomic_json, checked_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['inventory', 'observe', 'directions', 'summarize'], required=True)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--model-seeds', type=int, nargs='+')
    parser.add_argument('--noise-seeds', type=int, nargs='+')
    parser.add_argument('--condition', action='append')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--include-no-op-controls', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    scripts = Path(__file__).parent.resolve()
    root = Path(config['output_root'])
    if args.workers < 1 or args.workers > 8:
        raise ValueError('Choose 1–8 bounded workers; first measure memory on a representative case.')
    if args.stage == 'inventory':
        raise SystemExit(subprocess.call([sys.executable, str(scripts/'audit_checkpoints.py'), '--config', args.config]))
    if args.stage == 'summarize':
        out = root / 'summaries'
        subprocess.run([sys.executable, str(scripts/'summarize.py'), '--results', str(root), '--out', str(out)], check=True)
        plot_command = [sys.executable, str(scripts/'plot_results.py'), '--results', str(out), '--out', str(root/'figures')]
        if config.get('smoke_validation_only'):
            plot_command.append('--smoke-validation')
        subprocess.run(plot_command, check=True)
        return
    checked_device(args.device, config['allowed_gpu_ids'])
    invpath = root/'inventories/checkpoints.json'
    if not invpath.exists():
        raise FileNotFoundError('Run --stage inventory first; no checkpoint is silently assumed valid.')
    inventory = json.loads(invpath.read_text(encoding='utf-8'))
    records = inventory.get('rows', inventory.get('entries', []))
    bykey = {(r['dataset'], r['variant'], int(r['model_seed'])): r for r in records}
    datasets = args.datasets or config['datasets']
    variants = args.variants or config['variants']
    seeds = args.model_seeds or config['model_seeds']
    for requested, registered in ((datasets, config['datasets']), (variants, config['variants']), (seeds, config['model_seeds'])):
        if not set(requested).issubset(set(registered)):
            raise ValueError('Subset must be contained in the registered configuration.')
    tasks, excluded = [], []
    for dataset in datasets:
        for variant in variants:
            for seed in seeds:
                row = bykey.get((dataset, variant, seed))
                reason = None
                if row is None:
                    reason = 'not_in_checkpoint_inventory'
                elif row.get('status') not in {'ready', 'passed', 'verified', 'available', 'no_op_control', 'ok'}:
                    reason = 'inventory_status:' + str(row.get('status'))
                elif variant != 'full' and row.get('effective_intervention') is False and not args.include_no_op_controls:
                    reason = 'no_op_control_not_a_mechanism_intervention'
                if reason:
                    excluded.append(dict(dataset=dataset, variant=variant, model_seed=seed, reason=reason))
                    continue
                cmd = [sys.executable, str(scripts/'probe.py'), '--config', str(Path(args.config).resolve()),
                       '--stage', args.stage, '--dataset', dataset, '--variant', variant, '--model-seed', str(seed),
                       '--device', args.device]
                if args.noise_seeds:
                    cmd += ['--noise-seeds'] + list(map(str,args.noise_seeds))
                for condition in args.condition or []:
                    cmd += ['--condition',condition]
                tasks.append(dict(dataset=dataset, variant=variant, model_seed=seed, command=cmd))
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = root/'dispatch'/('{}-{}-{}'.format(args.stage,timestamp,os.getpid()))
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir/'manifest.json', dict(stage=args.stage, config_sha256=config['_config_sha256'], tasks=tasks, excluded=excluded))
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    def run_one(task):
        log = run_dir / '{}__{}__seed{}.log'.format(task['dataset'],task['variant'],task['model_seed'])
        with log.open('w',encoding='utf-8') as handle:
            process = subprocess.run(task['command'],stdout=handle,stderr=subprocess.STDOUT,env=env)
        return {**task,'status':'completed' if process.returncode==0 else 'failed',
                'returncode':process.returncode,'log':str(log)}
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_one,task):task for task in tasks}
        for job in concurrent.futures.as_completed(pending):
            try:
                result = job.result()
            except Exception as error:
                result = {**pending[job], 'status':'failed', 'error':repr(error)}
            completed.append(result)
            atomic_json(run_dir/'status.json',dict(total=len(tasks),completed=completed,excluded=excluded))
            print(json.dumps({k:result[k] for k in ('dataset','variant','model_seed','status')},ensure_ascii=False),flush=True)
    atomic_json(run_dir/'status.json',dict(status='finished_dispatching_audit_required',total=len(tasks),completed=completed,excluded=excluded))
    if any(row['status']=='failed' for row in completed):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
