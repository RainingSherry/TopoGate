from __future__ import annotations
import csv, os, subprocess, time
from pathlib import Path

ROOT = Path('/data/luolie/ToPoGate/result/v0_rg_biology_ablation_transductive_0911winners_20260916')
PY = '/data/luolie/conda/envs/scssl_bench_py310/bin/python'
GPUS = [1, 2, 3, 5]

def main():
    tasks = list(csv.DictReader((ROOT / 'scheduler' / 'tasks.tsv').open(), delimiter='\t'))
    pending = []
    for t in tasks:
        p = ROOT / 'datasets' / t['dataset'] / t['variant'] / t['estimator'] / ('seed_%s' % t['seed']) / 'final_summary.json'
        if not p.exists():
            pending.append(t)
    completed = len(tasks) - len(pending); failed = 0
    running = {}
    logs = ROOT / 'scheduler' / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for gpu in GPUS:
            if gpu in running or not pending:
                continue
            t = pending.pop(0)
            tag = '%s__%s__%s__seed%s' % (t['dataset'], t['variant'], t['estimator'], t['seed'])
            log = (logs / (tag + '.log')).open('w')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            cmd = [PY, 'scripts/V0_RG/run_abl_0911_11ds.py', '--dataset', t['dataset'], '--variant', t['variant'], '--estimator', t['estimator'], '--seed', t['seed'], '--gpu', str(gpu)]
            running[gpu] = (subprocess.Popen(cmd, cwd='/home/luolie/ToPoGate', env=env, stdout=log, stderr=subprocess.STDOUT), log, tag)
        for gpu, (proc, log, tag) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close(); del running[gpu]
            completed += rc == 0; failed += rc != 0
            with (ROOT / 'scheduler' / 'progress.log').open('a') as f:
                f.write('%s gpu=%s tag=%s rc=%s completed=%s failed=%s pending=%s\n' % (time.strftime('%F %T'), gpu, tag, rc, completed, failed, len(pending)))
        time.sleep(2)
    print(json_dumps_safe({'completed': completed, 'failed': failed, 'total': len(tasks)}))

def json_dumps_safe(x):
    import json
    return json.dumps(x)

if __name__ == '__main__':
    main()

