#!/usr/bin/env python3
"""Persistent dispatcher for the 131 CLUBench transductive tuning jobs."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT = Path('/data/luolie/ToPoGate/result/topogate_transductive_clubench131_scmae16_64x3_20260911')
REPO = Path('/tmp/topogate_transductive_20260911')
PYTHON = '/data/luolie/conda/envs/topogate_unified_20260910/bin/python'
SEARCH = REPO / 'methods/TopoGate/V0_RG/transductive_search.py'
FINAL = REPO / 'run_frozen_final_transductive.py'
GPUS = (4, 5, 6, 7)
POLL = 60

def datasets():
    rows = json.loads((ROOT/'manifest.json').read_text())['datasets']
    return [r for r in rows if r.get('panel') == 'clubench']

def active(kind, name):
    needle = 'transductive_search' if kind == 'search' else 'run_frozen_final_transductive.py'
    out = subprocess.run(['pgrep', '-af', needle], text=True, capture_output=True).stdout
    for line in out.splitlines():
        pid, _, command = line.partition(' ')
        if name not in command:
            continue
        try:
            # pgrep reports zombies.  A defunct child cannot write another
            # artifact, so it must not block the scheduler's retry path.
            state = (Path('/proc') / pid / 'stat').read_text().split()[2]
        except (FileNotFoundError, IndexError):
            continue
        if state != 'Z':
            return True
    return False

def launch(kind, row, gpu):
    name, k = row['dataset_id'], int(row['n_clusters'])
    out = ROOT/'datasets'/name
    log = ROOT/'scheduler'/f'{name}_{kind}_auto.log'
    if kind == 'search':
        cmd = [PYTHON, '-m', 'methods.TopoGate.V0_RG.transductive_search',
               str(row['canonical_path']), str(out), '--n-clusters', str(k),
               '--input-kind', 'general', '--device', 'cuda:0']
    else:
        cmd = [PYTHON, str(FINAL), '--data-path', str(row['canonical_path']),
               '--out', str(out), '--n-clusters', str(k), '--device', 'cuda:0']
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with log.open('a', encoding='utf-8') as fh:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
    (ROOT/'scheduler'/f'auto_{kind}_{name}.pid').write_text(str(p.pid))
    return p.pid

def main():
    (ROOT/'scheduler').mkdir(parents=True, exist_ok=True)
    rows = datasets(); cursor = 0
    log = ROOT/'scheduler/auto_dispatch.log'
    while True:
        # Start missing screen jobs first, then promote completed searches to final.
        for row in rows:
            name = row['dataset_id']; out = ROOT/'datasets'/name
            if not (out/'search_summary.json').exists() and not active('search', name):
                gpu = GPUS[cursor % len(GPUS)]; cursor += 1
                pid = launch('search', row, gpu)
                with log.open('a') as fh: fh.write(f'launch search {name} gpu={gpu} pid={pid}\n')
        for row in rows:
            name = row['dataset_id']; out = ROOT/'datasets'/name
            if (out/'search_summary.json').exists() and not (out/'final_summary.json').exists() and not active('final', name):
                gpu = GPUS[cursor % len(GPUS)]; cursor += 1
                pid = launch('final', row, gpu)
                with log.open('a') as fh: fh.write(f'launch final {name} gpu={gpu} pid={pid}\n')
        with log.open('a') as fh:
            s = sum((ROOT/'datasets'/r['dataset_id']/'search_summary.json').exists() for r in rows)
            f = sum((ROOT/'datasets'/r['dataset_id']/'final_summary.json').exists() for r in rows)
            fh.write(f'status search={s}/{len(rows)} final={f}/{len(rows)}\n')
        time.sleep(POLL)

if __name__ == '__main__': main()
