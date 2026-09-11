#!/usr/bin/env python3
"""Persistent dispatcher for verified raw-count biology tuning jobs."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/data/luolie/ToPoGate/result/topogate_transductive_clubench131_scmae16_64x3_20260911')
REPO = Path('/tmp/topogate_transductive_20260911')
PYTHON = '/data/luolie/conda/envs/topogate_unified_20260910/bin/python'
GPUS = (4, 5, 6, 7)
POLL_SECONDS = 60


def biology_rows() -> list[dict]:
    rows = json.loads((ROOT / 'manifest.json').read_text())['datasets']
    selected = [
        row for row in rows
        if row.get('panel') == 'biology'
        and row.get('eligible') == 'eligible'
        and row.get('input_kind') == 'raw_count'
    ]
    return sorted(selected, key=lambda row: (int(row['n_samples']), row['dataset_id']))


def is_active(kind: str, dataset_id: str) -> bool:
    needle = 'transductive_search' if kind == 'search' else 'run_frozen_final_transductive.py'
    output = subprocess.run(['pgrep', '-af', needle], text=True, capture_output=True).stdout
    for line in output.splitlines():
        pid, _, command = line.partition(' ')
        if dataset_id not in command:
            continue
        try:
            state = (Path('/proc') / pid / 'stat').read_text().split()[2]
        except (FileNotFoundError, IndexError):
            continue
        if state != 'Z':
            return True
    return False


def launch(kind: str, row: dict, gpu: int) -> int:
    dataset_id, clusters = row['dataset_id'], int(row['n_clusters'])
    output = ROOT / 'datasets' / dataset_id
    log = ROOT / 'scheduler' / f'{dataset_id}_{kind}_biology_auto.log'
    common = ['--n-clusters', str(clusters), '--input-kind', 'raw_count', '--device', 'cuda:0']
    if kind == 'search':
        command = [PYTHON, '-m', 'methods.TopoGate.V0_RG.transductive_search', row['canonical_path'], str(output), *common]
    else:
        command = [PYTHON, str(REPO / 'run_frozen_final_transductive.py'), '--data-path', row['canonical_path'], '--out', str(output), *common]
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with log.open('a', encoding='utf-8') as handle:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    (ROOT / 'scheduler' / f'bio_{kind}_{dataset_id}.pid').write_text(str(process.pid))
    return process.pid


def main() -> None:
    scheduler = ROOT / 'scheduler'
    scheduler.mkdir(parents=True, exist_ok=True)
    log = scheduler / 'biology_dispatch.log'
    rows = biology_rows()
    cursor = 0
    while True:
        for kind in ('search', 'final'):
            for row in rows:
                output = ROOT / 'datasets' / row['dataset_id']
                ready = not (output / 'search_summary.json').exists() if kind == 'search' else (output / 'search_summary.json').exists() and not (output / 'final_summary.json').exists()
                if ready and not is_active(kind, row['dataset_id']):
                    gpu = GPUS[cursor % len(GPUS)]
                    cursor += 1
                    pid = launch(kind, row, gpu)
                    with log.open('a', encoding='utf-8') as handle:
                        handle.write(f'launch {kind} {row["dataset_id"]} gpu={gpu} pid={pid}\n')
        searches = sum((ROOT / 'datasets' / row['dataset_id'] / 'search_summary.json').exists() for row in rows)
        finals = sum((ROOT / 'datasets' / row['dataset_id'] / 'final_summary.json').exists() for row in rows)
        with log.open('a', encoding='utf-8') as handle:
            handle.write(f'status eligible={len(rows)} search={searches}/{len(rows)} final={finals}/{len(rows)}\n')
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
