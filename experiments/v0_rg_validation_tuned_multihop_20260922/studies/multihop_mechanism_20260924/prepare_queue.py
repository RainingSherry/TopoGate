"""Freeze the mechanism tasks before any posthoc outcome is evaluated."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
SEEDS = (42, 123, 7)
ARMS = ('reference', 'h1_matched_amplitude', 'replace_far_le4',
        'kwide_countmatched', 'far_shuffle', 'rescue50', 'mixed_target')

def main():
    lock = json.loads((ROOT / 'inputs/selection_lock.json').read_text())
    tasks = []
    for row in lock['rows']:
        d = row['dataset']
        if d not in ('Tosches', 'fabert', 'Baron'):
            continue
        for seed in SEEDS:
            for arm, norm_match in [(a,False) for a in ARMS]+[(a,True) for a in ('replace_far_le4','far_shuffle')]:
                task = dict(dataset=d, seed=seed, arm=arm,
                            locked_record=row['selected_records'][str(seed)],
                            tuning_manifest=row['tuning_manifest_path'],
                            input_kind='general' if d == 'fabert' else 'raw_count',
                            feature_limit=2000, norm_match_reference=norm_match)
                output_arm=arm+('_normmatched' if norm_match else '')
                key = f'train_{d}_{seed}_{output_arm}'
                path = ROOT / 'tasks' / f'{key}.json'
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(task, indent=2))
                out = ROOT / 'outputs/step5' / d / output_arm / f'seed_{seed}'
                tasks.append(dict(id=key, kind='gpu', dataset=d, arm=arm, seed=seed,
                    out=str(out), command=['step5_interventions.py', '--task-json', str(path),
                                          '--out', str(out), '--device', 'cuda:0']))
            out = ROOT / 'outputs/steps234' / d / f'seed_{seed}'
            tasks.append(dict(id=f'info_{d}_{seed}', kind='analysis', dataset=d, seed=seed,
                out=str(out), command=['steps234_information.py', '--lock', str(ROOT/'inputs/selection_lock.json'),
                '--out', str(ROOT/'outputs/steps234'), '--dataset', d, '--seed', str(seed), '--core', str(BASE),
                '--max-anchors', '512', '--mc-repeats', '16']))
        out = ROOT / 'outputs/step1' / d
        tasks.append(dict(id=f'outcomes_{d}', kind='cpu', dataset=d, out=str(out),
            command=['step1_outcomes.py', '--lock', str(ROOT/'inputs/selection_lock.json'),
                     '--background', str(ROOT/'inputs/validation_best_multihop_vs_h1_full_x_60.json'),
                     '--out', str(out), '--dataset', d]))
    data = dict(protocol='multihop_mechanism_frozen_v1', seeds=list(SEEDS), arms=list(ARMS),
                selection='precommitted_controls_no_new_label_selection', tasks=tasks)
    target = ROOT / 'queue.json'
    encoded = json.dumps(data, ensure_ascii=False, indent=2).encode()
    if target.exists() and target.read_bytes() != encoded:
        if (ROOT/'controller_state.json').exists() or list((ROOT/'outputs').rglob('TASK_COMPLETED.json')):
            raise RuntimeError('Refuse to replace a queue after real evaluation begins')
        (ROOT/'queue_prelaunch_revision.json').write_bytes(target.read_bytes())
    target.write_bytes(encoded)
    (ROOT/'queue.sha256').write_text(hashlib.sha256(encoded).hexdigest()+'\n')
    print(json.dumps({'tasks':len(tasks),'training_and_reuse':sum(t['kind']=='gpu' for t in tasks)}))

if __name__ == '__main__':
    main()
