import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'multihop_mechanism_20260924'

def main():
    tasks=[]
    for dataset in ('Tosches','Baron','fabert'):
        for seed in (42,123,7):
            for step in ('readout','source','geometry'):
                out=ROOT/'outputs'/step/dataset/f'seed_{seed}'
                tasks.append({'id':f'{step}_{dataset}_{seed}', 'kind':'analysis' if step=='source' else 'cpu',
                    'dataset':dataset,'seed':seed,'out':str(out),
                    'command':[str(ROOT/f'{step}_analysis.py'),'--previous-root',str(PREVIOUS),
                               '--dataset',dataset,'--seed',str(seed),'--out',str(out)]})
    q={'protocol':'aggregation_case_analysis_v1','tasks':tasks,'no_encoder_training':True,
       'posthoc_label_use':True,'fixed_readout_states':list(range(20)), 'n_init':20}
    data=json.dumps(q,indent=2).encode()
    if (ROOT/'controller_state.json').exists():raise RuntimeError('Cannot replace dispatched queue')
    (ROOT/'queue.json').write_bytes(data)
    (ROOT/'queue.sha256').write_text(hashlib.sha256(data).hexdigest())
    print('Prepared',len(tasks),'fixed tasks')

if __name__=='__main__':main()
