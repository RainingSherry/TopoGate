"""Parent integration of Claude-written independent adapters. No outcome-based choice."""
from pathlib import Path
import numpy as np
from common import EXP,read_json,sha256,atomic_json,atomic_npz
from run_factorial import load_features
from run_schntl import official_hashes,official_source

def main():
    prereg=read_json(EXP/'prereg/prereg.json')
    seeds=[s['seed'] for s in prereg['seeds']]
    queue=[]
    for dataset in prereg['cases']:
        base=read_json(EXP/f'tasks/train_{dataset}_flat_anchor_s{seeds[0]}.json')
        prep,_=load_features(base)
        folder=EXP/'configlocks'/f'{dataset}_baseline_features'
        folder.mkdir(exist_ok=True)
        atomic_npz(folder/'preprocessor.npz',features=prep.features,mean=prep.scaler.mean_,scale=prep.scaler.scale_)
        atomic_json(folder/'preprocessor.json',{'fingerprint':prep.fingerprint,'input_kind':base['input_kind'],'width':prep.width})
        for seed in seeds:
            tid=f'schntl_{dataset}_s{seed}'
            task={k:base[k] for k in ['dataset','feature_path','feature_sha256','n_clusters','K_provenance','input_kind','expected_preprocessing_hash']}
            task.update(id=tid,seed=seed,algorithm='scHNTL_official',locked_preprocessor_npz=str(folder/'preprocessor.npz'),locked_preprocessor_json=str(folder/'preprocessor.json'),locked_preprocessor_sha256=sha256(folder/'preprocessor.npz'))
            task['official_source_sha256']=official_hashes(official_source(task))
            path=EXP/'tasks'/f'{tid}.json';atomic_json(path,task)
            out=f'outputs/schntl/{dataset}/seed_{seed}'
            queue.append(dict(id=tid,kind='cpu',stage='extension',out=out,task_file=str(path.relative_to(EXP)),validator='schntl',python='<REMOTE_PATH>',deps=[],command=['run_schntl.py','--task-json',str(path.relative_to(EXP)),'--out',out]))
            for arm in ['flat_anchor','geometric_anchor','flat_mixed','geometric_mixed']:
                run=f'outputs/train/{dataset}/{arm}/seed_{seed}'
                ref=f'outputs/train/{dataset}/H1_anchor/seed_{seed}'
                out=f'outputs/probes/{dataset}/{arm}/seed_{seed}'
                queue.append(dict(id=f'probe_{dataset}_{arm}_s{seed}',kind='cpu',stage='posthoc',out=out,summary='probe_summary.json',allow_blocked=True,deps=[f'train_{dataset}_{arm}_s{seed}',f'train_{dataset}_H1_anchor_s{seed}'],command=['probe_directions.py','--record',run+'/record.json','--reference-record',ref+'/record.json','--features',base['feature_path'],'--out',out,'--device','cpu']))
    queue.append(dict(id='extension_audit',kind='cpu',stage='cpu_analysis',out='outputs/extension_audit',summary='extension_audit.json',deps=[],command=['extension_audit.py','--out','outputs/extension_audit','--old-manifest','outputs/p0/selection_lock_v2.json']))
    atomic_json(EXP/'extension_tasks.json',{'tasks':queue,'policy':'fixed before new outcomes; scHNTL official native preprocessing; probes no labels'})
    print({'extension_tasks':len(queue)})
if __name__=='__main__':main()
