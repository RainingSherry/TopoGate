"""Fixed exploratory transfer: no candidate-label selection. Prepared X, default backbone, H4_B1."""
from pathlib import Path
from dataclasses import asdict
import numpy as np
from common import *
from v0rg_tunable_multihop.config import V0_RGConfig
from v0rg_tunable_multihop.tuning import SplitPreprocessor
from v0rg_tunable_multihop.input_adapter import load_npz_matrix_only
from prepare_factorial import train_task

def main():
    prereg=read_json(EXP/'prereg/prereg.json');seeds=[s['seed'] for s in prereg['seeds']]
    cfg=asdict(V0_RGConfig(hop_variant='H4_B1',high_hop_probability_multiplier=1.,mix_amplitude_multiplier=1.))
    lock={'status':'FIXED_EXPLORATORY_TRANSFER','datasets':{},'policy':cfg,'labels_used_for_selection':False,'independence':'not_yet_established','seeds':seeds}
    q=[]
    for dataset in ['Pollen','Wang','hrvatin_geo']:
        profile_path=Path('<PROJECT_ROOT>/datasets/eswa_prepared_v1')/f'{dataset}.profile.json'
        profile=read_json(profile_path);source=Path(profile['target_path'])
        # K from existing profile only; no label values loaded during preparation.
        k=len(profile['label_classes']);loaded=load_npz_matrix_only(source);assert loaded.profile['labels_accessed'] is False
        folder=EXP/'transfer_inputs'/dataset;folder.mkdir(parents=True,exist_ok=True)
        X=loaded.X.toarray() if hasattr(loaded.X,'toarray') else loaded.X
        atomic_npz(folder/'features_only.npz',X=X)
        prep=SplitPreprocessor('general',2000).fit(X)
        manifest={'feature_path':str(folder/'features_only.npz'),'feature_sha256':sha256(folder/'features_only.npz'),'n_clusters':k,'n_samples':len(X),'K_provenance':'benchmark_oracle_from_y','test_label_path_exposed':False}
        atomic_json(folder/'manifest.json',manifest)
        info={'tuning_manifest_path':str(folder/'manifest.json'),'posthoc_label_source_paths':[str(source)],'input_kind_candidates':['general'],'feature_limit_candidates':[2000],'expected_formal_records':10,'selected_multihop':{'config':cfg,'eff_hash':canonical_sha256(cfg),'preprocessing_hash':prep.fingerprint},'profile_sha256':sha256(profile_path),'prepared_input_note':'Use existing prepared1000-feature X without further log normalization; paired standard scaling; fixed default backbone80epochs','source_study_independence':'unknown, exploratory transfer'}
        lock['datasets'][dataset]=info
        files=[];ids=[]
        for seed in seeds:
            for arm in ARMS:
                task=train_task(dataset,info,manifest,{'input_kind':'general','feature_limit':2000},arm,seed,canonical_sha256(cfg))
                path=EXP/'tasks'/f"{task['id']}.json";write_once_json(path,task)
                if seed==seeds[0]:files.append(str(path.relative_to(EXP)))
                if arm not in ['H1_anchor','flat_anchor']:continue
                out=f'outputs/train/{dataset}/{arm}/seed_{seed}';ids.append(task['id'])
                q.append(dict(id=task['id'],kind='gpu',stage='train_transfer',out=out,task_file=str(path.relative_to(EXP)),deps=[f'refcheck_{dataset}'],command=['run_factorial.py','train','--task-json',str(path.relative_to(EXP)),'--out',out,'--device','cuda:0']))
        out=f'outputs/refcheck/{dataset}'
        q.append(dict(id=f'refcheck_{dataset}',kind='refcheck',stage='refcheck',out=out,deps=[],command=['run_factorial.py','refcheck','--task-json',*files,'--out',out,'--epochs','2']))
        out=f'outputs/factorial/{dataset}'
        q.append(dict(id=f'factorial_{dataset}',kind='cpu',stage='posthoc',out=out,deps=ids,dep_mode='all_terminal',allow_partial=True,command=['analyze_cases.py','factorial','--dataset',dataset,'--lock','prereg/transfer_lock.json','--out',out]))
    atomic_json(EXP/'prereg/transfer_lock.json',lock)
    atomic_json(EXP/'extension_tasks_transfer.json',{'tasks':q,'policy_sha256':canonical_sha256(cfg)})
    print({'transfer_tasks':len(q),'formal_paired_fits':30,'seeds':seeds,'independence':'exploratory'})
if __name__=='__main__':main()
