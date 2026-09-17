from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, '/home/luolie/ToPoGate')
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder

from methods.TopoGate.V0_RG.config import V0_RGConfig
from methods.TopoGate.V0_RG.graph import build_pca_knn_graph, compute_edge_reliability
from methods.TopoGate.V0_RG.mixing import compute_node_gate
from methods.TopoGate.V0_RG.run import clustering_metrics
from methods.TopoGate.V0_RG.trainer import fit_predict
from methods.TopoGate.V0_RG.tuning import SplitPreprocessor, split_rows

ROOT = Path('/data/luolie/ToPoGate/result/v0_rg_biology_ablation_transductive_old_winner_20260915')
OLD = Path('/data/luolie/ToPoGate/result/topogate_transductive_clubench131_scmae16_64x3_20260911/datasets')
DATA = Path('/data/luolie/biopipeline/scCluBench/data/scMAE')

def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False))
    tmp.replace(path)

def digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load_data(name):
    with h5py.File(DATA/f'{name}.h5','r') as h:
        X=np.asarray(h['X'][:],dtype=np.float32); raw=np.asarray(h['Y'][:]).reshape(-1)
    y=LabelEncoder().fit_transform(raw.astype(str)).astype(np.int64)
    if X.shape[0]!=len(y) or not np.isfinite(X).all(): raise ValueError('invalid X/Y')
    return X,y

def full_gate_mean(X, cfg, seed):
    graph=build_pca_knn_graph(X,k=min(cfg.neighbor_k,X.shape[0]-1),pca_dim=cfg.knn_pca_dim,tau=cfg.tau,seed=seed)
    _,weights,_=compute_edge_reliability(graph,mode=cfg.edge_reliability_mode,gamma_sim=cfg.gamma_sim,gamma_mutual=cfg.gamma_mutual,gamma_snn=cfg.gamma_snn,gamma_distance=cfg.gamma_distance)
    lo,hi=cfg.gate_bounds(); mode='constant' if lo==hi else 'topology'
    gate,_,_=compute_node_gate(graph,weights,mode,lo,hi,cfg.beta_mutual,cfg.beta_snn,cfg.beta_perturb,0.0,None)
    return float(gate.mean())

def resolve(base, variant, estimator, mean):
    p={**base,'neighbor_estimator':estimator,'protocol_id':'v0_rg_transductive_old_winner_ablation_20260915_v1',
       'anchor_auxiliary_weight':0.0,'neighbor_auxiliary_weight':1.0,'auxiliary_mask_mode':'independent',
       'auxiliary_input_mode':'mixed_neighbor','mix_strength_mode':'per_sample_gate','fixed_mix_strength':None,
       'mixed_observation_target':False,'edge_modulation_enabled':True}
    if variant=='base_scmae': p.update(variant='scmae_only',pseudo_weight=0.0,edge_reliability_mode='none')
    elif variant=='anchor_equal_mask': p.update(auxiliary_input_mode='anchor',anchor_auxiliary_weight=1.0,neighbor_auxiliary_weight=0.0,auxiliary_weighting='uniform')
    elif variant=='neighbor_equal': p.update(auxiliary_weighting='uniform')
    elif variant=='no_edge_modulation': p.update(edge_modulation_enabled=False,edge_reliability_mode='none')
    elif variant=='fixed_mix_original_aux': p.update(mix_strength_mode='fixed_train_mean',fixed_mix_strength=mean)
    elif variant=='mixed_observation_reconstruction': p.update(mixed_observation_target=True)
    elif variant!='full': raise ValueError('unknown variant')
    return V0_RGConfig(**p)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--variant',required=True)
    ap.add_argument('--estimator',default='current'); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--gpu',type=int,required=True)
    a=ap.parse_args()
    if a.gpu in {0,7}: raise SystemExit('GPU 0 and 7 forbidden')
    out=ROOT/'datasets'/a.dataset/a.variant/a.estimator/f'seed_{a.seed}'; done=out/'final_summary.json'
    if done.exists(): return
    out.mkdir(parents=True,exist_ok=True); started=time.time()
    X,y=load_data(a.dataset); train,val,test=split_rows(len(y),9102026)
    split={'split_seed':9102026,'train':train.tolist(),'validation':val.tolist(),'test':test.tolist(),'test_hash':digest(test.tolist())}
    split_path=ROOT/'datasets'/a.dataset/'split.json'
    if split_path.exists() and json.loads(split_path.read_text())!=split: raise ValueError('split identity changed')
    atomic_json(split_path,split)
    old_selected=json.loads((OLD/a.dataset/'selected.json').read_text()); base=old_selected['winner']['config']
    prep=SplitPreprocessor('raw_count',2000).fit(X); Xall=prep.transform(X)
    full_cfg=V0_RGConfig(**base); mean=full_gate_mean(Xall,full_cfg,a.seed) if a.variant=='fixed_mix_original_aux' else None
    cfg=resolve(base,a.variant,a.estimator,mean)
    pred,embedding,diag=fit_predict(Xall,fit_X=Xall,n_clusters=len(np.unique(y)),config=cfg,seed=a.seed,device='cuda:0')
    metrics=clustering_metrics(y[test],pred[test])
    np.save(out/'embedding_all.npy',embedding); np.save(out/'predictions_all.npy',pred); np.save(out/'test_indices.npy',test); np.save(out/'labels_true_test.npy',y[test])
    np.save(out/'cluster_centers.npy',diag['cluster_centers'])
    np.savez_compressed(out/'preprocessor.npz',features=prep.features,mean=prep.scaler.mean_,scale=prep.scaler.scale_)
    import torch; torch.save(diag['model_state_dict'],out/'model.pt')
    atomic_json(out/'preprocessor.json',{'fit_scope':'all_samples_unlabeled','fingerprint':prep.fingerprint,'selection':prep.selection,'n_features':len(prep.features)})
    atomic_json(out/'training_history.json',diag['training_history']); atomic_json(out/'graph_gate_summary.json',{'graph':diag['graph_profile'],'edge':diag['edge_summary'],'gate':diag['gate_summary'],'core':diag['core_summary']})
    old_metrics=json.loads((OLD/a.dataset/'final_summary.json').read_text())['final_test_metrics'].get(str(a.seed)) if a.variant=='full' and a.estimator=='current' else None
    result={'dataset':a.dataset,'variant':a.variant,'estimator':a.estimator,'seed':a.seed,'status':'completed','metrics':metrics,
            'mean_gate_train':mean,'fit_scope':'all_samples_unlabeled','evaluation_scope':'fixed_test_only','winner_source':str(OLD/a.dataset/'selected.json'),
            'winner_selection_hash':old_selected.get('selection_hash'),'config':asdict(cfg),'config_hash':digest(asdict(cfg)),
            'old_full_metrics':old_metrics,'full_reproduction_delta':({k:metrics[k]-old_metrics[k] for k in ('ari','nmi','acc')} if old_metrics else None),'wall_seconds':time.time()-started}
    atomic_json(done,result); print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
