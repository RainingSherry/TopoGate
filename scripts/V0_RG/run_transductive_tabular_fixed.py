from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, '/home/luolie/ToPoGate')
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from methods.TopoGate.V0_RG.config import V0_RGConfig
from methods.TopoGate.V0_RG.graph import build_pca_knn_graph, compute_edge_reliability
from methods.TopoGate.V0_RG.mixing import compute_node_gate
from methods.TopoGate.V0_RG.run import clustering_metrics
from methods.TopoGate.V0_RG.trainer import fit_predict
from methods.TopoGate.V0_RG.tuning import SplitPreprocessor, split_rows

ROOT = Path('/data/luolie/ToPoGate/result/v0_rg_ablation_0911winners_first26_20260917_tabular_fixed')
OLD = Path('/data/luolie/ToPoGate/result/topogate_transductive_clubench131_scmae16_64x3_20260911/datasets')
DATA = Path('/data/luolie/ToPoGate/datasets')
INPUT_KIND = 'general'  # all 20 first-26 new datasets ran under 'general' in the archive (verified)

def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False))
    tmp.replace(path)

def digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load_npz_xy(name):
    source = DATA / f'{name}.npz'
    if not source.exists():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as z:
        if {'data','indices','indptr','shape'}.issubset(z.files):
            shape = tuple(int(v) for v in np.asarray(z['shape']).reshape(-1))
            X = sp.csr_matrix((np.asarray(z['data']), np.asarray(z['indices'],dtype=np.int64),
                               np.asarray(z['indptr'],dtype=np.int64)), shape=shape)
        else:
            X = next((np.asarray(z[k]) for k in ('X','x','features','data') if k in z.files), None)
        y = next((np.asarray(z[k]).reshape(-1) for k in ('y','labels','label') if k in z.files), None)
    if X is None:
        raise ValueError(f'{name}: no matrix in npz')
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    if y is None:
        raise ValueError(f'{name}: no labels in npz')
    y = LabelEncoder().fit_transform(y.astype(str)).astype(np.int64)
    if X.shape[0] != len(y):
        raise ValueError(f'{name}: X/Y row mismatch')
    return X, y

def archive_winner(name):
    sel = OLD/name/'selected.json'
    if sel.exists():
        d = json.loads(sel.read_text())
        return d, d['winner']['config']
    fs = OLD/name/'final_summary.json'
    d = json.loads(fs.read_text())
    if 'winner' not in d:
        raise ValueError(f'{name}: no winner config in archive')
    return d, d['winner']['config']

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

def archive_old_metrics(name, seed):
    try:
        fs = json.loads((OLD/name/'final_summary.json').read_text())
        fsm = fs.get('final_test_metrics')
        if fsm:
            m = fsm.get(str(seed))
            return m if isinstance(m, dict) and 'ari' in m else None
        ps = fs.get('per_seed')
        if isinstance(ps, dict):
            m = ps.get(str(seed))
            return m if isinstance(m, dict) and 'ari' in m else None
        if isinstance(ps, list) and ps:
            if isinstance(ps[0], dict) and 'seed' in ps[0]:
                for r in ps:
                    if str(r.get('seed'))==str(seed) and isinstance(r.get('ari'),(int,float)):
                        return r
            elif isinstance(ps[0], dict) and 'ari' in ps[0]:
                order=[42,123,7]
                if str(seed) in {str(s) for s in order} and len(ps)>=order.index(seed)+1:
                    return ps[order.index(seed)]
    except Exception:
        return None
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--variant',required=True)
    ap.add_argument('--estimator',default='current'); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--gpu',type=int,required=True)
    a=ap.parse_args()
    if a.gpu in {0,7}: raise SystemExit('GPU 0 and 7 forbidden')
    out=ROOT/'datasets'/a.dataset/a.variant/a.estimator/f'seed_{a.seed}'; done=out/'final_summary.json'
    if done.exists(): return
    out.mkdir(parents=True,exist_ok=True); started=time.time()
    X,y=load_npz_xy(a.dataset); train,val,test=split_rows(len(y),9102026)
    split={'split_seed':9102026,'train':train.tolist(),'validation':val.tolist(),'test':test.tolist(),'test_hash':digest(test.tolist())}
    split_path=ROOT/'datasets'/a.dataset/'split.json'
    if split_path.exists() and json.loads(split_path.read_text())!=split: raise ValueError('split identity changed')
    atomic_json(split_path,split)
    old_selected,base=archive_winner(a.dataset)
    prep=SplitPreprocessor(INPUT_KIND,base['n_top_features']).fit(X); Xall=prep.transform(X)
    full_cfg=V0_RGConfig(**base); mean=full_gate_mean(Xall,full_cfg,a.seed) if a.variant=='fixed_mix_original_aux' else None
    cfg=resolve(base,a.variant,a.estimator,mean)
    pred,embedding,diag=fit_predict(Xall,fit_X=Xall,n_clusters=len(np.unique(y)),config=cfg,seed=a.seed,device='cuda:0')
    metrics=clustering_metrics(y[test],pred[test])
    np.save(out/'embedding_all.npy',embedding); np.save(out/'predictions_all.npy',pred); np.save(out/'test_indices.npy',test); np.save(out/'labels_true_test.npy',y[test])
    np.save(out/'cluster_centers.npy',diag['cluster_centers'])
    np.savez_compressed(out/'preprocessor.npz',features=prep.features,mean=prep.scaler.mean_,scale=prep.scaler.scale_)
    import torch; torch.save(diag['model_state_dict'],out/'model.pt')
    atomic_json(out/'preprocessor.json',{'fit_scope':'all_samples_unlabeled','input_kind':INPUT_KIND,'n_top_features':base['n_top_features'],'fingerprint':prep.fingerprint,'selection':prep.selection,'n_features':len(prep.features)})
    atomic_json(out/'training_history.json',diag['training_history']); atomic_json(out/'graph_gate_summary.json',{'graph':diag['graph_profile'],'edge':diag['edge_summary'],'gate':diag['gate_summary'],'core':diag['core_summary']})
    old_metrics=archive_old_metrics(a.dataset,a.seed) if a.variant=='full' and a.estimator=='current' else None
    arch_fp=None
    fp_ref=OLD/a.dataset/'final'/'seed_42'/'preprocessor_seed_42.json'
    if fp_ref.exists():
        try: arch_fp=json.loads(fp_ref.read_text()).get('fingerprint')
        except Exception: arch_fp=None
    result={'dataset':a.dataset,'variant':a.variant,'estimator':a.estimator,'seed':a.seed,'status':'completed','metrics':metrics,
            'input_kind':INPUT_KIND,'mean_gate_train':mean,'fit_scope':'all_samples_unlabeled','evaluation_scope':'fixed_test_only',
            'winner_source':str(OLD/a.dataset/'selected.json'),'winner_selection_hash':old_selected.get('selection_hash'),
            'archive_config_hash':old_selected.get('winner',{}).get('config_hash'),'archive_preprocessing_fingerprint':arch_fp,
            'config':asdict(cfg),'config_hash':digest(asdict(cfg)),
            'old_full_metrics':old_metrics,'full_reproduction_delta':({k:metrics[k]-old_metrics[k] for k in ('ari','nmi','acc')} if old_metrics else None),
            'wall_seconds':time.time()-started}
    atomic_json(done,result); print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
